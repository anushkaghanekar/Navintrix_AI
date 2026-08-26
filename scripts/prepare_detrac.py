"""Prepare UA-DETRAC data — primary detector/training/eval set for the four
normal vehicle classes (car / motorcycle / bus / truck).

Download (official host): https://detrac-db.rit.albany.edu/
Grab it early and keep a local copy — the official host is occasionally
slow. Roboflow Universe hosts several re-uploaded mirrors as a fallback,
but verify their license/completeness against the original before relying
on one.

Pipeline
--------
1. Discover sequences under --raw-dir. Stock layout:
       <raw>/DETRAC-train-Images/<SEQ>/imgNNNNN.jpg
       <raw>/DETRAC-train-Annotations-XML/<SEQ>.xml
   (same for test). Flattened/remixed mirror layouts are also understood as
   long as images map to a sequence XML by dir name or filename prefix;
   --images-dir / --annotations-dir override discovery for unusual trees.
2. Parse each sequence XML into per-frame box lists — the stock UA-DETRAC
   dialect (<frame num="N"><target_list><target>...</target_list>) plus
   re-uploaded wrapper/direct-target and frame_num-sibling dialects are
   handled.
3. Map DETRAC classes to project classes from configs/model.yaml
   (datasets.detrac.class_mapping), e.g. `van -> truck`. DETRAC's catch-all
   `others` maps to `null`: it is a heterogeneous remainder category, and
   folding it into a modelled class would inject label noise.
4. Write YOLO-format output under --out-dir:
       <out>/images/<train|test>/<SEQ>_<imagename>.jpg  (hardlink or copy)
       <out>/labels/<train|test>/<SEQ>_<imagename>.txt   (one box per line)
   The official train/test split is preserved (inferred from the annotation
   path token, forceable with --split). Sequence prefixes are kept so
   scripts/split_dataset.py can split sequences later without video
   leakage. Empty frames get a legitimately empty label file so
   scripts/validate_dataset.py can tell "no objects" from "missing label".
5. Write <out>/prep_summary.json — class distributions, per-sequence
   frame/box counts, drop map. Numbers in the report come from this file.

Usage:
  python scripts/prepare_detrac.py --raw-dir data/raw/detrac --out-dir data/processed/detrac
  python scripts/prepare_detrac.py --raw-dir data/raw/detrac --out-dir data/processed/detrac --link
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from PIL import Image

try:
    from .convert_annotations import to_yolo_format
except ImportError:  # invoked directly as `python scripts/prepare_detrac.py`
    from convert_annotations import to_yolo_format


SEQUENCE_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ---- XML PARSING ----


def _truncated_tag(tag: str) -> str:
    """Strip any XML namespace prefix: '{url}tag' -> 'tag'."""
    return tag.rsplit("}", 1)[-1]


def _children(elem: ET.Element, tag_name: str) -> list[ET.Element]:
    return [c for c in elem if _truncated_tag(c.tag) == tag_name]


def _find_child(elem: ET.Element, tag_name: str) -> ET.Element | None:
    for c in _children(elem, tag_name):
        return c
    return None


def _vehicle_class(target: ET.Element) -> str | None:
    """DETRAC stores its class on <attribute vehicle="car"/>."""
    attribute = _find_child(target, "attribute")
    if attribute is None:
        return None
    for key in ("vehicle", "vehicle_type", "class", "type"):
        value = attribute.attrib.get(key)
        if value:
            return value
    return None


def _parse_box(target: ET.Element) -> dict | None:
    """Return {class_name, x1, y1, x2, y2} or None if the target has no box."""
    box = _find_child(target, "box")
    if box is None:
        return None

    def coord(*names):
        for name in names:
            if name in box.attrib:
                return float(box.attrib[name])
        return None

    left = coord("left", "x")
    top = coord("top", "y")
    width = coord("width", "w")
    height = coord("height", "h")
    if None in (left, top, width, height) or width <= 0 or height <= 0:
        return None
    return {
        "class_name": _vehicle_class(target),
        "x1": left,
        "y1": top,
        "x2": left + width,
        "y2": top + height,
    }


def parse_detrac_xml(xml_path: Path) -> dict[int, list[dict]]:
    """Parse one DETRAC sequence XML into {frame_number: [boxes, ...]}.

    Stock UA-DETRAC annotations wrap targets as
    `<frame num="N"><target_list><target>...`; several mirrors use direct
    `<frame num="N"><target>...` children or interleave `<frame_num>N</frame_num>`
    with sibling `<target>` elements. All three dialects are handled. Boxes
    use a top-left origin to match YOLO.
    """
    root = ET.parse(str(xml_path)).getroot()
    frames: dict[int, list[dict]] = {}
    last_frame: int | None = None

    def add(target: ET.Element, force_frame: int | None = None) -> None:
        box = _parse_box(target)
        if box is None:
            return
        frame = force_frame if force_frame is not None else last_frame
        if frame is None:
            return
        frames.setdefault(frame, []).append(box)

    for elem in root:
        tag = _truncated_tag(elem.tag)
        if tag == "frame_num":
            text = (elem.text or "").strip()
            if text.isdigit():
                last_frame = int(text)
                frames.setdefault(last_frame, [])
        elif tag == "frame":
            raw = elem.attrib.get("num")
            if raw is None:
                inner = _find_child(elem, "frame_num")
                raw = inner.text if inner is not None else None
            if raw is None or not str(raw).strip().isdigit():
                continue
            frame = int(str(raw).strip())
            frames.setdefault(frame, [])
            for child in elem.iter():  # stock target_list or direct frame targets
                if child is not elem and _truncated_tag(child.tag) == "target":
                    add(child, force_frame=frame)
        elif tag == "target":
            add(elem)  # official dialect: target siblings after a <frame_num>

    return frames


# ---- DISCOVERY ----


def _frame_of_image(image: Path) -> int:
    """Extract DETRAC's frame number from a filename like img00042.jpg."""
    match = re.search(r"\d+", image.stem)
    if match is None:
        raise ValueError(f"cannot extract a frame number from image filename: {image}")
    return int(match.group())


def _match_seq_for_image(image: Path, sequences: dict) -> str | None:
    """Match an image to a sequence by parent dir name, then filename prefix."""
    parent_name = image.parent.name
    if parent_name in sequences:
        return parent_name
    for marker in ("_", "."):
        prefix = image.stem.split(marker, 1)[0]
        if prefix in sequences:
            return prefix
    return None


def infer_split(xml_path: Path, base_dir: Path) -> str | None:
    """Detect 'train'/'test' from the parts of the XML path under base_dir."""
    try:
        parts = xml_path.relative_to(base_dir).parts
    except ValueError:
        parts = (xml_path.name,)
    for part in parts:
        lowered = part.lower()
        if "test" in lowered:
            return "test"
        if "train" in lowered:
            return "train"
    return None


def discover_sequences(
    raw_dir: Path,
    images_dir: Path | None = None,
    annotations_dir: Path | None = None,
) -> tuple[dict[str, dict], list[Path]]:
    """Discover DETRAC sequences.

    Returns ({seq_name: {"xml": Path, "images": [Path,...],
    "split_hint": str|None}}, unmatched_images). Every `.xml` under the
    annotations root (or raw_dir) defines a sequence; every image under the
    images root (or raw_dir) that matches a sequence by parent dir name or
    filename prefix is attached to it.
    """
    raw = Path(raw_dir)
    xml_root = Path(annotations_dir) if annotations_dir is not None else raw
    image_root = Path(images_dir) if images_dir is not None else raw
    if not raw.is_dir():
        raise FileNotFoundError(f"raw dir not found: {raw}")
    if not xml_root.is_dir():
        raise FileNotFoundError(f"annotations dir not found: {xml_root}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"images dir not found: {image_root}")

    xml_files = sorted(p for p in xml_root.rglob("*") if p.suffix.lower() == ".xml")
    sequences: dict[str, dict] = {}
    for xml_path in xml_files:
        sequences[xml_path.stem] = {
            "xml": xml_path,
            "images": [],
            "split_hint": infer_split(xml_path, raw),
        }

    unmatched: list[Path] = []
    for image in image_root.rglob("*"):
        if image.suffix.lower() not in SEQUENCE_IMAGE_EXTS:
            continue
        seq_name = _match_seq_for_image(image, sequences)
        if seq_name is not None:
            sequences[seq_name]["images"].append(image)
        else:
            unmatched.append(image)
    return sequences, unmatched


# ---- CONFIG + OUTPUT HELPERS ----


def load_config(config_path: Path) -> tuple[list[str], dict[str, str | None]]:
    """Return (classes.normal, datasets.detrac.class_mapping) from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    try:
        normal_classes = cfg["classes"]["normal"]
        class_mapping = cfg["datasets"]["detrac"]["class_mapping"]
    except (KeyError, TypeError):
        raise SystemExit(
            "config error: model.yaml (configs/) must define both "
            "'classes.normal' and 'datasets.detrac.class_mapping'"
        )
    if not isinstance(normal_classes, list) or not normal_classes:
        raise SystemExit("config error: classes.normal must be a non-empty list")
    if not isinstance(class_mapping, dict):
        raise SystemExit("config error: datasets.detrac.class_mapping must be a mapping")
    for detrac_class, mapped in class_mapping.items():
        if mapped is not None and mapped not in normal_classes:
            raise SystemExit(
                f"config error: class_mapping value {mapped!r} (for "
                f"{detrac_class!r}) is not one of classes.normal"
            )
    return normal_classes, class_mapping


def emit_image(src: Path, dst: Path, link: bool) -> str:
    """Copy or hardlink an image file; returns the method used."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(src, dst)
            return "hardlinked"
        except OSError as exc:  # e.g. src/dst on different filesystems
            print(f"    (hardlink failed: {exc}; copying instead)")
    shutil.copy2(src, dst)
    return "copied"


def write_label(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


# ---- CORE FLOW ----


def run_conversion(
    raw_dir: Path,
    out_dir: Path,
    config_path: Path,
    link: bool,
    split_override: str | None,
    images_dir: Path | None = None,
    annotations_dir: Path | None = None,
) -> dict:
    """Convert the whole raw DETRAC tree to YOLO layout under out_dir.

    Returns the prep summary dict (also written to <out_dir>/prep_summary.json).
    """
    normal_classes, class_mapping = load_config(config_path)
    raw = Path(raw_dir).resolve()
    out = Path(out_dir).resolve()

    sequences, unmatched = discover_sequences(raw, images_dir, annotations_dir)
    if not sequences:
        raise SystemExit(
            "no DETRAC sequences found. Check --raw-dir (or --images-dir / "
            "--annotations-dir); expected <seq>/imgNNNNN.jpg + <seq>.xml"
        )
    if unmatched:
        print(f"[prep] ignoring {len(unmatched)} images that match no sequence XML")

    normal_ids = {name: i for i, name in enumerate(normal_classes)}
    detrac_to_id: dict[str, int] = {
        detrac_class: normal_ids[mapped]
        for detrac_class, mapped in class_mapping.items()
        if mapped is not None
    }
    dropped_classes = [
        detrac_class for detrac_class, mapped in class_mapping.items() if mapped is None
    ]

    summary: dict = {
        "source_dir": str(raw),
        "output_dir": str(out),
        "config_file": str(Path(config_path).resolve()),
        "classes_normal": normal_classes,
        "class_mapping": class_mapping,
        "classes_kept_counts": {name: 0 for name in normal_classes},
        "classes_dropped_counts": {name: 0 for name in dropped_classes},
        "images_per_split": {"train": 0, "test": 0},
        "sequences": {},
    }

    for seq_name, record in sorted(sequences.items()):
        split = split_override or record["split_hint"] or "train"
        if record["split_hint"] is None and split_override is None:
            print(
                f"  warning: no train/test token in path for {seq_name}; "
                f"treating it as 'train' (use --split to force)"
            )
        xml_frames = parse_detrac_xml(record["xml"])
        seq_summary = {
            "split": split,
            "images": 0,
            "xml_frames": len(xml_frames),
            "annotated_frames": sum(1 for boxes in xml_frames.values() if boxes),
            "boxes": 0,
            "dropped_boxes": 0,
            "link_tally": {"hardlinked": 0, "copied": 0},
        }

        for image in sorted(record["images"], key=lambda p: (_frame_of_image(p), str(p))):
            if not image.is_file():
                raise FileNotFoundError(f"image listed but not readable: {image}")
            with Image.open(image) as opened:
                width, height = opened.size

            frame = _frame_of_image(image)
            boxes: list[dict] = xml_frames.get(frame, [])
            if frame not in xml_frames:
                print(
                    f"  warning: {seq_name} frame {frame} has no XML entry; "
                    f"writing an empty label"
                )

            kept_boxes: list[dict] = []
            for box in boxes:
                class_id = detrac_to_id.get(box["class_name"])
                if class_id is None:
                    seq_summary["dropped_boxes"] += 1
                    dropped_class = box["class_name"] or "(unset)"
                    classes_dropped = summary["classes_dropped_counts"]
                    classes_dropped[dropped_class] = classes_dropped.get(dropped_class, 0) + 1
                else:
                    kept_boxes.append(box)
                    summary["classes_kept_counts"][normal_classes[class_id]] += 1

            new_name = f"{seq_name}_{image.name}"
            method = emit_image(image, out / "images" / split / new_name, link)
            seq_summary["images"] += 1
            seq_summary["link_tally"][method] += 1
            summary["images_per_split"][split] += 1
            seq_summary["boxes"] += len(kept_boxes)

            with_stem = f"{new_name.rsplit('.', 1)[0]}.txt"
            write_label(
                out / "labels" / split / with_stem,
                to_yolo_format(kept_boxes, detrac_to_id, float(width), float(height)),
            )

        summary["sequences"][seq_name] = seq_summary
        print(
            f"[prep] {seq_name:12s} ({split:5s}): {seq_summary['images']:5d} imgs, "
            f"{seq_summary['boxes']:6d} boxes, {seq_summary['dropped_boxes']} dropped"
        )

    (out / "prep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    _print_summary(summary)
    return summary


# ---- CLI REMAINS BELOW ----


def _print_summary(summary: dict) -> None:
    print("\nDataset prep complete.")
    print(f"  images per split : {summary['images_per_split']}")
    print(f"  kept classes     : {summary['classes_kept_counts']}")
    print(f"  dropped classes  : {summary['classes_dropped_counts']}")
    print(f"  prep summary     : {summary['output_dir']}/prep_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare UA-DETRAC: XML annotations -> YOLO txt + JPG layout."
    )
    parser.add_argument("--raw-dir", required=True, help="extracted DETRAC dataset dir")
    parser.add_argument("--out-dir", required=True, help="output YOLO-format dir")
    parser.add_argument("--config", default="configs/model.yaml", help="model config YAML")
    parser.add_argument("--images-dir", default=None, help="override images root")
    parser.add_argument("--annotations-dir", default=None, help="override annotations root")
    parser.add_argument("--link", action="store_true", help="hard-link images (default: copy)")
    parser.add_argument(
        "--split", choices=["train", "test"], default=None,
        help="force every sequence into this split (default: infer from path)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_conversion(
        raw_dir=Path(args.raw_dir),
        out_dir=out_dir,
        config_path=Path(args.config),
        link=args.link,
        split_override=args.split,
        images_dir=Path(args.images_dir) if args.images_dir else None,
        annotations_dir=Path(args.annotations_dir) if args.annotations_dir else None,
    )


if __name__ == "__main__":
    main()
