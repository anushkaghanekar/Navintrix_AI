"""Prepare the emergency vehicle dataset (ambulance / fire_truck /
police_vehicle).

No single clean public dataset covers all three classes well — expect to
combine a few small Roboflow Universe sets plus manual curation. Document
in the report, per source: name, license, classes covered, image/video
count, annotation format. Budget this as its own multi-day task, not a
subtask of prepare_detrac.py.

Pipeline
--------
1. Discover sources: every immediate subdirectory of --raw-dir is treated as
   one self-contained export. Sources are expected in the standard YOLO
   layout (what Roboflow ships): a class-name descriptor (classes.txt /
   _classes.txt listing one name per line in class-id order, or data.yaml
   with a ``names:`` list or {id: name} map) plus per-image ``*.txt`` labels
   in normalized YOLO form:
       <class_id> <cx> <cy> <w> <h>      (all in [0, 1])
2. Resolve class-name conflicts through configs/model.yaml's
   ``datasets.emergency.class_mapping`` (source-name -> target-name | null).
   A name folded to ``null`` is dropped; any target must be in
   ``classes.emergency``. A source name not listed falls back to identity:
   kept at its own index if it is one of the emergency classes, dropped
   otherwise (a foreign class we have no mapping for must never be silently
   merged into a target). This is where synonymous labels across sources
   (e.g. "police" vs "police_car") are reconciled without editing the script.
3. Preserve each source's own train/valid/test split instead of collapsing
   it: a label under a "train"/"valid"/"val" folder stays in the training/
   validation fold, "test" stays in test, so no validation or test image
   leaks into training. No split token -> "train".
4. Rewrite label class ids to the target order ``classes.emergency`` (0..N-1)
   and emit one consistent YOLO tree:
       <out>/images/<split>/<source>_<stem>.<ext>
       <out>/labels/<split>/<source>_<stem>.txt
   A label with no paired image is reported as unmatched (never silently
   dropped); a file that fails to parse as YOLO is treated as metadata and
   skipped; a parsed box with an unmapped class is dropped. Files with no
   kept boxes get a legitimately empty label so validate_dataset.py can tell
   "no objects" from "missing label".
5. Write <out>/prep_summary.json — per-source and per-class counts, link/
   copy tally. Numbers in the report come from this file.
6. Write <out>/data.yaml listing the target emergency classes in id order so
   a later fine-tune can point its data config directly at this dataset.

Usage:
  python scripts/prepare_emergency.py --raw-dir data/raw/emergency --out-dir data/processed/emergency
  python scripts/prepare_emergency.py --raw-dir data/raw/emergency --out-dir data/processed/emergency --link
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

try:
    from .prepare_detrac import emit_image
except ImportError:  # invoked directly as `python scripts/prepare_emergency.py`
    from prepare_detrac import emit_image


DEFAULT_CONFIG_PATH = "configs/model.yaml"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_CLASS_DESCRIPTOR_NAMES = {
    "classes.txt", "_classes.txt", "classes.yaml", "_classes.yaml", "classes.yml",
    "data.yaml", "_data.yaml", "data.yml",
}
_SPLIT_TO_DIR = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


# ---- CONFIG ----


def _load_emergency_config(config_path: str | Path) -> tuple[list[str], dict[str, str | None]]:
    """Return (classes.emergency, datasets.emergency.class_mapping).

    ``class_mapping`` is optional and, when absent, falls back to identity
    (ignored names are resolved by ``_target_for``).
    """
    with open(Path(config_path)) as f:
        cfg = yaml.safe_load(f) or {}
    try:
        emergency = cfg["classes"]["emergency"]
    except (KeyError, TypeError):
        raise SystemExit(f"config error: {config_path} must define 'classes.emergency'")
    if not isinstance(emergency, list) or not emergency:
        raise SystemExit(f"config error: {config_path}: classes.emergency must be a non-empty list")
    emergency = [str(name) for name in emergency]

    mapping: dict[str, str | None] = {}
    dataset_cfg = (cfg.get("datasets") or {}).get("emergency") or {}
    raw_mapping = (dataset_cfg or {}).get("class_mapping") or {}
    if not isinstance(raw_mapping, dict):
        raise SystemExit(
            f"config error: datasets.emergency.class_mapping must be a mapping "
            f"(source-class-name -> target-class-name | null)"
        )
    for source_name, target in raw_mapping.items():
        if target is None:
            mapping[str(source_name)] = None
            continue
        target = str(target)
        if target not in emergency:
            raise SystemExit(
                f"config error: class_mapping target {target!r} (for {source_name!r}) "
                f"is not one of classes.emergency"
            )
        mapping[str(source_name)] = target
    return emergency, mapping


def _target_for(source_name: str, emergency: list[str], mapping: dict[str, str | None]) -> int | None:
    """Return the target id (0..N) for a source class name, or None to drop."""
    if source_name in mapping:
        target = mapping[source_name]
        return None if target is None else emergency.index(target)
    if source_name in emergency:
        return emergency.index(source_name)
    return None


# ---- SOURCE DISCOVERY ----


def _parse_classes_desc(path: Path) -> list[str]:
    """Class-name list in id order from a YAML ``names:`` or one-name-per-line
    text descriptor."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError:
            data = {}
        if isinstance(data, dict):
            names = data.get("names")
            if isinstance(names, list):
                return [str(n) for n in names if str(n).strip()]
            if isinstance(names, dict):  # {id: name}
                return [str(names[k]) for k in sorted(names) if str(names[k]).strip()]
        # Fall through to bare-line parsing for yaml without a names key.
    lines = []
    for ln in text.splitlines():
        ln = ln.split("#")[0].strip()
        if ln and not ln.startswith(("-", "{", "}")):
            lines.append(ln)
    if not lines:
        raise ValueError(f"no class names found in {path}")
    return lines


def _find_classes_desc(source_root: Path) -> Path | None:
    for name in sorted(source_root.rglob("*"), key=lambda p: (len(p.parts), p.name)):
        if name.is_file() and name.name in _CLASS_DESCRIPTOR_NAMES:
            return name
    return None


def _looks_like_yolo_label(path: Path, source_names: list[str]) -> bool:
    """True if a file is a per-image YOLO label (>=5 numeric tokens per line)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    non_blank = [ln for ln in lines if ln.strip()]
    if not non_blank:
        return True  # empty label file is legitimately a label
    for ln in non_blank:
        parts = ln.split()
        if len(parts) < 5:
            return False
        try:
            class_id = int(float(parts[0]))
            floats = [float(p) for p in parts]
        except ValueError:
            return False
        if not (0 <= class_id < len(source_names)):
            return False
        if not all(0.0 <= v <= 1.0 for v in floats[1:]):
            return False
    return True


def _split_of(label_path: Path, source_root: Path) -> str:
    try:
        parts = label_path.relative_to(source_root).parts
    except ValueError:
        parts = (label_path.name,)
    for part in parts:
        key = part.lower()
        if key in _SPLIT_TO_DIR:
            return _SPLIT_TO_DIR[key]
    return "train"


def discover_sources(raw_dir: Path) -> dict[str, list[Path]]:
    """Map source-name -> YOLO label files for every usable source under raw_dir."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw dir not found: {raw_dir}")
    sources: dict[str, list[Path]] = {}
    for child in sorted(raw_dir.iterdir()):
        if not child.is_dir():
            continue
        desc = _find_classes_desc(child)
        if desc is None:
            print(f"[prep-emergency] skipping {child.name}: no classes descriptor found")
            continue
        try:
            names = _parse_classes_desc(desc)
        except ValueError as exc:
            print(f"[prep-emergency] skipping {child.name}: {exc}")
            continue
        labels = sorted(
            p for p in child.rglob("*.txt")
            if p.is_file() and p != desc and _looks_like_yolo_label(p, names)
        )
        if not labels:
            print(f"[prep-emergency] skipping {child.name}: no YOLO-format labels found")
            continue
        sources[child.name] = labels
    return sources


# ---- LABEL REWRITING ----


def _rewrite_labels(
    text: str,
    source_names: list[str],
    emergency: list[str],
    mapping: dict[str, str | None],
) -> tuple[str, dict[str, int], dict[str, int]]:
    """Rewrite a YOLO label body to target class ids.

    Returns (rewritten text, kept_count_by_class, dropped_count_by_class).
    A box whose source class is unmapped/dropped contributes no line and is
    counted in ``dropped`` (not silently re-typed).
    """
    kept: dict[str, int] = {}
    dropped: dict[str, int] = {}
    out_lines: list[str] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        class_id = int(float(parts[0]))
        source_class = source_names[class_id]
        target = _target_for(source_class, emergency, mapping)
        if target is None:
            dropped[source_class] = dropped.get(source_class, 0) + 1
            continue
        kept[emergency[target]] = kept.get(emergency[target], 0) + 1
        out_lines.append(f"{target} {' '.join(parts[1:])}")
    out_text = "\n".join(out_lines)
    return (out_text + "\n") if out_lines else "", kept, dropped


def _build_image_index(source_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for p in sorted(source_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.stem, []).append(p)
    return index


def _pair_image(label: Path, image_index: dict[str, list[Path]]) -> Path | None:
    cands = image_index.get(label.stem, [])
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    for c in cands:  # prefer same directory, then shortest relative distance
        if c.parent == label.parent:
            return c
    # ---- CORE FLOW ----


def run_prep(
    raw_dir: str | Path,
    out_dir: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    link: bool = False,
) -> dict:
    """Convert a tree of YOLO-format emergency sources into one consistent
    YOLO dataset; returns (and writes) the prep summary."""
    emergency, mapping = _load_emergency_config(config_path)
    raw = Path(raw_dir).resolve()
    out = Path(out_dir).resolve()
    sources = discover_sources(raw)
    if not sources:
        raise SystemExit(
            "no usable emergency sources found. Check --raw-dir: each source "
            "needs a classes descriptor and YOLO-format *.txt labels"
        )

    summary = {
        "source_dir": str(raw),
        "output_dir": str(out),
        "config_file": str(Path(config_path).resolve()),
        "classes_emergency": emergency,
        "class_mapping": mapping,
        "classes_kept_counts": {name: 0 for name in emergency},
        "classes_dropped_counts": {},
        "split_counts": {"train": 0, "valid": 0, "test": 0},
        "per_source": {},
        "link_tally": {"hardlinked": 0, "copied": 0},
    }
    dropped_total = summary["classes_dropped_counts"]
    dictref = summary["per_source"]

    for source_name, label_files in sorted(sources.items()):
        source_root = raw / source_name
        source_names = _parse_classes_desc(_find_classes_desc(source_root))
        image_index = _build_image_index(source_root)
        source_record = {
            "labels": 0,
            "images": 0,
            "unmatched_labels": 0,
            "boxes": 0,
            "dropped_boxes": 0,
            "split_counts": {"train": 0, "valid": 0, "test": 0},
        }

        for label in label_files:
            split = _split_of(label, source_root)
            image = _pair_image(label, image_index)
            if image is None:
                source_record["unmatched_labels"] += 1
                continue
            text, kept, dropped = _rewrite_labels(
                label.read_text(encoding="utf-8", errors="replace"),
                source_names,
                emergency,
                mapping,
            )
            for cls, count in kept.items():
                summary["classes_kept_counts"][cls] += count
            for cls, count in dropped.items():
                dropped_total[cls] = dropped_total.get(cls, 0) + count

            new_stem = f"{source_name}_{image.stem}"
            method = emit_image(
                image,
                out / "images" / split / f"{new_stem}{image.suffix.lower()}",
                link,
            )
            summary["link_tally"][method] = summary["link_tally"].get(method, 0) + 1
            dst_label = out / "labels" / split / f"{new_stem}.txt"
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text(text, encoding="utf-8")

            source_record["labels"] += 1
            source_record["images"] += 1
            source_record["boxes"] += sum(kept.values())
            source_record["dropped_boxes"] += sum(dropped.values())
            source_record["split_counts"][split] += 1
            summary["split_counts"][split] += 1

        dictref[source_name] = source_record
        print(
            f"[prep-emergency] {source_name}: {source_record['images']} imgs, "
            f"{source_record['labels']} labels, {source_record['boxes']} boxes, "
            f"{source_record['dropped_boxes']} dropped, "
            f"{source_record['unmatched_labels']} unmatched labels"
        )

    (out / "prep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out / "data.yaml").write_text(
        "names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(emergency)) + "\n",
        encoding="utf-8",
    )
    _print_summary(summary)
    return summary


def _print_summary(summary: dict) -> None:
    print("\nEmergency dataset prep complete.")
    print(f"  split counts    : {summary['split_counts']}")
    print(f"  kept classes    : {summary['classes_kept_counts']}")
    print(f"  dropped classes : {summary['classes_dropped_counts']}")
    print(f"  prep summary    : {summary['output_dir']}/prep_summary.json")
    print(f"  data.yaml       : {summary['output_dir']}/data.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge Roboflow-style emergency datasets into one YOLO-format layout."
    )
    parser.add_argument("--raw-dir", required=True, help="tree of one source dataset per child dir")
    parser.add_argument("--out-dir", required=True, help="output YOLO-format dir")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="model config YAML")
    parser.add_argument("--link", action="store_true", help="hard-link images (default: copy)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_prep(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        config_path=Path(args.config),
        link=args.link,
    )


if __name__ == "__main__":
    main()