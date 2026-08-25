"""Prepare AI City Challenge 2021 Track 1 data.

Used for counting/movement validation, NOT as primary detector training
data — it's a camera-viewpoint/ROI/movement-annotation dataset, treat it
as such rather than as an ordinary image classification set.

Download: https://www.aicitychallenge.org/2021-track1-download/
(no data-request form needed for this specific dataset as of the org's
current dataset-access page — re-check before assuming that's still true).

Usage: python scripts/prepare_aicity.py --raw-dir data/raw/aicity --out-dir data/processed/aicity
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ANNOTATION_EXTS = {".csv", ".json", ".txt", ".xml", ".yaml", ".yml"}
FRAME_DIR_NAMES = {"frame", "frames", "img", "imgs", "image", "images"}
ROI_MARKERS = ("roi", "region", "polygon", "mask", "calibration")
MOVEMENT_MARKERS = ("movement", "turn", "trajectory", "route", "count")


def _copy_or_link(src: Path, dst: Path, link: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(src, dst)
            return "hardlinked"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copied"


def _safe_id(path: Path) -> str:
    return "__".join(path.parts) if path.parts else "root"


def _media_root(path: Path) -> Path:
    """Return the camera/sequence directory that owns a media file."""
    if path.suffix.lower() in VIDEO_EXTS:
        return path.parent
    if path.parent.name.lower() in FRAME_DIR_NAMES:
        return path.parent.parent
    return path.parent


def _kind(path: Path) -> str:
    lowered = "/".join(part.lower() for part in path.parts)
    if any(marker in lowered for marker in ROI_MARKERS):
        return "roi"
    if any(marker in lowered for marker in MOVEMENT_MARKERS):
        return "movement"
    return "annotation"


def _relative_to_any(path: Path, roots: list[Path]) -> tuple[Path | None, Path | None]:
    for root in sorted(roots, key=lambda p: len(p.parts), reverse=True):
        try:
            return root, path.relative_to(root)
        except ValueError:
            continue
    return None, None


def discover_aicity_units(raw_dir: str | Path) -> tuple[dict[str, dict], list[Path]]:
    """Discover AI City camera/sequence units without flattening files.

    A unit is the closest directory that owns a video file or frame folder.
    Annotation-like files are attached to the nearest unit root when they
    live under it. Unmatched annotations are returned separately so the team
    can decide where they belong instead of silently dropping metadata.
    """
    raw = Path(raw_dir).resolve()
    if not raw.is_dir():
        raise FileNotFoundError(f"raw dir not found: {raw}")

    media_files = sorted(
        p
        for p in raw.rglob("*")
        if p.is_file() and p.suffix.lower() in (VIDEO_EXTS | IMAGE_EXTS)
    )
    if not media_files:
        raise SystemExit(f"no AI City media files found under {raw}")

    by_root: dict[Path, dict] = {}
    for media in media_files:
        root = _media_root(media)
        rel_root = root.relative_to(raw)
        unit = by_root.setdefault(
            root,
            {
                "id": _safe_id(rel_root),
                "source_root": root,
                "relative_root": rel_root,
                "media": [],
                "annotations": [],
                "roi_files": [],
                "movement_files": [],
            },
        )
        unit["media"].append(media)

    roots = list(by_root)
    unmatched_annotations: list[Path] = []
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ANNOTATION_EXTS:
            continue
        if path.suffix.lower() in IMAGE_EXTS:
            continue
        root, _ = _relative_to_any(path, roots)
        if root is None:
            unmatched_annotations.append(path)
            continue
        kind = _kind(path)
        if kind == "roi":
            by_root[root]["roi_files"].append(path)
        elif kind == "movement":
            by_root[root]["movement_files"].append(path)
        else:
            by_root[root]["annotations"].append(path)

    return {unit["id"]: unit for unit in by_root.values()}, unmatched_annotations


def prepare_aicity(
    raw_dir: str | Path,
    out_dir: str | Path,
    *,
    link: bool = False,
) -> dict:
    """Copy/link AI City data into a manifest-preserving layout.

    Output layout:
      <out>/sequences/<unit-id>/media/...        videos or frame images
      <out>/sequences/<unit-id>/annotations/...  gt/labels/etc.
      <out>/sequences/<unit-id>/metadata.json    per-camera manifest
      <out>/prep_summary.json                    project-level summary

    The unit-id is derived from the raw relative camera path, e.g.
    S01/c001 -> S01__c001, so sequence/camera identity survives.
    """
    raw = Path(raw_dir).resolve()
    out = Path(out_dir).resolve()
    units, unmatched = discover_aicity_units(raw)

    summary = {
        "source_dir": str(raw),
        "output_dir": str(out),
        "units": {},
        "unit_count": len(units),
        "media_files": 0,
        "annotation_files": 0,
        "roi_files": 0,
        "movement_files": 0,
        "unmatched_annotations": [str(p.relative_to(raw)) for p in unmatched],
        "link_tally": {"hardlinked": 0, "copied": 0},
    }

    for unit_id, unit in sorted(units.items()):
        unit_out = out / "sequences" / unit_id
        manifest = {
            "id": unit_id,
            "source_root": str(unit["source_root"]),
            "relative_root": str(unit["relative_root"]),
            "media": [],
            "annotations": [],
            "roi_files": [],
            "movement_files": [],
        }

        for bucket_name, dst_bucket in (
            ("media", "media"),
            ("annotations", "annotations"),
            ("roi_files", "annotations"),
            ("movement_files", "annotations"),
        ):
            for src in sorted(unit[bucket_name]):
                rel = src.relative_to(unit["source_root"])
                dst = unit_out / dst_bucket / rel
                method = _copy_or_link(src, dst, link)
                summary["link_tally"][method] += 1
                manifest[bucket_name].append(str(rel))

        summary["media_files"] += len(manifest["media"])
        summary["annotation_files"] += len(manifest["annotations"])
        summary["roi_files"] += len(manifest["roi_files"])
        summary["movement_files"] += len(manifest["movement_files"])
        (unit_out / "metadata.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        summary["units"][unit_id] = manifest
        print(
            f"[prep-aicity] {unit_id}: {len(manifest['media'])} media, "
            f"{len(manifest['annotations'])} annotations, "
            f"{len(manifest['roi_files'])} roi, "
            f"{len(manifest['movement_files'])} movement"
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "prep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[prep-aicity] wrote {out / 'prep_summary.json'}")
    return summary


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Prepare AI City Track 1 data while preserving camera/video metadata."
    )
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--link", action="store_true", help="hard-link files instead of copying")
    args = parser.parse_args(argv)
    prepare_aicity(args.raw_dir, args.out_dir, link=args.link)


if __name__ == "__main__":
    main()
