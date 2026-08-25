from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from PIL import Image, ImageDraw


def validate_dataset(
    data_dir: str | Path,
    sample_visual_checks: int = 20,
    out_visual_dir: str | Path | None = None,
    config_path: str | Path = "configs/model.yaml",
) -> dict:
    """Validate a YOLO format dataset:
    - Check all images are readable
    - Check matching label files exist
    - Verify bounding boxes are normalized in [0, 1]
    - Compute class distribution
    - Generate visual sample checks
    """
    data_path = Path(data_dir).resolve()
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    images = sorted(
        p for p in data_path.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    summary = {
        "data_dir": str(data_path),
        "total_images": len(images),
        "valid_images": 0,
        "corrupt_images": 0,
        "missing_labels": 0,
        "total_boxes": 0,
        "invalid_boxes": 0,
        "class_distribution": {},
        "errors": [],
    }

    if not images:
        summary["errors"].append("No images found in dataset directory.")
        return summary

    valid_annotated_samples = []

    for img_path in images:
        # Check image integrity
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                w, h = img.size
            summary["valid_images"] += 1
        except Exception as exc:
            summary["corrupt_images"] += 1
            summary["errors"].append(f"Corrupt image {img_path}: {exc}")
            continue

        # Look for matching label file
        # Common YOLO patterns: images/<split>/<name>.jpg -> labels/<split>/<name>.txt
        parts = list(img_path.parts)
        if "images" in parts:
            idx = parts.index("images")
            parts[idx] = "labels"
            label_path = Path(*parts).with_suffix(".txt")
        else:
            label_path = img_path.with_suffix(".txt")

        if not label_path.is_file():
            summary["missing_labels"] += 1
            summary["errors"].append(f"Missing label file for {img_path}")
            continue

        # Read label boxes
        content = label_path.read_text(encoding="utf-8").strip()
        boxes = []
        if content:
            for line_no, line in enumerate(content.splitlines(), 1):
                tokens = line.strip().split()
                if len(tokens) != 5:
                    summary["invalid_boxes"] += 1
                    summary["errors"].append(f"{label_path}:{line_no} invalid token count")
                    continue
                try:
                    cls_id = int(tokens[0])
                    cx, cy, bw, bh = (float(v) for v in tokens[1:])
                except ValueError:
                    summary["invalid_boxes"] += 1
                    summary["errors"].append(f"{label_path}:{line_no} non-numeric values")
                    continue

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    summary["invalid_boxes"] += 1
                    summary["errors"].append(f"{label_path}:{line_no} box out of [0, 1] bounds: {tokens}")
                    continue

                summary["total_boxes"] += 1
                cls_key = str(cls_id)
                summary["class_distribution"][cls_key] = summary["class_distribution"].get(cls_key, 0) + 1
                boxes.append((cls_id, cx, cy, bw, bh))

        if boxes:
            valid_annotated_samples.append((img_path, boxes))

    # Save visual verification images
    if sample_visual_checks > 0 and valid_annotated_samples:
        vis_dir = Path(out_visual_dir) if out_visual_dir else (data_path / "_visual_check")
        vis_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(42)
        samples = rng.sample(valid_annotated_samples, min(sample_visual_checks, len(valid_annotated_samples)))

        for i, (img_path, boxes) in enumerate(samples):
            try:
                with Image.open(img_path).convert("RGB") as img:
                    draw = ImageDraw.Draw(img)
                    iw, ih = img.size
                    for cls_id, cx, cy, bw, bh in boxes:
                        x1 = (cx - bw / 2.0) * iw
                        y1 = (cy - bh / 2.0) * ih
                        x2 = (cx + bw / 2.0) * iw
                        y2 = (cy + bh / 2.0) * ih
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                        draw.text((x1 + 2, max(0, y1 - 10)), f"cls:{cls_id}", fill="yellow")
                    img.save(vis_dir / f"check_{i:03d}_{img_path.name}")
            except Exception:
                pass

    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset and sanity-check labels.")
    parser.add_argument("--data-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--sample-visual-checks", type=int, default=20, help="Number of visual check images to draw")
    parser.add_argument("--out-visual-dir", default=None, help="Output directory for visual checks")
    args = parser.parse_args()

    summary = validate_dataset(
        data_dir=args.data_dir,
        sample_visual_checks=args.sample_visual_checks,
        out_visual_dir=args.out_visual_dir,
    )
    print("\nDataset Validation Summary:")
    print(f"  Total images      : {summary['total_images']}")
    print(f"  Valid images      : {summary['valid_images']}")
    print(f"  Corrupt images    : {summary['corrupt_images']}")
    print(f"  Missing labels    : {summary['missing_labels']}")
    print(f"  Total boxes       : {summary['total_boxes']}")
    print(f"  Invalid boxes     : {summary['invalid_boxes']}")
    print(f"  Class distribution: {summary['class_distribution']}")
    if summary["errors"]:
        print(f"  Errors encountered: {len(summary['errors'])} (showing first 5)")
        for err in summary["errors"][:5]:
            print(f"    - {err}")
    else:
        print("  All sanity checks PASSED.")


if __name__ == "__main__":
    main()

