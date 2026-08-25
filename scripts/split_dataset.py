from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import yaml


import re


def extract_sequence_name(filename: str) -> str:
    """Extract sequence prefix from image filename (e.g. 'MVI_20011_img00001.jpg' -> 'MVI_20011')."""
    stem = Path(filename).stem
    # Match standard patterns like <seq>_img00001, <seq>_frame_00001, <seq>_00001
    m = re.match(r"^(.+?)(?:_(?:img|frame_?)\d*|_\d+)$", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    if "_" in stem:
        parts = stem.rsplit("_", 1)
        if re.search(r"\d", parts[1]):
            return parts[0]
    return stem


def split_dataset(
    data_dir: str | Path,
    out_dir: str | Path | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    class_names: list[str] | None = None,
) -> dict[str, list[str]]:
    """Split dataset images and labels into train/val/test by SEQUENCE to avoid video leakage.

    Returns a dict mapping 'train', 'val', 'test' to lists of sequence names assigned to each.
    """
    if train_frac <= 0 or val_frac < 0 or (train_frac + val_frac) > 1.0:
        raise ValueError(f"Invalid split fractions: train={train_frac}, val={val_frac}")

    data_path = Path(data_dir).resolve()
    target_out = Path(out_dir).resolve() if out_dir else data_path

    # Discover all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [
        p for p in data_path.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
        and not any(part in ("train", "val", "test") for part in p.parts[:-1])
    ]
    # If already under train/val/test, gather them from everywhere
    if not all_images:
        all_images = [
            p for p in data_path.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

    if not all_images:
        raise FileNotFoundError(f"No images found under {data_path}")

    # Group images and labels by sequence name
    seq_to_files: dict[str, list[Path]] = {}
    for img_path in all_images:
        seq = extract_sequence_name(img_path.name)
        seq_to_files.setdefault(seq, []).append(img_path)

    sequences = sorted(seq_to_files.keys())
    rng = random.Random(seed)
    shuffled_seqs = list(sequences)
    rng.shuffle(shuffled_seqs)

    n_total = len(shuffled_seqs)
    n_train = max(1, int(round(n_total * train_frac))) if n_total > 1 else 1
    n_val = int(round(n_total * val_frac)) if n_total > 2 else (1 if n_total == 2 else 0)
    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)

    train_seqs = set(shuffled_seqs[:n_train])
    val_seqs = set(shuffled_seqs[n_train : n_train + n_val])
    test_seqs = set(shuffled_seqs[n_train + n_val :])
    # If test_seqs is empty and there are leftover items, allocate to test
    if not test_seqs and (n_total - n_train - n_val) > 0:
        test_seqs = set(shuffled_seqs[n_train + n_val :])

    splits = {
        "train": sorted(train_seqs),
        "val": sorted(val_seqs),
        "test": sorted(test_seqs),
    }

    # Verify no video leakage across splits
    assert train_seqs.isdisjoint(val_seqs), "Leakage detected between train and val"
    assert train_seqs.isdisjoint(test_seqs), "Leakage detected between train and test"
    assert val_seqs.isdisjoint(test_seqs), "Leakage detected between val and test"

    # Organize files into split subdirectories
    for split_name, seq_set in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
        img_out = target_out / "images" / split_name
        lbl_out = target_out / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for seq in seq_set:
            for src_img in seq_to_files.get(seq, []):
                dst_img = img_out / src_img.name
                if src_img.resolve() != dst_img.resolve():
                    shutil.copy2(src_img, dst_img)

                # Matching label file (.txt)
                # Look in corresponding labels folder or adjacent
                label_candidates = [
                    src_img.parent.parent / "labels" / src_img.parent.name / f"{src_img.stem}.txt",
                    src_img.parent.parent / "labels" / f"{src_img.stem}.txt",
                    src_img.parent / f"{src_img.stem}.txt",
                    data_path / "labels" / f"{src_img.stem}.txt",
                ]
                for cand in label_candidates:
                    if cand.is_file():
                        shutil.copy2(cand, lbl_out / f"{src_img.stem}.txt")
                        break
                else:
                    # If empty label, create empty file if not found
                    dst_lbl = lbl_out / f"{src_img.stem}.txt"
                    if not dst_lbl.exists():
                        dst_lbl.write_text("", encoding="utf-8")

    # Generate data.yaml for YOLO
    yaml_content = {
        "path": str(target_out),
        "train": f"images/train",
        "val": f"images/val" if val_seqs else "images/train",
        "test": f"images/test" if test_seqs else None,
        "names": {i: name for i, name in enumerate(class_names)} if class_names else {},
    }
    with open(target_out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False)

    return splits


def main():
    parser = argparse.ArgumentParser(description="Split dataset by video sequence without leakage.")
    parser.add_argument("--data-dir", required=True, help="Input dataset directory")
    parser.add_argument("--out-dir", default=None, help="Output directory (defaults to data-dir)")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = split_dataset(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print("Dataset split complete (leakage-free):")
    for name, seqs in splits.items():
        print(f"  {name:5s}: {len(seqs)} sequences -> {seqs}")


if __name__ == "__main__":
    main()

