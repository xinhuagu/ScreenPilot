#!/usr/bin/env python3
"""Prepare a collected dataset for YOLO training.

Takes a session directory with images/ and labels/ (from annotation tool),
splits into train/val, and updates dataset.yaml.

Usage:
    python -m gazefy.training.dataset_prep datasets/my_app/session_xxx [--split 0.8]

Expected input layout:
    session_dir/
    ├── images/          (flat: frame_000000.png, ...)
    ├── labels/          (flat: frame_000000.txt, ... from annotation export)
    └── dataset.yaml

Output layout:
    session_dir/
    ├── images/
    │   ├── train/       (80% of annotated images)
    │   └── val/         (20% of annotated images)
    ├── labels/
    │   ├── train/       (matching label files)
    │   └── val/
    └── dataset.yaml     (updated paths)
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def find_annotated_pairs(session_dir: Path) -> list[tuple[Path, Path]]:
    """Find image+label pairs. Only includes images that have a label file."""
    images_dir = session_dir / "images"
    labels_dir = session_dir / "labels"

    if not labels_dir.exists():
        return []

    pairs = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        stem = label_file.stem
        # Try common image extensions
        for ext in (".png", ".jpg", ".jpeg"):
            img_file = images_dir / f"{stem}{ext}"
            if img_file.exists():
                pairs.append((img_file, label_file))
                break
    return pairs


def split_dataset(
    session_dir: str | Path,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    """Split annotated images+labels into train/val and update dataset.yaml.

    Returns summary dict with counts.
    """
    session_dir = Path(session_dir)
    images_dir = session_dir / "images"
    labels_dir = session_dir / "labels"

    pairs = find_annotated_pairs(session_dir)
    if not pairs:
        print(f"No annotated pairs found in {session_dir}")
        print("  Expected: images/*.png + labels/*.txt (same stems)")
        print(f"  Have images: {len(list(images_dir.glob('*')))} files")
        if labels_dir.exists():
            print(f"  Have labels: {len(list(labels_dir.glob('*.txt')))} files")
        else:
            print("  labels/ not found")
        return {"train": 0, "val": 0, "unannotated": 0}

    # Shuffle and split
    random.seed(seed)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Create train/val subdirs
    for subdir in ["train", "val"]:
        (images_dir / subdir).mkdir(exist_ok=True)
        (labels_dir / subdir).mkdir(exist_ok=True)

    # Move files (not copy — avoid duplication)
    def move_pairs(pair_list: list[tuple[Path, Path]], split_name: str) -> int:
        for img, lbl in pair_list:
            shutil.move(str(img), str(images_dir / split_name / img.name))
            shutil.move(str(lbl), str(labels_dir / split_name / lbl.name))
        return len(pair_list)

    n_train = move_pairs(train_pairs, "train")
    n_val = move_pairs(val_pairs, "val")

    # Count unannotated images (images without labels, left in images/)
    remaining = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    n_unannotated = len(remaining)

    # Update dataset.yaml
    ds_yaml = session_dir / "dataset.yaml"
    if ds_yaml.exists():
        with open(ds_yaml) as f:
            ds = yaml.safe_load(f) or {}
    else:
        ds = {}

    ds["path"] = str(session_dir.resolve())
    ds["train"] = "images/train"
    ds["val"] = "images/val"
    with open(ds_yaml, "w") as f:
        yaml.dump(ds, f, default_flow_style=False)

    summary = {
        "train": n_train,
        "val": n_val,
        "unannotated": n_unannotated,
        "dataset_yaml": str(ds_yaml),
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset: split into train/val")
    parser.add_argument("session_dir", help="Path to session directory")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        print(f"Directory not found: {session_dir}")
        sys.exit(1)

    print(f"Preparing dataset: {session_dir}")
    print(f"Split ratio: {args.split:.0%} train / {1 - args.split:.0%} val\n")

    summary = split_dataset(session_dir, split_ratio=args.split, seed=args.seed)

    print("Results:")
    print(f"  Train:       {summary['train']} image+label pairs")
    print(f"  Val:         {summary['val']} image+label pairs")
    print(f"  Unannotated: {summary['unannotated']} images (not moved)")
    print(f"  dataset.yaml: {summary['dataset_yaml']}")

    if summary["train"] == 0:
        print("\nNo annotated data found. Annotation workflow:")
        print(f"  1. Import images from {session_dir}/images/ into Label Studio")
        print("  2. Annotate with bounding boxes (classes: button, menu_item, input_field, ...)")
        print(f"  3. Export as YOLO format into {session_dir}/labels/")
        print("  4. Re-run this script")


if __name__ == "__main__":
    main()
