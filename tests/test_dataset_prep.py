"""Tests for dataset_prep: train/val splitting."""

import tempfile
from pathlib import Path

import numpy as np
import yaml

from gazefy.training.dataset_prep import find_annotated_pairs, split_dataset


def _setup_session(tmp: Path, n_images: int = 10, n_labels: int = 8) -> Path:
    """Create a fake session with images and partial labels."""
    session = tmp / "session"
    (session / "images").mkdir(parents=True)
    (session / "labels").mkdir(parents=True)

    # Create images
    for i in range(n_images):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        import cv2

        cv2.imwrite(str(session / "images" / f"frame_{i:06d}.png"), img)

    # Create labels for first n_labels images
    for i in range(n_labels):
        (session / "labels" / f"frame_{i:06d}.txt").write_text("0 0.5 0.5 0.2 0.1\n")

    # Create dataset.yaml stub
    with open(session / "dataset.yaml", "w") as f:
        yaml.dump({"path": str(session), "names": {0: "button"}}, f)

    return session


def test_find_annotated_pairs():
    with tempfile.TemporaryDirectory() as tmp:
        session = _setup_session(Path(tmp), n_images=5, n_labels=3)
        pairs = find_annotated_pairs(session)
        assert len(pairs) == 3
        # Each pair is (image_path, label_path) with matching stems
        for img, lbl in pairs:
            assert img.stem == lbl.stem


def test_find_annotated_pairs_no_labels():
    with tempfile.TemporaryDirectory() as tmp:
        session = _setup_session(Path(tmp), n_images=5, n_labels=0)
        pairs = find_annotated_pairs(session)
        assert len(pairs) == 0


def test_split_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        session = _setup_session(Path(tmp), n_images=10, n_labels=10)
        summary = split_dataset(session, split_ratio=0.8, seed=42)

        assert summary["train"] == 8
        assert summary["val"] == 2
        assert summary["unannotated"] == 0

        # Check files exist in correct dirs
        assert len(list((session / "images" / "train").glob("*.png"))) == 8
        assert len(list((session / "images" / "val").glob("*.png"))) == 2
        assert len(list((session / "labels" / "train").glob("*.txt"))) == 8
        assert len(list((session / "labels" / "val").glob("*.txt"))) == 2

        # Check dataset.yaml updated
        with open(session / "dataset.yaml") as f:
            ds = yaml.safe_load(f)
        assert ds["train"] == "images/train"
        assert ds["val"] == "images/val"


def test_split_dataset_partial_annotation():
    with tempfile.TemporaryDirectory() as tmp:
        session = _setup_session(Path(tmp), n_images=10, n_labels=6)
        summary = split_dataset(session, split_ratio=0.8, seed=42)

        assert summary["train"] + summary["val"] == 6
        assert summary["unannotated"] == 4  # 10 - 6 images without labels
