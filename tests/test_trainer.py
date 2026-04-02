"""Tests for PackTrainer (packaging only — training requires GPU + ultralytics)."""

import tempfile
from pathlib import Path

import yaml

from gazefy.training.trainer import PackTrainer, TrainConfig, TrainResult


def test_package_pack_creates_structure():
    with tempfile.TemporaryDirectory() as tmp:
        # Create a fake trained model file
        model_path = Path(tmp) / "fake_model.pt"
        model_path.write_bytes(b"fake model data")

        config = TrainConfig(
            dataset_yaml="",
            imgsz=1024,
        )
        trainer = PackTrainer(config)

        result = TrainResult(best_model_path=str(model_path))
        pack_dir = trainer.package_pack(
            train_result=result,
            pack_name="test_app",
            output_dir=str(Path(tmp) / "packs"),
            labels=["button", "input_field", "menu_item"],
            window_match=["TestApp"],
        )

        assert pack_dir.exists()
        assert (pack_dir / "pack.yaml").exists()
        assert (pack_dir / "model.pt").exists()
        assert (pack_dir / "workflows").is_dir()

        # Verify pack.yaml content
        with open(pack_dir / "pack.yaml") as f:
            meta = yaml.safe_load(f)
        assert meta["name"] == "test_app"
        assert meta["labels"] == ["button", "input_field", "menu_item"]
        assert meta["window_match"] == ["TestApp"]
        assert meta["input_size"] == 1024


def test_package_pack_missing_model_warns(caplog):
    with tempfile.TemporaryDirectory() as tmp:
        config = TrainConfig()
        trainer = PackTrainer(config)
        result = TrainResult(best_model_path="/nonexistent/model.pt")

        pack_dir = trainer.package_pack(
            train_result=result,
            pack_name="no_model",
            output_dir=str(Path(tmp) / "packs"),
            labels=["a"],
        )
        assert (pack_dir / "pack.yaml").exists()
        assert not (pack_dir / "model.pt").exists()


def test_read_labels_from_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        ds_yaml = Path(tmp) / "dataset.yaml"
        data = {
            "path": tmp,
            "train": "images",
            "val": "images",
            "names": {0: "button", 1: "menu", 2: "input"},
        }
        with open(ds_yaml, "w") as f:
            yaml.dump(data, f)

        config = TrainConfig(dataset_yaml=str(ds_yaml))
        trainer = PackTrainer(config)
        labels = trainer._read_labels_from_dataset()
        assert labels == ["button", "menu", "input"]
