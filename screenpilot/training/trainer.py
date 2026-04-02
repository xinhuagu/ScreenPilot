"""PackTrainer: wraps Ultralytics training with VDI-specific augmentations.

Produces a trained model that can be packaged into an ApplicationPack.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    dataset_yaml: str = ""
    base_model: str = "yolov8m.pt"
    epochs: int = 50
    imgsz: int = 1024
    batch: int = 8
    device: str = "mps"  # mps, cuda, cpu
    project: str = "runs/detect"
    name: str = "train"
    # VDI augmentation
    vdi_augment: bool = True
    jpeg_quality_range: tuple[int, int] = (30, 80)


@dataclass
class TrainResult:
    """Result of a training run."""

    best_model_path: str = ""
    metrics: dict = field(default_factory=dict)
    epochs_completed: int = 0


class PackTrainer:
    """Trains a YOLO model and packages it into an ApplicationPack."""

    def __init__(self, config: TrainConfig | None = None):
        self._config = config or TrainConfig()

    def train(self) -> TrainResult:
        """Run training using Ultralytics API. Requires 'ml' extra installed."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "Ultralytics not installed. Install with: pip install screenpilot[ml]"
            )

        if not self._config.dataset_yaml:
            raise ValueError("dataset_yaml must be set in TrainConfig")

        logger.info(
            "Starting training: model=%s, epochs=%d, imgsz=%d, device=%s",
            self._config.base_model,
            self._config.epochs,
            self._config.imgsz,
            self._config.device,
        )

        model = YOLO(self._config.base_model)
        results = model.train(
            data=self._config.dataset_yaml,
            epochs=self._config.epochs,
            imgsz=self._config.imgsz,
            batch=self._config.batch,
            device=self._config.device,
            project=self._config.project,
            name=self._config.name,
        )

        best_path = Path(self._config.project) / self._config.name / "weights" / "best.pt"
        metrics = {}
        if hasattr(results, "results_dict"):
            metrics = dict(results.results_dict)

        return TrainResult(
            best_model_path=str(best_path),
            metrics=metrics,
            epochs_completed=self._config.epochs,
        )

    def package_pack(
        self,
        train_result: TrainResult,
        pack_name: str,
        output_dir: str = "packs",
        labels: list[str] | None = None,
        window_match: list[str] | None = None,
    ) -> Path:
        """Package a trained model into an ApplicationPack directory."""
        pack_dir = Path(output_dir) / pack_name
        pack_dir.mkdir(parents=True, exist_ok=True)

        # Copy model
        src_model = Path(train_result.best_model_path)
        if src_model.exists():
            shutil.copy2(src_model, pack_dir / "model.pt")
        else:
            logger.warning("Model file not found: %s", src_model)

        # Resolve labels
        if labels is None:
            labels = self._read_labels_from_dataset()

        # Write pack.yaml
        pack_meta = {
            "name": pack_name,
            "version": "0.1.0",
            "description": f"Auto-generated pack for {pack_name}",
            "model_file": "model.pt",
            "labels": labels,
            "window_match": window_match or [],
            "input_size": self._config.imgsz,
            "conf_threshold": 0.5,
            "iou_threshold": 0.45,
        }
        with open(pack_dir / "pack.yaml", "w") as f:
            yaml.dump(pack_meta, f, default_flow_style=False)

        # Create optional dirs
        (pack_dir / "workflows").mkdir(exist_ok=True)

        logger.info("Packaged ApplicationPack '%s' at %s", pack_name, pack_dir)
        return pack_dir

    def _read_labels_from_dataset(self) -> list[str]:
        """Read label names from dataset.yaml."""
        if not self._config.dataset_yaml:
            return []
        try:
            with open(self._config.dataset_yaml) as f:
                data = yaml.safe_load(f) or {}
            names = data.get("names", {})
            if isinstance(names, dict):
                return [names[k] for k in sorted(names.keys())]
            return list(names)
        except Exception:
            logger.warning("Could not read labels from %s", self._config.dataset_yaml)
            return []
