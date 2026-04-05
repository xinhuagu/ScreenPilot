"""ApplicationPack: self-contained per-application workspace.

Each pack is a complete, portable directory:

    packs/my_app/
    ├── pack.yaml              # Config: labels, window_match, thresholds
    ├── model.pt               # Current best model (copy of latest in models/)
    ├── icon_labels.json       # Accumulated icon semantic dictionary
    ├── element_registry.json  # Persistent element semantic identity (normalized coords)
    ├── recordings/            # All video recordings for this app
    │   ├── 20260404_103000/
    │   │   ├── video.mp4
    │   │   ├── events.jsonl
    │   │   └── annotations.jsonl
    │   └── 20260404_110500/
    ├── training_data/         # Accumulated YOLO training data
    │   ├── images/
    │   ├── labels/
    │   └── dataset.yaml
    ├── models/                # All trained models with timestamps
    │   ├── model_20260404_120000.pt
    │   └── model_20260404_150000.pt
    ├── knowledge/             # Downloaded HTML manuals for this app
    │   ├── html/              # Official docs (wget --recursive)
    │   └── readthedocs/       # Alternative doc sources
    └── logs/                  # Training + operation logs
        ├── train_20260404_120000.log
        └── train_20260404_150000.log
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PACK_MANIFEST = "pack.yaml"


@dataclass
class PackMetadata:
    """Metadata loaded from pack.yaml."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    window_match: list[str] = field(default_factory=list)  # Substrings to match window name
    model_file: str = "model.pt"
    labels: list[str] = field(default_factory=list)
    input_size: int = 1024
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45

    @classmethod
    def from_dict(cls, data: dict) -> PackMetadata:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ApplicationPack:
    """A loaded application pack ready for runtime use."""

    metadata: PackMetadata
    pack_dir: Path
    _model: object = field(default=None, repr=False)

    @property
    def model_path(self) -> Path:
        return self.pack_dir / self.metadata.model_file

    @property
    def has_model(self) -> bool:
        return self.model_path.exists()

    @property
    def label_map(self) -> dict[int, str]:
        return {i: name for i, name in enumerate(self.metadata.labels)}

    @property
    def recordings_dir(self) -> Path:
        return self.pack_dir / "recordings"

    @property
    def training_data_dir(self) -> Path:
        return self.pack_dir / "training_data"

    @property
    def models_dir(self) -> Path:
        return self.pack_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.pack_dir / "logs"

    @property
    def icon_labels_path(self) -> Path:
        return self.pack_dir / "icon_labels.json"

    @property
    def knowledge_dir(self) -> Path:
        return self.pack_dir / "knowledge"

    @property
    def has_knowledge(self) -> bool:
        """True if knowledge base has any HTML docs."""
        kd = self.knowledge_dir
        return kd.exists() and any(kd.rglob("*.html"))

    def ensure_dirs(self) -> None:
        """Create all pack subdirectories if they don't exist."""
        for d in [
            self.recordings_dir,
            self.training_data_dir,
            self.models_dir,
            self.logs_dir,
            self.knowledge_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def new_recording_dir(self) -> Path:
        """Create and return a new timestamped recording directory."""
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = self.recordings_dir / ts
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def load(cls, pack_dir: str | Path) -> ApplicationPack:
        """Load an ApplicationPack from a directory."""
        pack_dir = Path(pack_dir)
        manifest_path = pack_dir / PACK_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"No {PACK_MANIFEST} in {pack_dir}")

        with open(manifest_path) as f:
            raw = yaml.safe_load(f) or {}

        metadata = PackMetadata.from_dict(raw)
        logger.info("Loaded pack '%s' v%s from %s", metadata.name, metadata.version, pack_dir)
        return cls(metadata=metadata, pack_dir=pack_dir)

    def matches_window(self, window_name: str, owner_name: str = "") -> bool:
        """Check if this pack should handle the given window."""
        combined = f"{owner_name} {window_name}".lower()
        return any(pattern.lower() in combined for pattern in self.metadata.window_match)
