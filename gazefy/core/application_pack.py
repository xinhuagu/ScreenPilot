"""ApplicationPack: hot-swappable per-application model + config artifact.

An ApplicationPack is a directory containing everything needed to detect and
operate one specific application:

    packs/my_app/
    ├── pack.yaml          # PackMetadata (name, version, labels, thresholds)
    ├── model.pt           # Trained YOLO weights
    ├── labels.yaml        # Class name list
    ├── workflows/         # Optional workflow definitions
    └── verifier_rules/    # Optional pre/post-action verification rules
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
    def workflows_dir(self) -> Path:
        return self.pack_dir / "workflows"

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
