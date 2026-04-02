"""Gazefy configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CaptureRegion:
    top: int = 0
    left: int = 0
    width: int = 1920
    height: int = 1080


@dataclass
class GazefyConfig:
    # Capture
    region: CaptureRegion = field(default_factory=CaptureRegion)
    capture_fps: int = 20
    retina_scale: float = 2.0
    window_name: str = ""  # Auto-detect VDI window by name (empty = manual region)

    # Change detection
    similarity_threshold: float = 0.98
    downsample_size: tuple[int, int] = (160, 120)

    # Detection
    model_path: str = ""
    device: str = "coreml"  # coreml, mps, cpu
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    input_size: int = 1024

    # Tracker
    iou_match_threshold: float = 0.5
    stale_after_frames: int = 5
    min_stability: int = 2

    # Cursor
    cursor_poll_rate_hz: int = 60

    # Actions
    dry_run: bool = False
    inter_action_delay_ms: int = 100
    action_verify_timeout_ms: int = 2000

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"

    # Mode
    mode: str = "monitor"  # monitor, autopilot, collect, debug

    @classmethod
    def from_yaml(cls, path: str | Path) -> GazefyConfig:
        """Load config from a YAML file, merging with defaults."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        region_data = data.pop("region", {})
        region = CaptureRegion(**region_data) if region_data else CaptureRegion()

        # Handle tuple field
        ds = data.pop("downsample_size", None)
        if ds is not None:
            data["downsample_size"] = tuple(ds)

        return cls(region=region, **data)
