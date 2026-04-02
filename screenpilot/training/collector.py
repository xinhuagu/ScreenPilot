"""DataCollector: captures screenshots + user actions for building training datasets.

Runs in background while the user operates the target application normally.
Outputs a YOLO-format dataset directory.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActionEvent:
    """A recorded user action."""

    timestamp: float
    action_type: str  # "click", "key", "scroll"
    x: int = 0
    y: int = 0
    detail: str = ""  # key name, button, scroll direction


@dataclass
class CollectorConfig:
    """Configuration for data collection."""

    output_dir: str = "dataset"
    capture_interval_ms: int = 500  # Capture every N ms
    pack_name: str = "default"
    split_ratio: float = 0.8  # train/val split


class DataCollector:
    """Captures screenshots and logs user actions for training data."""

    def __init__(self, config: CollectorConfig | None = None):
        self._config = config or CollectorConfig()
        self._output = Path(self._config.output_dir)
        self._events: list[ActionEvent] = []
        self._frame_count = 0
        self._session_name = ""

    @property
    def output_dir(self) -> Path:
        return self._output

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start_session(self, session_name: str = "") -> Path:
        """Initialize a new data collection session. Returns session output dir."""
        self._session_name = session_name or f"session_{int(time.time())}"
        self._events = []
        self._frame_count = 0

        # Create directory structure
        session_dir = self._output / self._session_name
        (session_dir / "images").mkdir(parents=True, exist_ok=True)
        (session_dir / "labels").mkdir(parents=True, exist_ok=True)

        logger.info("Collection session '%s' started at %s", self._session_name, session_dir)
        return session_dir

    def save_frame(self, frame: np.ndarray, timestamp: float | None = None) -> Path:
        """Save a captured frame as PNG. Returns the saved file path."""
        ts = timestamp or time.time()
        name = f"frame_{self._frame_count:06d}_{int(ts * 1000)}"
        session_dir = self._output / self._session_name

        img_path = session_dir / "images" / f"{name}.png"
        # Convert BGRA to BGR for saving
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(img_path), frame)

        self._frame_count += 1
        return img_path

    def log_action(self, event: ActionEvent) -> None:
        """Record a user action event."""
        self._events.append(event)

    def finish_session(self) -> dict:
        """Finalize the session: save action log, generate dataset.yaml."""
        session_dir = self._output / self._session_name

        # Save action log
        log_path = session_dir / "actions.jsonl"
        with open(log_path, "w") as f:
            for evt in self._events:
                f.write(json.dumps(asdict(evt)) + "\n")

        # Generate dataset.yaml
        self._write_dataset_yaml(session_dir)

        summary = {
            "session": self._session_name,
            "frames": self._frame_count,
            "actions": len(self._events),
            "output_dir": str(session_dir),
        }
        logger.info("Session complete: %d frames, %d actions", self._frame_count, len(self._events))
        return summary

    def _write_dataset_yaml(self, session_dir: Path) -> None:
        """Generate Ultralytics-compatible dataset.yaml."""
        import yaml

        config = {
            "path": str(session_dir.resolve()),
            "train": "images",
            "val": "images",
            "names": {},  # Filled by user during annotation
        }
        yaml_path = session_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
