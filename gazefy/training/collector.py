"""DataCollector: captures screenshots from screen into a YOLO dataset directory.

Wires together window_finder + screen_capture to produce:
    datasets/<pack>/<session>/
    ├── images/
    │   ├── frame_000000.png
    │   ├── frame_000001.png
    │   └── ...
    ├── actions.jsonl        (click/key log)
    └── dataset.yaml         (Ultralytics stub)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from gazefy.config import CaptureRegion

logger = logging.getLogger(__name__)


@dataclass
class ActionEvent:
    """A recorded user action."""

    timestamp: float
    action_type: str  # "click", "key", "scroll"
    x: int = 0
    y: int = 0
    detail: str = ""


@dataclass
class CollectorConfig:
    output_dir: str = "datasets"
    pack_name: str = "default"
    capture_interval_ms: int = 500
    split_ratio: float = 0.8


class DataCollector:
    """Captures screenshots at intervals and logs user actions."""

    def __init__(self, config: CollectorConfig | None = None):
        self._config = config or CollectorConfig()
        self._events: list[ActionEvent] = []
        self._frame_count = 0
        self._session_dir: Path | None = None

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start_session(self, session_name: str = "") -> Path:
        """Create session directory structure. Returns session path."""
        name = session_name or f"session_{int(time.time())}"
        self._session_dir = Path(self._config.output_dir) / self._config.pack_name / name
        (self._session_dir / "images").mkdir(parents=True, exist_ok=True)
        self._events = []
        self._frame_count = 0
        logger.info("Session started: %s", self._session_dir)
        return self._session_dir

    def save_frame(self, frame: np.ndarray, timestamp: float | None = None) -> Path:
        """Save a BGRA frame as PNG. Returns saved path."""
        assert self._session_dir is not None, "Call start_session() first"
        img_path = self._session_dir / "images" / f"frame_{self._frame_count:06d}.png"
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
        cv2.imwrite(str(img_path), bgr)
        self._frame_count += 1
        return img_path

    def log_action(self, event: ActionEvent) -> None:
        self._events.append(event)

    def finish_session(self, labels: list[str] | None = None) -> dict:
        """Finalize: write actions log + dataset.yaml stub."""
        assert self._session_dir is not None
        # Actions log
        log_path = self._session_dir / "actions.jsonl"
        with open(log_path, "w") as f:
            for evt in self._events:
                f.write(json.dumps(asdict(evt)) + "\n")

        # dataset.yaml — points to images/ (labels/ filled after annotation)
        names = {}
        if labels:
            names = {i: n for i, n in enumerate(labels)}
        ds = {
            "path": str(self._session_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }
        with open(self._session_dir / "dataset.yaml", "w") as f:
            yaml.dump(ds, f, default_flow_style=False)

        summary = {
            "session": self._session_dir.name,
            "pack": self._config.pack_name,
            "frames": self._frame_count,
            "actions": len(self._events),
            "output_dir": str(self._session_dir),
        }
        logger.info("Session done: %d frames, %d actions", self._frame_count, len(self._events))
        return summary


def run_collect(
    region: CaptureRegion,
    pack_name: str = "default",
    output_dir: str = "datasets",
    interval_ms: int = 500,
    duration_s: float = 0,
    max_frames: int = 0,
    labels: list[str] | None = None,
) -> dict:
    """Run a collection session: capture screenshots at interval.

    Args:
        region: Screen region to capture.
        pack_name: Name of the application pack.
        output_dir: Root dataset directory.
        interval_ms: Capture interval in milliseconds.
        duration_s: Stop after N seconds (0 = unlimited, until max_frames or Ctrl+C).
        max_frames: Stop after N frames (0 = unlimited).
        labels: Optional class label list for dataset.yaml.

    Returns:
        Session summary dict.
    """
    import mss

    config = CollectorConfig(
        output_dir=output_dir,
        pack_name=pack_name,
        capture_interval_ms=interval_ms,
    )
    collector = DataCollector(config)
    session_dir = collector.start_session()

    monitor = {
        "top": region.top,
        "left": region.left,
        "width": region.width,
        "height": region.height,
    }
    interval = interval_ms / 1000.0
    start_time = time.monotonic()

    print(f"Collecting screenshots to {session_dir}")
    print(f"Region: ({region.left}, {region.top}) {region.width}x{region.height}")
    dur = "unlimited" if not duration_s else f"{duration_s}s"
    print(f"Interval: {interval_ms}ms | Duration: {dur}")
    print("Press Ctrl+C to stop.\n")

    try:
        with mss.mss() as sct:
            while True:
                frame = np.array(sct.grab(monitor))
                path = collector.save_frame(frame)
                print(f"\r  [{collector.frame_count}] {path.name}", end="", flush=True)

                if max_frames and collector.frame_count >= max_frames:
                    break
                if duration_s and (time.monotonic() - start_time) >= duration_s:
                    break

                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    print()
    summary = collector.finish_session(labels=labels)
    print(f"Done: {summary['frames']} frames saved to {summary['output_dir']}")
    print("\nNext steps:")
    print("  1. Annotate images in Label Studio / CVAT")
    print(f"  2. Export YOLO labels to: {session_dir}/labels/")
    print(f"  3. Run: python -m gazefy.training.dataset_prep {session_dir}")
    ds_yaml = f"{session_dir}/dataset.yaml"
    print(
        f"  4. Run: python -m gazefy.training.train_pack"
        f" --dataset {ds_yaml} --pack-name {pack_name}"
    )
    return summary
