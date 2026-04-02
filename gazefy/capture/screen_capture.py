"""Continuous screen capture using mss with a threaded ring buffer."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import mss
import numpy as np

from gazefy.config import CaptureRegion

logger = logging.getLogger(__name__)


@dataclass
class CapturedFrame:
    image: np.ndarray  # BGRA uint8, pixel coordinates
    timestamp: float
    frame_number: int


class ScreenCapture:
    """Captures a screen region in a background thread at a target FPS."""

    def __init__(
        self,
        region: CaptureRegion,
        target_fps: int = 20,
        buffer_size: int = 30,
        retina_scale: float = 2.0,
    ):
        self._region = region
        self._target_fps = target_fps
        self._retina_scale = retina_scale
        self._buffer: deque[CapturedFrame] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_count = 0

    @property
    def region(self) -> CaptureRegion:
        return self._region

    @region.setter
    def region(self, value: CaptureRegion) -> None:
        self._region = value

    def start(self) -> None:
        """Start the background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Capture started: region=(%d,%d) %dx%d, target_fps=%d",
            self._region.left,
            self._region.top,
            self._region.width,
            self._region.height,
            self._target_fps,
        )

    def stop(self) -> None:
        """Stop the background capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_latest_frame(self) -> CapturedFrame | None:
        """Return the most recent frame, or None if buffer is empty."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_frame_pair(self) -> tuple[CapturedFrame | None, CapturedFrame | None]:
        """Return (previous, current) frames for change detection."""
        with self._lock:
            if len(self._buffer) < 2:
                current = self._buffer[-1] if self._buffer else None
                return None, current
            return self._buffer[-2], self._buffer[-1]

    def grab_once(self) -> CapturedFrame:
        """Capture a single frame synchronously (useful for testing)."""
        monitor = {
            "top": self._region.top,
            "left": self._region.left,
            "width": self._region.width,
            "height": self._region.height,
        }
        with mss.mss() as sct:
            img = np.array(sct.grab(monitor))
        self._frame_count += 1
        return CapturedFrame(
            image=img,
            timestamp=time.monotonic(),
            frame_number=self._frame_count,
        )

    def _capture_loop(self) -> None:
        interval = 1.0 / self._target_fps
        monitor = {
            "top": self._region.top,
            "left": self._region.left,
            "width": self._region.width,
            "height": self._region.height,
        }
        with mss.mss() as sct:
            while self._running:
                loop_start = time.monotonic()
                try:
                    img = np.array(sct.grab(monitor))
                    self._frame_count += 1
                    frame = CapturedFrame(
                        image=img,
                        timestamp=loop_start,
                        frame_number=self._frame_count,
                    )
                    with self._lock:
                        self._buffer.append(frame)
                except Exception:
                    logger.exception("Capture error")

                elapsed = time.monotonic() - loop_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
