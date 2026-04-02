"""FPS counter and performance timing utilities."""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Sliding-window FPS counter."""

    def __init__(self, window_size: int = 60):
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> None:
        self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class Timer:
    """Context manager for measuring elapsed time in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000
