"""Three-tier frame change detection: perceptual hash → SSIM → dirty rect extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from screenpilot.utils.geometry import Rect

logger = logging.getLogger(__name__)


class ChangeLevel(Enum):
    NONE = 0  # Frame identical to previous
    MINOR = 1  # Small change (cursor blink, animation)
    MAJOR = 2  # Significant change (menu open, dialog, page transition)


@dataclass
class ChangeResult:
    changed: bool
    change_level: ChangeLevel
    dirty_rects: list[Rect] = field(default_factory=list)
    diff_score: float = 0.0  # 0.0 = identical, higher = more different


class ChangeDetector:
    """Detects screen content changes using a three-tier approach."""

    def __init__(
        self,
        similarity_threshold: float = 0.98,
        major_threshold: float = 0.85,
        downsample_size: tuple[int, int] = (160, 120),
    ):
        self._similarity_threshold = similarity_threshold
        self._major_threshold = major_threshold
        self._downsample_size = downsample_size
        self._prev_hash: int | None = None
        self._prev_gray_small: np.ndarray | None = None

    def reset(self) -> None:
        """Reset state (e.g., after capture region change)."""
        self._prev_hash = None
        self._prev_gray_small = None

    def check(self, frame: np.ndarray) -> ChangeResult:
        """Check if frame has changed compared to previous.

        Args:
            frame: BGRA uint8 image from mss capture.

        Returns:
            ChangeResult with change level and dirty rects.
        """
        # Stride-subsample FIRST (near-zero cost), THEN convert to gray
        target_w, target_h = self._downsample_size
        step_y = max(1, frame.shape[0] // target_h)
        step_x = max(1, frame.shape[1] // target_w)
        subsampled = frame[::step_y, ::step_x]
        gray_small = cv2.cvtColor(subsampled, cv2.COLOR_BGRA2GRAY)

        # --- Tier 1: Perceptual hash (< 1ms) ---
        current_hash = self._compute_dhash(gray_small)
        if self._prev_hash is not None and current_hash == self._prev_hash:
            self._prev_hash = current_hash
            self._prev_gray_small = gray_small
            return ChangeResult(changed=False, change_level=ChangeLevel.NONE)

        # --- Tier 2: Mean absolute difference as fast SSIM proxy (~1-2ms) ---
        if self._prev_gray_small is not None:
            diff = np.mean(
                np.abs(gray_small.astype(np.float32) - self._prev_gray_small.astype(np.float32))
            )
            similarity = 1.0 - (diff / 255.0)

            if similarity > self._similarity_threshold:
                # VDI compression noise — not a real change
                self._prev_hash = current_hash
                self._prev_gray_small = gray_small
                return ChangeResult(
                    changed=False,
                    change_level=ChangeLevel.NONE,
                    diff_score=1.0 - similarity,
                )

            # --- Tier 3: Dirty rect extraction ---
            if similarity < self._major_threshold:
                change_level = ChangeLevel.MAJOR
            else:
                change_level = ChangeLevel.MINOR
            dirty_rects = self._extract_dirty_rects(gray_small, frame.shape)

            self._prev_hash = current_hash
            self._prev_gray_small = gray_small
            return ChangeResult(
                changed=True,
                change_level=change_level,
                dirty_rects=dirty_rects,
                diff_score=1.0 - similarity,
            )

        # First frame — treat as major change
        self._prev_hash = current_hash
        self._prev_gray_small = gray_small
        return ChangeResult(changed=True, change_level=ChangeLevel.MAJOR, diff_score=1.0)

    def _compute_dhash(self, gray_small: np.ndarray, hash_size: int = 16) -> int:
        """Compute difference hash for fast frame comparison."""
        resized = cv2.resize(gray_small, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return int(np.packbits(diff.flatten()).tobytes().hex(), 16)

    def _extract_dirty_rects(
        self, current_gray_small: np.ndarray, frame_shape: tuple[int, ...]
    ) -> list[Rect]:
        """Find bounding rects of changed regions, scaled to full resolution."""
        if self._prev_gray_small is None:
            return [Rect(0, 0, float(frame_shape[1]), float(frame_shape[0]))]

        # Already at downsample resolution — direct diff
        h, w = frame_shape[:2]
        ds_w, ds_h = self._downsample_size
        diff = cv2.absdiff(current_gray_small, self._prev_gray_small)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Dilate to merge nearby changes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Scale contour bounding rects back to full resolution
        scale_x = w / ds_w
        scale_y = h / ds_h
        rects = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            rects.append(
                Rect(
                    x1=x * scale_x,
                    y1=y * scale_y,
                    x2=(x + cw) * scale_x,
                    y2=(y + ch) * scale_y,
                )
            )
        return rects
