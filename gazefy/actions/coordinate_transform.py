"""CoordinateTransform: converts between pixel coords and screen coords.

Two coordinate systems coexist on macOS:
    - Pixel coords: used by mss capture and YOLO detection (e.g. 3456x2160 on Retina)
    - Screen/logical coords: used by pyautogui for mouse actions (e.g. 1728x1080)

This module handles the translation, accounting for:
    - Retina scale factor (typically 2.0)
    - Capture region offset (VDI window may not be at screen origin)
"""

from __future__ import annotations

from dataclasses import dataclass

from gazefy.config import CaptureRegion
from gazefy.utils.geometry import Point


@dataclass(frozen=True)
class CoordinateTransform:
    """Bidirectional pixel ↔ screen coordinate conversion."""

    region: CaptureRegion  # Capture region in screen (logical) coords
    retina_scale: float = 2.0

    def pixel_to_screen(self, pixel_pt: Point) -> Point:
        """Convert pixel coords (from detection) to screen coords (for pyautogui).

        pixel (0,0) = top-left of captured frame
        screen result = absolute screen position for clicking
        """
        return Point(
            x=(pixel_pt.x / self.retina_scale) + self.region.left,
            y=(pixel_pt.y / self.retina_scale) + self.region.top,
        )

    def screen_to_pixel(self, screen_pt: Point) -> Point:
        """Convert screen coords (from mouse position) to pixel coords (for UIMap lookup)."""
        return Point(
            x=(screen_pt.x - self.region.left) * self.retina_scale,
            y=(screen_pt.y - self.region.top) * self.retina_scale,
        )
