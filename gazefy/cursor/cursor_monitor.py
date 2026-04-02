"""CursorMonitor: real-time cursor-to-element resolution.

    Mouse position (screen coords) + UIMap → CursorState

Runs in a dedicated thread at 60Hz. Reads the latest UIMap snapshot
(immutable, no lock needed) and resolves which element the cursor is over.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from gazefy.actions.coordinate_transform import CoordinateTransform
from gazefy.tracker.ui_map import UIElement, UIMap
from gazefy.utils.geometry import Point

logger = logging.getLogger(__name__)


@dataclass
class CursorState:
    """Current cursor state, updated at poll rate."""

    screen_position: Point = Point(0, 0)  # Absolute screen coords
    frame_position: Point = Point(0, 0)  # Relative to capture region (pixel)
    current_element: UIElement | None = None
    dwell_time_ms: float = 0.0  # How long cursor has been on current element


class CursorMonitor:
    """Polls cursor position and resolves it against the UIMap."""

    def __init__(
        self,
        transform: CoordinateTransform,
        poll_rate_hz: int = 60,
    ):
        self._transform = transform
        self._poll_interval = 1.0 / poll_rate_hz
        self._ui_map: UIMap = UIMap()
        self._state = CursorState()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_element_id: str = ""
        self._element_enter_time: float = 0.0

    @property
    def state(self) -> CursorState:
        return self._state

    def set_ui_map(self, ui_map: UIMap) -> None:
        """Update the UIMap reference (called by orchestrator after tracker update)."""
        self._ui_map = ui_map

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _poll_loop(self) -> None:
        try:
            import pyautogui
        except ImportError:
            logger.error("pyautogui not installed. Install: pip install gazefy[platform]")
            return

        while self._running:
            try:
                sx, sy = pyautogui.position()
                screen_pt = Point(float(sx), float(sy))
                frame_pt = self._transform.screen_to_pixel(screen_pt)
                element = self._ui_map.element_at(frame_pt)

                # Track dwell time
                eid = element.id if element else ""
                now = time.monotonic()
                if eid != self._last_element_id:
                    self._last_element_id = eid
                    self._element_enter_time = now

                dwell = (now - self._element_enter_time) * 1000 if eid else 0.0

                self._state = CursorState(
                    screen_position=screen_pt,
                    frame_position=frame_pt,
                    current_element=element,
                    dwell_time_ms=dwell,
                )
            except Exception:
                logger.exception("Cursor poll error")

            time.sleep(self._poll_interval)
