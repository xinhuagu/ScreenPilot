"""Transparent overlay: shows element label when cursor hovers for 1 second.

Only draws a highlight + label for the element under the cursor after
a 1-second dwell. Does not draw all boxes — keeps the view clean.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

DWELL_THRESHOLD = 0.8  # seconds before showing label


class OverlayWidget(QWidget):
    """Transparent overlay that shows element info on cursor dwell."""

    def __init__(self):
        super().__init__()
        self._elements: list[dict] = []
        self._cursor_element_id: str = ""
        self._cursor_enter_time: float = 0.0
        self._last_cursor_id: str = ""

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._timer = QTimer()
        self._timer.timeout.connect(self._repaint)
        self._timer.start(100)

    def _repaint(self) -> None:
        self.raise_()
        self.update()

    def set_region(self, left: int, top: int, width: int, height: int) -> None:
        self.setGeometry(left, top, width, height)

    def set_elements(
        self,
        elements: list[dict],
        cursor_element_id: str = "",
        retina_scale: float = 1.0,
    ) -> None:
        self._elements = elements
        self._update_cursor(cursor_element_id)

    def _update_cursor(self, cursor_id: str) -> None:
        if cursor_id != self._last_cursor_id:
            self._last_cursor_id = cursor_id
            self._cursor_enter_time = time.monotonic()
        self._cursor_element_id = cursor_id

    @property
    def _cursor_element(self) -> dict | None:
        if not self._cursor_element_id:
            return None
        for el in self._elements:
            if el.get("id") == self._cursor_element_id:
                return el
        return None

    @property
    def _dwell_time(self) -> float:
        if not self._cursor_element_id:
            return 0.0
        return time.monotonic() - self._cursor_enter_time

    def paintEvent(self, event) -> None:  # noqa: N802
        el = self._cursor_element
        if el is None:
            return

        # Only show after dwell threshold
        if self._dwell_time < DWELL_THRESHOLD:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        x1 = int(el["x1"])
        y1 = int(el["y1"])
        x2 = int(el["x2"])
        y2 = int(el["y2"])
        cls = el.get("class", "")
        text = el.get("text", "")

        # Draw highlight box (thin, green)
        color = QColor(0, 255, 0, 180)
        painter.setPen(QPen(color, 2))
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        # Draw label above the box
        label = f"[{cls}] {text}" if text else f"[{cls}]"
        font = QFont("Menlo", 11, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 10
        th = fm.height() + 6

        # Position label above box, or below if near top edge
        lx = x1
        ly = y1 - th - 2
        if ly < 0:
            ly = y2 + 2

        painter.fillRect(lx, ly, tw, th, QColor(0, 0, 0, 220))
        painter.setPen(QColor(0, 255, 0))
        painter.drawText(lx + 5, ly + th - 5, label)

        painter.end()
