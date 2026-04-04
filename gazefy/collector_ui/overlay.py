"""Transparent overlay window: draws detection bboxes on top of the screen.

Creates a frameless, transparent, click-through window that covers the
target application area. Draws bounding boxes and labels for all detected
UI elements in real-time.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class OverlayWidget(QWidget):
    """Transparent overlay that draws detection boxes on screen."""

    def __init__(self):
        super().__init__()
        self._elements: list[dict] = []  # [{x1,y1,x2,y2,class,text,conf}]
        self._cursor_element_id: str = ""
        self._offset_x: int = 0
        self._offset_y: int = 0

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Repaint timer
        self._timer = QTimer()
        self._timer.timeout.connect(self.update)
        self._timer.start(100)  # 10 Hz repaint

    def set_region(self, left: int, top: int, width: int, height: int) -> None:
        """Position the overlay to cover the target window."""
        self._offset_x = left
        self._offset_y = top
        self.setGeometry(left, top, width, height)

    def set_elements(
        self,
        elements: list[dict],
        cursor_element_id: str = "",
        retina_scale: float = 2.0,
    ) -> None:
        """Update the elements to draw. Coords are in pixel space."""
        self._elements = elements
        self._cursor_element_id = cursor_element_id
        self._retina_scale = retina_scale

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self._elements:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for el in self._elements:
            x1 = int(el["x1"])
            y1 = int(el["y1"])
            x2 = int(el["x2"])
            y2 = int(el["y2"])

            cls = el.get("class", "")
            text = el.get("text", "")
            el_id = el.get("id", "")
            is_cursor = el_id == self._cursor_element_id

            # Color by class
            color = _class_color(cls)
            if is_cursor:
                color = QColor(0, 255, 0)  # Green for cursor element

            # Draw box
            pen = QPen(color, 2 if not is_cursor else 3)
            painter.setPen(pen)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            # Draw label
            label = text if text else cls
            if label:
                font = QFont("Menlo", 9)
                font.setBold(is_cursor)
                painter.setFont(font)

                # Background for readability
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(label) + 6
                th = fm.height() + 2
                painter.fillRect(x1, y1 - th, tw, th, QColor(0, 0, 0, 180))

                painter.setPen(QColor(255, 255, 255))
                painter.drawText(x1 + 3, y1 - 3, label)

        painter.end()


def _class_color(cls: str) -> QColor:
    """Assign a consistent color to each element class."""
    colors = {
        "button": QColor(255, 100, 100),
        "menu_item": QColor(100, 150, 255),
        "input_field": QColor(100, 255, 100),
        "checkbox": QColor(200, 100, 255),
        "toolbar": QColor(255, 200, 50),
        "dialog": QColor(0, 200, 200),
        "icon": QColor(255, 150, 50),
        "tab": QColor(150, 255, 150),
        "label": QColor(200, 200, 200),
        "scrollbar": QColor(150, 150, 150),
        "dropdown": QColor(255, 100, 200),
    }
    return colors.get(cls, QColor(200, 200, 200))
