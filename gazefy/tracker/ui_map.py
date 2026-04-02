"""UIMap and UIElement: the central data structures of Gazefy.

Data flow through the system:
    Detector produces  →  [Detection]
    Tracker consumes   →  [Detection] and maintains → [UIMap]
    Cursor queries     →  [UIMap] by point
    LLM reads          →  [UIMap] serialized to text
    Actions target     →  [UIElement] by ID

UIMap is immutable once published — the tracker creates a new snapshot
on each update. Consumers hold a reference to the latest snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gazefy.utils.geometry import Point, Rect

# ---------------------------------------------------------------------------
# Detection output (produced by Detector, consumed by Tracker)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected UI element from one frame of YOLO inference."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Rect  # Pixel coordinates in the captured frame


# ---------------------------------------------------------------------------
# UIElement (managed by Tracker, consumed by everything downstream)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UIElement:
    """A tracked UI element with a stable identity across frames."""

    id: str  # Stable ID, e.g. "btn_0042"
    class_id: int
    class_name: str
    confidence: float
    bbox: Rect  # Pixel coordinates in the captured frame
    center: Point
    text: str = ""  # OCR-extracted text (empty until OCR runs)
    parent_id: str = ""  # ID of containing element (e.g. dialog)
    stability: int = 1  # Consecutive frames this element has been confirmed
    semantic_id: str = ""  # App-specific ID from knowledge module
    description: str = ""  # From manual/knowledge (optional)
    context: str = ""  # Current page/screen name (optional)


# ---------------------------------------------------------------------------
# UIMap (the system's view of the current screen state)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UIMap:
    """Immutable snapshot of all detected UI elements on the current screen.

    Published by the Tracker after each detection cycle. All downstream
    consumers (cursor monitor, LLM formatter, action executor) read from
    the latest UIMap without locking — they hold a reference to an
    immutable snapshot.
    """

    elements: dict[str, UIElement] = field(default_factory=dict)  # id → UIElement
    frame_width: int = 0
    frame_height: int = 0
    generation: int = 0  # Increments on each update
    timestamp: float = 0.0

    def element_at(self, point: Point) -> UIElement | None:
        """Find the smallest element containing the given point.

        When multiple elements overlap (e.g. a button inside a dialog),
        returns the most specific one (smallest area).
        """
        hits = [el for el in self.elements.values() if el.bbox.contains_point(point)]
        if not hits:
            return None
        return min(hits, key=lambda el: el.bbox.area)

    def elements_by_class(self, class_name: str) -> list[UIElement]:
        """Return all elements of a given class, sorted by position (top-left)."""
        return sorted(
            [el for el in self.elements.values() if el.class_name == class_name],
            key=lambda el: (el.bbox.y1, el.bbox.x1),
        )

    def get(self, element_id: str) -> UIElement | None:
        return self.elements.get(element_id)

    @property
    def element_count(self) -> int:
        return len(self.elements)

    @property
    def is_empty(self) -> bool:
        return len(self.elements) == 0


# ---------------------------------------------------------------------------
# UIMapDiff (what changed between two UIMap snapshots)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UIMapDiff:
    """Describes what changed between two UIMap generations."""

    added: list[str] = field(default_factory=list)  # New element IDs
    removed: list[str] = field(default_factory=list)  # Gone element IDs
    generation: int = 0
