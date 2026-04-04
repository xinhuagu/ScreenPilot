"""Element Registry: persistent semantic identity for every detected UI element.

Built during Annotate, loaded during Monitor. Maps bbox positions to
their semantic identity (text, icon label, function description).

Storage: packs/<app>/element_registry.json
{
  "abc123": {
    "class": "button",
    "text": "Playlist",
    "icon_label": "",
    "function": "Opens the playlist panel",
    "bbox": [10, 50, 100, 80],
    "source": "ocr"
  },
  "def456": {
    "class": "button",
    "text": "",
    "icon_label": "Play",
    "function": "Starts media playback",
    "bbox": [200, 300, 240, 340],
    "source": "vlm"
  }
}

Matching: elements are matched by bbox IoU, not exact position.
This handles minor position shifts between sessions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from gazefy.utils.geometry import Rect, iou

logger = logging.getLogger(__name__)

IOU_MATCH_THRESHOLD = 0.3  # Loose matching for registry lookup


def _bbox_hash(x1: int, y1: int, x2: int, y2: int) -> str:
    return hashlib.md5(f"{x1},{y1},{x2},{y2}".encode()).hexdigest()[:8]


class ElementRegistry:
    """Persistent semantic registry for UI elements."""

    def __init__(self, registry_path: Path | None = None):
        self._path = registry_path
        self._entries: dict[str, dict] = {}
        if registry_path and registry_path.exists():
            self._entries = json.loads(registry_path.read_text())
            logger.info("Loaded %d registry entries", len(self._entries))

    @property
    def entries(self) -> dict[str, dict]:
        return self._entries

    def register(
        self,
        bbox: tuple[int, int, int, int],
        element_class: str,
        text: str = "",
        icon_label: str = "",
        function: str = "",
        source: str = "ocr",
    ) -> str:
        """Register or update an element. Returns the entry key."""
        key = _bbox_hash(*bbox)

        # Check if already registered with same info
        if key in self._entries:
            existing = self._entries[key]
            # Only update if new info is better
            if not existing.get("text") and text:
                existing["text"] = text
            if not existing.get("icon_label") and icon_label:
                existing["icon_label"] = icon_label
            if not existing.get("function") and function:
                existing["function"] = function
            return key

        self._entries[key] = {
            "class": element_class,
            "text": text,
            "icon_label": icon_label,
            "function": function,
            "bbox": list(bbox),
            "source": source,
        }
        return key

    def lookup(self, bbox: Rect) -> dict | None:
        """Find a registered element by IoU or center-point proximity."""
        best_entry = None
        best_score = 0.0
        cx, cy = bbox.center.x, bbox.center.y

        for entry in self._entries.values():
            eb = entry["bbox"]
            entry_rect = Rect(eb[0], eb[1], eb[2], eb[3])

            # IoU matching
            score = iou(bbox, entry_rect)
            if score > best_score:
                best_score = score
                best_entry = entry

        # If IoU match found, return it
        if best_score >= IOU_MATCH_THRESHOLD:
            return best_entry

        # Fallback: center-point proximity (for small buttons that shift a few pixels)
        best_entry = None
        best_dist = 30.0  # Max 30 pixel distance
        for entry in self._entries.values():
            eb = entry["bbox"]
            ecx = (eb[0] + eb[2]) / 2
            ecy = (eb[1] + eb[3]) / 2
            dist = ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_entry = entry

        return best_entry

    def save(self) -> None:
        """Save registry to disk."""
        if self._path:
            self._path.write_text(json.dumps(self._entries, indent=2))
            logger.info("Saved %d registry entries", len(self._entries))

    def __len__(self) -> int:
        return len(self._entries)
