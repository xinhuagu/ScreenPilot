"""Element Registry: persistent semantic identity for UI elements.

All bbox coordinates are stored NORMALIZED (0-1), so they work
regardless of window size or position.

Storage: packs/<app>/element_registry.json
{
  "abc123": {
    "class": "button",
    "text": "Playlist",
    "icon_label": "",
    "function": "Opens the playlist panel",
    "bbox_norm": [0.14, 0.91, 0.22, 0.98],
    "source": "ocr"
  }
}
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IOU_MATCH_THRESHOLD = 0.3


def _norm_hash(nx1: float, ny1: float, nx2: float, ny2: float) -> str:
    """Hash from normalized coords (rounded to avoid float noise)."""
    s = f"{nx1:.3f},{ny1:.3f},{nx2:.3f},{ny2:.3f}"
    return hashlib.md5(s.encode()).hexdigest()[:8]


class ElementRegistry:
    """Persistent semantic registry with normalized coordinates."""

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
        frame_w: int,
        frame_h: int,
        element_class: str,
        text: str = "",
        icon_label: str = "",
        function: str = "",
        source: str = "ocr",
    ) -> str:
        """Register an element with normalized bbox. Returns entry key."""
        # Normalize to 0-1
        nx1 = bbox[0] / max(frame_w, 1)
        ny1 = bbox[1] / max(frame_h, 1)
        nx2 = bbox[2] / max(frame_w, 1)
        ny2 = bbox[3] / max(frame_h, 1)
        key = _norm_hash(nx1, ny1, nx2, ny2)

        if key in self._entries:
            existing = self._entries[key]
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
            "bbox_norm": [round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)],
            "source": source,
        }
        return key

    def lookup(
        self,
        bbox_x1: float,
        bbox_y1: float,
        bbox_x2: float,
        bbox_y2: float,
        frame_w: int,
        frame_h: int,
        element_class: str = "",
    ) -> dict | None:
        """Find a registered element by normalized bbox proximity."""
        # Normalize query bbox
        nx = (bbox_x1 / max(frame_w, 1) + bbox_x2 / max(frame_w, 1)) / 2
        ny = (bbox_y1 / max(frame_h, 1) + bbox_y2 / max(frame_h, 1)) / 2

        best_entry = None
        best_dist = 0.05  # Max 5% of frame distance

        for entry in self._entries.values():
            # Class filter
            if element_class and entry.get("class") != element_class:
                continue
            bn = entry.get("bbox_norm", entry.get("bbox", [0, 0, 0, 0]))
            # Handle old format (absolute pixels) — skip
            if any(v > 1.1 for v in bn):
                continue
            ecx = (bn[0] + bn[2]) / 2
            ecy = (bn[1] + bn[3]) / 2
            dist = ((nx - ecx) ** 2 + (ny - ecy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_entry = entry

        return best_entry

    def save(self) -> None:
        if self._path:
            self._path.write_text(json.dumps(self._entries, indent=2))
            logger.info("Saved %d registry entries", len(self._entries))

    def __len__(self) -> int:
        return len(self._entries)
