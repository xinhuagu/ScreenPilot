"""ElementTracker: maintains the UIMap by integrating detections across frames.

    list[Detection] + ChangeResult → ElementTracker → UIMap

Responsibilities:
    - Assign stable IDs to elements via IoU matching across frames
    - Build parent-child hierarchy (dialog contains button)
    - Filter unstable detections (require ≥ min_stability consecutive frames)
    - Remove stale elements after major screen changes
    - Publish a new immutable UIMap snapshot after each update
"""

from __future__ import annotations

import logging
import time

from gazefy.capture.change_detector import ChangeLevel, ChangeResult
from gazefy.tracker.ui_map import Detection, UIElement, UIMap, UIMapDiff
from gazefy.utils.geometry import iou

logger = logging.getLogger(__name__)


class ElementTracker:
    """Tracks UI elements across frames and publishes UIMap snapshots."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        min_stability: int = 2,
        stale_after_frames: int = 5,
    ):
        self._iou_threshold = iou_threshold
        self._min_stability = min_stability
        self._stale_after_frames = stale_after_frames
        self._current_map = UIMap()
        self._generation = 0
        self._next_id = 0
        # Mutable tracking state (stability counters, last-seen frame)
        self._stability: dict[str, int] = {}  # element_id → consecutive frames
        self._last_seen: dict[str, int] = {}  # element_id → generation when last detected

    @property
    def current_map(self) -> UIMap:
        """The latest published UIMap snapshot (immutable)."""
        return self._current_map

    def update(
        self,
        detections: list[Detection],
        change: ChangeResult,
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> UIMapDiff:
        """Integrate new detections into the UIMap.

        Args:
            detections: Raw detections from the current frame.
            change: Change detection result (used to decide full vs incremental update).
            frame_width: Width of the captured frame in pixels.
            frame_height: Height of the captured frame in pixels.

        Returns:
            UIMapDiff describing what changed.
        """
        self._generation += 1

        if change.change_level == ChangeLevel.MAJOR:
            return self._full_rebuild(detections, frame_width, frame_height)
        return self._incremental_update(detections, frame_width, frame_height)

    def _full_rebuild(self, detections: list[Detection], w: int, h: int) -> UIMapDiff:
        """Major change: discard old map, build fresh from detections."""
        old_ids = set(self._current_map.elements.keys())
        self._stability.clear()
        self._last_seen.clear()

        elements = {}
        for det in detections:
            eid = self._make_id(det.class_name)
            elements[eid] = self._det_to_element(eid, det)
            self._stability[eid] = 1
            self._last_seen[eid] = self._generation

        self._current_map = UIMap(
            elements=elements,
            frame_width=w,
            frame_height=h,
            generation=self._generation,
            timestamp=time.monotonic(),
        )
        new_ids = set(elements.keys())
        return UIMapDiff(
            added=sorted(new_ids),
            removed=sorted(old_ids),
            generation=self._generation,
        )

    def _incremental_update(self, detections: list[Detection], w: int, h: int) -> UIMapDiff:
        """Minor change: match detections to existing elements by IoU."""
        old_elements = dict(self._current_map.elements)
        matched_old: set[str] = set()
        matched_det: set[int] = set()
        new_elements: dict[str, UIElement] = {}
        added: list[str] = []

        # Match each detection to the best existing element
        for i, det in enumerate(detections):
            best_id, best_iou = "", 0.0
            for eid, el in old_elements.items():
                if eid in matched_old:
                    continue
                if el.class_name != det.class_name:
                    continue
                score = iou(det.bbox, el.bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = eid

            if best_iou >= self._iou_threshold and best_id:
                # Existing element — update it, increment stability
                matched_old.add(best_id)
                matched_det.add(i)
                stab = self._stability.get(best_id, 0) + 1
                self._stability[best_id] = stab
                self._last_seen[best_id] = self._generation
                new_elements[best_id] = self._det_to_element(best_id, det, stability=stab)
            else:
                # New element
                eid = self._make_id(det.class_name)
                self._stability[eid] = 1
                self._last_seen[eid] = self._generation
                new_elements[eid] = self._det_to_element(eid, det)
                added.append(eid)

        # Keep old elements that weren't matched (may be stale)
        removed: list[str] = []
        for eid, el in old_elements.items():
            if eid in matched_old:
                continue
            frames_since = self._generation - self._last_seen.get(eid, 0)
            if frames_since > self._stale_after_frames:
                removed.append(eid)
                self._stability.pop(eid, None)
                self._last_seen.pop(eid, None)
            else:
                new_elements[eid] = el  # Keep but don't update

        # Apply stability filter — only publish elements seen enough times
        published = {
            eid: el
            for eid, el in new_elements.items()
            if self._stability.get(eid, 0) >= self._min_stability
        }

        self._current_map = UIMap(
            elements=published,
            frame_width=w,
            frame_height=h,
            generation=self._generation,
            timestamp=time.monotonic(),
        )
        return UIMapDiff(
            added=sorted(added),
            removed=sorted(removed),
            generation=self._generation,
        )

    def _det_to_element(self, eid: str, det: Detection, stability: int = 1) -> UIElement:
        return UIElement(
            id=eid,
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            bbox=det.bbox,
            center=det.bbox.center,
            stability=stability,
        )

    def _make_id(self, class_name: str) -> str:
        prefix = class_name[:3]
        self._next_id += 1
        return f"{prefix}_{self._next_id:04d}"
