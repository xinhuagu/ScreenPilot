"""ScreenClassifier: identify which page/screen the application is currently showing.

Pack-specific lightweight classifier that maps a screenshot to a screen label
(e.g. "main_screen", "file_dialog", "preferences", "playback").

Two modes:
    1. Learned: fine-tuned image classifier from annotated screenshots
    2. Heuristic: match screen by which UI elements are visible (from UIMap)

Used for:
    - Post-action verification: confirm transition to expected page
    - LLM context: tell the LLM which screen we're on
    - Workflow slot: "expect: file_dialog_open" resolved by classifier
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreenState:
    """Classification result for the current screen."""

    label: str  # e.g. "main_screen", "file_dialog"
    confidence: float = 1.0
    matched_elements: list[str] = field(default_factory=list)  # semantic_ids that matched


@dataclass
class ScreenSignature:
    """A screen is defined by which elements are expected to be present."""

    label: str
    required_elements: list[str] = field(default_factory=list)  # semantic_ids that must be present
    forbidden_elements: list[str] = field(default_factory=list)  # must NOT be present
    min_match_ratio: float = 0.6  # fraction of required elements that must match


class ScreenClassifier:
    """Classify the current screen state from UIMap element signatures."""

    def __init__(self) -> None:
        self._signatures: list[ScreenSignature] = []
        self._history: list[ScreenState] = []

    @classmethod
    def load(cls, pack_dir: Path) -> ScreenClassifier:
        """Load screen signatures from pack_dir/screens.json."""
        classifier = cls()
        screens_path = pack_dir / "screens.json"
        if not screens_path.exists():
            logger.debug("No screens.json in %s", pack_dir)
            return classifier

        raw = json.loads(screens_path.read_text())
        for entry in raw:
            sig = ScreenSignature(
                label=entry.get("label", "unknown"),
                required_elements=entry.get("required_elements", []),
                forbidden_elements=entry.get("forbidden_elements", []),
                min_match_ratio=entry.get("min_match_ratio", 0.6),
            )
            classifier._signatures.append(sig)

        logger.info("Loaded %d screen signatures", len(classifier._signatures))
        return classifier

    def classify(self, ui_map, ontology_resolver=None) -> ScreenState:
        """Classify the current screen from UIMap elements.

        Matches element semantic_ids against screen signatures.
        """
        # Collect semantic IDs from current UIMap
        current_ids: set[str] = set()
        for el in ui_map.elements.values():
            # Use pre-enriched semantic_id if available
            if el.semantic_id:
                current_ids.add(el.semantic_id)
            # Resolve via ontology if provided
            if ontology_resolver:
                entry = ontology_resolver.resolve(el)
                if entry:
                    current_ids.add(entry.semantic_id)
            # Also use element text as fallback identifier
            if el.text:
                current_ids.add(el.text.lower().strip())

        best_state = ScreenState(label="unknown", confidence=0.0)

        for sig in self._signatures:
            # Check forbidden elements
            if any(fid in current_ids for fid in sig.forbidden_elements):
                continue

            # Count required element matches
            if not sig.required_elements:
                continue
            matched = [rid for rid in sig.required_elements if rid in current_ids]
            ratio = len(matched) / len(sig.required_elements)

            if ratio >= sig.min_match_ratio and ratio > best_state.confidence:
                best_state = ScreenState(
                    label=sig.label,
                    confidence=round(ratio, 3),
                    matched_elements=matched,
                )

        self._history.append(best_state)
        return best_state

    def classify_by_frame(
        self,
        frame: np.ndarray,
        detector,
        ontology_resolver=None,
    ) -> ScreenState:
        """Classify by running detection on a raw frame.

        Convenience method that runs YOLO → builds temp UIMap → classifies.
        """
        import cv2

        from gazefy.tracker.ui_map import UIElement, UIMap

        # Handle both BGR (3-channel) and BGRA (4-channel) inputs
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        else:
            frame_bgra = frame
        detections = detector.detect(frame_bgra)
        h, w = frame.shape[:2]

        elements = {}
        for i, det in enumerate(detections):
            eid = f"tmp_{i}"
            elements[eid] = UIElement(
                id=eid,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                center=det.bbox.center,
            )

        ui_map = UIMap(elements=elements, frame_width=w, frame_height=h)
        return self.classify(ui_map, ontology_resolver)

    def verify_transition(
        self,
        expected_label: str,
        ui_map,
        ontology_resolver=None,
    ) -> bool:
        """Verify the screen transitioned to the expected state."""
        state = self.classify(ui_map, ontology_resolver)
        return state.label == expected_label

    def add_signature(self, signature: ScreenSignature) -> None:
        """Add a screen signature dynamically."""
        self._signatures.append(signature)

    def learn_from_ui_map(self, label: str, ui_map, ontology_resolver=None) -> ScreenSignature:
        """Auto-generate a screen signature from current UIMap.

        Takes a snapshot of what elements are visible and creates a signature.
        """
        elements = []
        for el in ui_map.elements.values():
            # Use pre-enriched semantic_id
            if el.semantic_id:
                elements.append(el.semantic_id)
                continue
            # Try ontology resolver
            if ontology_resolver:
                entry = ontology_resolver.resolve(el)
                if entry:
                    elements.append(entry.semantic_id)
                    continue
            # Fallback: use element text
            if el.text:
                elements.append(el.text.lower().strip())

        sig = ScreenSignature(
            label=label,
            required_elements=elements,
            min_match_ratio=0.5,
        )
        self._signatures.append(sig)
        logger.info("Learned screen '%s' with %d elements", label, len(elements))
        return sig

    def save(self, pack_dir: Path) -> Path:
        """Save screen signatures to pack_dir/screens.json."""
        screens_path = pack_dir / "screens.json"
        data = []
        for sig in self._signatures:
            data.append(
                {
                    "label": sig.label,
                    "required_elements": sig.required_elements,
                    "forbidden_elements": sig.forbidden_elements,
                    "min_match_ratio": sig.min_match_ratio,
                }
            )
        screens_path.write_text(json.dumps(data, indent=2))
        logger.info("Saved %d screen signatures to %s", len(data), screens_path)
        return screens_path

    @property
    def last_state(self) -> ScreenState | None:
        return self._history[-1] if self._history else None

    @property
    def signatures(self) -> list[ScreenSignature]:
        return self._signatures

    def __len__(self) -> int:
        return len(self._signatures)
