"""ElementVerifier: pre-click bbox verification + post-action state check.

Before clicking, crops the target bbox and verifies via OCR that the
element text matches expectations. After clicking, uses ScreenClassifier
to verify the expected page transition occurred.

This is the safety net between detection and execution — catches:
    - Stale UIMap (element moved/disappeared since last detection)
    - Misidentified elements (OCR text doesn't match semantic label)
    - Failed transitions (clicked but nothing happened)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class VerifyResult(Enum):
    PASS = "pass"
    FAIL_TEXT_MISMATCH = "fail_text_mismatch"
    FAIL_ELEMENT_GONE = "fail_element_gone"
    FAIL_TRANSITION = "fail_transition"
    SKIP = "skip"  # verification not possible (no model/OCR)


@dataclass(frozen=True)
class VerificationReport:
    """Result of element verification."""

    result: VerifyResult
    expected_text: str = ""
    actual_text: str = ""
    detail: str = ""


class ElementVerifier:
    """Verifies elements before click and screen state after action."""

    def __init__(self, ocr=None, screen_classifier=None, ontology_resolver=None):
        """
        Args:
            ocr: ElementOCR instance for text verification
            screen_classifier: ScreenClassifier for post-action verification
            ontology_resolver: OntologyResolver for semantic lookup
        """
        self._ocr = ocr
        self._classifier = screen_classifier
        self._resolver = ontology_resolver

    def verify_before_click(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        expected_text: str = "",
        expected_class: str = "",
    ) -> VerificationReport:
        """Verify element at bbox matches expectations before clicking.

        Crops the bbox region and runs OCR to check text content.

        Args:
            frame: Current screen frame (RGB or BGR)
            bbox: (x1, y1, x2, y2) pixel coordinates
            expected_text: Text we expect to find (from ontology/workflow)
            expected_class: Expected element class (optional)

        Returns:
            VerificationReport with pass/fail status
        """
        if self._ocr is None:
            return VerificationReport(result=VerifyResult.SKIP, detail="No OCR available")

        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame.shape[:2]

        # Bounds check
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return VerificationReport(
                result=VerifyResult.FAIL_ELEMENT_GONE,
                detail=f"Invalid bbox: ({x1},{y1},{x2},{y2})",
            )

        # OCR the cropped region
        actual_text = self._ocr.read_element_text(frame, (x1, y1, x2, y2))

        if not expected_text:
            # No expected text — cannot verify, report SKIP not PASS
            return VerificationReport(
                result=VerifyResult.SKIP,
                actual_text=actual_text or "",
                detail="No expected text to verify against",
            )

        # Compare
        if not actual_text:
            return VerificationReport(
                result=VerifyResult.FAIL_TEXT_MISMATCH,
                expected_text=expected_text,
                actual_text="",
                detail="OCR returned empty — element may have moved",
            )

        # Fuzzy match: expected text should be contained in actual (or vice versa)
        exp_lower = expected_text.lower().strip()
        act_lower = actual_text.lower().strip()
        if exp_lower in act_lower or act_lower in exp_lower:
            return VerificationReport(
                result=VerifyResult.PASS,
                expected_text=expected_text,
                actual_text=actual_text,
            )

        return VerificationReport(
            result=VerifyResult.FAIL_TEXT_MISMATCH,
            expected_text=expected_text,
            actual_text=actual_text,
            detail=f"Expected '{expected_text}' but found '{actual_text}'",
        )

    def verify_after_action(
        self,
        ui_map,
        expected_screen: str = "",
        expected_element_appears: str = "",
        expected_element_disappears: str = "",
    ) -> VerificationReport:
        """Verify screen state after an action was executed.

        Args:
            ui_map: Current UIMap (freshly detected after action)
            expected_screen: Expected screen label from ScreenClassifier
            expected_element_appears: semantic_id that should now be visible
            expected_element_disappears: semantic_id that should be gone

        Returns:
            VerificationReport with pass/fail status
        """
        # Check screen transition
        if expected_screen:
            if not self._classifier:
                return VerificationReport(
                    result=VerifyResult.SKIP,
                    detail=f"Cannot verify screen '{expected_screen}': no classifier",
                )
            state = self._classifier.classify(ui_map, self._resolver)
            if state.label != expected_screen:
                return VerificationReport(
                    result=VerifyResult.FAIL_TRANSITION,
                    detail=f"Expected screen '{expected_screen}' but got '{state.label}'",
                )

        # Check element appearance
        if expected_element_appears:
            if not self._resolver:
                return VerificationReport(
                    result=VerifyResult.SKIP,
                    detail=f"Cannot verify element '{expected_element_appears}': no resolver",
                )
            found = False
            for el in ui_map.elements.values():
                entry = self._resolver.resolve(el)
                if entry and entry.semantic_id == expected_element_appears:
                    found = True
                    break
            if not found:
                return VerificationReport(
                    result=VerifyResult.FAIL_TRANSITION,
                    detail=f"Expected element '{expected_element_appears}' not found after action",
                )

        # Check element disappearance
        if expected_element_disappears and not self._resolver:
            return VerificationReport(
                result=VerifyResult.SKIP,
                detail=f"Cannot verify element '{expected_element_disappears}': no resolver",
            )
        if expected_element_disappears and self._resolver:
            for el in ui_map.elements.values():
                entry = self._resolver.resolve(el)
                if entry and entry.semantic_id == expected_element_disappears:
                    return VerificationReport(
                        result=VerifyResult.FAIL_TRANSITION,
                        detail=(
                            f"Element '{expected_element_disappears}' still present after action"
                        ),
                    )

        return VerificationReport(result=VerifyResult.PASS)

    def verify_workflow_step(
        self,
        frame: np.ndarray,
        ui_map,
        step: dict,
    ) -> VerificationReport:
        """Verify a workflow step's target before execution.

        Combines pre-click verification (OCR on bbox) with ontology lookup.
        """
        target = step.get("target", "")
        if not target or not self._resolver:
            return VerificationReport(result=VerifyResult.SKIP, detail="No target to verify")

        # Find the target element in UIMap
        for el in ui_map.elements.values():
            entry = self._resolver.resolve(el)
            if entry and entry.semantic_id == target:
                # Found the element — verify its text via OCR
                bbox = (el.bbox.x1, el.bbox.y1, el.bbox.x2, el.bbox.y2)
                # Only use visible text for OCR verification, not semantic description
                expected_text = el.text or ""
                return self.verify_before_click(frame, bbox, expected_text=expected_text)

        # Element not found in UIMap
        return VerificationReport(
            result=VerifyResult.FAIL_ELEMENT_GONE,
            detail=f"Target '{target}' not found in current UIMap",
        )
