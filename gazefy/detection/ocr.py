"""OCR: read text from detected UI element bounding boxes.

    Frame + list[Detection] → OCR → list[Detection with text]

Crops each element's bbox from the frame and runs EasyOCR to extract text.
Results are cached per element position — only re-OCR when the element moves.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ElementOCR:
    """Reads text from UI element bounding boxes using EasyOCR."""

    def __init__(self, languages: list[str] | None = None):
        self._languages = languages or ["en"]
        self._reader = None

    def _ensure_reader(self) -> None:
        if self._reader is not None:
            return
        try:
            import easyocr

            self._reader = easyocr.Reader(self._languages, gpu=False, verbose=False)
            logger.info("EasyOCR initialized: languages=%s", self._languages)
        except ImportError:
            raise RuntimeError("EasyOCR not installed. pip install easyocr")

    def read_element_text(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> str:
        """Read text from a single element's bounding box.

        Args:
            frame: BGR or BGRA image.
            bbox: (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            Extracted text string (empty if none found).
        """
        self._ensure_reader()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return ""

        crop = frame[y1:y2, x1:x2]
        if crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)

        try:
            results = self._reader.readtext(crop, detail=0, paragraph=True)
            return " ".join(results).strip() if results else ""
        except Exception:
            logger.debug("OCR failed for bbox (%d,%d,%d,%d)", x1, y1, x2, y2)
            return ""

    def read_all_elements(
        self,
        frame: np.ndarray,
        detections: list,
        min_area: int = 100,
    ) -> dict[int, str]:
        """Read text for all detected elements.

        Args:
            frame: BGR or BGRA image.
            detections: list of Detection objects (must have .bbox with x1,y1,x2,y2).
            min_area: Skip elements smaller than this area.

        Returns:
            Dict mapping detection index → extracted text.
        """
        self._ensure_reader()
        texts: dict[int, str] = {}

        # Convert frame once
        if frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr = frame

        for i, det in enumerate(detections):
            bbox = det.bbox
            area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
            if area < min_area:
                continue
            text = self.read_element_text(bgr, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
            if text:
                texts[i] = text

        return texts
