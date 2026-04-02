"""UIDetector: runs a pack's YOLO model on a frame, returns Detections.

    Frame (np.ndarray) → UIDetector → list[Detection]

The detector does NOT track elements or assign IDs — that's the Tracker's job.
It only runs inference and maps coordinates back to the original frame space.
"""

from __future__ import annotations

import logging

import numpy as np

from gazefy.core.application_pack import ApplicationPack
from gazefy.tracker.ui_map import Detection
from gazefy.utils.geometry import Rect

logger = logging.getLogger(__name__)


class UIDetector:
    """Runs YOLO inference using the active ApplicationPack's model."""

    def __init__(self, pack: ApplicationPack):
        self._pack = pack
        self._model = None
        self._label_map = pack.label_map

    def load_model(self) -> None:
        """Load the YOLO model from the pack. Requires 'ml' extra."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("Install ml extra: pip install gazefy[ml]")

        model_path = self._pack.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._model = YOLO(str(model_path))
        logger.info("Loaded model from %s", model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single BGRA frame.

        Args:
            frame: BGRA uint8 image from screen capture.

        Returns:
            List of Detection objects in frame pixel coordinates.
        """
        if self._model is None:
            raise RuntimeError("Call load_model() first")

        meta = self._pack.metadata
        results = self._model.predict(
            frame,
            imgsz=meta.input_size,
            conf=meta.conf_threshold,
            iou=meta.iou_threshold,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=self._label_map.get(cls_id, f"class_{cls_id}"),
                        confidence=float(box.conf[0]),
                        bbox=Rect(x1, y1, x2, y2),
                    )
                )
        return detections

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
