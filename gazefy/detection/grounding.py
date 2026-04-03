"""GroundingDINO zero-shot UI element detector.

Detects UI elements (buttons, menus, icons, inputs, …) in any screenshot
without task-specific training, using a text prompt.

Install the optional dependency:
    pip install gazefy[grounding]
    # requires transformers>=4.38; torch is already pulled in by ultralytics

Usage:
    detector = GroundingDetector()
    detector.load()                     # downloads ~700 MB on first run
    dets = detector.detect(frame_bgr)
    for d in dets:
        print(d.label, d.bbox, d.score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Text prompt — period-separated class names for GroundingDINO
_UI_PROMPT = (
    "button . menu . icon . input field . checkbox . tab . "
    "dropdown . scrollbar . toolbar . link . label"
)


@dataclass
class GroundingDetection:
    bbox: list[int]  # [x1, y1, x2, y2] in original image pixels
    label: str  # matched class from prompt
    score: float


class GroundingDetector:
    """Zero-shot UI element detector via GroundingDINO (HuggingFace transformers).

    Falls back gracefully when transformers is not installed.
    """

    MODEL_ID = "IDEA-Research/grounding-dino-tiny"

    def __init__(
        self,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: str = "cpu",
    ):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load model weights (downloads on first use, ~700 MB cached by HuggingFace)."""
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError:
            raise RuntimeError("transformers not installed. Run: pip install gazefy[grounding]")

        logger.info("Loading GroundingDINO %s on device=%s", self.MODEL_ID, self.device)
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.MODEL_ID).to(
            self.device
        )
        logger.info("GroundingDINO ready")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(
        self,
        frame_bgr: np.ndarray,
        prompt: str = _UI_PROMPT,
    ) -> list[GroundingDetection]:
        """Detect UI elements in a BGR frame.

        Returns list of GroundingDetection sorted by score descending.
        """
        if not self.is_loaded:
            self.load()

        import torch
        from PIL import Image

        h, w = frame_bgr.shape[:2]
        pil_image = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR → RGB

        inputs = self._processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]

        detections: list[GroundingDetection] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                detections.append(
                    GroundingDetection(
                        bbox=[x1, y1, x2, y2],
                        label=str(label),
                        score=float(score),
                    )
                )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
