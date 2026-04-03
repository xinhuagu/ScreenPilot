"""Learn mode: user clicks UI elements, VLM identifies them.

Usage:
    gazefy learn --window "gimp" --pack gimp_demo

User clicks an icon/button → system crops the bbox →
sends to Claude Vision → gets semantic label → saves to pack's icon dictionary.

Builds a persistent icon_labels.json in the pack directory:
    packs/gimp_demo/icon_labels.json
    {
        "bbox_hash_abc123": {
            "label": "Paintbrush Tool",
            "class": "button",
            "bbox": [50, 100, 80, 130],
            "image_file": "icons/abc123.png"
        },
        ...
    }
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from gazefy.config import CaptureRegion

logger = logging.getLogger(__name__)


def _bbox_hash(x1: int, y1: int, x2: int, y2: int) -> str:
    """Deterministic short hash for a bbox position."""
    s = f"{x1},{y1},{x2},{y2}"
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _crop_to_base64(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    """Crop bbox from frame and encode as base64 PNG."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    _, buf = cv2.imencode(".png", crop)
    return base64.standard_b64encode(buf).decode("utf-8")


def _crop_with_context(
    frame: np.ndarray, bbox: tuple[int, int, int, int], padding: int = 50
) -> str:
    """Crop bbox with surrounding context for better VLM understanding."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    cx1 = max(0, x1 - padding)
    cy1 = max(0, y1 - padding)
    cx2 = min(w, x2 + padding)
    cy2 = min(h, y2 + padding)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    _, buf = cv2.imencode(".png", crop)
    return base64.standard_b64encode(buf).decode("utf-8")


def _ask_vlm(icon_b64: str, context_b64: str, element_class: str) -> str:
    """Send icon image to VLM and get a label. Uses Copilot+gpt-4o."""
    from gazefy.llm.copilot import CopilotClient

    prompt = (
        f"This is a UI element (type: {element_class}) "
        "from a desktop application. "
        "The first image is the element itself. "
        "The second shows it in context.\n\n"
        "Reply with ONLY a short label, e.g. "
        "'Paintbrush Tool', 'Save Button'. "
        "No explanation."
    )
    # gpt-4o vision: send both images in one message
    copilot = CopilotClient(model="gpt-4o")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{icon_b64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{context_b64}"},
                },
            ],
        }
    ]
    return copilot.chat(messages, max_tokens=100)


def run_learn(
    region: CaptureRegion,
    pack_name: str,
    packs_dir: str = "packs",
) -> None:
    """Run learn mode: user clicks elements, VLM labels them.

    Args:
        region: Screen region to capture.
        pack_name: ApplicationPack name (must exist with a trained model).
        packs_dir: Packs directory.
    """
    pack_dir = Path(packs_dir) / pack_name
    if not (pack_dir / "pack.yaml").exists():
        print(f"Pack not found: {pack_dir}")
        sys.exit(1)

    # Load detection pipeline
    from gazefy.core.application_pack import ApplicationPack
    from gazefy.detection.detector import UIDetector

    pack = ApplicationPack.load(pack_dir)
    detector = UIDetector(pack)
    detector.load_model()

    # Load existing labels
    labels_path = pack_dir / "icon_labels.json"
    icon_labels: dict = {}
    if labels_path.exists():
        icon_labels = json.loads(labels_path.read_text())

    # Save icon images
    icons_dir = pack_dir / "icons"
    icons_dir.mkdir(exist_ok=True)

    print("Gazefy Learn Mode")
    print(f"  Pack: {pack_name}")
    print(f"  Region: ({region.left}, {region.top}) {region.width}x{region.height}")
    print(f"  Existing labels: {len(icon_labels)}")
    print()
    print("  Click on any UI element → VLM will identify it.")
    print("  Press Ctrl+C to stop.\n")

    # Capture initial frame + detect
    import mss

    monitor = {
        "top": region.top,
        "left": region.left,
        "width": region.width,
        "height": region.height,
    }

    try:
        from pynput import mouse
    except ImportError:
        print("pynput required: pip install pynput")
        sys.exit(1)

    # Shared state
    state = {"frame": None, "detections": [], "running": True}

    def capture_and_detect():
        with mss.mss() as sct:
            frame = np.array(sct.grab(monitor))
        state["frame"] = frame
        state["detections"] = detector.detect(frame)
        return len(state["detections"])

    # Initial detection
    n = capture_and_detect()
    print(f"  Detected {n} elements. Start clicking!\n")

    def on_click(x, y, button, pressed):
        if not pressed or not state["running"]:
            return

        frame = state["frame"]
        if frame is None:
            return

        # Convert screen coords to frame coords
        fx = (x - region.left) * 2  # Retina
        fy = (y - region.top) * 2

        # Find which detection bbox contains this click
        clicked_det = None
        min_area = float("inf")
        for det in state["detections"]:
            b = det.bbox
            if b.x1 <= fx <= b.x2 and b.y1 <= fy <= b.y2:
                area = (b.x2 - b.x1) * (b.y2 - b.y1)
                if area < min_area:
                    min_area = area
                    clicked_det = det

        if clicked_det is None:
            print(f"  Click ({x},{y}) → no detected element at this position")
            # Re-detect in case screen changed
            capture_and_detect()
            return

        b = clicked_det.bbox
        bbox = (int(b.x1), int(b.y1), int(b.x2), int(b.y2))
        bh = _bbox_hash(*bbox)

        # Skip if already labeled
        if bh in icon_labels:
            existing = icon_labels[bh]["label"]
            print(f'  Click → [{clicked_det.class_name}] already labeled: "{existing}"')
            return

        print(
            f"  Click → [{clicked_det.class_name}] at {bbox} ... asking VLM ...",
            end="",
            flush=True,
        )

        # Crop icon + context
        icon_b64 = _crop_to_base64(frame, bbox)
        context_b64 = _crop_with_context(frame, bbox, padding=80)

        # Ask VLM
        try:
            label = _ask_vlm(icon_b64, context_b64, clicked_det.class_name)
        except Exception as e:
            print(f" ERROR: {e}")
            return

        print(f' → "{label}"')

        # Save icon image
        icon_path = icons_dir / f"{bh}.png"
        crop = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        if crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(icon_path), crop)

        # Save label
        icon_labels[bh] = {
            "label": label,
            "class": clicked_det.class_name,
            "confidence": round(clicked_det.confidence, 3),
            "bbox": list(bbox),
            "image_file": f"icons/{bh}.png",
        }
        labels_path.write_text(json.dumps(icon_labels, indent=2))

    listener = mouse.Listener(on_click=on_click)
    listener.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        state["running"] = False
        listener.stop()

    print(f"\n\nDone: {len(icon_labels)} labeled elements saved to {labels_path}")
