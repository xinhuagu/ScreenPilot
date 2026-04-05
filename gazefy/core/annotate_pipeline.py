"""Annotate pipeline: extract video frames → GroundingDINO → OCR → VLM icons.

Extracts frames from ALL recordings in a pack, labels unlabeled ones,
skips already-labeled images. Accumulates training data across sessions.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def run_annotate(
    pack_dir: Path,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Run full annotation pipeline on all recordings in a pack.

    Steps:
        1. Extract frames from all video recordings (every 3s)
        2. Label new frames: GroundingDINO bbox → OCR text → VLM icons
        3. Write dataset.yaml

    Already-labeled images are skipped (incremental).

    Returns:
        {"extracted": N, "labeled": N, "total_images": N, "total_elements": N}
    """
    import cv2
    import numpy as np
    from PIL import Image

    from gazefy.detection.grounding_label import CLASSES, detections_to_yolo
    from gazefy.detection.ocr import ElementOCR

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    training_dir = pack_dir / "training_data"
    img_dir = training_dir / "images"
    lbl_dir = training_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Extract frames from ALL recordings ---
    log("Step 1/3: Extracting frames...")
    rec_dir = pack_dir / "recordings"
    all_recordings = []
    if rec_dir.exists():
        for d in sorted(rec_dir.iterdir()):
            if d.is_dir() and (d / "video.mp4").exists():
                all_recordings.append(d)

    extracted = []
    for rec in all_recordings:
        cap = cv2.VideoCapture(str(rec / "video.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        every = max(1, int(fps * 3))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every == 0:
                name = f"{rec.name}_f{idx:04d}.png"
                img_path = img_dir / name
                if not img_path.exists():
                    cv2.imwrite(str(img_path), frame)
                extracted.append(name)
            idx += 1
        cap.release()

    log(f"  {len(extracted)} frames from {len(all_recordings)} recordings")

    # --- Step 2: Label unlabeled images ---
    to_label = [n for n in extracted if not (lbl_dir / n.replace(".png", ".txt")).exists()]
    already = len(extracted) - len(to_label)
    log(f"Step 2/3: Labeling {len(to_label)} new frames ({already} already done)...")

    total_dets = 0
    if to_label:
        # Choose detector: YOLO if pack has model, else GroundingDINO
        use_yolo = (pack_dir / "model.pt").exists()
        if use_yolo:
            from gazefy.core.application_pack import ApplicationPack
            from gazefy.detection.detector import UIDetector

            pack = ApplicationPack.load(pack_dir)
            yolo_det = UIDetector(pack)
            yolo_det.load_model()
            log("  Using trained YOLO model (precise bboxes)")
        else:
            from gazefy.detection.grounding_label import load_model, predict_image

            processor, gd_model = load_model()
            log("  Using GroundingDINO (no trained model)")

        ocr = ElementOCR()

        # VLM for icons
        vlm = None
        try:
            from gazefy.llm.copilot import CopilotClient

            vlm = CopilotClient(model="gpt-4o")
        except Exception:
            log("  VLM not available — icons will be labeled as 'icon'")

        # Element registry — persistent semantic identity
        from gazefy.core.element_registry import ElementRegistry

        registry = ElementRegistry(pack_dir / "element_registry.json")

        for i, name in enumerate(to_label):
            img = Image.open(img_dir / name).convert("RGB")
            img_np = np.array(img)

            if use_yolo:
                # YOLO detection → convert to same dict format
                import cv2 as cv2_inner

                frame_bgra = cv2_inner.cvtColor(img_np, cv2_inner.COLOR_RGB2BGRA)
                raw_dets = yolo_det.detect(frame_bgra)
                dets = [
                    {
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox": (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2),
                    }
                    for d in raw_dets
                ]
            else:
                dets = predict_image(processor, gd_model, img, threshold=0.1)

            # OCR each detection
            n_ocr = 0
            for det in dets:
                bbox = det["bbox"]
                text = ocr.read_element_text(img_np, (bbox[0], bbox[1], bbox[2], bbox[3]))
                if text:
                    det["ocr_text"] = text
                    n_ocr += 1
                    # Register with OCR text
                    registry.register(
                        bbox=tuple(int(b) for b in bbox),
                        frame_w=img.width,
                        frame_h=img.height,
                        element_class=det["class_name"],
                        text=text,
                        source="ocr",
                    )

            # VLM for icons (no OCR text) — crop each icon individually
            icons = [d for d in dets if not d.get("ocr_text")]
            n_vlm = 0
            if icons and vlm:
                for icon_det in icons:
                    try:
                        bx1, by1, bx2, by2 = (int(v) for v in icon_det["bbox"])
                        # Pad crop by 50% of bbox size for context
                        pad_x = max(10, int((bx2 - bx1) * 0.5))
                        pad_y = max(10, int((by2 - by1) * 0.5))
                        cx1 = max(0, bx1 - pad_x)
                        cy1 = max(0, by1 - pad_y)
                        cx2 = min(img.width, bx2 + pad_x)
                        cy2 = min(img.height, by2 + pad_y)
                        crop = img_np[cy1:cy2, cx1:cx2]
                        if crop.size == 0:
                            continue
                        _, buf = cv2.imencode(".jpg", crop[:, :, ::-1])
                        b64 = base64.b64encode(buf).decode()
                        resp = vlm.chat_with_image(
                            "This is a cropped UI icon element. "
                            "What is this icon? Reply in format: Label | Function\n"
                            "Example: Play | Start playback",
                            b64,
                            max_tokens=100,
                        )
                        # Parse "Label | Function"
                        parts = resp.strip().split("|")
                        label = parts[0].strip()
                        func = parts[1].strip() if len(parts) > 1 else ""
                        # Clean up markdown/quotes
                        label = label.strip("*`\"'")
                        func = func.strip("*`\"'")
                        if label:
                            icon_det["vlm_label"] = label
                            registry.register(
                                bbox=(bx1, by1, bx2, by2),
                                frame_w=img.width,
                                frame_h=img.height,
                                element_class=icon_det["class_name"],
                                icon_label=label,
                                function=func,
                                source="vlm",
                            )
                            n_vlm += 1
                    except Exception:
                        continue

            # Save YOLO label
            yolo = detections_to_yolo(dets, img.width, img.height)
            (lbl_dir / name.replace(".png", ".txt")).write_text(yolo)
            total_dets += len(dets)
            log(
                f"  [{i + 1}/{len(to_label)}] {name}: "
                f"{len(dets)} dets ({n_ocr} OCR, {n_vlm} VLM, {len(icons)} icons)"
            )

        # Save registry
        registry.save()
        log(f"  Registry: {len(registry)} elements registered")

    # --- Step 3: Write dataset.yaml ---
    import yaml

    ds = {
        "path": str(training_dir.resolve()),
        "train": "images",
        "val": "images",
        "names": {i: c for i, c in enumerate(CLASSES)},
    }
    with open(training_dir / "dataset.yaml", "w") as f:
        yaml.dump(ds, f, default_flow_style=False)

    total_images = len(list(img_dir.glob("*.png")))
    total_labels = len(list(lbl_dir.glob("*.txt")))
    log(f"Step 3/3: Done! {total_images} images, {total_labels} labels, {total_dets} new elements")

    return {
        "extracted": len(extracted),
        "labeled": len(to_label),
        "total_images": total_images,
        "total_elements": total_dets,
    }
