"""Annotate pipeline: extract video frames → GroundingDINO → OCR → VLM icons.

Extracts frames from ALL recordings in a pack, labels unlabeled ones,
skips already-labeled images. Accumulates training data across sessions.
"""

from __future__ import annotations

import base64
import logging
import re
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

    from gazefy.detection.grounding_label import (
        CLASSES,
        detections_to_yolo,
        load_model,
        predict_image,
    )
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
        processor, gd_model = load_model()
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
            dets = predict_image(processor, gd_model, img, threshold=0.1)

            # OCR each detection
            img_np = np.array(img)
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
                        element_class=det["class_name"],
                        text=text,
                        source="ocr",
                    )

            # VLM for icons (no OCR text) — label + function
            icons = [d for d in dets if not d.get("ocr_text")]
            if icons and vlm:
                try:
                    _, buf = cv2.imencode(".jpg", img_np[:, :, ::-1])
                    b64 = base64.b64encode(buf).decode()
                    icon_desc = ", ".join(
                        f"#{j + 1} at ({d['bbox'][0]:.0f},{d['bbox'][1]:.0f})"
                        for j, d in enumerate(icons)
                    )
                    resp = vlm.chat_with_image(
                        f"UI screenshot with {len(icons)} icon elements. "
                        f"Icons: {icon_desc}.\n"
                        "For each icon, reply with its label AND function.\n"
                        "Format: #1: Label | Function, #2: Label | Function",
                        b64,
                        max_tokens=500,
                    )
                    for m in re.finditer(r"#(\d+):\s*(.+?)(?:\||,|$)", resp):
                        idx_match = int(m.group(1)) - 1
                        parts = m.group(2).strip().split("|")
                        label = parts[0].strip()
                        func = parts[1].strip() if len(parts) > 1 else ""
                        if 0 <= idx_match < len(icons):
                            icons[idx_match]["vlm_label"] = label
                            # Register icon with VLM label + function
                            bbox = icons[idx_match]["bbox"]
                            registry.register(
                                bbox=tuple(int(b) for b in bbox),
                                element_class=icons[idx_match]["class_name"],
                                icon_label=label,
                                function=func,
                                source="vlm",
                            )
                except Exception:
                    pass  # VLM error — continue

            # Save YOLO label
            yolo = detections_to_yolo(dets, img.width, img.height)
            (lbl_dir / name.replace(".png", ".txt")).write_text(yolo)
            total_dets += len(dets)
            log(
                f"  [{i + 1}/{len(to_label)}] {name}: "
                f"{len(dets)} dets ({n_ocr} OCR, {len(icons)} icons)"
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
