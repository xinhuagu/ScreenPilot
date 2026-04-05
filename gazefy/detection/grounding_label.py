#!/usr/bin/env python3
"""Auto-label UI screenshots using GroundingDINO (via HuggingFace transformers).

Generates YOLO-format label files for each image.

Usage:
    python3 scripts/auto_label.py datasets/gimp_demo/auto_collect/images/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# UI element classes — fine-grained prompts for precise bbox detection
CLASSES = [
    "button",
    "clickable text",
    "input field",
    "checkbox",
    "dropdown",
    "text label",
    "tab",
    "slider",
    "icon",
    "toggle",
    "search box",
]

# Map GroundingDINO text → YOLO class ID
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# The text prompt sent to GroundingDINO (period-separated)
TEXT_PROMPT = ". ".join(CLASSES) + "."


def load_model():
    """Load GroundingDINO model from HuggingFace."""
    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    if torch.backends.mps.is_available():
        model = model.to("mps")
        print("Using MPS (Apple Silicon)")
    else:
        print("Using CPU")
    return processor, model


def predict_image(
    processor,
    model,
    image: Image.Image,
    threshold: float = 0.25,
    text_prompt: str | None = None,
) -> list[dict]:
    """Run GroundingDINO on a single image. Returns list of detections.

    Args:
        text_prompt: Custom period-separated prompt. If None, uses default CLASSES.
    """
    prompt = text_prompt or TEXT_PROMPT
    # Parse custom prompt into class list for matching
    if text_prompt:
        custom_classes = [
            c.strip().lower() for c in text_prompt.rstrip(".").split(".") if c.strip()
        ]
    else:
        custom_classes = None

    device = next(model.parameters()).device
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    detections = []
    for box, score, label_text in zip(results["boxes"], results["scores"], results["text_labels"]):
        x1, y1, x2, y2 = box.tolist()
        label_text = label_text.strip().lower()

        if custom_classes:
            # Match against custom prompt classes
            class_id, class_name = _match_custom(label_text, custom_classes)
        else:
            # Match against default CLASSES
            cid = _match_class(label_text)
            class_id = cid
            class_name = CLASSES[cid] if cid is not None else None

        if class_id is not None and class_name:
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(score),
                    "bbox": (x1, y1, x2, y2),
                }
            )
    return detections


def _match_custom(text: str, classes: list[str]) -> tuple[int | None, str | None]:
    """Match GroundingDINO output to a custom class list."""
    text = text.strip().lower()
    # Exact match
    for i, cls in enumerate(classes):
        if text == cls:
            return i, cls
    # Partial match
    for i, cls in enumerate(classes):
        if cls in text or text in cls:
            return i, cls
    return None, None


def _match_class(text: str) -> int | None:
    """Match GroundingDINO output text to our class taxonomy."""
    text = text.strip().lower()
    # Exact match first
    if text in CLASS_TO_ID:
        return CLASS_TO_ID[text]
    # Partial match
    for cls_name, cls_id in CLASS_TO_ID.items():
        if cls_name in text or text in cls_name:
            return cls_id
    return None


def detections_to_yolo(detections: list[dict], img_w: int, img_h: int) -> str:
    """Convert detections to YOLO format string."""
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # YOLO format: class_id x_center y_center width height (normalized 0-1)
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label UI screenshots with GroundingDINO")
    parser.add_argument("images_dir", help="Directory containing PNG screenshots")
    parser.add_argument(
        "--threshold", type=float, default=0.25, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Labels output dir (default: sibling 'labels/' dir)",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Directory not found: {images_dir}")
        sys.exit(1)

    # Output labels next to images
    if args.output_dir:
        labels_dir = Path(args.output_dir)
    else:
        labels_dir = images_dir.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    images = sorted(images_dir.glob("*.png"))
    if not images:
        print(f"No PNG files found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images")
    print(f"Labels output: {labels_dir}")
    print(f"Classes: {CLASSES}")
    print(f"Threshold: {args.threshold}\n")

    # Load model
    processor, model = load_model()

    # Process each image
    total_detections = 0
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        detections = predict_image(processor, model, image, threshold=args.threshold)

        # Write YOLO label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        yolo_text = detections_to_yolo(detections, image.width, image.height)
        label_path.write_text(yolo_text)

        total_detections += len(detections)
        print(f"  {img_path.name}: {len(detections)} detections")

    print(f"\nDone: {total_detections} total detections across {len(images)} images")
    print(f"Labels saved to: {labels_dir}")

    # Write dataset.yaml
    session_dir = images_dir.parent
    ds_yaml = session_dir / "dataset.yaml"
    import yaml

    ds = {
        "path": str(session_dir.resolve()),
        "train": "images",
        "val": "images",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    with open(ds_yaml, "w") as f:
        yaml.dump(ds, f, default_flow_style=False)
    print(f"dataset.yaml: {ds_yaml}")


if __name__ == "__main__":
    main()
