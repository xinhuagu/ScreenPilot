"""AnnotationConverter: convert annotations.jsonl + video.mp4 → YOLO training dataset.

Takes a session directory that already has annotations.jsonl (produced by
HybridAnnotator or VideoAnnotator) and writes:

    output_dir/
    ├── images/          (extracted video frames as PNG, one per annotation)
    ├── labels/          (YOLO normalised .txt, one per image)
    └── dataset.yaml     (class taxonomy and paths)

The result plugs directly into ``gazefy prep`` (train/val split) and then
``gazefy train``.

Full pipeline
─────────────
    gazefy record-video          → session_dir/video.mp4 + events.jsonl
    gazefy annotate-video        → session_dir/annotations.jsonl
    gazefy convert-annotations   → output_dir/images/ + labels/ + dataset.yaml
    gazefy prep output_dir       → train/val split in-place
    gazefy train --dataset ...   → packs/my_app/

Usage
─────
    from gazefy.training.annotation_converter import AnnotationConverter
    result = AnnotationConverter().convert_session(Path("recordings/session_xxx"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default UI element class taxonomy (mirrors DESIGN.md training pipeline)
# ---------------------------------------------------------------------------

DEFAULT_CLASSES: list[str] = [
    "button",
    "menu",
    "input",
    "checkbox",
    "radio_button",
    "dropdown",
    "dialog",
    "menu_bar",
    "toolbar",
    "label",
    "tab",
    "scrollbar",
    "icon",
    "tab_bar",
    "status_bar",
    "other",
]

# Normalise alternative class names produced by HybridAnnotator / VLMs
_CLASS_ALIASES: dict[str, str] = {
    "menu_item": "menu",
    "input_field": "input",
    "text_field": "input",
    "text": "label",
    "image": "icon",
    "link": "button",
    "button_icon": "button",
    "combobox": "dropdown",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConvertResult:
    """Summary returned by AnnotationConverter.convert_session()."""

    output_dir: Path
    n_images: int = 0
    n_labels: int = 0
    n_elements: int = 0
    n_skipped: int = 0
    classes: list[str] = field(default_factory=list)
    dataset_yaml: Path = field(default_factory=Path)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class AnnotationConverter:
    """Convert annotations.jsonl + video.mp4 into a YOLO-format training dataset.

    Output is compatible with ``gazefy prep`` → ``gazefy train``.
    """

    def __init__(self, skip_unknown: bool = True):
        """
        Args:
            skip_unknown: Skip elements whose label is "unknown" or "unknown icon"
                          (produced when VLM/detector fails). Default True.
        """
        self.skip_unknown = skip_unknown

    def convert_session(
        self,
        session_dir: str | Path,
        output_dir: str | Path | None = None,
        class_names: list[str] | None = None,
        min_bbox_px: int = 4,
    ) -> ConvertResult:
        """Convert one recorded+annotated session into YOLO format.

        Args:
            session_dir:  Directory containing video.mp4 + annotations.jsonl.
            output_dir:   Destination for images/ labels/ dataset.yaml.
                          Defaults to ``session_dir/yolo_dataset``.
            class_names:  Custom class taxonomy. Defaults to DEFAULT_CLASSES.
            min_bbox_px:  Discard boxes whose width or height < this many pixels.

        Returns:
            ConvertResult with output path and element/image counts.
        """
        session_dir = Path(session_dir)
        output_dir = Path(output_dir) if output_dir else session_dir / "yolo_dataset"

        ann_path = session_dir / "annotations.jsonl"
        video_path = session_dir / "video.mp4"
        frame_times_path = session_dir / "frame_times.json"

        if not ann_path.exists():
            raise FileNotFoundError(f"annotations.jsonl not found in {session_dir}")
        if not video_path.exists():
            raise FileNotFoundError(f"video.mp4 not found in {session_dir}")

        classes = list(class_names) if class_names else list(DEFAULT_CLASSES)
        class_index: dict[str, int] = {c: i for i, c in enumerate(classes)}

        raw_lines = [ln for ln in ann_path.read_text().splitlines() if ln.strip()]
        annotations = [json.loads(ln) for ln in raw_lines]
        if not annotations:
            logger.warning("annotations.jsonl is empty: %s", ann_path)
            result = ConvertResult(output_dir=output_dir, classes=classes)
            result.dataset_yaml = output_dir / "dataset.yaml"
            return result

        frame_times: list[float] = []
        if frame_times_path.exists():
            frame_times = json.loads(frame_times_path.read_text())

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        result = ConvertResult(output_dir=output_dir, classes=classes)
        result.dataset_yaml = output_dir / "dataset.yaml"

        try:
            for ann_idx, ann in enumerate(annotations):
                t = float(ann.get("t", 0.0))
                elements = ann.get("elements", [])
                if not elements:
                    result.n_skipped += 1
                    continue

                frame = _extract_frame(cap, t, fps, frame_times, total_frames)
                if frame is None:
                    result.n_skipped += 1
                    continue

                h, w = frame.shape[:2]
                stem = f"frame_{ann_idx:06d}_t{int(t * 100):07d}"

                lines: list[str] = []
                for el in elements:
                    yolo_line = _element_to_yolo(
                        el, w, h, class_index, min_bbox_px, self.skip_unknown
                    )
                    if yolo_line is not None:
                        lines.append(yolo_line)
                        result.n_elements += 1

                if not lines:
                    result.n_skipped += 1
                    continue

                img_path = images_dir / f"{stem}.png"
                cv2.imwrite(str(img_path), frame)
                result.n_images += 1

                lbl_path = labels_dir / f"{stem}.txt"
                lbl_path.write_text("\n".join(lines) + "\n")
                result.n_labels += 1

        finally:
            cap.release()

        _write_dataset_yaml(result.dataset_yaml, output_dir, classes)
        logger.info(
            "Converted %d images, %d elements → %s",
            result.n_images,
            result.n_elements,
            output_dir,
        )
        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_frame(
    cap: cv2.VideoCapture,
    t: float,
    fps: float,
    frame_times: list[float],
    total_frames: int,
) -> np.ndarray | None:
    """Seek to timestamp t and return the frame (BGR)."""
    if frame_times:
        idx = min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - t))
    else:
        idx = int(t * fps)
    idx = max(0, min(idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ret, frame = cap.read()
    return frame if ret else None


def _normalise_class(raw: str) -> str:
    """Map annotation class name to a canonical DEFAULT_CLASSES entry."""
    raw = raw.strip().lower()
    return _CLASS_ALIASES.get(raw, raw)


def _element_to_yolo(
    el: dict,
    img_w: int,
    img_h: int,
    class_index: dict[str, int],
    min_bbox_px: int,
    skip_unknown: bool,
) -> str | None:
    """Convert one element dict to a YOLO label line, or None if it should be skipped."""
    label = str(el.get("label", "")).strip()
    if skip_unknown and label.lower() in ("unknown", "unknown icon", ""):
        return None

    raw_class = str(el.get("class", el.get("element_class", "other")))
    norm_class = _normalise_class(raw_class)
    class_id = class_index.get(norm_class, class_index.get("other", 0))

    bbox = el.get("bbox", [])
    if len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    x2 = min(x2, img_w)
    y2 = min(y2, img_h)
    bw = x2 - x1
    bh = y2 - y1
    if bw < min_bbox_px or bh < min_bbox_px:
        return None

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def _write_dataset_yaml(dst: Path, output_dir: Path, classes: list[str]) -> None:
    """Write a dataset.yaml compatible with Ultralytics YOLO training."""
    ds = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: c for i, c in enumerate(classes)},
    }
    dst.write_text(yaml.dump(ds, default_flow_style=False))
