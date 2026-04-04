"""Hybrid annotator: GroundingDINO bboxes + EasyOCR text + Claude Vision for icons.

Pipeline per key frame
──────────────────────
Stage 1  Detect all UI element bounding boxes
         • If a trained YOLO pack is supplied → use it (highest quality)
         • Else → GroundingDINO zero-shot detection
         • Else → skip detection, fall back to pure-VLM (VideoAnnotator)

Stage 2  EasyOCR (already in project) on every detected bbox
         Elements whose text is readable → labelled for free, no API call.

Stage 3  Claude Vision for icon-only elements (no OCR text found)
         All icons in a frame are sent in ONE API call:
         • Frame is drawn with numbered red boxes around each icon
         • Claude sees full spatial context and labels every numbered element
         • Response: {"labels": [{"n": 1, "label": "Pencil Tool"}, …]}

Output  annotations.jsonl  (same FrameAnnotation format as VideoAnnotator)
        {
          "t": 1.23,
          "mouse_x": 452, "mouse_y": 310,
          "action": "click_left",
          "elements": [
            {"label": "File",        "class": "menu",   "bbox": [10, 5, 50, 25],  "source": "ocr"},
            {"label": "Brush Tool",  "class": "icon",   "bbox": [5, 70, 45, 100], "source": "vlm"},
            {"label": "Save Button", "class": "button", "bbox": [80, 5, 130, 25], "source": "ocr"},
          ]
        }

Usage
─────
    annotator = HybridAnnotator()
    annotator.annotate_session(Path("recordings/session_xxx"))
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _sanitize(obj):
    """Convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Data models  (same as video_annotator for compatibility)
# ---------------------------------------------------------------------------


@dataclass
class UIElement:
    label: str
    element_class: str
    bbox: list[int]  # [x1, y1, x2, y2] original pixels
    source: str = "vlm"  # "ocr" | "vlm" | "yolo+ocr" | "yolo+vlm"


@dataclass
class FrameAnnotation:
    t: float
    mouse_x: int
    mouse_y: int
    action: str | None
    elements: list[UIElement] = field(default_factory=list)
    # Populated for click frames: offline before/after ChangeDetector comparison
    click_verified: bool = False  # True if screen changed after this click
    diff_score: float = 0.0  # Change magnitude (0=none, higher=more change)

    def to_dict(self) -> dict:
        d: dict = {
            "t": float(self.t),
            "mouse_x": int(self.mouse_x),
            "mouse_y": int(self.mouse_y),
            "action": self.action,
            "elements": [_sanitize(asdict(e)) for e in self.elements],
        }
        if self.action:
            d["click_verified"] = self.click_verified
            d["diff_score"] = round(float(self.diff_score), 4)
        return d


# ---------------------------------------------------------------------------
# HybridAnnotator
# ---------------------------------------------------------------------------


class HybridAnnotator:
    """Three-stage annotator: detector → OCR → LLM (icons only)."""

    _MAX_VLM_WIDTH = 1280  # resize before sending to Claude
    _MAX_ICONS_PER_CALL = 30  # max numbered boxes per Claude call

    def __init__(
        self,
        sample_interval: float = 3.0,
        box_threshold: float = 0.30,
        grounding_device: str = "cpu",
        pack_dir: str | Path | None = None,
    ):
        """
        Args:
            sample_interval: Annotate a periodic frame every N seconds in
                             addition to click frames. 0 = clicks only.
            box_threshold: GroundingDINO confidence threshold.
            grounding_device: "cpu" or "mps" or "cuda".
            pack_dir: If provided, load the YOLO model from this pack directory
                      instead of using GroundingDINO.
        """
        self.sample_interval = sample_interval
        self.box_threshold = box_threshold
        self.grounding_device = grounding_device
        self.pack_dir = Path(pack_dir) if pack_dir else None

        self._grounding: object | None = None  # GroundingDetector
        self._yolo: object | None = None  # UIDetector
        self._ocr: object | None = None  # ElementOCR
        self._client = None  # anthropic.Anthropic

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_session(
        self,
        session_dir: Path,
        sample_interval: float | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[FrameAnnotation]:
        """Annotate all key frames in session_dir.

        Args:
            session_dir: Must contain video.mp4 + events.jsonl.
            sample_interval: Override instance default.
            on_progress: Callback(current, total, description).
        """
        interval = sample_interval if sample_interval is not None else self.sample_interval

        events_path = session_dir / "events.jsonl"
        video_path = session_dir / "video.mp4"
        frame_times_path = session_dir / "frame_times.json"

        if not events_path.exists():
            raise FileNotFoundError(f"Missing events.jsonl in {session_dir}")
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video.mp4 in {session_dir}")

        with open(events_path) as f:
            events = [json.loads(line) for line in f if line.strip()]

        frame_times: list[float] = []
        if frame_times_path.exists():
            frame_times = json.loads(frame_times_path.read_text())

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        duration = frame_times[-1] if frame_times else (total_frames / fps)

        key_frames = self._select_key_frames(events, duration, interval)

        self._init_detector()
        self._init_ocr()
        self._init_llm()

        annotations: list[FrameAnnotation] = []
        try:
            for i, kf in enumerate(key_frames):
                t = kf["t"]
                mx, my = kf["mouse_x"], kf["mouse_y"]
                action = kf.get("action")

                if on_progress:
                    on_progress(
                        i + 1, len(key_frames), f"t={t:.1f}s  {action or 'scan'}  ({mx},{my})"
                    )

                frame = self._get_frame_at_time(cap, t, fps, frame_times, total_frames)
                if frame is None:
                    continue

                elements = self._process_frame(frame, mx, my, action)
                ann = FrameAnnotation(t=t, mouse_x=mx, mouse_y=my, action=action, elements=elements)

                # For click frames: verify using before/after frame comparison
                # (same ChangeDetector as ActionExecutor, applied offline)
                if action:
                    verified, diff = self._verify_click(cap, t, fps, frame_times, total_frames)
                    ann.click_verified = verified
                    ann.diff_score = diff

                annotations.append(ann)

                n_ocr = sum(1 for e in elements if "ocr" in e.source)
                n_vlm = sum(1 for e in elements if "vlm" in e.source)
                if on_progress:
                    verify_str = ""
                    if action:
                        verify_str = "  ✓" if ann.click_verified else "  ✗ no change"
                    on_progress(
                        i + 1,
                        len(key_frames),
                        f"t={t:.1f}s → {len(elements)} elements "
                        f"({n_ocr} OCR, {n_vlm} VLM){verify_str}",
                    )
        finally:
            cap.release()

        ann_path = session_dir / "annotations.jsonl"
        with open(ann_path, "w") as f:
            for ann in annotations:
                f.write(json.dumps(ann.to_dict()) + "\n")

        return annotations

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame: np.ndarray,
        mouse_x: int,
        mouse_y: int,
        action: str | None,
    ) -> list[UIElement]:
        """Stage 1→2→3: detect → OCR → VLM for icons."""
        orig_h, orig_w = frame.shape[:2]

        # --- Stage 1: detect bboxes ---
        detected = self._detect_bboxes(frame)  # list of (bbox, class_hint)

        if not detected:
            # No detector available: fall back to full-frame VLM
            return self._fallback_vlm(frame, mouse_x, mouse_y, action)

        # --- Stage 2: OCR each bbox ---
        text_elements: list[UIElement] = []
        icon_items: list[tuple[int, list[int], str]] = []  # (idx, bbox, class_hint)

        for idx, (bbox, class_hint) in enumerate(detected):
            text = self._ocr_bbox(frame, bbox)
            if text:
                # Readable text → label directly from OCR
                el_class = self._infer_class_from_text(bbox, class_hint)
                text_elements.append(
                    UIElement(
                        label=text,
                        element_class=el_class,
                        bbox=bbox,
                        source="ocr" if not class_hint else "yolo+ocr",
                    )
                )
            else:
                # No text → send to VLM
                icon_items.append((idx, bbox, class_hint))

        # --- Stage 3: batch VLM for icons ---
        icon_elements: list[UIElement] = []
        if icon_items:
            icon_elements = self._label_icons_on_frame(frame, icon_items)

        return text_elements + icon_elements

    # ------------------------------------------------------------------
    # Stage 1: detection
    # ------------------------------------------------------------------

    def _init_detector(self) -> None:
        if self.pack_dir and (self.pack_dir / "pack.yaml").exists():
            try:
                from gazefy.core.application_pack import ApplicationPack
                from gazefy.detection.detector import UIDetector

                pack = ApplicationPack.load(self.pack_dir)
                self._yolo = UIDetector(pack)
                self._yolo.load_model()
                logger.info("YOLO detector loaded from pack: %s", self.pack_dir)
                return
            except Exception as e:
                logger.warning("Failed to load YOLO pack: %s — falling back to GroundingDINO", e)

        try:
            from gazefy.detection.grounding import GroundingDetector

            self._grounding = GroundingDetector(
                box_threshold=self.box_threshold,
                device=self.grounding_device,
            )
            self._grounding.load()
            logger.info("GroundingDINO detector ready")
        except Exception as e:
            logger.warning("GroundingDINO unavailable: %s — will use full-frame VLM", e)

    def _detect_bboxes(self, frame: np.ndarray) -> list[tuple[list[int], str]]:
        """Returns list of (bbox, class_hint). class_hint may be empty string."""
        if self._yolo is not None:
            try:
                dets = self._yolo.detect(frame)
                return [
                    ([int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2)], d.class_name)
                    for d in dets
                ]
            except Exception as e:
                logger.warning("YOLO detection failed: %s", e)

        if self._grounding is not None:
            try:
                dets = self._grounding.detect(frame)
                return [(d.bbox, d.label) for d in dets]
            except Exception as e:
                logger.warning("GroundingDINO detection failed: %s", e)

        return []

    # ------------------------------------------------------------------
    # Stage 2: OCR
    # ------------------------------------------------------------------

    def _init_ocr(self) -> None:
        try:
            from gazefy.detection.ocr import ElementOCR

            self._ocr = ElementOCR()
            logger.info("EasyOCR ready")
        except Exception as e:
            logger.warning("EasyOCR unavailable: %s", e)

    def _ocr_bbox(self, frame: np.ndarray, bbox: list[int]) -> str:
        if self._ocr is None:
            return ""
        try:
            return self._ocr.read_element_text(frame, tuple(bbox))
        except Exception:
            return ""

    @staticmethod
    def _infer_class_from_text(bbox: list[int], class_hint: str) -> str:
        """Infer element class when OCR text is available."""
        if class_hint:
            return class_hint
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        # Heuristic: wide short elements = button/menu, tall narrow = scrollbar
        if h > 0 and w / h > 4:
            return "menu"
        if h > 0 and h / w > 3:
            return "scrollbar"
        return "button"

    # ------------------------------------------------------------------
    # Stage 3: VLM batch for icons
    # ------------------------------------------------------------------

    def _init_llm(self) -> None:
        try:
            from gazefy.llm.copilot import CopilotClient

            self._client = CopilotClient(model="gpt-4o")
            logger.info("Copilot+gpt-4o client ready")
        except Exception as e:
            logger.warning("LLM init failed: %s", e)

    def _label_icons_on_frame(
        self,
        frame: np.ndarray,
        icon_items: list[tuple[int, list[int], str]],
    ) -> list[UIElement]:
        """Draw numbered boxes on frame, send to Claude, return UIElement list.

        Processes in batches of _MAX_ICONS_PER_CALL if there are many icons.
        """
        elements: list[UIElement] = []
        # Process in batches
        for batch_start in range(0, len(icon_items), self._MAX_ICONS_PER_CALL):
            batch = icon_items[batch_start : batch_start + self._MAX_ICONS_PER_CALL]
            batch_elements = self._label_icon_batch(frame, batch, offset=batch_start)
            elements.extend(batch_elements)
        return elements

    def _label_icon_batch(
        self,
        frame: np.ndarray,
        batch: list[tuple[int, list[int], str]],
        offset: int = 0,
    ) -> list[UIElement]:
        """One Claude call for up to _MAX_ICONS_PER_CALL icon elements."""
        if self._client is None:
            # No LLM: return unknown placeholders
            return [
                UIElement(
                    label=class_hint or "unknown icon",
                    element_class=class_hint or "icon",
                    bbox=bbox,
                    source="grounding",
                )
                for _, bbox, class_hint in batch
            ]

        # Draw numbered red boxes on a copy of the frame
        vis = frame.copy()
        for local_n, (_, bbox, _) in enumerate(batch, start=1):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Number label background
            lbl = str(local_n)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(lbl, font, 0.55, 2)
            cv2.rectangle(vis, (x1, y1), (x1 + tw + 4, y1 + th + 4), (0, 0, 255), -1)
            cv2.putText(vis, lbl, (x1 + 2, y1 + th + 2), font, 0.55, (255, 255, 255), 2)

        # Resize for VLM
        h, w = vis.shape[:2]
        if w > self._MAX_VLM_WIDTH:
            scale = self._MAX_VLM_WIDTH / w
            vis = cv2.resize(
                vis, (self._MAX_VLM_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA
            )

        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.standard_b64encode(buf).decode()

        n = len(batch)
        class_hints = [class_hint or "icon" for _, _, class_hint in batch]
        hint_str = ", ".join(f"{i + 1}={h}" for i, h in enumerate(class_hints))

        prompt = (
            f"This is a desktop application screenshot. "
            f"{n} UI elements are marked with numbered red boxes ({hint_str}).\n"
            "These elements have no readable text — icons, symbols, or graphic buttons.\n\n"
            "For each numbered element, provide:\n"
            '- "label": short name, e.g. "Pencil Tool", "Undo", "File Menu", "Close Tab"\n'
            '- "class": one of icon | button | menu | toolbar | scrollbar | other\n\n'
            "Reply with JSON only:\n"
            '{"labels": [{"n": 1, "label": "...", "class": "..."}, ...]}'
        )

        try:
            text = self._client.chat_with_image(
                prompt, frame_b64, media_type="image/jpeg", max_tokens=1024
            )
        except Exception as e:
            logger.warning("VLM API error: %s", e)
            return [
                UIElement(
                    label=class_hint or "icon",
                    element_class=class_hint or "icon",
                    bbox=bbox,
                    source="grounding",
                )
                for _, bbox, class_hint in batch
            ]

        # Parse response
        label_map = self._parse_icon_labels(text)

        elements: list[UIElement] = []
        for local_n, (_, bbox, class_hint) in enumerate(batch, start=1):
            item = label_map.get(local_n, {})
            known = ("icon", "button", "menu")
            source = "yolo+vlm" if class_hint and class_hint not in known else "vlm"
            elements.append(
                UIElement(
                    label=item.get("label", class_hint or "unknown"),
                    element_class=item.get("class", class_hint or "icon"),
                    bbox=bbox,
                    source=source,
                )
            )
        return elements

    @staticmethod
    def _parse_icon_labels(text: str) -> dict[int, dict]:
        """Parse {'labels': [{'n': 1, 'label': '...', 'class': '...'}, ...]}"""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return {}
        result = {}
        for item in data.get("labels", []):
            n = item.get("n")
            if isinstance(n, int):
                result[n] = item
        return result

    # ------------------------------------------------------------------
    # Fallback: full-frame VLM (no detector available)
    # ------------------------------------------------------------------

    def _fallback_vlm(
        self,
        frame: np.ndarray,
        mouse_x: int,
        mouse_y: int,
        action: str | None,
    ) -> list[UIElement]:
        """Send entire frame to Claude, ask for all elements with bboxes."""
        if self._client is None:
            return []

        h, w = frame.shape[:2]
        vis = frame.copy()
        color = (0, 0, 255) if action else (0, 200, 0)
        cv2.circle(vis, (mouse_x, mouse_y), 10, color, 2)
        cv2.circle(vis, (mouse_x, mouse_y), 4, color, -1)

        if w > self._MAX_VLM_WIDTH:
            scale = self._MAX_VLM_WIDTH / w
            vis = cv2.resize(vis, (self._MAX_VLM_WIDTH, int(h * scale)))

        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.standard_b64encode(buf).decode()

        cursor_desc = (
            f"action: {action} at ({mouse_x},{mouse_y})"
            if action
            else f"mouse at ({mouse_x},{mouse_y})"
        )
        prompt = (
            f"Desktop application screenshot ({cursor_desc}, marked with a dot).\n"
            "Identify ALL visible UI elements with approximate bounding boxes.\n\n"
            '{"elements": [{"label": "...", "class": "...", "bbox": [x1,y1,x2,y2]}, ...]}\n'
            "Classes: button | menu | icon | input | tab | checkbox | dropdown | scrollbar | other"
        )

        try:
            text = self._client.chat_with_image(
                prompt, frame_b64, media_type="image/jpeg", max_tokens=4096
            )
        except Exception as e:
            logger.warning("Fallback VLM error: %s", e)
            return []

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return []

        elements = []
        for item in data.get("elements", []):
            bbox = item.get("bbox", [0, 0, w, h])
            if len(bbox) == 4:
                elements.append(
                    UIElement(
                        label=str(item.get("label", "unknown")),
                        element_class=str(item.get("class", "other")),
                        bbox=[max(0, int(v)) for v in bbox],
                        source="vlm",
                    )
                )
        return elements

    # ------------------------------------------------------------------
    # Key frame selection (shared with VideoAnnotator)
    # ------------------------------------------------------------------

    def _select_key_frames(
        self, events: list[dict], duration: float, interval: float
    ) -> list[dict]:
        positions = sorted((e["t"], e["x"], e["y"]) for e in events)

        def mouse_at(t: float) -> tuple[int, int]:
            if not positions:
                return 0, 0
            best = min(positions, key=lambda p: abs(p[0] - t))
            return best[1], best[2]

        candidates: list[dict] = []

        if interval > 0 and duration > 0:
            t = 0.0
            while t <= duration:
                mx, my = mouse_at(t)
                candidates.append({"t": round(t, 2), "mouse_x": mx, "mouse_y": my})
                t += interval

        for ev in events:
            if ev.get("click"):
                candidates.append(
                    {
                        "t": ev["t"],
                        "mouse_x": ev["x"],
                        "mouse_y": ev["y"],
                        "action": f"click_{ev['click']}",
                    }
                )

        candidates.sort(key=lambda c: c["t"])

        result: list[dict] = []
        last_t = -999.0
        for c in candidates:
            if c.get("action") or (c["t"] - last_t >= 0.5):
                result.append(c)
                last_t = c["t"]

        return result

    # ------------------------------------------------------------------
    # Video helpers
    # ------------------------------------------------------------------

    def _verify_click(
        self,
        cap,
        t: float,
        fps: float,
        frame_times: list[float],
        total_frames: int,
        before_s: float = 0.4,
        after_s: float = 1.0,
    ) -> tuple[bool, float]:
        """Offline click verification: compare frames before and after a click.

        Uses the same ChangeDetector as ActionExecutor._verify_change(),
        applied to pre-recorded frames instead of live screen captures.

        Returns:
            (click_verified, diff_score)
        """
        from gazefy.capture.change_detector import ChangeDetector

        before = self._get_frame_at_time(
            cap, max(0.0, t - before_s), fps, frame_times, total_frames
        )
        after = self._get_frame_at_time(cap, t + after_s, fps, frame_times, total_frames)

        if before is None or after is None:
            return False, 0.0

        # ChangeDetector expects BGRA; VideoCapture frames are BGR
        detector = ChangeDetector()
        detector.check(cv2.cvtColor(before, cv2.COLOR_BGR2BGRA))
        result = detector.check(cv2.cvtColor(after, cv2.COLOR_BGR2BGRA))
        return result.changed, result.diff_score

    def _get_frame_at_time(
        self,
        cap,
        t: float,
        fps: float,
        frame_times: list[float],
        total_frames: int,
    ):
        if frame_times:
            idx = min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - t))
        else:
            idx = int(t * fps)
        idx = max(0, min(idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        return frame if ret else None
