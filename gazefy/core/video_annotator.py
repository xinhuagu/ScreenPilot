"""Annotate a recorded video session using VLM full-frame analysis.

For each key frame (every click + periodic interval):
  1. Seek to the nearest video frame at time t
  2. Overlay the mouse cursor position as a red dot
  3. Send the full screenshot (resized) to Claude Vision
  4. Claude identifies ALL visible UI elements (buttons, menus, icons, ...)
     and returns their labels, classes, and approximate bounding boxes
  5. Write one rich annotation record per frame to annotations.jsonl

Output format (one JSON object per line in annotations.jsonl):
    {
        "t": 1.23,
        "mouse_x": 452,
        "mouse_y": 310,
        "action": "click_left",     // or null for periodic samples
        "elements": [
            {"label": "File Menu",   "class": "menu",   "bbox": [10, 5, 50, 25]},
            {"label": "Brush Tool",  "class": "icon",   "bbox": [5, 70, 45, 100]},
            ...
        ]
    }

Usage:
    annotator = VideoAnnotator()
    annotator.annotate_session(
        Path("recordings/session_xxx"),
        sample_interval=3.0,       # also annotate every 3 s
        on_progress=my_callback,
    )
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class UIElement:
    label: str
    element_class: str
    bbox: list[int]  # [x1, y1, x2, y2] in original frame pixels


@dataclass
class FrameAnnotation:
    t: float
    mouse_x: int
    mouse_y: int
    action: str | None  # "click_left", "click_right", or None
    elements: list[UIElement] = field(default_factory=list)
    # Populated for click frames: offline before/after ChangeDetector comparison
    click_verified: bool = False  # True if screen changed after this click
    diff_score: float = 0.0  # Change magnitude (0=none, higher=more change)

    def to_dict(self) -> dict:
        d: dict = {
            "t": self.t,
            "mouse_x": self.mouse_x,
            "mouse_y": self.mouse_y,
            "action": self.action,
            "elements": [asdict(e) for e in self.elements],
        }
        if self.action:  # Only include verification fields for click frames
            d["click_verified"] = self.click_verified
            d["diff_score"] = round(float(self.diff_score), 4)
        return d


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------


class VideoAnnotator:
    """Annotates a recording session: for each key frame, identifies all
    visible UI elements using Claude Vision.

    Key frames = all click events + periodic samples every sample_interval seconds.
    """

    # Max width to send to VLM (resize for speed; bboxes are scaled back)
    _MAX_VLM_WIDTH = 1280

    def __init__(self, sample_interval: float = 3.0):
        """
        Args:
            sample_interval: Annotate a frame every this many seconds in addition
                             to click frames. Set to 0 to only annotate click frames.
        """
        self.sample_interval = sample_interval

    def annotate_session(
        self,
        session_dir: Path,
        sample_interval: float | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[FrameAnnotation]:
        """Annotate all key frames in a session directory.

        Args:
            session_dir: Path containing video.mp4 + events.jsonl.
            sample_interval: Override instance default.
            on_progress: Callback(current, total, description).

        Returns:
            List of FrameAnnotation, also written to annotations.jsonl.
        """
        import cv2

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

        annotations: list[FrameAnnotation] = []
        try:
            for i, kf in enumerate(key_frames):
                t = kf["t"]
                mouse_x = kf["mouse_x"]
                mouse_y = kf["mouse_y"]
                action = kf.get("action")

                if on_progress:
                    action_str = action or "scan"
                    on_progress(
                        i + 1,
                        len(key_frames),
                        f"t={t:.1f}s  {action_str}  ({mouse_x},{mouse_y})",
                    )

                frame = self._get_frame_at_time(cap, t, fps, frame_times, total_frames)
                if frame is None:
                    continue

                # Draw cursor on frame copy
                vis = self._draw_cursor(frame, mouse_x, mouse_y, action)

                # Resize for VLM, remember scale
                vis_small, scale = self._resize_for_vlm(vis)
                frame_b64 = self._to_b64(vis_small)

                orig_h, orig_w = frame.shape[:2]

                try:
                    elements = self._ask_vlm(
                        frame_b64, mouse_x, mouse_y, action, orig_w, orig_h, scale
                    )
                except Exception as e:
                    elements = [
                        UIElement(
                            label=f"VLM error: {e}",
                            element_class="other",
                            bbox=[0, 0, orig_w, orig_h],
                        )
                    ]

                ann = FrameAnnotation(
                    t=t, mouse_x=mouse_x, mouse_y=mouse_y, action=action, elements=elements
                )
                # For click frames: verify using before/after frame comparison
                # (same ChangeDetector as ActionExecutor, but applied offline)
                if action:
                    verified, diff = self._verify_click(cap, t, fps, frame_times, total_frames)
                    ann.click_verified = verified
                    ann.diff_score = diff

                annotations.append(ann)

                if on_progress:
                    verify_str = ""
                    if action:
                        verify_str = "  ✓" if ann.click_verified else "  ✗ no change"
                    on_progress(
                        i + 1,
                        len(key_frames),
                        f"t={t:.1f}s → {len(elements)} elements{verify_str}",
                    )
        finally:
            cap.release()

        ann_path = session_dir / "annotations.jsonl"
        with open(ann_path, "w") as f:
            for ann in annotations:
                f.write(json.dumps(ann.to_dict()) + "\n")

        return annotations

    # --- key frame selection ---

    def _select_key_frames(
        self, events: list[dict], duration: float, interval: float
    ) -> list[dict]:
        """Build list of {t, mouse_x, mouse_y, action?} to annotate.

        Includes:
        - All click events
        - Periodic samples every `interval` seconds (if interval > 0)

        Mouse position for non-click frames is interpolated from move events.
        Frames closer than 0.5 s to each other are deduplicated.
        """
        # Build a sorted list of (t, x, y) from all events for interpolation
        positions: list[tuple[float, int, int]] = [(e["t"], e["x"], e["y"]) for e in events]
        positions.sort()

        def mouse_at(t: float) -> tuple[int, int]:
            if not positions:
                return 0, 0
            # Find nearest position
            best = min(positions, key=lambda p: abs(p[0] - t))
            return best[1], best[2]

        candidates: list[dict] = []

        # Periodic samples
        if interval > 0 and duration > 0:
            t = 0.0
            while t <= duration:
                mx, my = mouse_at(t)
                candidates.append({"t": round(t, 2), "mouse_x": mx, "mouse_y": my})
                t += interval

        # Click events always included
        for ev in events:
            if ev.get("click"):
                btn = f"click_{ev['click']}"
                candidates.append(
                    {
                        "t": ev["t"],
                        "mouse_x": ev["x"],
                        "mouse_y": ev["y"],
                        "action": btn,
                    }
                )

        # Sort by time
        candidates.sort(key=lambda c: c["t"])

        # Deduplicate: skip frames within 0.5 s of a previous one
        # but always keep clicks even if close
        result: list[dict] = []
        last_t = -999.0
        for c in candidates:
            if c.get("action") or (c["t"] - last_t >= 0.5):
                result.append(c)
                last_t = c["t"]

        return result

    # --- video helpers ---

    def _get_frame_at_time(
        self,
        cap,
        t: float,
        fps: float,
        frame_times: list[float],
        total_frames: int,
    ):
        import cv2

        if frame_times:
            idx = min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - t))
        else:
            idx = int(t * fps)

        idx = max(0, min(idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        return frame if ret else None

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

        Args:
            before_s: How many seconds before the click to take the "before" frame.
            after_s:  How many seconds after the click to check for change.

        Returns:
            (click_verified, diff_score)
        """
        import cv2

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

    def _draw_cursor(self, frame, mx: int, my: int, action: str | None):
        """Draw a cursor marker on a copy of frame."""
        import cv2

        vis = frame.copy()
        h, w = vis.shape[:2]
        mx = max(0, min(mx, w - 1))
        my = max(0, min(my, h - 1))

        if action:  # click: red filled circle + ring
            cv2.circle(vis, (mx, my), 14, (0, 0, 255), 2)
            cv2.circle(vis, (mx, my), 5, (0, 0, 255), -1)
        else:  # move: small green dot
            cv2.circle(vis, (mx, my), 6, (0, 200, 0), -1)
            cv2.circle(vis, (mx, my), 6, (255, 255, 255), 1)

        return vis

    def _resize_for_vlm(self, frame) -> tuple:
        """Resize frame so width <= _MAX_VLM_WIDTH. Returns (resized, scale)."""
        import cv2

        h, w = frame.shape[:2]
        if w <= self._MAX_VLM_WIDTH:
            return frame, 1.0
        scale = self._MAX_VLM_WIDTH / w
        new_w = self._MAX_VLM_WIDTH
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _to_b64(self, frame) -> str:
        import cv2

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.standard_b64encode(buf).decode()

    # --- VLM ---

    def _ask_vlm(
        self,
        frame_b64: str,
        mouse_x: int,
        mouse_y: int,
        action: str | None,
        orig_w: int,
        orig_h: int,
        scale: float,
    ) -> list[UIElement]:
        """Send a full frame to Claude Vision; return all detected UI elements."""
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("Install LLM extra: pip install gazefy[llm]")

        from gazefy.llm.credentials import get_api_key

        api_key = get_api_key()
        if not api_key:
            raise RuntimeError("No API key. Set ANTHROPIC_API_KEY or use Claude Code login.")

        vlm_w = int(orig_w * scale)
        vlm_h = int(orig_h * scale)

        if action:
            cursor_desc = (
                f"The user performed **{action}** at pixel ({mouse_x}, {mouse_y}) "
                f"(marked with a red circle in the image)."
            )
        else:
            cursor_desc = (
                f"The mouse cursor is at pixel ({mouse_x}, {mouse_y}) "
                f"(marked with a green dot in the image)."
            )

        prompt = f"""This is a screenshot of a desktop application.
Image size: {vlm_w}×{vlm_h} px (original: {orig_w}×{orig_h} px).
{cursor_desc}

Identify ALL visible interactive UI elements:
buttons, menus, icons, toolbars, inputs, checkboxes, tabs, dropdowns, scrollbars, etc.

For each element provide:
- "label": short name, e.g. "File Menu", "Save Button", "Brush Tool"
- "class": button|menu|menu_item|icon|toolbar|input|checkbox|tab|dropdown|scrollbar|other
- "bbox": [x1, y1, x2, y2] in original image pixels ({orig_w}×{orig_h})

Reply with JSON only, no explanation:
{{"elements": [{{"label": "...", "class": "...", "bbox": [x1, y1, x2, y2]}}, ...]}}"""

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": frame_b64,
                            },
                        },
                    ],
                }
            ],
        )

        text = response.content[0].text.strip()
        return self._parse_elements(text, orig_w, orig_h)

    def _parse_elements(self, text: str, orig_w: int, orig_h: int) -> list[UIElement]:
        """Parse VLM JSON response into UIElement list."""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return []

        elements = []
        for item in data.get("elements", []):
            bbox = item.get("bbox", [0, 0, orig_w, orig_h])
            if len(bbox) != 4:
                continue
            # Clamp to frame bounds
            x1 = max(0, min(int(bbox[0]), orig_w))
            y1 = max(0, min(int(bbox[1]), orig_h))
            x2 = max(0, min(int(bbox[2]), orig_w))
            y2 = max(0, min(int(bbox[3]), orig_h))
            elements.append(
                UIElement(
                    label=str(item.get("label", "Unknown")),
                    element_class=str(item.get("class", "other")),
                    bbox=[x1, y1, x2, y2],
                )
            )
        return elements
