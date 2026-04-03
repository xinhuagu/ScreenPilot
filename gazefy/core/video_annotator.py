"""Annotate a recorded video session using VLM analysis.

For each click event in events.jsonl:
  1. Seek to the nearest video frame at click time t
  2. Crop around the click position (with red crosshair overlay)
  3. Send to Claude Vision to identify the UI element
  4. Write results to annotations.jsonl

Usage:
    annotator = VideoAnnotator()
    annotations = annotator.annotate_session(Path("recordings/session_xxx"))
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Annotation:
    t: float
    x: int
    y: int
    click: str
    label: str
    element_class: str


class VideoAnnotator:
    """Annotates click events in a recording session using Claude Vision."""

    def __init__(self, crop_size: int = 200, context_size: int = 450):
        self.crop_size = crop_size
        self.context_size = context_size

    def annotate_session(
        self,
        session_dir: Path,
        on_progress: Callable[[int, int, Annotation], None] | None = None,
    ) -> list[Annotation]:
        """Annotate all clicks in a session directory.

        Args:
            session_dir: Path to session directory (must contain video.mp4 + events.jsonl).
            on_progress: Optional callback(current, total, annotation) for UI updates.

        Returns:
            List of Annotation dataclasses, also written to annotations.jsonl.
        """
        import cv2

        events_path = session_dir / "events.jsonl"
        video_path = session_dir / "video.mp4"
        frame_times_path = session_dir / "frame_times.json"

        if not events_path.exists():
            raise FileNotFoundError(f"Missing events.jsonl in {session_dir}")
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video.mp4 in {session_dir}")

        with open(events_path) as f:
            events = [json.loads(line) for line in f if line.strip()]

        clicks = [e for e in events if e.get("click")]
        if not clicks:
            return []

        frame_times: list[float] = []
        if frame_times_path.exists():
            frame_times = json.loads(frame_times_path.read_text())

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

        annotations: list[Annotation] = []
        try:
            for i, click in enumerate(clicks):
                frame = self._get_frame_at_time(
                    cap, click["t"], fps, frame_times, total_frames
                )
                if frame is None:
                    continue

                x, y = click["x"], click["y"]
                icon_b64 = self._crop_b64(frame, x, y, self.crop_size)
                context_b64 = self._crop_b64(frame, x, y, self.context_size)

                try:
                    label, el_class = self._ask_vlm(icon_b64, context_b64)
                except Exception as e:
                    label, el_class = f"error: {e}", "unknown"

                ann = Annotation(
                    t=click["t"],
                    x=x,
                    y=y,
                    click=click["click"],
                    label=label,
                    element_class=el_class,
                )
                annotations.append(ann)

                if on_progress:
                    on_progress(i + 1, len(clicks), ann)
        finally:
            cap.release()

        ann_path = session_dir / "annotations.jsonl"
        with open(ann_path, "w") as f:
            for ann in annotations:
                f.write(json.dumps(asdict(ann)) + "\n")

        return annotations

    # --- helpers ---

    def _get_frame_at_time(
        self,
        cap,
        t: float,
        fps: float,
        frame_times: list[float],
        total_frames: int,
    ):
        """Extract the video frame closest to time t."""
        import cv2

        if frame_times:
            idx = min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - t))
        else:
            idx = int(t * fps)

        idx = max(0, min(idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        return frame if ret else None

    def _crop_b64(self, frame, cx: int, cy: int, size: int) -> str:
        """Crop a square region centred on (cx, cy), draw a red crosshair, return base64 PNG."""
        import cv2

        h, w = frame.shape[:2]
        half = size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        crop = frame[y1:y2, x1:x2].copy()
        rx, ry = cx - x1, cy - y1
        cv2.drawMarker(crop, (rx, ry), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        _, buf = cv2.imencode(".png", crop)
        return base64.standard_b64encode(buf).decode()

    def _ask_vlm(self, icon_b64: str, context_b64: str) -> tuple[str, str]:
        """Send two crops to Claude Vision, return (label, element_class)."""
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("Install LLM extra: pip install gazefy[llm]")

        from gazefy.llm.credentials import get_api_key

        api_key = get_api_key()
        if not api_key:
            raise RuntimeError("No API key. Set ANTHROPIC_API_KEY or use Claude Code login.")

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "A user clicked a UI element in a desktop application. "
                                "The red crosshair marks the exact click position.\n\n"
                                "First image: close-up around the click.\n"
                                "Second image: wider context.\n\n"
                                "Reply with JSON only:\n"
                                '{"label": "Save Button", "class": "button"}\n\n'
                                "Valid classes: button, menu, input, checkbox, icon, "
                                "tab, dropdown, text, scrollbar, other"
                            ),
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": icon_b64,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": context_b64,
                            },
                        },
                    ],
                }
            ],
        )
        text = response.content[0].text.strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return data.get("label", "Unknown"), data.get("class", "button")
        return text, "button"
