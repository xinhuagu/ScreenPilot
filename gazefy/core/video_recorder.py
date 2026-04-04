"""Video recorder: captures screen as video + mouse events simultaneously.

No YOLO model required during recording. Semantic annotation is done
post-hoc by VideoAnnotator using VLM analysis of video frames.

Session directory structure:
    recordings/session_YYYYMMDD_HHMMSS/
        video.mp4         - Screen recording at specified fps
        events.jsonl      - Mouse events:
                              moves: {t, x, y}            (throttled, ~5/s)
                              clicks: {t, x, y, click}    (all clicks)
        frame_times.json  - [t0, t1, ...] monotonic time of each video frame
        annotations.jsonl - Added by VideoAnnotator after analysis
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable


class VideoRecorder:
    """Records screen to video + mouse events to JSONL simultaneously.

    No YOLO or VLM dependency at record time.
    Mouse moves are throttled to ~5/s; all clicks are captured.
    """

    # Minimum interval between recorded move events (seconds)
    _MOVE_INTERVAL = 0.2

    def __init__(
        self,
        fps: int = 10,
        monitor_index: int = 1,
        window_name: str = "",
    ):
        self.fps = fps
        self.monitor_index = monitor_index
        self.window_name = window_name  # Track a specific window
        self._session_dir: Path | None = None
        self._recording = False
        self._start_time = 0.0
        self._events: list[dict] = []
        self._frame_times: list[float] = []
        self._frame_windows: list[dict] = []  # Window rect per frame
        self._video_writer = None
        self._video_thread: threading.Thread | None = None
        self._mouse_listener = None
        self._on_click: Callable[[dict], None] | None = None
        self._last_move_t: float = -1.0

    def start(self, session_dir: Path, on_click: Callable[[dict], None] | None = None) -> None:
        """Start recording. on_click is called on the listener thread for each click."""
        session_dir.mkdir(parents=True, exist_ok=True)
        self._session_dir = session_dir
        self._recording = True
        self._start_time = time.monotonic()
        self._events = []
        self._frame_times = []
        self._frame_windows = []
        self._last_move_t = -1.0
        self._on_click = on_click

        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._video_thread.start()
        self._start_mouse_listener()

    def stop(self) -> Path:
        """Stop recording, flush files, return session_dir."""
        self._recording = False

        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None

        if self._video_thread:
            self._video_thread.join(timeout=5)
            self._video_thread = None

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        assert self._session_dir is not None
        self._flush(self._session_dir)
        return self._session_dir

    # --- internal ---

    def _video_loop(self) -> None:
        try:
            import cv2
            import mss
            import numpy as np
        except ImportError as e:
            raise RuntimeError(f"Missing dependency: {e}") from e

        interval = 1.0 / self.fps
        assert self._session_dir is not None

        with mss.mss() as sct:
            # Determine capture region
            if self.window_name:
                from gazefy.capture.window_finder import find_window

                win = find_window(self.window_name)
                if win:
                    monitor = {
                        "left": win.region.left,
                        "top": win.region.top,
                        "width": win.region.width,
                        "height": win.region.height,
                    }
                else:
                    monitor = sct.monitors[self.monitor_index]
            else:
                monitor = sct.monitors[self.monitor_index]

            w, h = monitor["width"], monitor["height"]
            video_path = str(self._session_dir / "video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))

            next_t = time.monotonic()
            while self._recording:
                now = time.monotonic()
                if now >= next_t:
                    # Re-find window each frame (handles move/resize)
                    if self.window_name:
                        win = find_window(self.window_name)
                        if win:
                            monitor = {
                                "left": win.region.left,
                                "top": win.region.top,
                                "width": win.region.width,
                                "height": win.region.height,
                            }

                    img = np.array(sct.grab(monitor))
                    # Resize if window changed size
                    frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    fh, fw = frame_bgr.shape[:2]
                    if fw != w or fh != h:
                        frame_bgr = cv2.resize(frame_bgr, (w, h))

                    self._video_writer.write(frame_bgr)
                    self._frame_times.append(round(now - self._start_time, 3))
                    self._frame_windows.append(
                        {
                            "left": monitor["left"],
                            "top": monitor["top"],
                            "width": monitor["width"],
                            "height": monitor["height"],
                        }
                    )
                    next_t += interval
                else:
                    time.sleep(0.002)

    def _start_mouse_listener(self) -> None:
        try:
            from pynput import mouse
        except ImportError:
            return

        def on_move(x: float, y: float) -> None:
            if not self._recording:
                return
            now = time.monotonic()
            t = round(now - self._start_time, 3)
            # Throttle moves to _MOVE_INTERVAL
            if now - self._last_move_t < self._MOVE_INTERVAL:
                return
            self._last_move_t = now
            self._events.append({"t": t, "x": int(x), "y": int(y)})

        def on_click(x: float, y: float, button, pressed: bool) -> None:
            if not self._recording or not pressed:
                return
            t = round(time.monotonic() - self._start_time, 3)
            btn = "left" if button == mouse.Button.left else "right"
            ev = {"t": t, "x": int(x), "y": int(y), "click": btn}
            self._events.append(ev)
            if self._on_click:
                self._on_click(ev)

        scroll_accum = [0.0]  # Accumulate trackpad fractional scrolls

        def on_scroll(x: float, y: float, dx: int, dy: int) -> None:
            if not self._recording:
                return
            # Mac trackpad sends fractional dy — accumulate until >= 1
            scroll_accum[0] += dy
            if abs(scroll_accum[0]) < 1:
                return
            t = round(time.monotonic() - self._start_time, 3)
            steps = int(scroll_accum[0])
            scroll_accum[0] -= steps
            direction = "up" if steps > 0 else "down"
            ev = {"t": t, "x": int(x), "y": int(y), "scroll": direction, "dy": steps}
            self._events.append(ev)
            if self._on_click:
                self._on_click(ev)

        self._mouse_listener = mouse.Listener(
            on_move=on_move, on_click=on_click, on_scroll=on_scroll
        )
        self._mouse_listener.start()

    def _flush(self, session_dir: Path) -> None:
        # Sort events by time before writing
        self._events.sort(key=lambda e: e["t"])

        def _sanitize(obj):
            """Convert numpy types to Python natives for JSON."""
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if hasattr(obj, "__float__"):
                f = float(obj)
                return int(f) if f == int(f) else f
            return obj

        events_path = session_dir / "events.jsonl"
        with open(events_path, "w") as f:
            for ev in self._events:
                f.write(json.dumps(_sanitize(ev)) + "\n")

        times_path = session_dir / "frame_times.json"
        times_path.write_text(json.dumps(_sanitize(self._frame_times)))

        # Save per-frame window rects
        windows_path = session_dir / "frame_windows.json"
        windows_path.write_text(json.dumps(_sanitize(self._frame_windows)))

    # --- properties for UI ---

    @property
    def click_count(self) -> int:
        return sum(1 for e in self._events if e.get("click"))

    @property
    def elapsed(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time
