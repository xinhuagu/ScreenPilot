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

    def __init__(self, fps: int = 10, monitor_index: int = 1):
        self.fps = fps
        self.monitor_index = monitor_index
        self._session_dir: Path | None = None
        self._recording = False
        self._start_time = 0.0
        self._events: list[dict] = []
        self._frame_times: list[float] = []
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
            monitor = sct.monitors[self.monitor_index]
            w, h = monitor["width"], monitor["height"]
            video_path = str(self._session_dir / "video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))

            next_t = time.monotonic()
            while self._recording:
                now = time.monotonic()
                if now >= next_t:
                    img = np.array(sct.grab(monitor))
                    frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    self._video_writer.write(frame_bgr)
                    self._frame_times.append(round(now - self._start_time, 3))
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

        self._mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
        self._mouse_listener.start()

    def _flush(self, session_dir: Path) -> None:
        # Sort events by time before writing
        self._events.sort(key=lambda e: e["t"])

        events_path = session_dir / "events.jsonl"
        with open(events_path, "w") as f:
            for ev in self._events:
                f.write(json.dumps(ev) + "\n")

        times_path = session_dir / "frame_times.json"
        times_path.write_text(json.dumps(self._frame_times))

    # --- properties for UI ---

    @property
    def click_count(self) -> int:
        return sum(1 for e in self._events if e.get("click"))

    @property
    def elapsed(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time
