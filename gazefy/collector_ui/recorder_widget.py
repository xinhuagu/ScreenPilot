"""Floating recorder widget: semantic recording with YOLO + OCR.

Records mouse movements and clicks as element identities, not raw coordinates.
Replay finds elements by class+text using live YOLO detection.

┌─────────────────────────────────────┐
│ Gazefy Recorder             ● REC  │
│ [Start] [Stop] [Replay] [Open]     │
│ Pack: [_________]  Frames: 0 00:00 │
│ → [button] "Save"                  │
└─────────────────────────────────────┘
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class RecorderWidget(QMainWindow):
    """Compact floating window for semantic recording."""

    _frame_update = Signal(int, str)

    def __init__(self):
        super().__init__()
        self._recording = False
        self._replaying = False
        self._frames: list[dict] = []
        self._record_start = 0.0
        self._record_path: Path | None = None
        self._worker_thread: threading.Thread | None = None
        # Detection pipeline (loaded on start)
        self._detector = None
        self._ocr = None
        self._tracker = None
        self._ui_map = None

        self._init_ui()
        self._frame_update.connect(self._on_frame_update)
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._scan_packs()

    def _init_ui(self) -> None:
        self.setWindowTitle("Gazefy Recorder")
        self.setFixedSize(380, 120)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Pack selector row
        pack_row = QHBoxLayout()
        pack_row.addWidget(QLabel("Pack:"))
        self.pack_combo = QComboBox()
        self.pack_combo.setMinimumWidth(150)
        pack_row.addWidget(self.pack_combo, 1)
        layout.addLayout(pack_row)

        # Controls row
        ctrl = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.replay_btn = QPushButton("Replay")
        self.open_btn = QPushButton("Open")
        self.stop_btn.setEnabled(False)
        self.replay_btn.setEnabled(False)
        for btn in [self.start_btn, self.stop_btn, self.replay_btn, self.open_btn]:
            btn.setFixedHeight(28)
            ctrl.addWidget(btn)
        layout.addLayout(ctrl)

        # Status row
        status = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        self.frame_label = QLabel("Frames: 0")
        self.time_label = QLabel("00:00")
        status.addWidget(self.status_label)
        status.addWidget(self.frame_label)
        status.addWidget(self.time_label)
        status.addStretch()
        layout.addLayout(status)

        # Element display
        self.element_label = QLabel("")
        self.element_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
        layout.addWidget(self.element_label)

        # Signals
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.replay_btn.clicked.connect(self._on_replay)
        self.open_btn.clicked.connect(self._on_open)

    def _scan_packs(self) -> None:
        packs_dir = Path("packs")
        self.pack_combo.clear()
        self.pack_combo.addItem("(no model)")
        if packs_dir.exists():
            for p in sorted(packs_dir.iterdir()):
                if (p / "pack.yaml").exists():
                    self.pack_combo.addItem(p.name)

    def _load_pipeline(self) -> bool:
        """Load YOLO detector + OCR for the selected pack."""
        pack_name = self.pack_combo.currentText()
        if pack_name == "(no model)":
            return False
        try:
            from gazefy.core.application_pack import ApplicationPack
            from gazefy.detection.detector import UIDetector
            from gazefy.detection.ocr import ElementOCR
            from gazefy.tracker.element_tracker import ElementTracker

            pack = ApplicationPack.load(f"packs/{pack_name}")
            self._detector = UIDetector(pack)
            self._detector.load_model()
            self._ocr = ElementOCR()
            self._tracker = ElementTracker(min_stability=1)
            return True
        except Exception as e:
            self.element_label.setText(f"Load failed: {e}")
            return False

    def _detect_and_ocr(self, frame) -> None:
        """Run YOLO + OCR on a frame, update internal UIMap."""
        if self._detector is None:
            return
        from gazefy.capture.change_detector import ChangeLevel, ChangeResult

        detections = self._detector.detect(frame)
        if self._ocr and detections:
            texts = self._ocr.read_all_elements(frame, detections)
            # Inject text into detections via tracker
            # We'll read text after tracking
            self._texts = texts
        else:
            self._texts = {}

        h, w = frame.shape[:2]
        change = ChangeResult(changed=True, change_level=ChangeLevel.MAJOR)
        self._tracker.update(detections, change, frame_width=w, frame_height=h)
        # Bootstrap stability
        change2 = ChangeResult(changed=True, change_level=ChangeLevel.MINOR)
        self._tracker.update(detections, change2, frame_width=w, frame_height=h)
        self._ui_map = self._tracker.current_map
        self._detections = detections

    def _resolve_element(self, x: float, y: float) -> dict:
        """Find which element the cursor is on, return semantic info."""
        if self._ui_map is None:
            return {}
        from gazefy.utils.geometry import Point

        el = self._ui_map.element_at(Point(x, y))
        if el is None:
            return {}

        # Find OCR text for this element
        text = el.text
        if not text and hasattr(self, "_texts") and hasattr(self, "_detections"):
            # Match by bbox overlap
            for idx, det_text in self._texts.items():
                if idx < len(self._detections):
                    det = self._detections[idx]
                    if abs(det.bbox.x1 - el.bbox.x1) < 10 and abs(det.bbox.y1 - el.bbox.y1) < 10:
                        text = det_text
                        break

        return {
            "element_id": el.id,
            "element_class": el.class_name,
            "text": text,
            "confidence": round(el.confidence, 3),
        }

    # --- Actions ---

    def _on_start(self) -> None:
        if self._recording:
            return

        has_model = self._load_pipeline()
        self._recording = True
        self._frames = []
        self._record_start = time.monotonic()
        rec_dir = Path("recordings")
        rec_dir.mkdir(exist_ok=True)
        self._record_path = rec_dir / f"session_{int(time.time())}.jsonl"

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.replay_btn.setEnabled(False)
        self.pack_combo.setEnabled(False)
        label = "● REC (semantic)" if has_model else "● REC (coords only)"
        self.status_label.setText(label)
        self.status_label.setStyleSheet("font-weight: bold; color: red;")
        self._elapsed_timer.start(200)

        self._worker_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._worker_thread.start()

    def _on_stop(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._elapsed_timer.stop()

        if self._record_path and self._frames:
            with open(self._record_path, "w") as f:
                for frame in self._frames:
                    f.write(json.dumps(frame) + "\n")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.replay_btn.setEnabled(bool(self._frames))
        self.pack_combo.setEnabled(True)
        n_clicks = sum(1 for f in self._frames if f.get("click"))
        n_semantic = sum(1 for f in self._frames if f.get("element_class"))
        self.status_label.setText(
            f"Saved ({len(self._frames)} frames, {n_clicks} clicks, {n_semantic} semantic)"
        )
        self.status_label.setStyleSheet("font-weight: bold; color: #333;")
        self.element_label.setText(f"→ {self._record_path}")

    def _on_replay(self) -> None:
        if self._replaying or not self._frames:
            return
        self._replaying = True
        self.replay_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.status_label.setText("▶ Replaying...")
        self.status_label.setStyleSheet("font-weight: bold; color: blue;")
        t = threading.Thread(target=self._replay_loop, daemon=True)
        t.start()

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Recording", "recordings", "JSONL (*.jsonl)"
        )
        if path:
            with open(path) as f:
                self._frames = [json.loads(line) for line in f if line.strip()]
            self._record_path = Path(path)
            self.replay_btn.setEnabled(bool(self._frames))
            n_semantic = sum(1 for f in self._frames if f.get("element_class"))
            self.status_label.setText(f"Loaded ({len(self._frames)} frames, {n_semantic} semantic)")
            self.frame_label.setText(f"Frames: {len(self._frames)}")

    # --- Background loops ---

    def _record_loop(self) -> None:
        try:
            import pyautogui

            pyautogui.FAILSAFE = False
            from pynput import mouse
        except ImportError:
            return

        # Capture one frame for detection
        has_model = self._detector is not None
        if has_model:
            try:
                import mss
                import numpy as np

                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    img = np.array(sct.grab(monitor))
                self._detect_and_ocr(img)
                n = self._ui_map.element_count if self._ui_map else 0
                self._frame_update.emit(0, f"Detected {n} elements")
            except Exception as e:
                self._frame_update.emit(0, f"Detection error: {e}")

        # Click listener
        def on_click(x, y, button, pressed):
            if not self._recording:
                return False
            if pressed:
                t = time.monotonic() - self._record_start
                btn = "left" if button == mouse.Button.left else "right"
                frame = {
                    "t": round(t, 3),
                    "x": int(x),
                    "y": int(y),
                    "click": btn,
                }
                # Resolve element at click position
                if has_model:
                    el = self._resolve_element(float(x), float(y))
                    frame.update(el)
                self._frames.append(frame)
                el_desc = frame.get("element_class", "")
                el_text = frame.get("text", "")
                desc = f'CLICK {btn} [{el_desc}] "{el_text}"' if el_desc else f"CLICK {btn}"
                self._frame_update.emit(len(self._frames), desc)

        listener = mouse.Listener(on_click=on_click)
        listener.start()

        # Position polling
        while self._recording:
            x, y = pyautogui.position()
            t = time.monotonic() - self._record_start
            frame = {"t": round(t, 3), "x": x, "y": y}
            if has_model:
                el = self._resolve_element(float(x), float(y))
                frame.update(el)
            self._frames.append(frame)

            el_class = frame.get("element_class", "")
            el_text = frame.get("text", "")
            if el_class:
                desc = f'[{el_class}] "{el_text}"'
            else:
                desc = f"({x}, {y})"
            self._frame_update.emit(len(self._frames), desc)
            time.sleep(0.05)

        listener.stop()

    def _replay_loop(self) -> None:
        try:
            import pyautogui

            pyautogui.FAILSAFE = False
        except ImportError:
            return

        # For semantic replay, load model and detect current screen
        has_model = self._detector is not None
        if not has_model:
            has_model = self._load_pipeline()

        if has_model:
            try:
                import mss
                import numpy as np

                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    img = np.array(sct.grab(monitor))
                self._detect_and_ocr(img)
            except Exception:
                pass

        for i, frame in enumerate(self._frames):
            if not self._replaying:
                break

            click = frame.get("click", "")
            target_class = frame.get("element_class", "")
            target_text = frame.get("text", "")

            # Semantic targeting: find element by class+text
            if target_class and has_model and self._ui_map:
                resolved = self._find_element_by_identity(target_class, target_text)
                if resolved:
                    x, y = int(resolved.x), int(resolved.y)
                else:
                    x, y = int(frame["x"]), int(frame["y"])
            else:
                x, y = int(frame["x"]), int(frame["y"])

            if x <= 5 and y <= 5:
                continue

            if click:
                if click == "right":
                    pyautogui.rightClick(x, y, _pause=False)
                else:
                    pyautogui.click(x, y, _pause=False)
                desc = f'CLICK {click} [{target_class}] "{target_text}"'
                self._frame_update.emit(i + 1, desc)
            else:
                pyautogui.moveTo(x, y, _pause=False)
                self._frame_update.emit(i + 1, f"({x}, {y})")

            if i + 1 < len(self._frames):
                dt = self._frames[i + 1]["t"] - frame["t"]
                if dt > 0:
                    time.sleep(dt)

        self._replaying = False
        self._frame_update.emit(len(self._frames), "done")

    def _find_element_by_identity(self, target_class: str, target_text: str):
        """Find an element in current UIMap by class + text match."""
        if self._ui_map is None:
            return None

        # Try exact text match first
        if target_text:
            for el in self._ui_map.elements.values():
                # Check OCR texts
                if el.class_name == target_class:
                    # Get text for this element
                    el_text = ""
                    if hasattr(self, "_texts") and hasattr(self, "_detections"):
                        for idx, dt in self._texts.items():
                            if idx < len(self._detections):
                                det = self._detections[idx]
                                if (
                                    abs(det.bbox.x1 - el.bbox.x1) < 10
                                    and abs(det.bbox.y1 - el.bbox.y1) < 10
                                ):
                                    el_text = dt
                                    break
                    if target_text.lower() in el_text.lower():
                        return el.center

        # Fallback: match by class only, pick first
        candidates = self._ui_map.elements_by_class(target_class)
        if candidates:
            return candidates[0].center
        return None

    # --- UI updates ---

    def _on_frame_update(self, count: int, desc: str) -> None:
        self.frame_label.setText(f"Frames: {count}")
        if desc == "done":
            self.status_label.setText("Replay done")
            self.status_label.setStyleSheet("font-weight: bold; color: #333;")
            self.replay_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
        else:
            self.element_label.setText(desc)

    def _update_elapsed(self) -> None:
        if self._recording:
            elapsed = time.monotonic() - self._record_start
            m, s = divmod(int(elapsed), 60)
            self.time_label.setText(f"{m:02d}:{s:02d}")


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Gazefy Recorder")
    w = RecorderWidget()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
