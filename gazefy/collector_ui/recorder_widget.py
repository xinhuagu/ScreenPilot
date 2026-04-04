"""Floating recorder widget: semantic recording with YOLO + OCR.

Records mouse movements and clicks as element identities, not raw coordinates.
Replay finds elements by class+text using live YOLO detection.

Video mode: records screen as MP4 + click events (no YOLO required).
Annotate: after recording, VLM analyses video frames at each click → fills semantic labels.

┌──────────────────────────────────────────┐
│ Gazefy Recorder                 ● REC   │
│ Pack: [_________]      [☑ Video mode]   │
│ [Start] [Stop] [Replay] [Open] [Annotate]│
│ Frames: 0   00:00                        │
│ → [button] "Save"                        │
└──────────────────────────────────────────┘
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
    QCheckBox,
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
    """Compact floating window for semantic recording + auto icon labeling."""

    _frame_update = Signal(int, str)

    def __init__(self):
        super().__init__()
        self._recording = False
        self._replaying = False
        self._annotating = False
        self._frames: list[dict] = []
        self._record_start = 0.0
        self._record_path: Path | None = None
        self._worker_thread: threading.Thread | None = None
        # Detection pipeline (loaded on start)
        self._detector = None
        self._ocr = None
        self._tracker = None
        self._ui_map = None
        self._pack_name = ""
        # Icon labels dictionary (auto-learn on click)
        self._icon_labels: dict = {}
        self._last_frame: object = None  # numpy array for VLM cropping
        # Video mode
        self._video_recorder = None
        self._video_session_dir: Path | None = None
        # Monitor mode
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None

        self._init_ui()
        self._frame_update.connect(self._on_frame_update)
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._scan_packs()

    def _init_ui(self) -> None:
        self.setWindowTitle("Gazefy Recorder")
        self.setMinimumSize(480, 140)
        self.setMaximumWidth(600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Pack selector row + mode toggles
        pack_row = QHBoxLayout()
        pack_row.addWidget(QLabel("Pack:"))
        self.pack_combo = QComboBox()
        self.pack_combo.setMinimumWidth(120)
        pack_row.addWidget(self.pack_combo, 1)
        self.video_check = QCheckBox("Video")
        self.video_check.setToolTip("Record screen as MP4 + mouse events")
        self.monitor_check = QCheckBox("Monitor")
        self.monitor_check.setToolTip(
            "Live cursor-to-element tracking.\nShows which UI element the mouse is hovering over."
        )
        pack_row.addWidget(self.video_check)
        pack_row.addWidget(self.monitor_check)
        layout.addLayout(pack_row)

        # Annotator mode row (visible only in video mode)
        ann_row = QHBoxLayout()
        ann_row.addWidget(QLabel("Annotator:"))
        self.detector_combo = QComboBox()
        self.detector_combo.addItem("Hybrid: GroundingDINO + OCR + VLM", "grounding")
        self.detector_combo.addItem("Full-frame VLM only", "none")
        self.detector_combo.setToolTip(
            "Hybrid (recommended): GroundingDINO detects precise bboxes, "
            "EasyOCR reads text labels for free, Claude Vision only labels icons.\n\n"
            "Full-frame VLM: sends entire screenshot to Claude (simpler, less precise bboxes)."
        )
        ann_row.addWidget(self.detector_combo, 1)
        self._ann_row_widget = QWidget()
        self._ann_row_widget.setLayout(ann_row)
        self._ann_row_widget.setVisible(False)
        layout.addWidget(self._ann_row_widget)

        # Controls row 1: Record
        ctrl1 = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.replay_btn = QPushButton("Replay")
        self.open_btn = QPushButton("Open")
        self.stop_btn.setEnabled(False)
        self.replay_btn.setEnabled(False)
        for btn in [self.start_btn, self.stop_btn, self.replay_btn, self.open_btn]:
            btn.setFixedHeight(28)
            ctrl1.addWidget(btn)
        layout.addLayout(ctrl1)

        # Controls row 2: Annotate + Train
        ctrl2 = QHBoxLayout()
        self.annotate_btn = QPushButton("Annotate")
        self.train_btn = QPushButton("Train")
        self.annotate_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.annotate_btn.setToolTip(
            "Label all UI elements in the recording.\n"
            "GroundingDINO for bboxes, OCR for text, VLM for icons.\n"
            "Results are appended to the pack's training data."
        )
        self.train_btn.setToolTip(
            "Train YOLO on all accumulated training data.\n"
            "Each session's annotations stack — model gets better over time."
        )
        for btn in [self.annotate_btn, self.train_btn]:
            btn.setFixedHeight(28)
            ctrl2.addWidget(btn)
        layout.addLayout(ctrl2)

        # Status row
        status = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        self.frame_label = QLabel("Clicks: 0")
        self.time_label = QLabel("00:00")
        status.addWidget(self.status_label)
        status.addWidget(self.frame_label)
        status.addWidget(self.time_label)
        status.addStretch()
        layout.addLayout(status)

        # Element / info display
        self.element_label = QLabel("")
        self.element_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
        layout.addWidget(self.element_label)

        # Signals
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.replay_btn.clicked.connect(self._on_replay)
        self.open_btn.clicked.connect(self._on_open)
        self.annotate_btn.clicked.connect(self._on_annotate)
        self.train_btn.clicked.connect(self._on_train)
        self.video_check.toggled.connect(self._on_video_mode_toggled)
        self.monitor_check.toggled.connect(self._on_monitor_toggled)

    def _scan_packs(self) -> None:
        packs_dir = Path("packs")
        self.pack_combo.clear()
        self.pack_combo.addItem("(no model)")
        if packs_dir.exists():
            for p in sorted(packs_dir.iterdir()):
                if (p / "pack.yaml").exists():
                    self.pack_combo.addItem(p.name)

    def _load_pipeline(self) -> bool:
        """Load YOLO detector + OCR + icon labels for the selected pack."""
        self._pack_name = self.pack_combo.currentText()
        if self._pack_name == "(no model)":
            return False
        try:
            from gazefy.core.application_pack import ApplicationPack
            from gazefy.detection.detector import UIDetector
            from gazefy.detection.ocr import ElementOCR
            from gazefy.tracker.element_tracker import ElementTracker

            pack = ApplicationPack.load(f"packs/{self._pack_name}")
            self._detector = UIDetector(pack)
            self._detector.load_model()
            self._ocr = ElementOCR()
            self._tracker = ElementTracker(min_stability=1)
            # Load existing icon labels
            labels_path = Path(f"packs/{self._pack_name}/icon_labels.json")
            if labels_path.exists():
                self._icon_labels = json.loads(labels_path.read_text())
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
        self._last_frame = frame

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

    def _auto_label_element(self, el_info: dict, x: float, y: float) -> str:
        """If element has no text (icon), ask VLM to label it. Returns label."""
        text = el_info.get("text", "")
        if text:
            return text  # OCR already got it

        el_class = el_info.get("element_class", "")
        if not el_class or self._last_frame is None:
            return ""

        # Check if already labeled by bbox position

        el_id = el_info.get("element_id", "")
        if el_id in self._icon_labels:
            return self._icon_labels[el_id].get("label", "")

        # Crop icon + context from last frame
        import base64

        import cv2

        frame = self._last_frame
        if self._ui_map is None:
            return ""
        from gazefy.utils.geometry import Point

        el = self._ui_map.element_at(Point(x, y))
        if el is None:
            return ""

        b = el.bbox
        x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
        x2, y2 = min(frame.shape[1], int(b.x2)), min(frame.shape[0], int(b.y2))
        if x2 - x1 < 5 or y2 - y1 < 5:
            return ""

        crop = frame[y1:y2, x1:x2]
        if crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        _, buf = cv2.imencode(".png", crop)
        icon_b64 = base64.standard_b64encode(buf).decode()

        # Context crop
        pad = 80
        cy1, cx1 = max(0, y1 - pad), max(0, x1 - pad)
        cy2 = min(frame.shape[0], y2 + pad)
        cx2 = min(frame.shape[1], x2 + pad)
        ctx = frame[cy1:cy2, cx1:cx2]
        if ctx.shape[2] == 4:
            ctx = cv2.cvtColor(ctx, cv2.COLOR_BGRA2BGR)
        _, buf2 = cv2.imencode(".png", ctx)
        ctx_b64 = base64.standard_b64encode(buf2).decode()

        # Ask VLM
        try:
            from gazefy.core.learner import _ask_vlm

            label = _ask_vlm(icon_b64, ctx_b64, el_class)
        except Exception as e:
            label = f"(VLM error: {e})"
            return label

        # Save to icon_labels
        self._icon_labels[el_id] = {
            "label": label,
            "class": el_class,
            "bbox": [x1, y1, x2, y2],
        }
        pack_dir = Path(f"packs/{self._pack_name}")
        if pack_dir.exists():
            (pack_dir / "icon_labels.json").write_text(json.dumps(self._icon_labels, indent=2))
        return label

    # --- Actions ---

    def _on_video_mode_toggled(self, checked: bool) -> None:
        self._ann_row_widget.setVisible(checked)

    def _on_monitor_toggled(self, checked: bool) -> None:
        if checked:
            if not self._load_pipeline():
                self.monitor_check.setChecked(False)
                self.element_label.setText("Select a pack with a model first")
                return
            self._monitoring = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.replay_btn.setEnabled(False)
            self.open_btn.setEnabled(False)
            self.annotate_btn.setEnabled(False)
            self.train_btn.setEnabled(False)
            self.video_check.setEnabled(False)
            self.pack_combo.setEnabled(False)
            self.status_label.setText("Monitoring...")
            self.status_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
        else:
            self._monitoring = False
            self.start_btn.setEnabled(True)
            self.open_btn.setEnabled(True)
            self.video_check.setEnabled(True)
            self.pack_combo.setEnabled(True)
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("font-weight: bold;")

    def _monitor_loop(self) -> None:
        """Continuously detect + resolve cursor to element."""
        try:
            import mss
            import numpy as np
            import pyautogui

            pyautogui.FAILSAFE = False
        except ImportError:
            return

        last_element_id = ""
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            # Initial detection
            img = np.array(sct.grab(monitor))
            self._detect_and_ocr(img)

            while self._monitoring:
                x, y = pyautogui.position()
                el_info = self._resolve_element(float(x), float(y))
                el_class = el_info.get("element_class", "")
                el_text = el_info.get("text", "")
                el_id = el_info.get("element_id", "")

                if el_class:
                    desc = f'[{el_class}] "{el_text}" ({el_id})'
                else:
                    desc = f"({x}, {y}) no element"

                if el_id != last_element_id:
                    last_element_id = el_id
                    self._frame_update.emit(0, f"→ {desc}")

                time.sleep(0.05)

    def _on_start(self) -> None:
        if self._recording:
            return

        if self.video_check.isChecked():
            self._start_video_mode()
        else:
            self._start_semantic_mode()

    def _start_video_mode(self) -> None:
        """Start recording: screen video + click events only (no YOLO)."""
        from gazefy.core.video_recorder import VideoRecorder

        rec_dir = Path("recordings")
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = rec_dir / f"session_{ts}"

        self._video_recorder = VideoRecorder(fps=10)
        self._recording = True
        self._record_start = time.monotonic()
        self._video_session_dir = session_dir

        def on_click(ev: dict) -> None:
            self._frame_update.emit(
                self._video_recorder.click_count if self._video_recorder else 0,
                f"CLICK {ev['click']} at ({ev['x']}, {ev['y']})",
            )

        self._video_recorder.start(session_dir, on_click=on_click)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.replay_btn.setEnabled(False)
        self.annotate_btn.setEnabled(False)
        self.pack_combo.setEnabled(False)
        self.video_check.setEnabled(False)
        self.status_label.setText("● REC (video)")
        self.status_label.setStyleSheet("font-weight: bold; color: red;")
        self._elapsed_timer.start(200)

    def _start_semantic_mode(self) -> None:
        """Start recording: semantic JSONL mode (existing behaviour)."""
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
        self.annotate_btn.setEnabled(False)
        self.pack_combo.setEnabled(False)
        self.video_check.setEnabled(False)
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

        if self.video_check.isChecked() and self._video_recorder:
            session_dir = self._video_recorder.stop()
            n_clicks = self._video_recorder.click_count
            self._video_recorder = None
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.replay_btn.setEnabled(False)
            self.annotate_btn.setEnabled(True)
            self.pack_combo.setEnabled(True)
            self.video_check.setEnabled(True)
            self.status_label.setText(f"Saved ({n_clicks} clicks)")
            self.status_label.setStyleSheet("font-weight: bold; color: #333;")
            self.element_label.setText(f"→ {session_dir}")
            self.frame_label.setText(f"Clicks: {n_clicks}")
        else:
            if self._record_path and self._frames:
                with open(self._record_path, "w") as f:
                    for frame in self._frames:
                        f.write(json.dumps(frame) + "\n")

            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.replay_btn.setEnabled(bool(self._frames))
            self.annotate_btn.setEnabled(False)
            self.pack_combo.setEnabled(True)
            self.video_check.setEnabled(True)
            n_clicks = sum(1 for f in self._frames if f.get("click"))
            n_semantic = sum(1 for f in self._frames if f.get("element_class"))
            self.status_label.setText(
                f"Saved ({len(self._frames)} frames, {n_clicks} clicks, {n_semantic} semantic)"
            )
            self.status_label.setStyleSheet("font-weight: bold; color: #333;")
            self.element_label.setText(f"→ {self._record_path}")

    def _on_annotate(self) -> None:
        """Annotate video → convert to YOLO → append to pack training data."""
        if self._annotating:
            return
        if self._video_session_dir is None or not self._video_session_dir.exists():
            from PySide6.QtWidgets import QFileDialog

            path = QFileDialog.getExistingDirectory(self, "Select session directory", "recordings")
            if not path:
                return
            self._video_session_dir = Path(path)

        pack_name = self.pack_combo.currentText()
        if pack_name == "(no model)":
            self.element_label.setText("Select a pack first")
            return

        self._annotating = True
        self.annotate_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.status_label.setText("Annotating...")
        self.status_label.setStyleSheet("font-weight: bold; color: #FF8C00;")

        session_dir = self._video_session_dir
        detector_mode = self.detector_combo.currentData()
        pack_dir = Path("packs") / pack_name

        def run() -> None:
            def on_progress(current: int, total: int, desc: str) -> None:
                self._frame_update.emit(current, f"{current}/{total}  {desc}")

            try:
                # Step 1: Annotate
                self._frame_update.emit(0, "Step 1/2: Annotating...")
                if detector_mode == "grounding":
                    from gazefy.core.hybrid_annotator import HybridAnnotator

                    annotator = HybridAnnotator(pack_dir=pack_dir)
                else:
                    from gazefy.core.video_annotator import VideoAnnotator

                    annotator = VideoAnnotator()

                annotator.annotate_session(session_dir, on_progress=on_progress)

                # Step 2: Convert to YOLO + append to pack training data
                self._frame_update.emit(0, "Step 2/2: Converting to YOLO...")
                from gazefy.training.annotation_converter import (
                    AnnotationConverter,
                )

                training_dir = pack_dir / "training_data"
                converter = AnnotationConverter()
                result = converter.convert(session_dir, output_dir=training_dir)
                n_images = result.get("images", 0)
                n_elements = result.get("elements", 0)

                # Count total accumulated data
                img_dir = training_dir / "images"
                total = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0

                self._frame_update.emit(
                    n_images,
                    f"Done: +{n_images} images, +{n_elements} elements "
                    f"(total: {total} images accumulated)",
                )
            except Exception as e:
                self._frame_update.emit(0, f"Error: {e}")
            finally:
                self._annotating = False

        t = threading.Thread(target=run, daemon=True)
        t.start()

    def _on_train(self) -> None:
        """Train YOLO on all accumulated training data in the pack."""
        pack_name = self.pack_combo.currentText()
        if pack_name == "(no model)":
            self.element_label.setText("Select a pack first")
            return

        pack_dir = Path("packs") / pack_name
        training_dir = pack_dir / "training_data"
        dataset_yaml = training_dir / "dataset.yaml"

        if not dataset_yaml.exists():
            self.element_label.setText("No training data. Annotate first.")
            return

        self.train_btn.setEnabled(False)
        self.annotate_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.status_label.setText("Training...")
        self.status_label.setStyleSheet("font-weight: bold; color: #FF8C00;")

        def run() -> None:
            try:
                # Prep: split into train/val
                self._frame_update.emit(0, "Splitting train/val...")
                from gazefy.training.dataset_prep import split_dataset

                split_dataset(training_dir, split_ratio=0.8)

                # Train
                self._frame_update.emit(0, "Training YOLO (this takes a few minutes)...")
                from gazefy.training.trainer import PackTrainer, TrainConfig

                config = TrainConfig(
                    dataset_yaml=str(dataset_yaml),
                    imgsz=640,
                    epochs=50,
                    batch=8,
                    device="mps",
                )
                trainer = PackTrainer(config)
                result = trainer.train()

                # Package: update model in pack
                import shutil

                model_src = Path(result.best_model_path)
                if not model_src.exists():
                    # Ultralytics may put it elsewhere
                    import glob

                    candidates = glob.glob("**/best.pt", recursive=True)
                    if candidates:
                        model_src = Path(sorted(candidates)[-1])

                if model_src.exists():
                    shutil.copy2(model_src, pack_dir / "model.pt")

                map50 = result.metrics.get("metrics/mAP50(B)", "?")
                self._frame_update.emit(
                    1, f"Training done! mAP@0.5={map50} Model saved to {pack_dir}/model.pt"
                )
            except Exception as e:
                self._frame_update.emit(0, f"Training error: {e}")
            finally:
                pass

        t = threading.Thread(target=run, daemon=True)
        t.start()

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

        # Click listener — records click + auto-labels icons via VLM
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
                # Resolve element + auto-label if icon
                if has_model:
                    el = self._resolve_element(float(x), float(y))
                    if el and not el.get("text"):
                        # No OCR text — auto-label via VLM
                        label = self._auto_label_element(el, float(x), float(y))
                        if label:
                            el["text"] = label
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
        self.element_label.setText(desc)

        # Detect completion signals
        if desc.startswith("Done:") or desc.startswith("Training done"):
            self.status_label.setText(desc[:60])
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
            self.annotate_btn.setEnabled(True)
            self.train_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self._annotating = False
        elif desc.startswith("Error:") or desc.startswith("Training error"):
            self.status_label.setText(desc[:60])
            self.status_label.setStyleSheet("font-weight: bold; color: red;")
            self.annotate_btn.setEnabled(True)
            self.train_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self._annotating = False
        elif desc == "done":
            self.status_label.setText("Replay done")
            self.status_label.setStyleSheet("font-weight: bold; color: #333;")
            self.replay_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
        elif self.video_check.isChecked():
            self.frame_label.setText(f"Clicks: {count}")
        else:
            self.frame_label.setText(f"Frames: {count}")

    def _update_elapsed(self) -> None:
        if self._recording:
            if self.video_check.isChecked() and self._video_recorder:
                elapsed = self._video_recorder.elapsed
            else:
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
