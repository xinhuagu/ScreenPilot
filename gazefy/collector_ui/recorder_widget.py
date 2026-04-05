"""Floating recorder widget: semantic recording with YOLO + OCR.

Records mouse movements and clicks as element identities, not raw coordinates.
Replay finds elements by class+text using live YOLO detection.

Video mode: records screen as MP4 + click events (no YOLO required).
Annotate: after recording, VLM analyses video frames at each click → fills semantic labels.

┌──────────────────────────────────────────┐
│ Gazefy                 ● REC   │
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


class _GroundingDINOWrapper:
    """Wraps GroundingDINO to match UIDetector interface (detect + is_loaded)."""

    def __init__(self):
        self._processor = None
        self._model = None

    def load_model(self) -> None:
        from gazefy.detection.grounding_label import load_model

        self._processor, self._model = load_model()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(self, frame):
        """Run GroundingDINO on a frame, return list[Detection]."""
        import cv2
        from PIL import Image

        from gazefy.detection.grounding_label import predict_image
        from gazefy.tracker.ui_map import Detection
        from gazefy.utils.geometry import Rect

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dets = predict_image(self._processor, self._model, img, threshold=0.15)
        return [
            Detection(
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=Rect(*d["bbox"]),
            )
            for d in dets
        ]


class RecorderWidget(QMainWindow):
    """Compact floating window for semantic recording + auto icon labeling."""

    _frame_update = Signal(int, str)
    _progress_update = Signal(int, int, str)  # (current, total, description)

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
        # overlay removed — status bar shows element info

        self._init_ui()
        self._frame_update.connect(self._on_frame_update)
        # overlay removed
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._progress_update.connect(self._on_progress)
        self._scan_windows()

    def closeEvent(self, event) -> None:  # noqa: N802
        """Prevent closing while annotate or train is running."""
        if self._annotating or self._recording:
            from PySide6.QtWidgets import QMessageBox

            task = "Annotating" if self._annotating else "Recording"
            reply = QMessageBox.question(
                self,
                "Task in progress",
                f"{task} is still running. Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        event.accept()

    def _init_ui(self) -> None:
        self.setWindowTitle("Gazefy")
        self.setMinimumSize(480, 140)
        self.setMaximumWidth(600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Window selector + refresh + mode toggles
        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("App:"))
        self.window_combo = QComboBox()
        self.window_combo.setMinimumWidth(200)
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(30)
        self.refresh_btn.setToolTip("Refresh window list")
        win_row.addWidget(self.window_combo, 1)
        win_row.addWidget(self.refresh_btn)
        self.monitor_check = QCheckBox("Monitor")
        self.monitor_check.setToolTip(
            "Live cursor-to-element tracking.\nShows which UI element the mouse is hovering over."
        )
        win_row.addWidget(self.monitor_check)
        layout.addLayout(win_row)

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
        self.replay_btn.setToolTip("Replay mouse actions (move + click + scroll)")
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

        # Progress bar (hidden until annotate/train)
        from PySide6.QtWidgets import QProgressBar

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Element / info display
        self.element_label = QLabel("")
        self.element_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
        layout.addWidget(self.element_label)

        # Log area (scrollable, selectable, copyable)
        from PySide6.QtWidgets import QTextEdit

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFixedHeight(120)
        self.log_area.setStyleSheet(
            "background-color: #1a1a1a; color: #ccc; font-family: Menlo; font-size: 11px;"
        )
        layout.addWidget(self.log_area)

        # Signals
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.replay_btn.clicked.connect(self._on_replay)
        self.open_btn.clicked.connect(self._on_open)
        self.annotate_btn.clicked.connect(self._on_annotate)
        self.train_btn.clicked.connect(self._on_train)
        self.refresh_btn.clicked.connect(self._scan_windows)
        self.window_combo.currentIndexChanged.connect(self._on_app_selected)
        # video_check removed — video is always recorded
        self.monitor_check.toggled.connect(self._on_monitor_toggled)

    def _scan_windows(self) -> None:
        """Populate window dropdown with all visible application windows."""
        from gazefy.capture.window_finder import list_windows

        self.window_combo.clear()
        self._window_list = list_windows()
        seen = set()
        for w in self._window_list:
            name = w.owner_name
            if name and name not in seen:
                seen.add(name)
                has_pack = (Path("packs") / name.lower() / "pack.yaml").exists()
                label = f"{name} ●" if has_pack else name
                self.window_combo.addItem(label, name)

    def _on_app_selected(self) -> None:
        """Bring the selected application to front + check pack state."""
        app_name = self.window_combo.currentData()
        if not app_name:
            return

        # Activate the app
        try:
            import subprocess

            subprocess.run(
                ["osascript", "-e", f'tell application "{app_name}" to activate'],
                timeout=3,
                capture_output=True,
            )
        except Exception:
            pass

        # Check if pack has recordings → enable Annotate/Train
        pack_name = app_name.lower().replace(" ", "_")
        pack_dir = Path("packs") / pack_name
        rec_dir = pack_dir / "recordings"
        has_recordings = False
        if rec_dir.exists():
            # Find latest recording with video.mp4
            for d in sorted(rec_dir.iterdir(), reverse=True):
                if d.is_dir() and (d / "video.mp4").exists():
                    self._video_session_dir = d
                    has_recordings = True
                    break

        self.annotate_btn.setEnabled(has_recordings)
        has_training = (pack_dir / "training_data" / "dataset.yaml").exists()
        self.train_btn.setEnabled(has_training)

        if has_recordings:
            self.element_label.setText(f"Latest: {self._video_session_dir}")
        elif pack_dir.exists():
            self.element_label.setText("Pack exists, no recordings yet")
        else:
            self.element_label.setText("New app — Start to record")

    def _get_selected_app(self) -> str:
        """Get the selected application name (lowercase, for pack dir)."""
        data = self.window_combo.currentData()
        return str(data).lower().replace(" ", "_") if data else ""

    def _ensure_pack(self, app_name: str) -> Path:
        """Create pack directory + pack.yaml if it doesn't exist."""
        import yaml

        pack_dir = Path("packs") / app_name
        pack_yaml = pack_dir / "pack.yaml"

        if not pack_yaml.exists():
            pack_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "name": app_name,
                "version": "0.1.0",
                "description": f"Auto-created pack for {app_name}",
                "window_match": [self.window_combo.currentData() or app_name],
                "model_file": "model.pt",
                "labels": [
                    "button",
                    "menu_item",
                    "input_field",
                    "checkbox",
                    "dropdown",
                    "dialog",
                    "toolbar",
                    "label",
                    "tab",
                    "icon",
                    "slider",
                ],
                "input_size": 640,
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
            }
            with open(pack_yaml, "w") as f:
                yaml.dump(meta, f, default_flow_style=False)

            # Create subdirs
            from gazefy.core.application_pack import ApplicationPack

            pack = ApplicationPack.load(pack_dir)
            pack.ensure_dirs()

            self.element_label.setText(f"Created new pack: {pack_dir}")

        return pack_dir

    def _load_pipeline(self) -> bool:
        """Load detector + OCR + element registry."""
        self._pack_name = self._get_selected_app()
        if not self._pack_name:
            return False
        try:
            pack_dir = self._ensure_pack(self._pack_name)
            from gazefy.core.application_pack import ApplicationPack
            from gazefy.core.element_registry import ElementRegistry
            from gazefy.detection.ocr import ElementOCR
            from gazefy.tracker.element_tracker import ElementTracker

            pack = ApplicationPack.load(pack_dir)

            if pack.has_model:
                from gazefy.detection.detector import UIDetector

                self._detector = UIDetector(pack)
                self._detector.load_model()
                self._log(f"Loaded YOLO model: {pack.model_path}")
                self.element_label.setText(f"Loaded YOLO: {self._pack_name}")
            else:
                self._detector = _GroundingDINOWrapper()
                self._detector.load_model()
                self._log("No trained model — using GroundingDINO (slower)")
                self.element_label.setText("No model — using GroundingDINO (slower)")

            self._ocr = ElementOCR()
            self._tracker = ElementTracker(min_stability=1)
            self._registry = ElementRegistry(pack_dir / "element_registry.json")
            if pack.icon_labels_path.exists():
                self._icon_labels = json.loads(pack.icon_labels_path.read_text())
            n_reg = len(self._registry)
            if n_reg:
                self.element_label.setText(f"Registry: {n_reg} known elements")
            return True
        except Exception as e:
            self.element_label.setText(f"Load failed: {e}")
            return False

    def _detect_and_ocr(self, frame, force_rebuild: bool = False) -> None:
        """Run detection + OCR on a frame, update UIMap with stable text."""
        if self._detector is None:
            return
        from gazefy.capture.change_detector import ChangeLevel, ChangeResult

        detections = self._detector.detect(frame)
        h, w = frame.shape[:2]

        # First detection or forced: MAJOR rebuild. Otherwise: MINOR (IoU match)
        is_first = self._ui_map is None or self._ui_map.is_empty
        if is_first or force_rebuild:
            change = ChangeResult(changed=True, change_level=ChangeLevel.MAJOR)
            self._tracker.update(detections, change, frame_width=w, frame_height=h)
            # Bootstrap stability
            change2 = ChangeResult(changed=True, change_level=ChangeLevel.MINOR)
            self._tracker.update(detections, change2, frame_width=w, frame_height=h)
        else:
            # Incremental: preserves element IDs and cached text
            change = ChangeResult(changed=True, change_level=ChangeLevel.MINOR)
            self._tracker.update(detections, change, frame_width=w, frame_height=h)

        # Look up element names from registry (stable, no OCR needed)
        if hasattr(self, "_registry") and self._registry and self._tracker.current_map.elements:
            reg_texts = {}
            fw = self._tracker.current_map.frame_width or 1
            fh = self._tracker.current_map.frame_height or 1
            for eid, el in self._tracker.current_map.elements.items():
                if not el.text:
                    reg = self._registry.lookup(
                        el.bbox.x1, el.bbox.y1, el.bbox.x2, el.bbox.y2, fw, fh, el.class_name
                    )
                    if reg:
                        name = reg.get("text") or reg.get("icon_label") or ""
                        if name:
                            reg_texts[eid] = name
            if reg_texts:
                self._tracker.set_element_texts(reg_texts)

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

        result = {
            "element_id": el.id,
            "element_class": el.class_name,
            "text": el.text,
            "confidence": round(el.confidence, 3),
        }

        # Enrich from registry
        if hasattr(self, "_registry") and self._registry:
            fw = self._ui_map.frame_width if self._ui_map else 1
            fh = self._ui_map.frame_height if self._ui_map else 1
            reg = self._registry.lookup(
                el.bbox.x1,
                el.bbox.y1,
                el.bbox.x2,
                el.bbox.y2,
                fw,
                fh,
                el.class_name,
            )
            if reg:
                if not result["text"] and reg.get("text"):
                    result["text"] = reg["text"]
                if reg.get("icon_label"):
                    result["icon_label"] = reg["icon_label"]
                if reg.get("function"):
                    result["function"] = reg["function"]

        return result

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
            # video always on
            self.window_combo.setEnabled(False)

            # Restore window + bring app to front
            self._restore_window()
            app_name = self.window_combo.currentData()
            if app_name:
                try:
                    import subprocess

                    subprocess.run(
                        ["osascript", "-e", f'tell application "{app_name}" to activate'],
                        timeout=3,
                        capture_output=True,
                    )
                except Exception:
                    pass

            self.status_label.setText("Monitoring...")
            self.status_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
        else:
            self._monitoring = False
            self.start_btn.setEnabled(True)
            self.open_btn.setEnabled(True)
            # video always on
            self.window_combo.setEnabled(True)
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("font-weight: bold;")

    def _monitor_loop(self) -> None:
        """Continuously detect + resolve cursor, update overlay.

        On each cycle:
        1. Find target window (current position + size)
        2. Capture only that window region
        3. Run YOLO detection
        4. Position overlay exactly on the window
        5. Resolve cursor to element
        """
        try:
            import mss
            import numpy as np
            import pyautogui

            pyautogui.FAILSAFE = False
        except ImportError:
            return

        from gazefy.capture.change_detector import ChangeDetector
        from gazefy.capture.window_finder import find_window

        change_detector = ChangeDetector()
        last_element_id = ""
        detect_interval = 0  # Force first detection

        with mss.mss() as sct:
            while self._monitoring:
                # 1. Find window every cycle (handles move/resize)
                pack_name = self._pack_name
                win = find_window(pack_name) if pack_name else None
                if win is None:
                    # Try window_match from pack
                    try:
                        from gazefy.core.application_pack import ApplicationPack

                        pack = ApplicationPack.load(f"packs/{pack_name}")
                        for pattern in pack.metadata.window_match:
                            win = find_window(pattern)
                            if win:
                                break
                    except Exception:
                        pass

                if win is None:
                    time.sleep(0.2)
                    continue

                region = win.region

                # 2. Capture window region only
                monitor = {
                    "top": region.top,
                    "left": region.left,
                    "width": region.width,
                    "height": region.height,
                }
                img = np.array(sct.grab(monitor))

                # 3. Detect on change (or every ~1 second)
                change = change_detector.check(img)
                detect_interval += 1
                if change.changed or detect_interval >= 20:
                    detect_interval = 0
                    self._detect_and_ocr(img)
                    # Bootstrap stability (need 2 updates for elements to appear)
                    if self._ui_map and self._ui_map.element_count == 0:
                        from gazefy.capture.change_detector import (
                            ChangeLevel,
                            ChangeResult,
                        )

                        boot = ChangeResult(changed=True, change_level=ChangeLevel.MINOR)
                        if self._detector and hasattr(self._detector, "detect"):
                            dets2 = self._detector.detect(img)
                            h, w = img.shape[:2]
                            self._tracker.update(dets2, boot, frame_width=w, frame_height=h)
                            self._ui_map = self._tracker.current_map

                    # Don't overwrite element_label — cursor info goes there

                # 4. Resolve cursor (screen coords → window-relative pixel coords)
                x, y = pyautogui.position()
                # Detect actual retina scale
                actual_retina = img.shape[1] / max(region.width, 1)
                px = (x - region.left) * actual_retina
                py = (y - region.top) * actual_retina
                el_info = self._resolve_element(px, py)
                el_class = el_info.get("element_class", "")
                el_text = el_info.get("text", "")
                el_id = el_info.get("element_id", "")
                icon_label = el_info.get("icon_label", "")
                func = el_info.get("function", "")

                if el_class:
                    name = el_text or icon_label or el_class
                    desc = f'[{el_class}] "{name}"'
                    if func:
                        desc += f" — {func}"
                else:
                    desc = ""

                # Always update element_label with current cursor element
                self._frame_update.emit(0, f"→ {desc}" if desc else "")

                # Log only on element change
                if el_id != last_element_id:
                    last_element_id = el_id
                    if desc:
                        self._frame_update.emit(0, f"LOG:→ {desc}")

                time.sleep(0.05)

    def _on_start(self) -> None:
        if self._recording:
            return

        self._start_video_mode()

    def _start_video_mode(self) -> None:
        """Start recording: screen video + click events into pack directory."""
        from gazefy.core.video_recorder import VideoRecorder

        pack_name = self._get_selected_app()
        if pack_name:
            pack_dir = self._ensure_pack(pack_name)
            from gazefy.core.application_pack import ApplicationPack

            pack = ApplicationPack.load(pack_dir)
            session_dir = pack.new_recording_dir()
            win_name = pack.metadata.window_match[0] if pack.metadata.window_match else pack_name
        else:
            import datetime

            rec_dir = Path("recordings")
            rec_dir.mkdir(exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = rec_dir / f"untagged_{ts}"
            session_dir.mkdir(parents=True, exist_ok=True)
            win_name = ""
        self._video_recorder = VideoRecorder(fps=10, window_name=win_name)
        self._recording = True
        self._record_start = time.monotonic()
        self._video_session_dir = session_dir

        def on_click(ev: dict) -> None:
            try:
                if ev.get("click"):
                    action = ev.get("action", "press")
                    if action == "release":
                        desc = f"RELEASE {ev['click']} ({ev['x']},{ev['y']})"
                    else:
                        desc = f"CLICK {ev['click']} ({ev['x']},{ev['y']})"
                elif ev.get("scroll"):
                    desc = f"SCROLL {ev['scroll']} dy={ev['dy']} ({ev['x']},{ev['y']})"
                else:
                    return
                count = self._video_recorder.click_count if self._video_recorder else 0
                self._frame_update.emit(count, desc)
            except Exception:
                pass  # Never crash the pynput listener

        self._video_recorder.start(session_dir, on_click=on_click)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.replay_btn.setEnabled(False)
        self.annotate_btn.setEnabled(False)
        self.window_combo.setEnabled(False)
        # video always on
        self.status_label.setText("● REC (video)")
        self.status_label.setStyleSheet("font-weight: bold; color: red;")
        self._elapsed_timer.start(200)

    def _start_semantic_mode(self) -> None:
        """Start recording: semantic JSONL into pack directory."""
        import datetime

        has_model = self._load_pipeline()
        self._recording = True
        self._frames = []
        self._record_start = time.monotonic()

        pack_name = self._get_selected_app()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if pack_name:
            pack_dir = self._ensure_pack(pack_name)
            from gazefy.core.application_pack import ApplicationPack

            pack = ApplicationPack.load(pack_dir)
            rec_dir = pack.recordings_dir
        else:
            rec_dir = Path("recordings")
            rec_dir.mkdir(exist_ok=True)
        self._record_path = rec_dir / f"semantic_{ts}.jsonl"

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.replay_btn.setEnabled(False)
        self.annotate_btn.setEnabled(False)
        self.window_combo.setEnabled(False)
        # video always on
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

        if self._video_recorder:
            session_dir = self._video_recorder.stop()
            n_clicks = self._video_recorder.click_count
            self._video_recorder = None
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.replay_btn.setEnabled(True)
            self.annotate_btn.setEnabled(True)
            self.window_combo.setEnabled(True)
            self.status_label.setText(f"Saved ({n_clicks} clicks)")
            self.status_label.setStyleSheet("font-weight: bold; color: #333;")
            self.element_label.setText(f"→ {session_dir}")
            self.frame_label.setText(f"Clicks: {n_clicks}")

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

        pack_name = self._get_selected_app()
        if not pack_name:
            self.element_label.setText("Select a pack first")
            return

        self._annotating = True
        self.annotate_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.status_label.setText("Annotating...")
        self.status_label.setStyleSheet("font-weight: bold; color: #FF8C00;")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        pack_dir = self._ensure_pack(pack_name)

        def run() -> None:
            try:
                from gazefy.core.annotate_pipeline import run_annotate

                def on_progress(msg: str) -> None:
                    self._frame_update.emit(0, msg)
                    # Parse "[N/M]" from message for progress bar
                    import re

                    m = re.search(r"\[(\d+)/(\d+)\]", msg)
                    if m:
                        self._progress_update.emit(int(m.group(1)), int(m.group(2)), msg)

                result = run_annotate(pack_dir, on_progress=on_progress)
                self._progress_update.emit(100, 100, "Done")
                self._frame_update.emit(
                    result["total_images"],
                    f"Done: {result['total_images']} images, "
                    f"{result['labeled']} newly labeled, "
                    f"{result['total_elements']} elements",
                )
            except Exception as e:
                self._frame_update.emit(0, f"Error: {e}")
            finally:
                self._annotating = False

        t = threading.Thread(target=run, daemon=True)
        t.start()

    def _on_train(self) -> None:
        """Train YOLO on all accumulated data. Save timestamped model + log."""
        import datetime

        pack_name = self._get_selected_app()
        if not pack_name:
            self.element_label.setText("Select a pack first")
            return

        pack_dir = self._ensure_pack(pack_name)
        training_dir = pack_dir / "training_data"
        dataset_yaml = training_dir / "dataset.yaml"

        if not dataset_yaml.exists():
            self.element_label.setText("No training data. Annotate first.")
            return

        self.train_btn.setEnabled(False)
        self.annotate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)  # Indeterminate (spinning)
        self.progress_bar.setVisible(True)
        self.start_btn.setEnabled(False)
        self.status_label.setText("Training...")
        self.status_label.setStyleSheet("font-weight: bold; color: #FF8C00;")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = pack_dir / "models"
        logs_dir = pack_dir / "logs"
        models_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)

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

                # Find trained model — search multiple locations
                import glob
                import shutil

                model_src = Path(result.best_model_path)
                if not model_src.exists():
                    # Ultralytics may save to a different root
                    search_paths = [
                        "/opt/homebrew/runs/**/best.pt",
                        "runs/**/best.pt",
                        str(Path.home() / "runs/**/best.pt"),
                    ]
                    for pattern in search_paths:
                        candidates = glob.glob(pattern, recursive=True)
                        if candidates:
                            # Pick the most recently modified
                            model_src = max(
                                (Path(c) for c in candidates),
                                key=lambda p: p.stat().st_mtime,
                            )
                            break

                if model_src.exists():
                    # Save timestamped copy to models/
                    ts_model = models_dir / f"model_{ts}.pt"
                    shutil.copy2(model_src, ts_model)
                    # Update current model.pt
                    shutil.copy2(model_src, pack_dir / "model.pt")
                else:
                    self._frame_update.emit(
                        0, f"Error: trained model not found at {result.best_model_path}"
                    )
                    return

                # Write training log
                map50 = result.metrics.get("metrics/mAP50(B)", "?")
                log_path = logs_dir / f"train_{ts}.log"
                import json as json_mod

                log_data = {
                    "timestamp": ts,
                    "dataset": str(dataset_yaml),
                    "epochs": config.epochs,
                    "imgsz": config.imgsz,
                    "metrics": {
                        k: (float(v) if hasattr(v, "__float__") else v)
                        for k, v in result.metrics.items()
                    },
                    "model_file": f"models/model_{ts}.pt",
                }
                log_path.write_text(json_mod.dumps(log_data, indent=2))

                self._frame_update.emit(
                    1,
                    f"Training done! mAP@0.5={map50} → models/model_{ts}.pt",
                )
            except Exception as e:
                self._frame_update.emit(0, f"Training error: {e}")

        t = threading.Thread(target=run, daemon=True)
        t.start()

    def _restore_window(self) -> None:
        """Restore target app window to the position/size from recording."""
        # Find frame_windows.json from session or latest recording
        win_path = None
        if self._video_session_dir:
            win_path = self._video_session_dir / "frame_windows.json"
        if not win_path or not win_path.exists():
            # Try latest recording in pack
            pack_name = self._get_selected_app()
            if pack_name:
                rec_dir = Path(f"packs/{pack_name}/recordings")
                if rec_dir.exists():
                    for d in sorted(rec_dir.iterdir(), reverse=True):
                        p = d / "frame_windows.json"
                        if p.exists():
                            win_path = p
                            break
        if not win_path or not win_path.exists():
            return
        try:
            windows = json.loads(win_path.read_text())
            if not windows:
                return
            # Use first frame's window rect
            w = windows[0]
            app_name = self.window_combo.currentData() or ""
            if not app_name:
                return
            import subprocess

            script = (
                f'tell application "System Events"\n'
                f'  tell process "{app_name}"\n'
                f"    set position of front window to {{{w['left']}, {w['top']}}}\n"
                f"    set size of front window to {{{w['width']}, {w['height']}}}\n"
                f"  end tell\n"
                f"end tell"
            )
            subprocess.run(["osascript", "-e", script], timeout=3, capture_output=True)
            self.element_label.setText(
                f"Window restored to ({w['left']},{w['top']}) {w['width']}x{w['height']}"
            )
        except Exception as e:
            self.element_label.setText(f"Window restore failed: {e}")

    def _on_replay(self) -> None:
        if self._replaying:
            return

        # Load events from video session
        if self._video_session_dir and (self._video_session_dir / "events.jsonl").exists():
            with open(self._video_session_dir / "events.jsonl") as f:
                self._frames = [json.loads(line) for line in f if line.strip()]

        if not self._frames:
            self.element_label.setText("No events to replay")
            return

        # Restore window to recording position/size before replaying
        self._restore_window()
        import time as time_mod

        time_mod.sleep(0.5)  # Wait for window to settle

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

        def on_scroll(x, y, dx, dy):
            if not self._recording or dy == 0:
                return
            t = time.monotonic() - self._record_start
            direction = "up" if dy > 0 else "down"
            frame = {
                "t": round(t, 3),
                "x": int(x),
                "y": int(y),
                "scroll": direction,
                "dy": round(float(dy), 2),
            }
            if has_model:
                el = self._resolve_element(float(x), float(y))
                frame.update(el)
            self._frames.append(frame)
            el_desc = frame.get("element_class", "")
            desc = f"SCROLL {direction} [{el_desc}]" if el_desc else f"SCROLL {direction}"
            self._frame_update.emit(len(self._frames), desc)

        listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
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
        """Replay all mouse actions: move + click + scroll."""
        try:
            import pyautogui

            pyautogui.FAILSAFE = False
        except ImportError:
            return

        scroll_accum = 0.0

        for i, ev in enumerate(self._frames):
            if not self._replaying:
                break

            x, y = int(ev.get("x", 0)), int(ev.get("y", 0))
            if x <= 5 and y <= 5:
                continue

            click = ev.get("click", "")
            action = ev.get("action", "press")  # press/release
            scroll = ev.get("scroll", "")

            if click:
                pyautogui.moveTo(x, y, _pause=False)
                if action == "release":
                    pyautogui.mouseUp(x, y, button=click, _pause=False)
                    self._frame_update.emit(i + 1, f"RELEASE {click} ({x},{y})")
                elif click == "right":
                    pyautogui.rightClick(x, y, _pause=False)
                    self._frame_update.emit(i + 1, f"CLICK {click} ({x},{y})")
                else:
                    # Check if next event is a release at different position (drag)
                    is_drag = False
                    for j in range(i + 1, min(i + 50, len(self._frames))):
                        nxt = self._frames[j]
                        if nxt.get("click") == click and nxt.get("action") == "release":
                            is_drag = abs(nxt["x"] - x) > 5 or abs(nxt["y"] - y) > 5
                            break
                    if is_drag:
                        pyautogui.mouseDown(x, y, button=click, _pause=False)
                        self._frame_update.emit(i + 1, f"DRAG START ({x},{y})")
                    else:
                        pyautogui.click(x, y, _pause=False)
                        self._frame_update.emit(i + 1, f"CLICK {click} ({x},{y})")
            elif scroll:
                pyautogui.moveTo(x, y, _pause=False)
                raw_dy = ev.get("dy", 0)
                scroll_accum += float(raw_dy)
                # Execute when accumulated >= 1 full step
                if abs(scroll_accum) >= 1:
                    steps = int(scroll_accum)
                    scroll_accum -= steps
                    pyautogui.scroll(steps, x, y)
                    self._frame_update.emit(i + 1, f"SCROLL {scroll} dy={steps} ({x},{y})")
            else:
                pyautogui.moveTo(x, y, _pause=False)

            # Wait for next event timing
            if i + 1 < len(self._frames):
                dt = self._frames[i + 1]["t"] - ev["t"]
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

    def _log(self, msg: str) -> None:
        """Append a line to the log area and auto-scroll."""
        self.log_area.append(msg)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def _on_progress(self, current: int, total: int, desc: str) -> None:
        """Update progress bar from background thread."""
        if total > 0:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        if current >= total and total > 0:
            self.progress_bar.setVisible(False)
        self._log(desc)

    def _on_frame_update(self, count: int, desc: str) -> None:
        if desc.startswith("LOG:"):
            # Log only, don't update element_label
            self._log(desc[4:])
            return
        self.element_label.setText(desc)
        if desc and not desc.startswith("→"):
            self._log(desc)

        if desc.startswith("Done:") or desc.startswith("Training done"):
            self.status_label.setText(desc[:60])
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
            self.annotate_btn.setEnabled(True)
            self.train_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self._annotating = False
        elif desc.startswith("Error:") or desc.startswith("Training error"):
            self.status_label.setText(desc[:60])
            self.status_label.setStyleSheet("font-weight: bold; color: red;")
            self.annotate_btn.setEnabled(True)
            self.train_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self._annotating = False
        elif desc == "done":
            self.status_label.setText("Replay done")
            self.status_label.setStyleSheet("font-weight: bold; color: #333;")
            self.replay_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
        else:
            self.frame_label.setText(f"Clicks: {count}")

    def _update_elapsed(self) -> None:
        if self._recording and self._video_recorder:
            elapsed = self._video_recorder.elapsed
            m, s = divmod(int(elapsed), 60)
            self.time_label.setText(f"{m:02d}:{s:02d}")


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Gazefy")
    w = RecorderWidget()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
