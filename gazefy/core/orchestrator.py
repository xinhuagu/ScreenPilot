"""Orchestrator: the main runtime loop that wires all modules together.

Complete data flow:

    ┌──────────────────────────────────────────────────────────────────────┐
    │                        ORCHESTRATOR LOOP                            │
    │                                                                     │
    │  ScreenCapture ──→ ChangeDetector ──→ UIDetector ──→ ElementTracker │
    │       (frame)         (changed?)       (detections)     (UIMap)     │
    │                                                           │         │
    │                    ┌──────────────────────────────────────┤         │
    │                    │                │                     │         │
    │              CursorMonitor    LLMInterface          ActionExecutor  │
    │              (element hover) (reason → actions)    (click/type)     │
    │                                     │                     │         │
    │                                     └──→ Actions ──→──────┘         │
    └──────────────────────────────────────────────────────────────────────┘

Module ownership:
    - ScreenCapture:    capture/ — produces frames
    - ChangeDetector:   capture/ — gates model inference
    - UIDetector:       detection/ — runs pack's YOLO model
    - ElementTracker:   tracker/ — maintains UIMap
    - CursorMonitor:    cursor/ — real-time cursor-to-element
    - LLMInterface:     llm/ — reasons about next action
    - ActionExecutor:   actions/ — executes mouse/keyboard
    - ApplicationPack:  core/ — provides model + labels + config
    - AppRouter:        core/ — selects active pack
"""

from __future__ import annotations

import logging
import time

from gazefy.actions.coordinate_transform import CoordinateTransform
from gazefy.actions.executor import ActionExecutor
from gazefy.capture.change_detector import ChangeDetector
from gazefy.capture.screen_capture import ScreenCapture
from gazefy.config import GazefyConfig
from gazefy.core.app_router import AppRouter
from gazefy.core.model_registry import ModelRegistry
from gazefy.cursor.cursor_monitor import CursorMonitor
from gazefy.detection.detector import UIDetector
from gazefy.tracker.element_tracker import ElementTracker

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main runtime: capture → detect → track → act loop."""

    def __init__(self, config: GazefyConfig):
        self.config = config

        # --- Capture ---
        self.capture = ScreenCapture(
            config.region,
            target_fps=config.capture_fps,
            retina_scale=config.retina_scale,
        )
        self.change_detector = ChangeDetector(
            similarity_threshold=config.similarity_threshold,
            downsample_size=config.downsample_size,
        )

        # --- Pack routing ---
        self.registry = ModelRegistry(packs_dir="packs")
        self.router = AppRouter(self.registry)

        # --- Detection + Tracking ---
        self.detector: UIDetector | None = None
        self.tracker = ElementTracker(
            iou_threshold=config.iou_match_threshold,
            min_stability=config.min_stability,
            stale_after_frames=config.stale_after_frames,
        )

        # --- Coordinate transform ---
        self.transform = CoordinateTransform(
            region=config.region,
            retina_scale=config.retina_scale,
        )

        # --- Cursor ---
        self.cursor = CursorMonitor(
            transform=self.transform,
            poll_rate_hz=config.cursor_poll_rate_hz,
        )

        # --- Actions ---
        self.executor = ActionExecutor(
            transform=self.transform,
            dry_run=config.dry_run,
            inter_action_delay_ms=config.inter_action_delay_ms,
        )

    def setup(self) -> None:
        """Initialize: scan packs, load model, start capture + cursor threads."""
        self.registry.scan()
        logger.info("Registry: %d pack(s) loaded", len(self.registry.packs))

        # Route to pack (by window name or forced)
        if self.config.window_name:
            pack = self.router.route(self.config.window_name)
        else:
            # Use first available pack
            packs = list(self.registry.packs.values())
            pack = packs[0] if packs else None
            if pack:
                self.router.force_pack(pack.metadata.name)

        if pack and pack.has_model:
            self.detector = UIDetector(pack)
            self.detector.load_model()
            logger.info("Detector ready: %s", pack.metadata.name)
        else:
            logger.warning("No model available — running in monitor-only mode")

        self.capture.start()
        self.cursor.start()

    def shutdown(self) -> None:
        """Stop all threads."""
        self.cursor.stop()
        self.capture.stop()

    def step(self) -> None:
        """Run one iteration of the main loop.

        Called repeatedly by run_loop(). Separated out for testability.
        """
        frame = self.capture.get_latest_frame()
        if frame is None:
            return

        # --- Change detection ---
        change = self.change_detector.check(frame.image)
        if not change.changed:
            return

        # --- Detection ---
        if self.detector and self.detector.is_loaded:
            detections = self.detector.detect(frame.image)
            h, w = frame.image.shape[:2]
            self.tracker.update(detections, change, frame_width=w, frame_height=h)

        # --- Publish UIMap to cursor monitor ---
        self.cursor.set_ui_map(self.tracker.current_map)

    def run_loop(self, duration_s: float = 0) -> None:
        """Run the main loop. Blocks until duration or Ctrl+C.

        Args:
            duration_s: Run for N seconds (0 = run forever).
        """
        interval = 1.0 / self.config.capture_fps
        start = time.monotonic()

        try:
            while True:
                self.step()

                if duration_s and (time.monotonic() - start) >= duration_s:
                    break

                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()
