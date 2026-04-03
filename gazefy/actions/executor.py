"""ActionExecutor: translates Actions into mouse/keyboard operations.

    Action + UIMap + CoordinateTransform → ActionExecutor → ActionResult

Execution flow per action:
    1. Resolve target coordinates (from element ID or explicit coords)
    2. Pre-check: verify element still exists in current UIMap
    3. Capture "before" screenshot (when capture_fn is injected)
    4. Execute via pyautogui
    5. Poll for screen change up to verify_timeout_s (20 Hz, ChangeDetector)
    6. Return ActionResult with status + screen_changed + diff_score

Verification is opt-in: pass capture_fn=... to enable it.
Without capture_fn the executor behaves as before (backward-compatible).
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from gazefy.actions.action_types import Action, ActionResult, ActionStatus, ActionType
from gazefy.actions.coordinate_transform import CoordinateTransform
from gazefy.capture.change_detector import ChangeDetector, ChangeLevel
from gazefy.tracker.ui_map import UIMap
from gazefy.utils.geometry import Point

logger = logging.getLogger(__name__)

# Minimum change level to consider an action "verified"
_VERIFY_THRESHOLD = ChangeLevel.MINOR

# Action types that do not produce a screen change (skip verification)
_NO_VERIFY = {ActionType.WAIT}


class ActionExecutor:
    """Executes actions on screen with optional post-action change verification."""

    def __init__(
        self,
        transform: CoordinateTransform,
        dry_run: bool = False,
        inter_action_delay_ms: int = 100,
        capture_fn: Callable[[], np.ndarray] | None = None,
        verify_timeout_s: float = 2.0,
    ):
        """
        Args:
            transform: Pixel ↔ screen coordinate converter.
            dry_run: Log but do not execute actions.
            inter_action_delay_ms: Minimum delay between actions in a sequence.
            capture_fn: Optional callable returning a fresh BGRA screen frame.
                        When provided, enables post-action change verification.
                        When None, verification is skipped (backward-compatible).
            verify_timeout_s: Maximum time to wait for a screen change after acting.
        """
        self._transform = transform
        self._dry_run = dry_run
        self._delay = inter_action_delay_ms / 1000.0
        self._capture_fn = capture_fn
        self._verify_timeout = verify_timeout_s

    def execute(self, action: Action, ui_map: UIMap) -> ActionResult:
        """Execute a single action and optionally verify screen changed.

        Status semantics:
          SUCCESS  — executed; screen changed (or verification not configured)
          FAILED   — executed but screen did not change within verify_timeout_s
          ABORTED  — not executed (target element missing from UIMap)
          SKIPPED  — not executed (dry_run mode)
        """
        # --- 1. Resolve coordinates ---
        screen_pt = self._resolve_target(action, ui_map)
        if screen_pt is None:
            return ActionResult(
                action=action,
                status=ActionStatus.ABORTED,
                error=f"Cannot resolve target: {action.target_element_id!r}",
            )

        # --- 2. Pre-check: element must still be present ---
        if action.target_element_id and ui_map.get(action.target_element_id) is None:
            return ActionResult(
                action=action,
                status=ActionStatus.ABORTED,
                error=(
                    f"Element {action.target_element_id!r} disappeared from UIMap before execution"
                ),
            )

        # --- 3. Dry run ---
        if self._dry_run:
            logger.info(
                "[DRY RUN] %s at (%.0f, %.0f) target=%s",
                action.type.value,
                screen_pt.x,
                screen_pt.y,
                action.target_element_id,
            )
            return ActionResult(
                action=action,
                status=ActionStatus.SKIPPED,
                executed_at=screen_pt,
            )

        # --- 4. Capture "before" frame ---
        before_frame: np.ndarray | None = None
        if self._capture_fn is not None and action.type not in _NO_VERIFY:
            before_frame = self._capture_fn()

        # --- 5. Execute ---
        try:
            self._do_action(action, screen_pt)
        except Exception as exc:
            logger.error("Action %s raised: %s", action.type.value, exc)
            return ActionResult(
                action=action,
                status=ActionStatus.FAILED,
                executed_at=screen_pt,
                error=str(exc),
            )

        time.sleep(self._delay)

        # --- 6. Verify screen changed ---
        if before_frame is not None:
            changed, diff_score = self._verify_change(before_frame)
            if not changed:
                logger.warning(
                    "%s at (%d,%d): no screen change within %.1fs",
                    action.type.value,
                    int(screen_pt.x),
                    int(screen_pt.y),
                    self._verify_timeout,
                )
            return ActionResult(
                action=action,
                status=ActionStatus.SUCCESS if changed else ActionStatus.FAILED,
                executed_at=screen_pt,
                screen_changed=changed,
                diff_score=diff_score,
                error=(
                    ""
                    if changed
                    else (
                        f"No screen change after {action.type.value} "
                        f"within {self._verify_timeout:.1f}s"
                    )
                ),
            )

        # Verification not configured — assume success
        return ActionResult(
            action=action,
            status=ActionStatus.SUCCESS,
            executed_at=screen_pt,
        )

    def execute_sequence(self, actions: list[Action], ui_map: UIMap) -> list[ActionResult]:
        """Execute a sequence, stopping on first FAILED or ABORTED result."""
        results: list[ActionResult] = []
        for action in actions:
            result = self.execute(action, ui_map)
            results.append(result)
            if result.status in (ActionStatus.FAILED, ActionStatus.ABORTED):
                logger.warning(
                    "Sequence halted after %d/%d actions: %s",
                    len(results),
                    len(actions),
                    result.error,
                )
                break
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_target(self, action: Action, ui_map: UIMap) -> Point | None:
        """Resolve action target to screen coordinates."""
        if action.coordinates is not None:
            return action.coordinates

        if action.target_element_id:
            element = ui_map.get(action.target_element_id)
            if element is None:
                logger.warning("Element %s not found in UIMap", action.target_element_id)
                return None
            return self._transform.pixel_to_screen(element.center)

        # No spatial target needed for keyboard/wait actions
        if action.type in (ActionType.PRESS_KEY, ActionType.HOTKEY, ActionType.WAIT):
            return Point(0, 0)

        return None

    def _do_action(self, action: Action, screen_pt: Point) -> None:
        """Execute the actual pyautogui call. Requires 'platform' extra."""
        try:
            import pyautogui
        except ImportError:
            raise RuntimeError("Install platform extra: pip install gazefy[platform]")

        pyautogui.FAILSAFE = False
        x, y = int(screen_pt.x), int(screen_pt.y)

        match action.type:
            case ActionType.CLICK:
                pyautogui.click(x, y)
            case ActionType.DOUBLE_CLICK:
                pyautogui.doubleClick(x, y)
            case ActionType.RIGHT_CLICK:
                pyautogui.rightClick(x, y)
            case ActionType.TYPE_TEXT:
                pyautogui.click(x, y)
                time.sleep(0.05)  # Let focus settle before typing
                pyautogui.write(action.text, interval=0.02)
            case ActionType.PRESS_KEY:
                for key in action.keys:
                    pyautogui.press(key)
            case ActionType.HOTKEY:
                pyautogui.hotkey(*action.keys)
            case ActionType.SCROLL:
                pyautogui.scroll(action.scroll_amount, x, y)
            case ActionType.WAIT:
                time.sleep(0.5)

    def _verify_change(self, before_frame: np.ndarray) -> tuple[bool, float]:
        """Poll for screen change after an action.

        Seeds a fresh ChangeDetector with before_frame, then polls every 50 ms
        (≈20 Hz) for up to verify_timeout_s.

        Returns:
            (changed, diff_score) — changed is True if >= _VERIFY_THRESHOLD detected.
        """
        detector = ChangeDetector()
        detector.check(before_frame)  # Seed with pre-action state

        deadline = time.monotonic() + self._verify_timeout
        best_score = 0.0

        while time.monotonic() < deadline:
            time.sleep(0.05)
            frame = self._capture_fn()  # type: ignore[misc]  # guaranteed non-None here
            result = detector.check(frame)
            if result.diff_score > best_score:
                best_score = result.diff_score
            if result.change_level.value >= _VERIFY_THRESHOLD.value:
                return True, result.diff_score

        return False, best_score
