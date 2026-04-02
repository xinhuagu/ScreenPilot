"""ActionExecutor: translates Actions into mouse/keyboard operations.

    Action + UIMap + CoordinateTransform → ActionExecutor → ActionResult

Execution flow per action:
    1. Resolve target coordinates (from element ID or explicit coords)
    2. Pre-check: verify element still exists in UIMap
    3. Execute via pyautogui
    4. Wait for screen change (verification)
    5. Return ActionResult with status
"""

from __future__ import annotations

import logging
import time

from gazefy.actions.action_types import Action, ActionResult, ActionStatus, ActionType
from gazefy.actions.coordinate_transform import CoordinateTransform
from gazefy.tracker.ui_map import UIMap
from gazefy.utils.geometry import Point

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Executes actions on screen via pyautogui with coordinate translation."""

    def __init__(
        self,
        transform: CoordinateTransform,
        dry_run: bool = False,
        inter_action_delay_ms: int = 100,
    ):
        self._transform = transform
        self._dry_run = dry_run
        self._delay = inter_action_delay_ms / 1000.0

    def execute(self, action: Action, ui_map: UIMap) -> ActionResult:
        """Execute a single action.

        Args:
            action: The action to execute.
            ui_map: Current UIMap for resolving element targets.

        Returns:
            ActionResult with execution status.
        """
        # 1. Resolve screen coordinates
        screen_pt = self._resolve_target(action, ui_map)
        if screen_pt is None:
            return ActionResult(
                action=action,
                status=ActionStatus.ABORTED,
                error=f"Cannot resolve target: {action.target_element_id}",
            )

        # 2. Dry run — log but don't execute
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

        # 3. Execute
        try:
            self._do_action(action, screen_pt)
        except Exception as e:
            return ActionResult(
                action=action,
                status=ActionStatus.FAILED,
                executed_at=screen_pt,
                error=str(e),
            )

        time.sleep(self._delay)
        return ActionResult(
            action=action,
            status=ActionStatus.SUCCESS,
            executed_at=screen_pt,
        )

    def execute_sequence(self, actions: list[Action], ui_map: UIMap) -> list[ActionResult]:
        """Execute a sequence of actions with inter-action delays."""
        results = []
        for action in actions:
            result = self.execute(action, ui_map)
            results.append(result)
            if result.status in (ActionStatus.FAILED, ActionStatus.ABORTED):
                break  # Stop on failure
        return results

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

        # No target needed for some actions (PRESS_KEY, HOTKEY, WAIT)
        if action.type in (ActionType.PRESS_KEY, ActionType.HOTKEY, ActionType.WAIT):
            return Point(0, 0)

        return None

    def _do_action(self, action: Action, screen_pt: Point) -> None:
        """Execute the actual pyautogui call. Requires 'platform' extra."""
        try:
            import pyautogui
        except ImportError:
            raise RuntimeError("Install platform extra: pip install gazefy[platform]")

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
