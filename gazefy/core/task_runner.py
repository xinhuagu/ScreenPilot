"""TaskRunner: M6-lite single-task execution loop.

Wires LLMInterface + ActionExecutor + Orchestrator for natural-language task
execution:

    task (str)
      ↓
    LLMInterface.get_actions(UIMap, task)  → list[Action]
      ↓
    for each action:
        ActionExecutor.execute(action, UIMap) → ActionResult
        wait → Orchestrator.step() → fresh UIMap
      ↓
    TaskResult

This is intentionally minimal (M6-lite): one LLM call, sequential execution,
stop on first failure. Multi-step planning and retry logic come later.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from gazefy.actions.action_types import ActionResult, ActionStatus

if TYPE_CHECKING:
    from gazefy.core.orchestrator import Orchestrator
    from gazefy.llm.interface import LLMInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Outcome of a single task execution."""

    task: str
    status: Literal["success", "partial", "failed", "no_actions"]
    actions_planned: int = 0
    actions_executed: int = 0
    actions_succeeded: int = 0
    results: list[ActionResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Task:    {self.task}",
            f"Status:  {self.status.upper()}",
            f"Actions: {self.actions_succeeded}/{self.actions_planned} succeeded",
        ]
        for i, r in enumerate(self.results, 1):
            ok = r.status == ActionStatus.SUCCESS
            mark = "+" if ok else "-"
            target = r.action.target_element_id or "(no target)"
            delta = f"  diff={r.diff_score:.2f}" if r.screen_changed else ""
            err = f"  [{r.error}]" if r.error else ""
            lines.append(
                f"  {i}. [{mark}] {r.action.type.value} -> {target}{delta}{err}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TaskRunner:
    """Execute a natural-language task against the live screen.

    Requires an already-setup Orchestrator (capture + cursor threads running).
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        llm: LLMInterface | None = None,
        re_detect_wait_s: float = 0.4,
        uimap_timeout_s: float = 5.0,
    ):
        """
        Args:
            orchestrator: Running Orchestrator (call setup() before TaskRunner).
            llm:          LLM interface. Defaults to Anthropic Claude if None.
            re_detect_wait_s: After each action, wait before re-detecting screen.
            uimap_timeout_s:  How long to wait for UIMap before giving up.
        """
        self._orch = orchestrator
        self._llm = llm or LLMInterface()
        self._re_detect_wait = re_detect_wait_s
        self._uimap_timeout = uimap_timeout_s

        # Inject capture_fn into executor so it can verify screen changes.
        # The executor's capture_fn receives a BGRA frame; get_latest_frame()
        # returns CapturedFrame with .image — use a safe wrapper.
        def _capture_fn():
            frame = orchestrator.capture.get_latest_frame()
            return frame.image if frame is not None else None

        orchestrator.executor._capture_fn = _capture_fn

    def run(self, task: str) -> TaskResult:
        """Execute a natural-language task.

        Args:
            task: What to do, e.g. "Click the Export button" or
                  "Type 'hello' into the search field and press Enter".

        Returns:
            TaskResult describing what happened.
        """
        result = TaskResult(task=task, status="no_actions")

        # --- 1. Wait for UIMap to have detections ---
        ui_map = self._wait_for_uimap()
        if ui_map.is_empty:
            logger.warning(
                "UIMap empty after %.1fs — no elements detected. "
                "Is the YOLO model loaded?",
                self._uimap_timeout,
            )
            result.status = "failed"
            return result

        logger.info("UIMap ready: %d element(s)", ui_map.element_count)

        # --- 2. Ask LLM to plan actions ---
        cursor = self._orch.cursor.state
        try:
            actions = self._llm.get_actions(ui_map, task, cursor=cursor)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            result.status = "failed"
            return result

        result.actions_planned = len(actions)
        if not actions:
            logger.warning("LLM returned no actions for: %r", task)
            return result  # status stays "no_actions"

        logger.info("LLM planned %d action(s):", len(actions))
        for i, a in enumerate(actions, 1):
            logger.info("  %d. %s -> %s", i, a.type.value, a.target_element_id or "(no target)")

        # --- 3. Execute each action ---
        for i, action in enumerate(actions):
            # Always use the latest UIMap before each action
            ui_map = self._orch.tracker.current_map

            logger.info(
                "Executing %d/%d: %s -> %s",
                i + 1,
                len(actions),
                action.type.value,
                action.target_element_id or "(no target)",
            )

            action_result = self._orch.executor.execute(action, ui_map)
            result.results.append(action_result)
            result.actions_executed += 1

            if action_result.status == ActionStatus.SUCCESS:
                result.actions_succeeded += 1
                logger.info(
                    "  -> SUCCESS  diff=%.2f",
                    action_result.diff_score,
                )
            else:
                logger.warning(
                    "  -> %s: %s",
                    action_result.status.value,
                    action_result.error or "no detail",
                )
                result.status = "partial" if result.actions_succeeded > 0 else "failed"
                return result

            # Between actions: wait, then force a fresh detection cycle
            if i < len(actions) - 1:
                time.sleep(self._re_detect_wait)
                self._orch.step()

        # --- 4. Final status ---
        if result.actions_succeeded == result.actions_planned:
            result.status = "success"
        elif result.actions_succeeded > 0:
            result.status = "partial"
        else:
            result.status = "failed"

        return result

    def run_interactive(self) -> None:
        """Read tasks from stdin in a loop until EOF or 'quit'.

        Each task is executed sequentially. Useful for quick manual testing.
        """
        print("Gazefy interactive mode. Type a task and press Enter. 'quit' to exit.\n")
        while True:
            try:
                task = input("task> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not task:
                continue
            if task.lower() in ("quit", "exit", "q"):
                break
            result = self.run(task)
            print("\n" + result.summary() + "\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_for_uimap(self):
        """Poll until UIMap has elements, or return empty map after timeout."""
        deadline = time.monotonic() + self._uimap_timeout
        while time.monotonic() < deadline:
            self._orch.step()
            ui_map = self._orch.tracker.current_map
            if not ui_map.is_empty:
                return ui_map
            time.sleep(0.1)
        return self._orch.tracker.current_map
