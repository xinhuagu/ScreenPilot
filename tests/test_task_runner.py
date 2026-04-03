"""Tests for TaskRunner (M6-lite)."""

from __future__ import annotations

from unittest.mock import MagicMock

from gazefy.actions.action_types import Action, ActionResult, ActionStatus, ActionType
from gazefy.core.task_runner import TaskResult, TaskRunner
from gazefy.tracker.ui_map import UIMap
from gazefy.utils.geometry import Point

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui_map(n_elements: int = 2) -> UIMap:
    """Return a UIMap pre-populated with stub elements."""
    ui_map = MagicMock(spec=UIMap)
    ui_map.is_empty = n_elements == 0
    ui_map.element_count = n_elements
    ui_map.elements = {}
    ui_map.get.return_value = MagicMock()  # element found
    return ui_map


def _make_action(action_id: str = "btn_001") -> Action:
    return Action(type=ActionType.CLICK, target_element_id=action_id)


def _make_success_result(action: Action) -> ActionResult:
    return ActionResult(
        action=action,
        status=ActionStatus.SUCCESS,
        executed_at=Point(100, 100),
        screen_changed=True,
        diff_score=0.45,
    )


def _make_failed_result(action: Action) -> ActionResult:
    return ActionResult(
        action=action,
        status=ActionStatus.FAILED,
        executed_at=Point(100, 100),
        screen_changed=False,
        error="No screen change",
    )


def _make_aborted_result(action: Action) -> ActionResult:
    return ActionResult(
        action=action,
        status=ActionStatus.ABORTED,
        error="Element not found",
    )


def _make_orchestrator(ui_map: UIMap | None = None) -> MagicMock:
    """Return a mock Orchestrator with sensible defaults."""
    orch = MagicMock()
    orch.capture.get_latest_frame.return_value = MagicMock(image=MagicMock())
    orch.cursor.state = MagicMock()
    orch.tracker.current_map = ui_map or _make_ui_map()
    orch.executor._capture_fn = None  # will be overwritten by TaskRunner.__init__
    return orch


def _make_llm(actions: list[Action]) -> MagicMock:
    llm = MagicMock()
    llm.get_actions.return_value = actions
    return llm


# ---------------------------------------------------------------------------
# TaskResult.summary()
# ---------------------------------------------------------------------------


def test_task_result_summary_success():
    action = _make_action()
    result = TaskResult(
        task="Click Export",
        status="success",
        actions_planned=1,
        actions_executed=1,
        actions_succeeded=1,
        results=[_make_success_result(action)],
    )
    text = result.summary()
    assert "SUCCESS" in text
    assert "1/1" in text
    assert "[+]" in text


def test_task_result_summary_failed():
    action = _make_action()
    result = TaskResult(
        task="Click Export",
        status="failed",
        actions_planned=1,
        actions_executed=1,
        actions_succeeded=0,
        results=[_make_failed_result(action)],
    )
    text = result.summary()
    assert "FAILED" in text
    assert "[-]" in text


# ---------------------------------------------------------------------------
# TaskRunner construction
# ---------------------------------------------------------------------------


def test_task_runner_injects_capture_fn():
    orch = _make_orchestrator()
    llm = _make_llm([])
    TaskRunner(orch, llm=llm)
    # capture_fn was injected into executor
    assert orch.executor._capture_fn is not None
    assert callable(orch.executor._capture_fn)


# ---------------------------------------------------------------------------
# TaskRunner.run() — happy path
# ---------------------------------------------------------------------------


def test_run_single_action_success():
    action = _make_action("btn_export")
    ui_map = _make_ui_map(3)
    orch = _make_orchestrator(ui_map)
    orch.executor.execute.return_value = _make_success_result(action)

    runner = TaskRunner(orch, llm=_make_llm([action]))
    result = runner.run("Click Export")

    assert result.status == "success"
    assert result.actions_planned == 1
    assert result.actions_succeeded == 1
    orch.executor.execute.assert_called_once_with(action, ui_map)


def test_run_multi_action_all_success():
    actions = [_make_action("btn_a"), _make_action("btn_b")]
    ui_map = _make_ui_map(4)
    orch = _make_orchestrator(ui_map)
    orch.executor.execute.side_effect = [
        _make_success_result(actions[0]),
        _make_success_result(actions[1]),
    ]

    runner = TaskRunner(orch, llm=_make_llm(actions), re_detect_wait_s=0)
    result = runner.run("Click A then B")

    assert result.status == "success"
    assert result.actions_succeeded == 2
    assert orch.executor.execute.call_count == 2


# ---------------------------------------------------------------------------
# TaskRunner.run() — failure cases
# ---------------------------------------------------------------------------


def test_run_stops_on_failed_action():
    actions = [_make_action("btn_a"), _make_action("btn_b")]
    ui_map = _make_ui_map(4)
    orch = _make_orchestrator(ui_map)
    orch.executor.execute.side_effect = [
        _make_failed_result(actions[0]),
        _make_success_result(actions[1]),  # should never be called
    ]

    runner = TaskRunner(orch, llm=_make_llm(actions), re_detect_wait_s=0)
    result = runner.run("Click A then B")

    assert result.status == "failed"
    assert result.actions_succeeded == 0
    assert orch.executor.execute.call_count == 1  # stopped after first failure


def test_run_partial_when_first_succeeds_second_fails():
    actions = [_make_action("btn_a"), _make_action("btn_b")]
    ui_map = _make_ui_map(4)
    orch = _make_orchestrator(ui_map)
    orch.executor.execute.side_effect = [
        _make_success_result(actions[0]),
        _make_aborted_result(actions[1]),
    ]

    runner = TaskRunner(orch, llm=_make_llm(actions), re_detect_wait_s=0)
    result = runner.run("Click A then B")

    assert result.status == "partial"
    assert result.actions_succeeded == 1
    assert result.actions_executed == 2


def test_run_no_actions_from_llm():
    orch = _make_orchestrator(_make_ui_map(2))
    runner = TaskRunner(orch, llm=_make_llm([]))
    result = runner.run("Do something")

    assert result.status == "no_actions"
    assert result.actions_planned == 0
    orch.executor.execute.assert_not_called()


def test_run_empty_uimap_returns_failed():
    orch = _make_orchestrator(_make_ui_map(0))
    # step() never populates UIMap (still empty)
    orch.tracker.current_map = _make_ui_map(0)

    runner = TaskRunner(orch, llm=_make_llm([]), uimap_timeout_s=0.0)
    result = runner.run("Do something")

    assert result.status == "failed"
    orch.executor.execute.assert_not_called()


def test_run_llm_exception_returns_failed():
    ui_map = _make_ui_map(2)
    orch = _make_orchestrator(ui_map)
    llm = MagicMock()
    llm.get_actions.side_effect = RuntimeError("API error")

    runner = TaskRunner(orch, llm=llm, uimap_timeout_s=0.0)
    # Directly set UIMap so _wait_for_uimap returns immediately
    orch.tracker.current_map = ui_map

    # Patch _wait_for_uimap to skip polling
    runner._wait_for_uimap = lambda: ui_map
    result = runner.run("Do something")

    assert result.status == "failed"


# ---------------------------------------------------------------------------
# Re-detect between actions
# ---------------------------------------------------------------------------


def test_run_calls_step_between_actions():
    actions = [_make_action("btn_a"), _make_action("btn_b")]
    ui_map = _make_ui_map(4)
    orch = _make_orchestrator(ui_map)
    orch.executor.execute.side_effect = [
        _make_success_result(actions[0]),
        _make_success_result(actions[1]),
    ]

    runner = TaskRunner(orch, llm=_make_llm(actions), re_detect_wait_s=0)
    # Bypass _wait_for_uimap so we can count only the between-action step() calls
    runner._wait_for_uimap = lambda: ui_map
    orch.step.reset_mock()

    runner.run("Click A then B")

    # step() called once between the two actions (not after the last one)
    assert orch.step.call_count == 1
