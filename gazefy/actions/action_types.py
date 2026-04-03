"""Action vocabulary: the commands that the LLM can issue.

LLM response → Parser → list[Action] → Executor → list[ActionResult]
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gazefy.utils.geometry import Point


class ActionType(Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    WAIT = "wait"


@dataclass(frozen=True)
class Action:
    """A single action to execute on the screen."""

    type: ActionType
    target_element_id: str = ""  # Resolved via UIMap
    coordinates: Point | None = None  # Explicit coords (alternative to target)
    text: str = ""  # For TYPE_TEXT
    keys: tuple[str, ...] = ()  # For PRESS_KEY / HOTKEY
    scroll_amount: int = 0  # For SCROLL (positive = down)


class ActionStatus(Enum):
    SUCCESS = "success"  # Action executed and verified
    FAILED = "failed"  # Action executed but verification failed
    ABORTED = "aborted"  # Action not executed (safety check failed)
    SKIPPED = "skipped"  # Action skipped (dry_run mode)


@dataclass
class ActionResult:
    """Result of executing an action."""

    action: Action
    status: ActionStatus
    executed_at: Point | None = None  # Actual screen coords used
    screen_changed: bool = False  # True if ChangeDetector saw >= MINOR change
    diff_score: float = 0.0  # Change magnitude (0.0 = none, higher = more)
    error: str = ""
