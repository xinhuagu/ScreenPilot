"""Parse LLM responses into executable Action sequences.

LLM response (str/JSON) → parse_actions() → list[Action]
"""

from __future__ import annotations

import json
import logging

from gazefy.actions.action_types import Action, ActionType

logger = logging.getLogger(__name__)


def parse_actions(response: str) -> list[Action]:
    """Parse an LLM response into a list of Actions.

    Expects JSON in the format:
        {"actions": [{"type": "click", "target": "btn_0042"}, ...]}

    Falls back to extracting JSON from markdown code blocks if wrapped.
    """
    text = _extract_json(response)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON")
        return []

    raw_actions = data.get("actions", [])
    if not isinstance(raw_actions, list):
        return []

    actions = []
    for item in raw_actions:
        try:
            action = _parse_single(item)
            if action:
                actions.append(action)
        except Exception:
            logger.warning("Skipping unparseable action: %s", item)
    return actions


def _parse_single(item: dict) -> Action | None:
    """Parse a single action dict."""
    action_type_str = item.get("type", "")
    try:
        action_type = ActionType(action_type_str)
    except ValueError:
        logger.warning("Unknown action type: %s", action_type_str)
        return None

    return Action(
        type=action_type,
        target_element_id=item.get("target", ""),
        text=item.get("text", ""),
        keys=tuple(item.get("keys", [])),
        scroll_amount=item.get("scroll", 0),
    )


def _extract_json(text: str) -> str:
    """Extract JSON from a response that might have markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        inner = [line for line in lines[1:] if not line.strip().startswith("```")]
        return "\n".join(inner)
    return text
