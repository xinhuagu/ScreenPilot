"""LLM interface: send screen state to an LLM, get back action decisions.

    UIMap + task description → LLMInterface → list[Action]

Abstracts the LLM provider. V1 supports Anthropic Claude API.
"""

from __future__ import annotations

import logging

from gazefy.actions.action_types import Action
from gazefy.cursor.cursor_monitor import CursorState
from gazefy.llm.formatters import format_state
from gazefy.llm.parsers import parse_actions
from gazefy.tracker.ui_map import UIMap

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Gazefy, an AI that operates a software application by \
analyzing detected UI elements and issuing precise mouse/keyboard actions.

You receive a structured list of UI elements with their IDs, types, positions, and text.
You must respond with a JSON object containing an "actions" array.

Each action has:
- "type": one of "click", "double_click", "right_click", "type_text", \
"press_key", "hotkey", "scroll", "wait"
- "target": the element ID to act on (e.g. "btn_0042")
- "text": for type_text actions
- "keys": for press_key/hotkey actions (array of key names)
- "scroll": for scroll actions (positive = down)

Respond ONLY with JSON. No explanation."""


class LLMInterface:
    """Sends screen state to an LLM and parses action responses."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
    ):
        self._provider = provider
        self._model = model
        self._client = None

    def get_actions(
        self,
        ui_map: UIMap,
        task: str,
        cursor: CursorState | None = None,
    ) -> list[Action]:
        """Send current state to LLM and return parsed actions.

        Args:
            ui_map: Current screen element map.
            task: Natural language task description.
            cursor: Optional cursor state.

        Returns:
            List of parsed actions to execute.
        """
        state_text = format_state(ui_map, cursor)
        prompt = f"TASK: {task}\n\n{state_text}"

        response = self._call_llm(prompt)
        return parse_actions(response)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API. Requires 'llm' extra."""
        if self._provider == "anthropic":
            return self._call_anthropic(prompt)
        raise ValueError(f"Unsupported provider: {self._provider}")

    def _call_anthropic(self, prompt: str) -> str:
        from gazefy.llm.client import call_with_retry, get_client

        if self._client is None:
            self._client = get_client()

        response = call_with_retry(
            lambda: self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        return response.content[0].text
