"""WorkflowExtractor: action trace -> parameterized workflow steps.

Converts a grounded action trace into a canonical workflow by:
1. Filtering noise (consecutive duplicate scrolls, ungrounded actions)
2. Collapsing repeated actions into single steps
3. Detecting parameterizable values (file paths, text input)
4. Generating a named workflow with slots
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def extract_workflow(
    trace_path: Path,
    workflow_name: str = "",
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Extract a workflow from an action trace.

    Args:
        trace_path: Path to action_trace.json
        workflow_name: Name for the workflow (auto-generated if empty)
        on_progress: Progress callback

    Returns:
        Workflow dict ready for YAML serialization
    """

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    if not trace_path.exists():
        log(f"No action trace found: {trace_path}")
        return {}

    trace = json.loads(trace_path.read_text())
    log(f"Step 1: Loaded {len(trace)} actions from trace")

    # Step 1: Filter and collapse actions
    steps = _collapse_actions(trace)
    log(f"Step 2: Collapsed to {len(steps)} canonical steps")

    if not steps:
        log("No meaningful steps found")
        return {}

    # Step 2: Detect slots (parameterizable values)
    slots = _detect_slots(steps)
    log(f"Step 3: Detected {len(slots)} slots")

    # Step 3: Generate workflow name if not provided
    if not workflow_name:
        workflow_name = _infer_name(steps)
    log(f"Step 4: Workflow name: {workflow_name}")

    # Build workflow
    workflow = {
        "name": workflow_name,
        "intent_examples": [workflow_name.replace("_", " ")],
        "slots": [{"name": s["name"], "type": s["type"]} for s in slots],
        "steps": steps,
    }

    return workflow


def _collapse_actions(trace: list[dict]) -> list[dict]:
    """Collapse raw trace into canonical workflow steps."""
    steps = []
    prev_target = ""
    prev_action = ""
    scroll_count = 0

    for action in trace:
        action_type = action.get("action", "")
        target_text = action.get("target_text", "")
        target_class = action.get("target_class", "")
        target_semantic = action.get("target_semantic_id", "")
        target_bbox = action.get("target_bbox", [])

        # Skip ungrounded actions (no target element found)
        if not target_class and action_type != "scroll":
            continue

        # Use semantic_id as target if available, else text, else class
        target = target_semantic or target_text or target_class

        # Collapse consecutive scrolls on same element
        if action_type == "scroll":
            if prev_action == "scroll" and target == prev_target:
                scroll_count += 1
                continue
            elif prev_action == "scroll" and scroll_count > 0:
                # Emit accumulated scroll
                steps.append(
                    {
                        "action": "scroll",
                        "target": prev_target,
                        "details": {"count": scroll_count + 1},
                    }
                )
                scroll_count = 0

        # Flush pending scroll if action changed
        if prev_action == "scroll" and action_type != "scroll" and scroll_count > 0:
            steps.append(
                {
                    "action": "scroll",
                    "target": prev_target,
                    "details": {"count": scroll_count + 1},
                }
            )
            scroll_count = 0

        if action_type == "scroll":
            scroll_count = 1
            prev_target = target
            prev_action = "scroll"
            continue

        # Skip duplicate consecutive clicks on same target
        if action_type == prev_action and target == prev_target:
            continue

        step: dict = {
            "action": action_type,
            "target": target,
        }

        # Add expect hint from screen_changed
        if action.get("screen_changed"):
            step["expect"] = f"{target}_changed"

        # Add bbox for unresolved targets
        if not target_semantic and target_bbox:
            step["target_bbox_norm"] = target_bbox

        steps.append(step)
        prev_target = target
        prev_action = action_type

    # Flush final scroll
    if prev_action == "scroll" and scroll_count > 0:
        steps.append(
            {
                "action": "scroll",
                "target": prev_target,
                "details": {"count": scroll_count + 1},
            }
        )

    return steps


def _detect_slots(steps: list[dict]) -> list[dict]:
    """Detect parameterizable values in workflow steps."""
    slots = []
    for step in steps:
        if step.get("action") == "type":
            # Typed text is likely a parameter
            text = step.get("details", {}).get("text", "")
            if text:
                slot_name = f"input_{len(slots)}"
                slots.append(
                    {
                        "name": slot_name,
                        "type": "string",
                        "example": text,
                    }
                )
                step["slot"] = slot_name

        # File paths in targets
        target = step.get("target", "")
        if "/" in target or "\\" in target or target.endswith((".mp4", ".mkv", ".avi", ".pdf")):
            slot_name = "file_path"
            if not any(s["name"] == "file_path" for s in slots):
                slots.append(
                    {
                        "name": slot_name,
                        "type": "string",
                        "example": target,
                    }
                )
                step["slot"] = slot_name

    return slots


def _infer_name(steps: list[dict]) -> str:
    """Infer a workflow name from the step targets."""
    # Use first and last click targets
    clicks = [s for s in steps if s.get("action") in ("click", "double_click")]
    if not clicks:
        return "unnamed_workflow"

    first = clicks[0].get("target", "unknown").lower().replace(" ", "_")
    if len(clicks) > 1:
        last = clicks[-1].get("target", "").lower().replace(" ", "_")
        return f"{first}_to_{last}"
    return f"{first}_action"
