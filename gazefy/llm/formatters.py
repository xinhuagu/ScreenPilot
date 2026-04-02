"""UIMap → text serialization for LLM consumption.

    UIMap + CursorState → format_state() → str (for LLM prompt)

Produces structured text that an LLM can reason over to decide actions.
"""

from __future__ import annotations

from gazefy.cursor.cursor_monitor import CursorState
from gazefy.tracker.ui_map import UIMap


def format_state(
    ui_map: UIMap,
    cursor: CursorState | None = None,
    screen_context: str = "",
) -> str:
    """Serialize current screen state for LLM consumption.

    Args:
        ui_map: Current UI element map.
        cursor: Optional cursor state.
        screen_context: Optional context string (e.g. "Report Export Dialog").

    Returns:
        Structured text representation of the screen state.
    """
    lines: list[str] = []

    # Header
    lines.append(f"SCREEN STATE [generation={ui_map.generation}, elements={ui_map.element_count}]")
    if ui_map.frame_width:
        lines.append(f"Resolution: {ui_map.frame_width}x{ui_map.frame_height}")
    if screen_context:
        lines.append(f"Context: {screen_context}")

    # Cursor
    if cursor and cursor.current_element:
        el = cursor.current_element
        lines.append(
            f"Cursor: ({cursor.screen_position.x:.0f}, {cursor.screen_position.y:.0f}) "
            f'on [{el.class_name}] "{el.text or el.id}" '
            f"(dwell {cursor.dwell_time_ms:.0f}ms)"
        )
    elif cursor:
        lines.append(
            f"Cursor: ({cursor.screen_position.x:.0f}, {cursor.screen_position.y:.0f}) "
            f"not on any element"
        )

    # Elements grouped by class
    lines.append("")
    lines.append("ELEMENTS:")
    if ui_map.is_empty:
        lines.append("  (none detected)")
    else:
        for el in sorted(ui_map.elements.values(), key=lambda e: (e.bbox.y1, e.bbox.x1)):
            cursor_marker = ""
            if cursor and cursor.current_element and cursor.current_element.id == el.id:
                cursor_marker = " **CURSOR**"
            text_part = f' "{el.text}"' if el.text else ""
            lines.append(
                f"  [{el.id}] {el.class_name}{text_part} "
                f"at ({el.bbox.x1:.0f},{el.bbox.y1:.0f})-"
                f"({el.bbox.x2:.0f},{el.bbox.y2:.0f}) "
                f"conf={el.confidence:.2f}{cursor_marker}"
            )

    return "\n".join(lines)
