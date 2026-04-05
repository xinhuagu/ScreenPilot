"""Parse downloaded HTML manuals → extract UI element knowledge → build GroundingDINO prompts.

Pipeline:
    1. Parse all HTML files in pack/knowledge/ → extract plain text
    2. Filter pages mentioning UI controls (button, menu, slider, etc.)
    3. Send relevant text to VLM → get app-specific UI element list
    4. Save as pack-level ui_elements.json + grounding_prompt.txt
"""

from __future__ import annotations

import json
import logging
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Keywords that indicate a page describes UI elements
UI_KEYWORDS = [
    "button",
    "menu",
    "toolbar",
    "slider",
    "checkbox",
    "toggle",
    "panel",
    "dialog",
    "tab",
    "icon",
    "control",
    "interface",
    "playlist",
    "equalizer",
    "volume",
    "seek",
    "playback",
    "fullscreen",
    "subtitle",
    "audio",
    "video",
    "preferences",
    "settings",
    "window",
    "statusbar",
    "sidebar",
]

# Max characters to send to VLM per chunk
VLM_CHUNK_SIZE = 12000


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags, keep text content."""

    def __init__(self) -> None:
        super().__init__()
        self._texts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = data.strip()
            if text:
                self._texts.append(text)

    def get_text(self) -> str:
        return " ".join(self._texts)


def extract_text_from_html(html_path: Path) -> str:
    """Extract plain text from an HTML file."""
    try:
        raw = html_path.read_text(encoding="utf-8", errors="ignore")
        parser = _HTMLTextExtractor()
        parser.feed(raw)
        return parser.get_text()
    except Exception:
        return ""


def find_ui_pages(knowledge_dir: Path, max_pages: int = 30) -> list[tuple[Path, str]]:
    """Find HTML pages that describe UI elements. Returns (path, text) pairs."""
    results = []
    for html_file in sorted(knowledge_dir.rglob("*.html")):
        text = extract_text_from_html(html_file)
        if not text or len(text) < 100:
            continue
        text_lower = text.lower()
        # Score by number of UI keyword matches
        score = sum(1 for kw in UI_KEYWORDS if kw in text_lower)
        if score >= 2:
            results.append((html_file, text, score))

    # Sort by relevance score descending
    results.sort(key=lambda x: x[2], reverse=True)
    return [(path, text) for path, text, _ in results[:max_pages]]


def build_ui_prompt_from_docs(
    pack_dir: Path,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Parse knowledge base → VLM → app-specific UI elements + GroundingDINO prompt.

    Saves to pack_dir/knowledge/:
        - ui_elements.json: structured list of UI elements
        - grounding_prompt.txt: GroundingDINO text prompt

    Returns: {"elements": [...], "prompt": "..."}
    """
    from gazefy.llm.copilot import CopilotClient

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    knowledge_dir = pack_dir / "knowledge"
    if not knowledge_dir.exists():
        log("No knowledge directory found")
        return {"elements": [], "prompt": ""}

    # Step 1: Find relevant UI pages
    log("Step 1: Scanning HTML docs for UI element descriptions...")
    pages = find_ui_pages(knowledge_dir)
    log(f"  Found {len(pages)} relevant pages")

    if not pages:
        log("No UI-related pages found in knowledge base")
        return {"elements": [], "prompt": ""}

    # Step 2: Concatenate relevant text into chunks
    all_text = ""
    for path, text in pages:
        # Add page separator with filename for context
        rel_path = path.relative_to(knowledge_dir)
        all_text += f"\n\n=== {rel_path} ===\n{text}"

    # Trim to manageable size
    if len(all_text) > VLM_CHUNK_SIZE * 3:
        all_text = all_text[: VLM_CHUNK_SIZE * 3]

    log(f"  Extracted {len(all_text)} chars of UI documentation")

    # Step 3: Ask VLM to extract UI elements
    log("Step 2: Asking VLM to extract UI elements from documentation...")
    vlm = CopilotClient(model="gpt-4o")

    # Split into chunks if needed
    chunks = []
    for i in range(0, len(all_text), VLM_CHUNK_SIZE):
        chunks.append(all_text[i : i + VLM_CHUNK_SIZE])

    all_elements_text = ""
    for i, chunk in enumerate(chunks):
        log(f"  Processing chunk {i + 1}/{len(chunks)}...")
        resp = vlm.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "You are analyzing software documentation to identify VISIBLE UI "
                        "controls that a user can see and interact with on screen.\n\n"
                        f"Documentation excerpt:\n{chunk}\n\n"
                        "List ALL visible, clickable UI controls mentioned in this text.\n"
                        "Focus on controls visible in the main interface, toolbars, and "
                        "panels — NOT buried config options or registry settings.\n\n"
                        "For each element, give:\n"
                        "- name: SHORT visual name, max 3 words "
                        "(e.g. 'play button', 'volume slider', 'seek bar')\n"
                        "- type: button/slider/toggle/menu/panel/input/icon/tab/"
                        "checkbox/dropdown/label/toolbar\n"
                        "- function: what it does (brief)\n\n"
                        "Format each as: name | type | function\n"
                        "One per line. Keep names SHORT for object detection."
                    ),
                }
            ],
            max_tokens=2000,
        )
        all_elements_text += resp + "\n"

    # Step 4: Parse VLM response into structured elements
    log("Step 3: Building element list and GroundingDINO prompt...")
    elements = []
    seen_names = set()
    for line in all_elements_text.strip().split("\n"):
        parts = line.split("|")
        if len(parts) >= 2:
            name = parts[0].strip().strip("-*• ").lower()
            elem_type = parts[1].strip().lower()
            func = parts[2].strip() if len(parts) > 2 else ""
            # Clean up
            name = re.sub(r"[*`\"']", "", name)
            if name and name not in seen_names and len(name) < 50:
                seen_names.add(name)
                elements.append(
                    {
                        "name": name,
                        "type": elem_type,
                        "function": func,
                    }
                )

    # Deduplicate similar names
    elements = _deduplicate_elements(elements)

    # Build GroundingDINO prompt — use specific element names
    # Group by type, take most important ones (GroundingDINO works best with <20 classes)
    prompt_classes = _build_grounding_classes(elements)
    grounding_prompt = ". ".join(prompt_classes) + "."

    # Save results
    output = {
        "elements": elements,
        "prompt_classes": prompt_classes,
        "grounding_prompt": grounding_prompt,
        "source_pages": len(pages),
        "total_chars_parsed": len(all_text),
    }
    (knowledge_dir / "ui_elements.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False)
    )
    (knowledge_dir / "grounding_prompt.txt").write_text(grounding_prompt)

    log(f"  {len(elements)} UI elements extracted")
    log(f"  {len(prompt_classes)} GroundingDINO classes")
    log(f"  Prompt: {grounding_prompt[:100]}...")
    log("  Saved to knowledge/ui_elements.json + grounding_prompt.txt")

    return {"elements": elements, "prompt": grounding_prompt}


def _deduplicate_elements(elements: list[dict]) -> list[dict]:
    """Remove near-duplicate elements (e.g. 'play button' and 'play')."""
    result = []
    names = set()
    for el in elements:
        name = el["name"]
        # Skip if a more specific version already exists
        skip = False
        for existing in names:
            if name in existing or existing in name:
                # Keep the longer (more specific) one
                if len(name) <= len(existing):
                    skip = True
                    break
        if not skip:
            result.append(el)
            names.add(name)
    return result


def _build_grounding_classes(elements: list[dict], max_classes: int = 20) -> list[str]:
    """Build GroundingDINO class list from extracted elements.

    GroundingDINO works best with short, visual names and <20 classes.
    Filters out long names, config-only items, and duplicates.
    """
    # Always include these generic fallbacks
    base_classes = [
        "button",
        "clickable text",
        "input field",
        "icon",
        "checkbox",
        "dropdown",
        "slider",
        "tab",
        "toggle",
    ]

    # Collect short, specific element names suitable for GroundingDINO
    specific = []
    for el in elements:
        name = el["name"]
        # Skip generic names already in base
        if name in base_classes:
            continue
        # GroundingDINO needs short names (max ~25 chars, ~3 words)
        if len(name) > 25 or name.count(" ") > 3:
            continue
        # Skip names with special chars (menu paths, keyboard shortcuts)
        if any(c in name for c in "->()[]{}"):
            continue
        specific.append(name)

    # Take top specific ones + base classes, up to max
    remaining = max_classes - len(base_classes)
    classes = specific[:remaining] + base_classes

    # Deduplicate while preserving order
    seen: set[str] = set()
    result = []
    for c in classes:
        if c not in seen:
            seen.add(c)
            result.append(c)

    return result


def load_pack_prompt(pack_dir: Path) -> str | None:
    """Load app-specific GroundingDINO prompt if available."""
    prompt_file = pack_dir / "knowledge" / "grounding_prompt.txt"
    if prompt_file.exists():
        return prompt_file.read_text().strip()
    return None
