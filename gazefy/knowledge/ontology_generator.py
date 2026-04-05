"""OntologyGenerator: element registry + manual -> ontology.yaml draft.

Generates a business-semantic ontology from detected elements and optional
manual context, using LLM to infer semantic IDs and descriptions.

Output: packs/<app>/ontology.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import yaml

logger = logging.getLogger(__name__)

# Max elements per LLM batch (to stay within token limits)
BATCH_SIZE = 30


def generate_ontology(
    pack_dir: Path,
    manual_path: Path | str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Generate ontology.yaml from element registry + optional manual.

    Args:
        pack_dir: Pack directory containing element_registry.json
        manual_path: Optional HTML/PDF manual for context
        on_progress: Progress callback

    Returns:
        {"elements": N, "output": path_to_ontology_yaml}
    """
    from gazefy.knowledge.manual_parser import ManualParser
    from gazefy.llm.copilot import CopilotClient

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    # Step 1: Load element registry
    registry_path = pack_dir / "element_registry.json"
    if not registry_path.exists():
        log("No element_registry.json found — run Annotate first")
        return {"elements": 0, "output": ""}

    registry = json.loads(registry_path.read_text())
    log(f"Step 1: Loaded {len(registry)} elements from registry")

    # Step 2: Deduplicate elements by (class, text/icon_label)
    unique_elements = _deduplicate_registry(registry)
    log(f"Step 2: {len(unique_elements)} unique elements after dedup")

    # Step 3: Load manual for context (optional)
    parser = ManualParser()
    knowledge_dir = pack_dir / "knowledge"
    if knowledge_dir.exists():
        n = parser.load_knowledge_dir(knowledge_dir)
        log(f"Step 3: Loaded {n} manual chunks from knowledge base")
    elif manual_path:
        manual_path = Path(manual_path)
        if manual_path.suffix == ".pdf":
            parser.load_pdf(manual_path)
        else:
            parser.load_html_file(manual_path)
        log(f"Step 3: Loaded {len(parser)} manual chunks from {manual_path}")
    else:
        log("Step 3: No manual — LLM will infer from OCR/VLM context only")

    # Step 4: Generate ontology via LLM
    log("Step 4: Generating ontology via LLM...")
    vlm = CopilotClient(model="gpt-4o")

    ontology: dict[str, dict] = {}
    batches = [
        unique_elements[i : i + BATCH_SIZE] for i in range(0, len(unique_elements), BATCH_SIZE)
    ]

    for batch_idx, batch in enumerate(batches):
        log(f"  Batch {batch_idx + 1}/{len(batches)} ({len(batch)} elements)...")

        # Build context for each element
        elements_desc = []
        for el in batch:
            label = el.get("text") or el.get("icon_label") or ""
            el_class = el.get("class", "")
            func = el.get("function", "")
            desc = f'- class={el_class}, label="{label}"'
            if func:
                desc += f', function="{func}"'

            # Search manual for relevant context
            if label and len(parser) > 0:
                manual_hits = parser.search(label, top_k=1)
                if manual_hits:
                    snippet = manual_hits[0].body[:200]
                    desc += f"\n  Manual context: {snippet}"

            elements_desc.append(desc)

        prompt = (
            "You are analyzing a software application's UI elements to generate "
            "a semantic ontology.\n\n"
            "For each element below, generate a YAML entry with its semantic_id "
            "as the top-level key.\n\n"
            "Elements:\n"
            + "\n".join(elements_desc)
            + "\n\nRespond ONLY with YAML. Each element gets its own top-level "
            "key (snake_case). Example:\n"
            "```yaml\nplay_button:\n  detection_class: button\n"
            "  description: Starts media playback\n"
            "  interaction: click\n"
            "  expected_outcome: Media begins playing\n"
            "  confirmation_required: false\n"
            "volume_slider:\n  detection_class: slider\n"
            "  description: Adjusts audio volume\n"
            "  interaction: drag\n"
            "  expected_outcome: Volume level changes\n"
            "  confirmation_required: false\n```"
        )

        resp = vlm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=4000,
        )

        # Parse YAML from response
        parsed = _parse_yaml_response(resp)
        for key, value in parsed.items():
            if isinstance(value, dict):
                ontology[key] = value

    # Step 5: Write ontology.yaml
    output_path = pack_dir / "ontology.yaml"
    with open(output_path, "w") as f:
        yaml.dump(ontology, f, default_flow_style=False, allow_unicode=True)

    log(f"Step 5: Written {len(ontology)} entries to {output_path}")
    return {"elements": len(ontology), "output": str(output_path)}


def _deduplicate_registry(registry: dict) -> list[dict]:
    """Deduplicate registry entries by (class, text/icon_label)."""
    seen: set[str] = set()
    unique = []
    for entry in registry.values():
        el_class = entry.get("class", "")
        label = entry.get("text") or entry.get("icon_label") or ""
        key = f"{el_class}:{label.lower().strip()}"
        if key in seen or not label:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _parse_yaml_response(resp: str) -> dict:
    """Extract YAML from LLM response (may be wrapped in ```yaml blocks)."""
    # Strip markdown code fences
    resp = resp.strip()
    if "```yaml" in resp:
        resp = resp.split("```yaml", 1)[1]
        if "```" in resp:
            resp = resp.split("```", 1)[0]
    elif "```" in resp:
        resp = resp.split("```", 1)[1]
        if "```" in resp:
            resp = resp.split("```", 1)[0]

    try:
        parsed = yaml.safe_load(resp)
        if isinstance(parsed, dict):
            return parsed
    except yaml.YAMLError:
        logger.warning("Failed to parse YAML from LLM response")

    return {}
