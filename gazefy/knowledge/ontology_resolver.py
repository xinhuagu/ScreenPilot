"""OntologyResolver: enrich UIMap with semantic IDs from ontology.yaml.

Maps each UIElement in a UIMap to its semantic entry from ontology.yaml,
producing an enriched UIMap where elements have semantic_id, description,
and expected_outcome metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from gazefy.tracker.ui_map import UIElement, UIMap

logger = logging.getLogger(__name__)


@dataclass
class OntologyEntry:
    """A single entry from ontology.yaml."""

    semantic_id: str
    detection_class: str = ""
    description: str = ""
    interaction: str = "click"
    expected_outcome: str = ""
    confirmation_required: bool = False


@dataclass
class OntologyResolver:
    """Matches UIMap elements to ontology entries."""

    _entries: dict[str, OntologyEntry] = field(default_factory=dict)
    # Lookup indices
    _by_class_text: dict[str, OntologyEntry] = field(default_factory=dict)
    _by_text: dict[str, OntologyEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, ontology_path: Path | str) -> OntologyResolver:
        """Load ontology from YAML file."""
        ontology_path = Path(ontology_path)
        if not ontology_path.exists():
            logger.warning("Ontology not found: %s", ontology_path)
            return cls()

        with open(ontology_path) as f:
            raw = yaml.safe_load(f) or {}

        resolver = cls()
        for semantic_id, data in raw.items():
            if not isinstance(data, dict):
                continue
            entry = OntologyEntry(
                semantic_id=semantic_id,
                detection_class=data.get("detection_class", ""),
                description=data.get("description", ""),
                interaction=data.get("interaction", "click"),
                expected_outcome=str(data.get("expected_outcome", "")),
                confirmation_required=bool(data.get("confirmation_required", False)),
            )
            resolver._entries[semantic_id] = entry

            # Build lookup indices from semantic_id patterns
            # e.g. "play_button" -> match element with text "play" and class "button"
            parts = semantic_id.rsplit("_", 1)
            if len(parts) == 2:
                label_part, class_part = parts
                label_key = label_part.replace("_", " ").lower()
                # Index by (class, text)
                resolver._by_class_text[f"{class_part}:{label_key}"] = entry
                # Index by text alone
                resolver._by_text[label_key] = entry

        logger.info("Loaded %d ontology entries", len(resolver._entries))
        return resolver

    def resolve(self, element: UIElement) -> OntologyEntry | None:
        """Find the best ontology match for a UIElement."""
        text = (element.text or "").lower().strip()
        class_name = element.class_name.lower().replace(" ", "_")

        # Strategy 1: exact (class, text) match
        key = f"{class_name}:{text}"
        if key in self._by_class_text:
            return self._by_class_text[key]

        # Strategy 2: text match (ignore class)
        if text and text in self._by_text:
            return self._by_text[text]

        # Strategy 3: text contains / is contained by an ontology label
        if text:
            for label, entry in self._by_text.items():
                if label in text or text in label:
                    # Prefer same class
                    if entry.detection_class.replace(" ", "_") == class_name:
                        return entry
            # Relaxed: any class
            for label, entry in self._by_text.items():
                if label in text or text in label:
                    return entry

        return None

    def enrich_map(self, ui_map: UIMap) -> UIMap:
        """Return a new UIMap with semantic IDs injected into elements."""
        if not self._entries:
            return ui_map

        enriched: dict[str, UIElement] = {}
        matched = 0
        for eid, el in ui_map.elements.items():
            entry = self.resolve(el)
            if entry:
                enriched[eid] = UIElement(
                    id=el.id,
                    class_id=el.class_id,
                    class_name=el.class_name,
                    confidence=el.confidence,
                    bbox=el.bbox,
                    center=el.center,
                    text=el.text,
                    parent_id=el.parent_id,
                    stability=el.stability,
                    semantic_id=entry.semantic_id,
                    description=entry.description,
                    context=el.context,
                )
                matched += 1
            else:
                enriched[eid] = el

        logger.debug("Enriched %d/%d elements", matched, len(ui_map.elements))
        return UIMap(
            elements=enriched,
            frame_width=ui_map.frame_width,
            frame_height=ui_map.frame_height,
            generation=ui_map.generation,
            timestamp=ui_map.timestamp,
        )

    def get_entry(self, semantic_id: str) -> OntologyEntry | None:
        """Look up an ontology entry by semantic_id."""
        return self._entries.get(semantic_id)

    def __len__(self) -> int:
        return len(self._entries)
