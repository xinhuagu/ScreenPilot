"""TaskStore: persist and load learned workflows for an ApplicationPack.

Workflows are stored as YAML files under packs/<app>/workflows/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class TaskStore:
    """Manages pack-local workflow storage."""

    def __init__(self, pack_dir: Path):
        self._workflows_dir = pack_dir / "workflows"
        self._workflows_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}

    def save(self, workflow: dict) -> Path:
        """Save a workflow to the store. Returns the file path."""
        name = workflow.get("name", "unnamed")
        path = self._workflows_dir / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(workflow, f, default_flow_style=False, allow_unicode=True)
        self._cache[name] = workflow
        logger.info("Saved workflow '%s' to %s", name, path)
        return path

    def load(self, name: str) -> dict | None:
        """Load a workflow by name."""
        if name in self._cache:
            return self._cache[name]

        path = self._workflows_dir / f"{name}.yaml"
        if not path.exists():
            return None

        with open(path) as f:
            workflow = yaml.safe_load(f) or {}
        self._cache[name] = workflow
        return workflow

    def list_workflows(self) -> list[str]:
        """List all available workflow names."""
        return [p.stem for p in sorted(self._workflows_dir.glob("*.yaml"))]

    def load_all(self) -> dict[str, dict]:
        """Load all workflows."""
        workflows = {}
        for name in self.list_workflows():
            w = self.load(name)
            if w:
                workflows[name] = w
        return workflows

    def delete(self, name: str) -> bool:
        """Delete a workflow. Returns True if deleted."""
        path = self._workflows_dir / f"{name}.yaml"
        if path.exists():
            path.unlink()
            self._cache.pop(name, None)
            logger.info("Deleted workflow '%s'", name)
            return True
        return False

    def __len__(self) -> int:
        return len(self.list_workflows())
