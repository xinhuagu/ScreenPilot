"""ModelRegistry: discovers and loads ApplicationPacks from a packs directory."""

from __future__ import annotations

import logging
from pathlib import Path

from screenpilot.core.application_pack import PACK_MANIFEST, ApplicationPack

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Scans a directory for ApplicationPack subdirectories and indexes them."""

    def __init__(self, packs_dir: str | Path = "packs"):
        self._packs_dir = Path(packs_dir)
        self._packs: dict[str, ApplicationPack] = {}

    @property
    def packs(self) -> dict[str, ApplicationPack]:
        return dict(self._packs)

    def scan(self) -> int:
        """Scan packs directory and load all valid packs. Returns count loaded."""
        self._packs.clear()
        if not self._packs_dir.exists():
            logger.warning("Packs directory does not exist: %s", self._packs_dir)
            return 0

        count = 0
        for child in sorted(self._packs_dir.iterdir()):
            if child.is_dir() and (child / PACK_MANIFEST).exists():
                try:
                    pack = ApplicationPack.load(child)
                    self._packs[pack.metadata.name] = pack
                    count += 1
                except Exception:
                    logger.exception("Failed to load pack from %s", child)
        logger.info("Registry loaded %d pack(s) from %s", count, self._packs_dir)
        return count

    def get(self, name: str) -> ApplicationPack | None:
        """Get a pack by name."""
        return self._packs.get(name)

    def find_for_window(self, window_name: str, owner_name: str = "") -> ApplicationPack | None:
        """Find the first pack whose window_match patterns match the given window."""
        for pack in self._packs.values():
            if pack.matches_window(window_name, owner_name):
                logger.info("Matched pack '%s' for window '%s'", pack.metadata.name, window_name)
                return pack
        return None
