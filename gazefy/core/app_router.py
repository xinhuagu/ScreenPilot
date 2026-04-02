"""AppRouter: resolves which ApplicationPack should handle the current screen.

Routes based on window metadata. Fails closed — returns None if no confident match.
"""

from __future__ import annotations

import logging

from gazefy.core.application_pack import ApplicationPack
from gazefy.core.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class AppRouter:
    """Routes the active VDI window to the correct ApplicationPack."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._active_pack: ApplicationPack | None = None

    @property
    def active_pack(self) -> ApplicationPack | None:
        return self._active_pack

    def route(self, window_name: str, owner_name: str = "") -> ApplicationPack | None:
        """Determine which pack to activate for the given window.

        Returns the matched pack, or None if no confident match (fail closed).
        If the matched pack is the same as the currently active one, returns it
        without re-logging.
        """
        matched = self._registry.find_for_window(window_name, owner_name)

        if matched is None:
            if self._active_pack is not None:
                logger.warning("No pack matches window '%s' — deactivating", window_name)
                self._active_pack = None
            return None

        if self._active_pack is not matched:
            logger.info(
                "Routing to pack '%s' for window '%s'",
                matched.metadata.name,
                window_name,
            )
            self._active_pack = matched

        return self._active_pack

    def force_pack(self, pack_name: str) -> ApplicationPack | None:
        """Force-activate a pack by name, bypassing window matching."""
        pack = self._registry.get(pack_name)
        if pack is None:
            logger.error("Pack '%s' not found in registry", pack_name)
            return None
        self._active_pack = pack
        logger.info("Force-activated pack '%s'", pack_name)
        return pack
