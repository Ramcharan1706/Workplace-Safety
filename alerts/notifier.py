"""Visual and optional sound alert helpers."""

from __future__ import annotations

import logging
import sys
from core.config import ENABLE_SOUND_ALERTS

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, sound_enabled: bool = ENABLE_SOUND_ALERTS) -> None:
        self.sound_enabled = sound_enabled
        self._beep_available = self._check_beep_available()

    @staticmethod
    def _check_beep_available() -> bool:
        """Check if platform supports winsound beep."""
        return sys.platform == "win32"

    def trigger(self, message: str, high_risk: bool) -> str:
        if high_risk and self.sound_enabled:
            self._play_beep()
        return message

    def _play_beep(self) -> None:
        """Play beep on Windows. No-op on Linux/macOS."""
        if not self._beep_available:
            return
        try:
            import winsound
            winsound.Beep(1500, 250)
        except Exception as e:
            logger.warning(f"Failed to play alert beep: {e}")

