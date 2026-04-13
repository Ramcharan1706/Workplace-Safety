"""Visual and optional sound alert helpers."""

from __future__ import annotations

from core.config import ENABLE_SOUND_ALERTS


class AlertManager:
    def __init__(self, sound_enabled: bool = ENABLE_SOUND_ALERTS) -> None:
        self.sound_enabled = sound_enabled

    def trigger(self, message: str, high_risk: bool) -> str:
        if high_risk and self.sound_enabled:
            self._play_beep()
        return message

    def _play_beep(self) -> None:
        try:
            import winsound

            winsound.Beep(1500, 250)
        except Exception:
            return

