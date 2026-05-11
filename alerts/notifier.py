"""Visual and optional sound alert helpers."""

from __future__ import annotations

import logging
import sys
from core.config import ENABLE_SOUND_ALERTS

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, sound_enabled: bool = ENABLE_SOUND_ALERTS) -> None:
        self.sound_enabled = sound_enabled

    def trigger(self, message: str, high_risk: bool) -> str:
        return message

