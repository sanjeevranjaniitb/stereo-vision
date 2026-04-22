from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SanitizedText:
    value: str


class InputGuard:
    """Lightweight guardrails; not a substitute for full prompt-injection defenses."""

    _control = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

    def sanitize_user_message(self, text: str, *, max_len: int = 16_000) -> SanitizedText:
        cleaned = text.strip()
        cleaned = self._control.sub("", cleaned)
        if len(cleaned) > max_len:
            cleaned = cleaned[:max_len]
        return SanitizedText(cleaned)
