from __future__ import annotations

import re
from dataclasses import dataclass

from copilot_iitb.config.settings import Settings


@dataclass(frozen=True, slots=True)
class PolicyResult:
    allowed: bool
    block_reason: str | None = None


def _parse_blocklist(raw: str) -> tuple[str, ...]:
    parts = [p.strip() for p in raw.split("|")]
    return tuple(p for p in parts if p)


def evaluate_input_policy(settings: Settings, text: str) -> PolicyResult:
    """Configurable phrase blocklist and injection heuristics (driven by .env)."""
    hay = text.casefold()
    if settings.guardrail_enable_phrase_blocklist:
        for phrase in _parse_blocklist(settings.guardrail_blocked_phrases):
            if phrase.casefold() in hay:
                return PolicyResult(allowed=False, block_reason="phrase_blocklist")
    if settings.guardrail_enable_injection_regex:
        pattern = (settings.guardrail_injection_regex or "").strip()
        if pattern:
            try:
                if re.search(pattern, text):
                    return PolicyResult(allowed=False, block_reason="injection_heuristic")
            except re.error:
                # Misconfigured regex should not hard-block all traffic; skip this check.
                pass
    return PolicyResult(allowed=True)
