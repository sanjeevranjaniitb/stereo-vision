from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass
class InMemoryRateLimiter:
    """Very small in-process limiter; swap for Redis in multi-node deployments."""

    per_minute: int

    def __post_init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.time()
        window_start = now - 60.0
        dq = self._hits[key]
        while dq and dq[0] < window_start:
            dq.popleft()
        if len(dq) >= self.per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
        dq.append(now)


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "anonymous"
