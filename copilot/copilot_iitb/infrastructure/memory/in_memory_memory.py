from __future__ import annotations

import asyncio
from collections import defaultdict, deque

from copilot_iitb.config.settings import Settings
from copilot_iitb.core.interfaces.embeddings import IEmbeddingProvider
from copilot_iitb.core.interfaces.memory import IMemoryStore
from copilot_iitb.domain.models import ChatMessage, EpisodicTurn, LongTermMemoryItem


class InMemoryMemoryStore(IMemoryStore):
    """Process-local memory with async locks (swap for Redis/DB without changing callers)."""

    def __init__(self, settings: Settings, embeddings: IEmbeddingProvider) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._short: dict[str, deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=settings.short_term_max_messages)
        )
        self._episodic: dict[str, deque[EpisodicTurn]] = defaultdict(lambda: deque(maxlen=200))
        self._long_term: dict[str, list[LongTermMemoryItem]] = defaultdict(list)
        self._long_vectors: dict[str, list[tuple[LongTermMemoryItem, list[float]]]] = defaultdict(list)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _lock(self, session_id: str) -> asyncio.Lock:
        return self._locks[session_id]

    async def aappend_short_term(self, session_id: str, message: ChatMessage) -> None:
        async with self._lock(session_id):
            self._short[session_id].append(message)

    async def aload_short_term(self, session_id: str, limit: int) -> list[ChatMessage]:
        async with self._lock(session_id):
            items = list(self._short[session_id])
        return items[-limit:] if limit > 0 else items

    async def aappend_episodic(self, session_id: str, turn: EpisodicTurn) -> None:
        async with self._lock(session_id):
            self._episodic[session_id].append(turn)

    async def aload_episodic(self, session_id: str, limit: int) -> list[EpisodicTurn]:
        async with self._lock(session_id):
            items = list(self._episodic[session_id])
        return items[-limit:] if limit > 0 else items

    async def aupsert_long_term(self, user_id: str, item: LongTermMemoryItem) -> None:
        vec = (await self._embeddings.aembed_texts([item.text]))[0]
        async with self._locks[f"user:{user_id}"]:
            self._long_term[user_id].append(item)
            self._long_vectors[user_id].append((item, vec))

    async def aload_long_term_hints(self, user_id: str, query: str, limit: int) -> list[str]:
        async with self._locks[f"user:{user_id}"]:
            pairs = list(self._long_vectors[user_id])
        if not pairs or limit <= 0:
            return []
        q = (await self._embeddings.aembed_texts([query]))[0]

        def cos(a: list[float], b: list[float]) -> float:
            import math

            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        scored = sorted(((cos(q, vec), item.text) for item, vec in pairs), key=lambda t: t[0], reverse=True)
        return [t for _, t in scored[:limit]]
