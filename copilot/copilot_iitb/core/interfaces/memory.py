from __future__ import annotations

from abc import ABC, abstractmethod

from copilot_iitb.domain.models import ChatMessage, EpisodicTurn, LongTermMemoryItem


class IMemoryStore(ABC):
    """Layered memory: episodic summaries, short-term transcript, long-term facts."""

    @abstractmethod
    async def aappend_short_term(self, session_id: str, message: ChatMessage) -> None:
        raise NotImplementedError

    @abstractmethod
    async def aload_short_term(self, session_id: str, limit: int) -> list[ChatMessage]:
        raise NotImplementedError

    @abstractmethod
    async def aappend_episodic(self, session_id: str, turn: EpisodicTurn) -> None:
        raise NotImplementedError

    @abstractmethod
    async def aload_episodic(self, session_id: str, limit: int) -> list[EpisodicTurn]:
        raise NotImplementedError

    @abstractmethod
    async def aupsert_long_term(self, user_id: str, item: LongTermMemoryItem) -> None:
        raise NotImplementedError

    @abstractmethod
    async def aload_long_term_hints(self, user_id: str, query: str, limit: int) -> list[str]:
        raise NotImplementedError
