from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass(slots=True)
class SessionRecord:
    """Server-side chat session: owned by SessionManager / repository."""

    session_id: str
    user_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class ISessionRepository(ABC):
    @abstractmethod
    async def acreate(self, user_id: str | None = None) -> SessionRecord:
        raise NotImplementedError

    @abstractmethod
    async def aget(self, session_id: str) -> SessionRecord | None:
        raise NotImplementedError

    @abstractmethod
    async def atouch(self, session_id: str) -> None:
        raise NotImplementedError


def new_session_id() -> str:
    return str(uuid4())
