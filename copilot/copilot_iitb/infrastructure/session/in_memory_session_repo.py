from __future__ import annotations

import asyncio
from datetime import datetime

from copilot_iitb.core.interfaces.session import ISessionRepository, SessionRecord, new_session_id


class InMemorySessionRepository(ISessionRepository):
    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()

    async def acreate(self, user_id: str | None = None) -> SessionRecord:
        rec = SessionRecord(session_id=new_session_id(), user_id=user_id)
        async with self._lock:
            self._sessions[rec.session_id] = rec
        return rec

    async def aget(self, session_id: str) -> SessionRecord | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def atouch(self, session_id: str) -> None:
        async with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].updated_at = datetime.utcnow()
