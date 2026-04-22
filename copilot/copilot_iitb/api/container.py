"""Typed dependency bag attached to ``FastAPI`` app state for route handlers."""

from __future__ import annotations

from dataclasses import dataclass

from copilot_iitb.application.chat_service import ChatService
from copilot_iitb.core.interfaces.session import ISessionRepository
from copilot_iitb.infrastructure.langchain.knowledge_index_service import KnowledgeIndexService


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Wired services built in ``api.main.lifespan`` and injected per HTTP request.

    Why: Keeps ``chat`` and ``kb`` routes declarative—pull ``chat_service`` for RAG,
    ``knowledge_index`` for ingestion, ``sessions`` where session APIs need the same repo.
    """

    chat_service: ChatService
    knowledge_index: KnowledgeIndexService
    sessions: ISessionRepository
