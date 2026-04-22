"""Shared API and domain types for chat, RAG answers, and knowledge-base ingestion.

These models are the contract between FastAPI routes (``/v1/chat``,
``/v1/kb/...``), application services, and infrastructure adapters.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MemoryKind(StrEnum):
    EPISODIC = "episodic"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class ChatRole(StrEnum):
    """Who produced a :class:`ChatMessage` (mirrors common chat-ML conventions)."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SourceCitation(BaseModel):
    """One attributable slice of evidence the assistant relied on (provenance for RAG)."""

    chunk_id: str
    document_id: str | None = None
    title: str | None = None
    snippet: str = Field(description="Short excerpt from the source used.")
    score: float | None = None


class RAGAnswer(BaseModel):
    """Assistant-facing payload: natural language plus citation/evidence metadata.

    Why: Separates the user-visible ``answer`` from structured fields the UI can
    use (confidence, insufficient-evidence flag, follow-up prompt).
    """

    answer: str
    citations: list[SourceCitation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    insufficient_evidence: bool = False
    follow_up_question: str | None = None


class ChatMessage(BaseModel):
    """Single turn in short-term conversational memory (ordered deque per session)."""

    role: ChatRole
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EpisodicTurn(BaseModel):
    """One compressed episode (turn-level) for auditing and personalization."""

    summary: str
    user_intent: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LongTermMemoryItem(BaseModel):
    """Stable facts/preferences distilled from sessions."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Inbound body for ``POST /v1/chat`` (one new user message per request).

    Why ``session_id`` / ``user_id``:
        - ``session_id`` continues an existing server-side chat (memory + retrieval context).
        - ``user_id`` optionally scopes long-term memory hints when no session yet.
    """

    session_id: str | None = None
    user_id: str | None = Field(default=None, description="Optional stable user key for long-term memory.")
    message: str = Field(min_length=1, max_length=16_000)
    metadata_filters: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """Outbound chat result: always returns the canonical ``session_id`` for follow-ups."""

    session_id: str
    result: RAGAnswer
    retrieval_debug: dict[str, Any] | None = None


class IndexDocumentRequest(BaseModel):
    """Normalized document ready for chunking + embedding (JSON ingest or upload pipeline).

    Why: Both ``POST /v1/kb/documents`` and multipart upload converge on this shape
    so :class:`~copilot_iitb.infrastructure.langchain.knowledge_index_service.KnowledgeIndexService`
    has one ingestion code path.
    """

    document_id: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class UploadedFileIngestResult(BaseModel):
    """Per-file outcome row inside :class:`UploadDocumentsIngestResponse`."""

    file_name: str
    ok: bool
    document_id: str | None = None
    chunks: int | None = None
    error: str | None = None


class UploadDocumentsIngestResponse(BaseModel):
    """Aggregate response for ``POST /v1/kb/documents/upload`` (totals + per-file detail)."""

    documents_indexed: int
    chunks_written: int
    files: list[UploadedFileIngestResult]
