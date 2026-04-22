from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from copilot_iitb.domain.models import IndexDocumentRequest


class IKnowledgeIndex(ABC):
    """Ingestion and persistence for the KB (separate from conversational memory)."""

    @abstractmethod
    async def aindex_document(self, doc: IndexDocumentRequest) -> dict[str, int]:
        """Returns simple stats, e.g. chunks written."""
        raise NotImplementedError

    @abstractmethod
    async def aindex_documents(self, docs: list[IndexDocumentRequest]) -> dict[str, Any]:
        """Ingest multiple documents in one vector-store batch (single embed/add pass per batch)."""
        raise NotImplementedError
