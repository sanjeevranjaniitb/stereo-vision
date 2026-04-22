from __future__ import annotations

import asyncio
from typing import Sequence

from langchain_core.embeddings import Embeddings

from copilot_iitb.core.interfaces.embeddings import IEmbeddingProvider


class LangChainEmbeddingBridge(IEmbeddingProvider):
    """Bridges LangChain `Embeddings` to `IEmbeddingProvider` for application modules."""

    def __init__(self, embed_model: Embeddings) -> None:
        self._embed_model = embed_model
        probe = self._embed_model.embed_documents(["dimension-probe"])[0]
        self._dim = len(probe)

    @property
    def langchain_embeddings(self) -> Embeddings:
        return self._embed_model

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    async def aembed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        em = self._embed_model
        if hasattr(em, "aembed_documents"):
            return await em.aembed_documents(list(texts))
        return await asyncio.to_thread(em.embed_documents, list(texts))
