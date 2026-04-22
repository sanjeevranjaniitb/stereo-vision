from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.documents import Document

from copilot_iitb.config.settings import Settings
from copilot_iitb.core.interfaces.embeddings import IEmbeddingProvider
from copilot_iitb.core.interfaces.retrieval import IRetriever, IReranker, RetrievedChunk


class NoOpReranker(IReranker):
    async def arerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return list(chunks)


def _chroma_distance_to_similarity(distance: float) -> float:
    """Chroma cosine distance is lower for more similar vectors; map to [0, 1] similarity."""
    return max(0.0, min(1.0, 1.0 - float(distance)))


def _vectorstore_score_to_similarity(settings: Settings, raw_score: float) -> float:
    if settings.vector_store_provider == "chroma":
        return _chroma_distance_to_similarity(raw_score)
    return max(0.0, min(1.0, float(raw_score)))


def _metadata_matches(meta: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if meta.get(key) != expected:
            return False
    return True


def _doc_identity(doc: Document) -> str:
    mid = doc.metadata.get("id")
    if mid is not None:
        return str(mid)
    if doc.id:
        return str(doc.id)
    did = doc.metadata.get("document_id")
    if did is not None:
        return str(did)
    return ""


def _chunks_from_scored_ids(
    merged: list[tuple[str, float]],
    by_id: dict[str, Document],
    vector_scores: dict[str, float],
) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    for doc_id, score in merged:
        doc = by_id.get(doc_id)
        if doc is None:
            continue
        md = dict(doc.metadata or {})
        chunk_id = str(md.get("id", doc_id))
        if doc_id in vector_scores:
            md["vector_similarity"] = vector_scores[doc_id]
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=doc.page_content,
                score=float(score),
                metadata=md,
            )
        )
    return out


class LangChainVectorRetriever(IRetriever):
    """Embed the query, search Chroma/Pinecone by vector only, then optional reranking."""

    def __init__(
        self,
        settings: Settings,
        vectorstore: Any,
        embedding_provider: IEmbeddingProvider,
        reranker: IReranker,
    ) -> None:
        self._settings = settings
        self._vectorstore = vectorstore
        self._embedding_provider = embedding_provider
        self._reranker = reranker

    async def _vector_rows_for_query_embedding(
        self, query_embedding: list[float]
    ) -> list[tuple[Document, float]]:
        k = self._settings.retrieval_top_k
        vs = self._vectorstore
        if self._settings.vector_store_provider == "pinecone":
            # Async Pinecone (aiohttp) can raise "Session is closed" when the async client lifecycle
            # does not match request scope; sync index.query in a worker thread is stable.
            return await asyncio.to_thread(
                vs.similarity_search_by_vector_with_score,
                query_embedding,
                k=k,
            )
        return await asyncio.to_thread(vs.similarity_search_by_vector_with_relevance_scores, query_embedding, k)

    def _rows_to_vector_hits(
        self,
        rows: list[tuple[Document, float]],
        filters: dict[str, Any] | None,
    ) -> tuple[list[str], dict[str, Document], dict[str, float]]:
        ordered_ids: list[str] = []
        by_id: dict[str, Document] = {}
        vector_scores: dict[str, float] = {}
        for doc, raw in rows:
            if not _metadata_matches(dict(doc.metadata or {}), filters):
                continue
            did = _doc_identity(doc)
            if not did:
                continue
            sim = _vectorstore_score_to_similarity(self._settings, raw)
            vector_scores[did] = sim
            ordered_ids.append(did)
            by_id[did] = doc
        return ordered_ids, by_id, vector_scores

    async def aretrieve(self, query: str, *, filters: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        try:
            vecs = await self._embedding_provider.aembed_texts([query])
            query_embedding = vecs[0]
            rows = await self._vector_rows_for_query_embedding(query_embedding)

            vector_ids, by_id, vec_scores = self._rows_to_vector_hits(rows, filters)
            top_ids = vector_ids[: self._settings.fusion_top_k]
            merged = [(did, vec_scores[did]) for did in top_ids]

            chunks = _chunks_from_scored_ids(merged, by_id, vec_scores)
        except Exception as e:
            raise e
               
        return await self._reranker.arerank(query, chunks)
