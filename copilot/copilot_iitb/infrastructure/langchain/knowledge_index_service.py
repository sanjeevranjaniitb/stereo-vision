"""Vector KB persistence: split documents, embed chunks, upsert to Chroma or Pinecone.

Why this module exists: HTTP upload/JSON routes produce
:class:`~copilot_iitb.domain.models.IndexDocumentRequest`; this service is the
only place that should know about chunk sizes, embedding batches, and store-specific
metadata flattening—keeping routes and domain models store-agnostic.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from copilot_iitb.config.settings import Settings
from copilot_iitb.core.interfaces.embeddings import IEmbeddingProvider
from copilot_iitb.core.interfaces.knowledge_index import IKnowledgeIndex
from copilot_iitb.domain.models import IndexDocumentRequest


def _pinecone_flat_metadata(meta: dict[str, Any], text: str, text_key: str) -> dict[str, Any]:
    """Normalize metadata for Pinecone upserts.

    Why: Pinecone rejects nested structures and ``None`` values; we still want rich
    provenance from uploads (strings/numbers) attached to each vector.

    Operation: Copy scalar fields, stringify other values, skip ``$`` keys and ``None``.
    """
    out: dict[str, Any] = {text_key: text}
    for k, v in meta.items():
        if k == text_key or v is None:
            continue
        key = str(k)
        if key.startswith("$"):
            continue
        if isinstance(v, (str, int, float, bool)):
            out[key] = v
        else:
            out[key] = str(v)
    return out


class KnowledgeIndexService(IKnowledgeIndex):
    """Chunk + enrich documents, embed chunks, then upsert into Chroma or Pinecone.

    Called from:
        - ``POST /v1/kb/documents`` → :meth:`aindex_document`
        - ``POST /v1/kb/documents/upload`` → :meth:`aindex_documents` (batch)
    """

    def __init__(
        self,
        settings: Settings,
        vectorstore: Chroma | PineconeVectorStore,
        embedding_provider: IEmbeddingProvider,
    ) -> None:
        """Capture store + embedding ports and configure the recursive text splitter.

        Why: Chunk size/overlap are deployment tuning knobs (``settings``); the
        splitter must stay consistent with how retrieval expects paragraph boundaries.
        """
        self._settings = settings
        self._vectorstore = vectorstore
        self._embedding_provider = embedding_provider
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.kb_chunk_size,
            chunk_overlap=settings.kb_chunk_overlap,
        )
        self._lock = asyncio.Lock()

    def _split_document(self, doc: IndexDocumentRequest) -> tuple[Document, list[Document], list[str]]:
        """Build LangChain :class:`~langchain_core.documents.Document` nodes + stable chunk ids.

        Why: Every chunk needs consistent metadata (``document_id``, ``title``, ACL
        hints) for filtering at chat time; prepending a header improves embedding
        quality versus raw body text alone.

        Operation:
            Merge caller metadata with defaults, prefix ``title``/ownership header,
            run ``RecursiveCharacterTextSplitter``, assign UUID per chunk into ``metadata['id']``.

        Returns:
            ``(parent_document, chunk_documents, chunk_ids)`` — parent is mostly for debugging;
            upsert uses ``chunk_documents`` and parallel ``chunk_ids``.
        """
        now = datetime.utcnow().isoformat() + "Z"
        base_meta = dict(doc.metadata)
        meta: dict[str, Any] = {
            "document_id": doc.document_id,
            "title": doc.title,
            "source": base_meta.get("source", "user_upload"),
            "updated_at": base_meta.get("updated_at", now),
            "owner": base_meta.get("owner", "unknown"),
            "language": base_meta.get("language", "und"),
            "access_scope": base_meta.get("access_scope", "public"),
            "product_area": base_meta.get("product_area", "general"),
            "version": str(base_meta.get("version", "1")),
        }
        for k, v in base_meta.items():
            if k not in meta:
                meta[k] = v

        header = f"# {doc.title}\n\nUpdated: {meta['updated_at']} | Owner: {meta['owner']}\n\n"
        full_text = header + doc.text
        parent = Document(page_content=full_text, metadata=meta)

        nodes = self._splitter.split_documents([parent])
        chunk_ids: list[str] = []
        for node in nodes:
            cid = str(uuid.uuid4())
            node.metadata["id"] = cid
            chunk_ids.append(cid)
        return parent, nodes, chunk_ids

    async def _upsert_embedded_chunks(self, nodes: list[Document], chunk_ids: list[str]) -> int:
        """Embed chunk texts through ``IEmbeddingProvider``, then upsert vectors into Chroma or Pinecone.

        Why: Embedding APIs are async-friendly but Chroma/Pinecone client calls may
        block; we batch Pinecone to respect API limits.

        Operation:
            ``IEmbeddingProvider.aembed_texts`` → validate counts → dispatch to
            Chroma ``collection.upsert`` or Pinecone ``index.upsert`` inside
            ``asyncio.to_thread``.
        """
        if not nodes:
            return 0
        texts = [n.page_content for n in nodes]
        metadatas = [dict(n.metadata) for n in nodes]
        embeddings = await self._embedding_provider.aembed_texts(texts)
        if len(embeddings) != len(chunk_ids):
            msg = f"embedding count ({len(embeddings)}) does not match chunk count ({len(chunk_ids)})"
            raise RuntimeError(msg)

        def _upsert_chroma() -> None:
            self._vectorstore._collection.upsert(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        def _upsert_pinecone() -> None:
            vs = self._vectorstore
            assert isinstance(vs, PineconeVectorStore)
            text_key = vs._text_key
            idx = vs.index
            ns = vs._namespace
            upsert_kw: dict[str, str] = {}
            if ns is not None:
                upsert_kw["namespace"] = ns
            batch = 64
            for start in range(0, len(chunk_ids), batch):
                sl = slice(start, start + batch)
                vectors: list[dict[str, Any]] = []
                for cid, emb, text, meta in zip(
                    chunk_ids[sl], embeddings[sl], texts[sl], metadatas[sl], strict=True
                ):
                    md = _pinecone_flat_metadata(dict(meta), text, text_key)
                    vectors.append({"id": cid, "values": list(emb), "metadata": md})
                idx.upsert(vectors=vectors, **upsert_kw)

        def _upsert() -> None:
            if isinstance(self._vectorstore, Chroma):
                _upsert_chroma()
            else:
                _upsert_pinecone()

        await asyncio.to_thread(_upsert)
        return len(nodes)

    async def aindex_documents(self, docs: list[IndexDocumentRequest]) -> dict[str, Any]:
        """Ingest many documents in one locked section: split all, embed once, upsert once.

        Why: Upload routes pass whole batches—one embed+upsert round-trip reduces
        latency and keeps vector dimension ordering deterministic vs per-file calls.

        Returns:
            Dict with ``documents``, ``chunks``, and ``items`` (per-document chunk counts
            keyed by ``document_id`` / ``file_name`` for API response shaping).
        """
        if not docs:
            return {"documents": 0, "chunks": 0, "items": []}

        all_nodes: list[Document] = []
        all_chunk_ids: list[str] = []
        items: list[dict[str, Any]] = []
        for doc in docs:
            _, nodes, chunk_ids = self._split_document(doc)
            all_nodes.extend(nodes)
            all_chunk_ids.extend(chunk_ids)
            items.append(
                {
                    "document_id": doc.document_id,
                    "file_name": doc.metadata.get("file_name"),
                    "chunks": len(nodes),
                }
            )

        async with self._lock:
            # Serialize writes so concurrent uploads/chat reindex jobs cannot corrupt batch state.
            written = await self._upsert_embedded_chunks(all_nodes, all_chunk_ids)
        return {"documents": len(docs), "chunks": written, "items": items}

    async def aindex_document(self, doc: IndexDocumentRequest) -> dict[str, int]:
        """Convenience wrapper for single-document JSON ingest (``POST /v1/kb/documents``).

        Why: Keeps the public interface symmetrical with
        :class:`~copilot_iitb.core.interfaces.knowledge_index.IKnowledgeIndex` while
        reusing the batched implementation.

        Operation: Delegates to :meth:`aindex_documents` with a one-element list.
        """
        stats = await self.aindex_documents([doc])
        return {"chunks": int(stats["chunks"])}
