from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore

from copilot_iitb.config.settings import Settings


def _normalize_pinecone_host(host: str) -> str:
    h = str(host).strip().rstrip("/")
    return h


def build_pinecone_vectorstore(settings: Settings, lc_embeddings: Embeddings) -> PineconeVectorStore:
    """Connect to a Pinecone index by host (serverless / custom endpoint) plus API key."""
    host = _normalize_pinecone_host(settings.pinecone_index_host or "")
    name = (settings.pinecone_index_name or "").strip()
    ns = settings.pinecone_namespace
    namespace = ns.strip() if ns and str(ns).strip() else None
    key = str(settings.pinecone_api_key).strip()
    return PineconeVectorStore(
        embedding=lc_embeddings,
        pinecone_api_key=key,
        index_name=name,
        host=host,
        namespace=namespace,
    )
