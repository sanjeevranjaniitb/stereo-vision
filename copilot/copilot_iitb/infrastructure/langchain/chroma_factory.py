from __future__ import annotations

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from copilot_iitb.config.settings import Settings


def get_chromadb_client(settings: Settings) -> chromadb.ClientAPI:
    """Return a persistent (local) or Chroma Cloud client, matching `CHROMA_MODE`."""
    if settings.chroma_mode == "local":
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    args: dict[str, str] = {"api_key": settings.chroma_cloud_api_key}  # validated in Settings
    if settings.chroma_tenant:
        args["tenant"] = settings.chroma_tenant
    if settings.chroma_database:
        args["database"] = settings.chroma_database
    return chromadb.CloudClient(**args)


def build_chroma_vectorstore(
    settings: Settings,
    lc_embeddings: Embeddings,
    client: chromadb.ClientAPI | None = None,
) -> Chroma:
    c = client or get_chromadb_client(settings)
    return Chroma(
        client=c,
        collection_name=settings.chroma_collection,
        embedding_function=lc_embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
