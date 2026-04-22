"""FastAPI application factory and dependency wiring (retriever, KB, chat).

Routers:
    ``/v1/chat`` → :mod:`copilot_iitb.api.routes.chat`
    ``/v1/kb/...`` (documents + upload) → :mod:`copilot_iitb.api.routes.kb`
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from copilot_iitb.api.container import AppContainer
from copilot_iitb.api.rate_limit import InMemoryRateLimiter
from copilot_iitb.api.routes import chat, health, kb, sessions
from copilot_iitb.application.chat_service import ChatService
from copilot_iitb.config.settings import settings
from copilot_iitb.infrastructure.langchain.chroma_factory import build_chroma_vectorstore, get_chromadb_client
from copilot_iitb.infrastructure.langchain.pinecone_factory import build_pinecone_vectorstore
from copilot_iitb.infrastructure.langchain.embedding_adapter import build_embedding_provider
from copilot_iitb.infrastructure.langchain.embedding_reranker import EmbeddingCosineReranker
from copilot_iitb.infrastructure.langchain.vector_retriever import LangChainVectorRetriever, NoOpReranker
from copilot_iitb.infrastructure.langchain.knowledge_index_service import KnowledgeIndexService
from copilot_iitb.infrastructure.llm.query_rewriter import build_query_rewriter
from copilot_iitb.infrastructure.llm.retrieval_planner import build_retrieval_planner
from copilot_iitb.infrastructure.llm.structured_synthesizer import build_synthesizer
from copilot_iitb.infrastructure.memory.in_memory_memory import InMemoryMemoryStore
from copilot_iitb.infrastructure.security.input_guard import InputGuard
from copilot_iitb.infrastructure.session.in_memory_session_repo import InMemorySessionRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    embed_provider = build_embedding_provider(settings)
    lc_embeddings = embed_provider.langchain_embeddings

    if settings.vector_store_provider == "pinecone":
        vectorstore = build_pinecone_vectorstore(settings, lc_embeddings)
    else:
        chroma_client = get_chromadb_client(settings)
        vectorstore = build_chroma_vectorstore(settings, lc_embeddings, chroma_client)

    knowledge_index = KnowledgeIndexService(
        settings=settings,
        vectorstore=vectorstore,
        embedding_provider=embed_provider,
    )
    reranker = (
        EmbeddingCosineReranker(settings, embed_provider)
        if settings.enable_embedding_rerank
        else NoOpReranker()
    )
    retriever = LangChainVectorRetriever(
        settings=settings,
        vectorstore=vectorstore,
        embedding_provider=embed_provider,
        reranker=reranker,
    )

    sessions_repo = InMemorySessionRepository()
    memory = InMemoryMemoryStore(settings, embed_provider)
    synthesizer = build_synthesizer(settings)
    query_rewriter = build_query_rewriter(settings)
    retrieval_planner = build_retrieval_planner(settings)
    # Chat pipeline: shared retriever + memory + LLM ports (see ChatService docstring).
    chat_service = ChatService(
        settings=settings,
        sessions=sessions_repo,
        memory=memory,
        retriever=retriever,
        synthesizer=synthesizer,
        input_guard=InputGuard(),
        query_rewriter=query_rewriter,
        retrieval_planner=retrieval_planner,
    )

    app.state.limiter = InMemoryRateLimiter(per_minute=settings.rate_limit_per_minute)
    # Routes read `request.app.state.container` for chat + KB without global singletons.
    app.state.container = AppContainer(
        chat_service=chat_service,
        knowledge_index=knowledge_index,
        sessions=sessions_repo,
    )
    yield


def _patch_openapi_file_upload_widgets(schema: dict[str, Any]) -> None:
    """Swagger UI shows a file picker only when file parts use ``format: binary``.

    Pydantic v2 emits ``contentMediaType: application/octet-stream`` on array items
    instead, which OpenAPI tooling often surfaces as ``array<string>`` with no browse
    control. Normalize those nodes for multipart file fields.
    """

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            items = node.get("items")
            if isinstance(items, dict):
                if (
                    items.get("type") == "string"
                    and items.get("contentMediaType") == "application/octet-stream"
                ):
                    items.setdefault("format", "binary")
                    items.pop("contentMediaType", None)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)


def create_app() -> FastAPI:
    app = FastAPI(title="Copilot IITB RAG API", version="0.1.0", lifespan=lifespan)

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        )
        _patch_openapi_file_upload_widgets(openapi_schema)
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(chat.router)  # POST /v1/chat
    app.include_router(kb.router)  # POST /v1/kb/documents, /v1/kb/documents/upload
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("copilot_iitb.api.main:app", host="0.0.0.0", port=8000, reload=False)
