"""Single factory for LangChain embeddings and ``IEmbeddingProvider`` (``build_embedding_provider``).

Call sites should use :func:`build_embedding_provider` so vector stores, KB ingest, reranking, and memory share
one embedding implementation derived from :class:`~copilot_iitb.config.settings.Settings`.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from pydantic import SecretStr

from copilot_iitb.config.settings import Settings
from copilot_iitb.infrastructure.langchain.embedding_bridge import LangChainEmbeddingBridge
from copilot_iitb.infrastructure.langchain.local_hash_embedding import LocalHashEmbeddings


def _azure_openai_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_deployment=settings.azure_openai_embed_deployment,
        openai_api_version=settings.azure_openai_api_version,
        openai_api_key=settings.azure_openai_api_key,
        model=settings.azure_openai_embed_model,
    )


def build_langchain_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "local_hash":
        return LocalHashEmbeddings()
    if settings.embedding_provider == "pinecone":
        from langchain_pinecone import PineconeEmbeddings

        key = str(settings.pinecone_api_key).strip()
        return PineconeEmbeddings(
            model=str(settings.pinecone_embed_model).strip(),
            pinecone_api_key=SecretStr(key),
        )
    use_azure = settings.embedding_provider == "azure_openai" or (
        settings.embedding_provider == "auto" and settings.llm_provider == "azure_openai"
    )
    if use_azure:
        return _azure_openai_embeddings(settings)
    if settings.openai_api_key:
        return OpenAIEmbeddings(openai_api_key=settings.openai_api_key, model=settings.openai_embed_model)
    return LocalHashEmbeddings()


def build_embedding_provider(settings: Settings) -> LangChainEmbeddingBridge:
    return LangChainEmbeddingBridge(build_langchain_embeddings(settings))
