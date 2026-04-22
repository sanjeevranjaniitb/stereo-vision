"""Knowledge-base HTTP API: JSON document indexing and multipart file upload.

Why this module exists: ingestion must accept both structured text
(``POST /v1/kb/documents``) and binary uploads (``POST /v1/kb/documents/upload``),
then hand normalized :class:`~copilot_iitb.domain.models.IndexDocumentRequest`
objects to :class:`~copilot_iitb.infrastructure.langchain.knowledge_index_service.KnowledgeIndexService`.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import PurePath
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from copilot_iitb.api.container import AppContainer
from copilot_iitb.api.rate_limit import InMemoryRateLimiter, client_key
from copilot_iitb.config.settings import settings
from copilot_iitb.domain.models import (
    IndexDocumentRequest,
    UploadDocumentsIngestResponse,
    UploadedFileIngestResult,
)
from copilot_iitb.infrastructure.documents.extraction import extract_text_from_bytes

router = APIRouter(prefix="/v1/kb", tags=["knowledge-base"])


@router.post("/documents")
async def index_document(request: Request, body: IndexDocumentRequest) -> dict[str, int]:
    """Index a single document that already arrives as title + plain text + metadata.

    Why: Programmatic clients (ETL, admin tools) can push text without file I/O;
    the same chunking, embedding, and vector upsert path as uploads is reused.

    Operation: rate-limit, then delegate to
    :meth:`copilot_iitb.core.interfaces.knowledge_index.IKnowledgeIndex.aindex_document`
    (implemented by :class:`~copilot_iitb.infrastructure.langchain.knowledge_index_service.KnowledgeIndexService`).

    Returns:
        Stats such as ``{"chunks": N}`` describing how many vectors were written.
    """
    limiter: InMemoryRateLimiter = request.app.state.limiter
    limiter.check(client_key(request))

    container: AppContainer = request.app.state.container
    return await container.knowledge_index.aindex_document(body)


async def _extract_to_request(
    upload: UploadFile,
    user_metadata: dict[str, Any],
    sem: asyncio.Semaphore,
) -> IndexDocumentRequest | tuple[str, str]:
    """Turn one uploaded file into an :class:`~copilot_iitb.domain.models.IndexDocumentRequest` or an error pair.

    Why: Upload handlers must decode bytes off the wire, run CPU-heavy parsing in a
    thread pool, and respect ``kb_ingest_concurrency`` so many large PDFs cannot
    exhaust workers at once.

    Operation:
        1. Read raw bytes from the ``UploadFile`` (under ``sem``).
        2. Call :func:`copilot_iitb.infrastructure.documents.extraction.extract_text_from_bytes`
           in ``asyncio.to_thread`` so event loop stays responsive.
        3. Build metadata (merge client JSON + always set ``file_name``) and assign
           a fresh ``document_id`` (UUID) for vector-store identity.

    Returns:
        Either a ready-to-index request, or ``(file_name, error_message)`` on failure.
    """
    fname = upload.filename or "unnamed"
    async with sem:
        try:
            raw = await upload.read()
            # Blocking PDF/DOCX parsers must not run on the asyncio event loop.
            text = await asyncio.to_thread(extract_text_from_bytes, fname, raw)
        except Exception as e:
            return (fname, str(e))

    doc_id = str(uuid.uuid4())
    merged: dict[str, Any] = {**user_metadata, "file_name": fname}
    title_raw = merged.get("title")
    if title_raw is not None and str(title_raw).strip():
        title = str(title_raw)
    else:
        title = PurePath(fname).stem or fname

    return IndexDocumentRequest(
        document_id=doc_id,
        title=title,
        text=text,
        metadata=merged,
    )


@router.post("/documents/upload", response_model=UploadDocumentsIngestResponse)
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(..., description="One or more .pdf, .docx, or .txt files"),
    metadata: str = Form(
        default="{}",
        description='JSON object merged into each document\'s metadata; `file_name` is always set from the upload.',
    ),
) -> UploadDocumentsIngestResponse:
    """Ingest one or more files: extract text, chunk+embed+upsert, return per-file status.

    Why: Multipart upload is the primary way humans add PDFs/DOCX to the KB; callers
    need per-file success/failure plus aggregate chunk counts for UI feedback.

    Operation:
        1. Rate-limit and validate ``metadata`` form field as a JSON object.
        2. Concurrently run :func:`_extract_to_request` for each file (bounded by semaphore).
        3. Batch successful extractions through
           :meth:`~copilot_iitb.core.interfaces.knowledge_index.IKnowledgeIndex.aindex_documents`
           for efficiency (single embed/upsert batch in the service).
        4. Zip outcomes back to original ``files`` order to build
           :class:`~copilot_iitb.domain.models.UploadDocumentsIngestResponse`.

    Raises:
        HTTPException: 400 if no files; 422 if ``metadata`` is not a JSON object.
    """
    limiter: InMemoryRateLimiter = request.app.state.limiter
    limiter.check(client_key(request))

    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    try:
        parsed: Any = json.loads(metadata) if metadata.strip() else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"metadata must be valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="metadata JSON must be an object")

    user_metadata: dict[str, Any] = parsed
    sem = asyncio.Semaphore(settings.kb_ingest_concurrency)
    # Parallel extract+decode; semaphore caps simultaneous heavy file work.
    outcomes = await asyncio.gather(*[_extract_to_request(f, user_metadata, sem) for f in files])

    to_index: list[IndexDocumentRequest] = []
    for o in outcomes:
        if isinstance(o, IndexDocumentRequest):
            to_index.append(o)

    container: AppContainer = request.app.state.container
    stats: dict[str, Any] = {"documents": 0, "chunks": 0, "items": []}
    if to_index:
        # One vector-store batch for all successfully parsed documents.
        stats = await container.knowledge_index.aindex_documents(to_index)

    chunk_by_doc: dict[str, int] = {
        str(i["document_id"]): int(i["chunks"]) for i in stats.get("items", []) if i.get("document_id") is not None
    }

    file_results: list[UploadedFileIngestResult] = []
    for upload, outcome in zip(files, outcomes, strict=True):
        fname = upload.filename or "unnamed"
        if isinstance(outcome, tuple):
            file_results.append(UploadedFileIngestResult(file_name=fname, ok=False, error=outcome[1]))
            continue
        chunks = chunk_by_doc.get(outcome.document_id)
        file_results.append(
            UploadedFileIngestResult(
                file_name=fname,
                ok=True,
                document_id=outcome.document_id,
                chunks=chunks,
            )
        )

    return UploadDocumentsIngestResponse(
        documents_indexed=int(stats["documents"]),
        chunks_written=int(stats["chunks"]),
        files=file_results,
    )
