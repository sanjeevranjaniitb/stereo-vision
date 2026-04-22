"""Binary → plain text helpers for KB upload ingestion.

Used by :func:`copilot_iitb.api.routes.kb._extract_to_request` (multipart upload path).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import PurePath


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    """Return plain text from PDF, DOCX, or common text encodings.

    Why: Upload handlers receive opaque bytes; we must branch on filename extension
    before choosing a parser so the KB only stores searchable UTF-8 text.

    Operation: Dispatch by suffix to ``_extract_pdf``, ``_extract_docx``, or
    ``_extract_plain_text``; raises ``ValueError`` for empty bodies or unsupported types.

    Note: Typically invoked via ``asyncio.to_thread`` from the FastAPI route so PDF
    parsing does not block the event loop.
    """
    if not data:
        raise ValueError("Empty file body")

    suffix = PurePath(filename or "").suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(data)
    if suffix == ".docx":
        return _extract_docx(data)
    if suffix in (".txt", ".text", ".md"):
        return _extract_plain_text(data)
    raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}; use .pdf, .docx, or .txt")


def _extract_pdf(data: bytes) -> str:
    """Extract page text from a PDF byte stream (pypdf)."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    text = "\n\n".join(parts).strip()
    if not text:
        raise ValueError("No extractable text found in PDF")
    return text


def _extract_docx(data: bytes) -> str:
    """Extract paragraph text from a DOCX byte stream (python-docx)."""
    from docx import Document

    doc = Document(BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("No extractable text found in DOCX")
    return text


def _extract_plain_text(data: bytes) -> str:
    """Decode `.txt`/`.md` uploads with tolerant encoding fallbacks."""
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            text = data.decode(encoding).strip()
            if text:
                return text
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode text file as UTF-8, UTF-16, or Latin-1")
