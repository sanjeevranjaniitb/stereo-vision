from __future__ import annotations

import math

from copilot_iitb.config.settings import Settings
from copilot_iitb.core.interfaces.embeddings import IEmbeddingProvider
from copilot_iitb.core.interfaces.retrieval import IReranker, RetrievedChunk


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 0.0:
        return 0.0
    v = dot / denom
    return max(0.0, min(1.0, float(v)))


class EmbeddingCosineReranker(IReranker):
    """Rerank vector-retrieval hits using query–chunk embedding cosine similarity."""

    def __init__(self, settings: Settings, embed: IEmbeddingProvider) -> None:
        self._settings = settings
        self._embed = embed

    async def arerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        lim = self._settings.rerank_chunk_char_limit
        chunk_texts: list[str] = []
        for c in chunks:
            t = c.text
            if len(t) > lim:
                t = t[:lim]
            chunk_texts.append(t)
        vecs = await self._embed.aembed_texts([query, *chunk_texts])
        qv = vecs[0]
        cvecs = vecs[1:]
        scored: list[tuple[float, RetrievedChunk]] = []
        for c, cv in zip(chunks, cvecs):
            s = _cosine_sim(qv, cv)
            md = dict(c.metadata or {})
            md["rerank_similarity"] = s
            scored.append(
                (
                    s,
                    RetrievedChunk(
                        chunk_id=c.chunk_id,
                        text=c.text,
                        score=float(c.score),
                        metadata=md,
                    ),
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        top_n = self._settings.rerank_top_n
        return [c for _, c in scored[:top_n]]
