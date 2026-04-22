from __future__ import annotations

import json
import re
from typing import Any

from copilot_iitb.config.settings import Settings
from copilot_iitb.core.interfaces.query_planning import IQueryRewriter, QueryRewriteResult
from copilot_iitb.infrastructure.llm.openai_json_client import achat_json_object


def _dedupe_nonempty(queries: list[str], *, original: str) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        t = re.sub(r"\s+", " ", q.strip())
        if not t:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    if not out:
        o = original.strip()
        return (o,) if o else tuple()
    return tuple(out)


class HeuristicQueryRewriter(IQueryRewriter):
    """No LLM: pass through the user query as the only retrieval string."""

    async def arewrite(
        self,
        *,
        user_query: str,
        recent_dialogue: str,
        episodic_summary: str,
    ) -> QueryRewriteResult:
        _ = recent_dialogue, episodic_summary
        return QueryRewriteResult(queries=_dedupe_nonempty([user_query], original=user_query))


class OpenAIQueryRewriter(IQueryRewriter):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def arewrite(
        self,
        *,
        user_query: str,
        recent_dialogue: str,
        episodic_summary: str,
    ) -> QueryRewriteResult:
        payload = {
            "user_query": user_query,
            "recent_dialogue": recent_dialogue[-4000:],
            "episodic_summary": episodic_summary[:1200],
        }
        user = json.dumps(payload, ensure_ascii=False)
        data: dict[str, Any] = await achat_json_object(
            self._settings,
            system=self._settings.query_rewrite_system_prompt,
            user=user,
            temperature=self._settings.query_rewrite_temperature,
        )
        raw_notes = data.get("notes")
        notes = raw_notes if isinstance(raw_notes, str) else None
        raw_list = data.get("search_queries")
        queries: list[str] = []
        if isinstance(raw_list, list):
            queries = [str(x) for x in raw_list if isinstance(x, (str, int, float))]
        elif isinstance(data.get("search_query"), str):
            queries = [str(data["search_query"])]
        merged = _dedupe_nonempty(queries, original=user_query)
        if len(merged) > self._settings.query_rewrite_max_variants:
            merged = merged[: self._settings.query_rewrite_max_variants]
        return QueryRewriteResult(queries=merged, raw_model_notes=notes)


def build_query_rewriter(settings: Settings) -> IQueryRewriter:
    if not settings.enable_query_rewrite:
        return HeuristicQueryRewriter()
    if settings.llm_provider == "azure_openai":
        return OpenAIQueryRewriter(settings)
    if settings.openai_api_key:
        return OpenAIQueryRewriter(settings)
    return HeuristicQueryRewriter()
