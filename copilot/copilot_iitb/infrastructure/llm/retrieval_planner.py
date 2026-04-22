from __future__ import annotations

import json
from typing import Any

from copilot_iitb.config.settings import Settings
from copilot_iitb.core.interfaces.query_planning import IRetrievalPlanner, RetrievalPlannerDecision
from copilot_iitb.infrastructure.llm.openai_json_client import achat_json_object


class NoOpRetrievalPlanner(IRetrievalPlanner):
    """Skips extra retrieval rounds (single-shot hybrid search after rewrite)."""

    async def aplan(
        self,
        *,
        user_query: str,
        memory_hints: str,
        tried_queries: tuple[str, ...],
        evidence_excerpts: tuple[str, ...],
    ) -> RetrievalPlannerDecision:
        _ = user_query, memory_hints, tried_queries, evidence_excerpts
        return RetrievalPlannerDecision(
            chain_of_thought="Reasoning agent disabled by configuration.",
            sufficient=True,
            follow_up_search_queries=tuple(),
        )


class OpenAIRetrievalPlanner(IRetrievalPlanner):
    """Temporary 'agent' skills: only structured follow-up vector queries, bounded by caller."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def aplan(
        self,
        *,
        user_query: str,
        memory_hints: str,
        tried_queries: tuple[str, ...],
        evidence_excerpts: tuple[str, ...],
    ) -> RetrievalPlannerDecision:
        payload = {
            "user_query": user_query,
            "memory_hints": memory_hints[:3000],
            "already_tried_search_queries": list(tried_queries),
            "top_evidence_excerpts": list(evidence_excerpts[:12]),
        }
        user = json.dumps(payload, ensure_ascii=False)
        data: dict[str, Any] = await achat_json_object(
            self._settings,
            system=self._settings.reasoning_planner_system_prompt,
            user=user,
            temperature=self._settings.reasoning_planner_temperature,
        )
        cot = data.get("chain_of_thought")
        if not isinstance(cot, str):
            cot = ""
        suff = data.get("sufficient")
        sufficient = bool(suff) if isinstance(suff, bool) else True
        raw_follow = data.get("follow_up_search_queries")
        follow: list[str] = []
        if isinstance(raw_follow, list):
            follow = [str(x).strip() for x in raw_follow if str(x).strip()]
        tried_set = {t.casefold() for t in tried_queries}
        unique: list[str] = []
        for q in follow:
            if q.casefold() in tried_set:
                continue
            tried_set.add(q.casefold())
            unique.append(q)
        max_q = self._settings.reasoning_max_follow_up_queries_per_step
        return RetrievalPlannerDecision(
            chain_of_thought=cot.strip(),
            sufficient=sufficient,
            follow_up_search_queries=tuple(unique[:max_q]),
        )


def build_retrieval_planner(settings: Settings) -> IRetrievalPlanner:
    if not settings.enable_reasoning_retrieval:
        return NoOpRetrievalPlanner()
    if settings.llm_provider == "azure_openai":
        return OpenAIRetrievalPlanner(settings)
    if settings.openai_api_key:
        return OpenAIRetrievalPlanner(settings)
    return NoOpRetrievalPlanner()
