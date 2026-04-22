from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class QueryRewriteResult:
    """One or more retrieval-focused query strings (deduped, non-empty)."""

    queries: tuple[str, ...]
    raw_model_notes: str | None = None


class IQueryRewriter(ABC):
    """Rewrite user text into search-friendly queries for hybrid retrieval."""

    @abstractmethod
    async def arewrite(
        self,
        *,
        user_query: str,
        recent_dialogue: str,
        episodic_summary: str,
    ) -> QueryRewriteResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RetrievalPlannerDecision:
    """Planner output: optional follow-up searches before final answering."""

    chain_of_thought: str
    sufficient: bool
    follow_up_search_queries: tuple[str, ...] = field(default_factory=tuple)


class IRetrievalPlanner(ABC):
    """CoT-style planner that may request extra vector searches (bounded iterations)."""

    @abstractmethod
    async def aplan(
        self,
        *,
        user_query: str,
        memory_hints: str,
        tried_queries: tuple[str, ...],
        evidence_excerpts: tuple[str, ...],
    ) -> RetrievalPlannerDecision:
        raise NotImplementedError
