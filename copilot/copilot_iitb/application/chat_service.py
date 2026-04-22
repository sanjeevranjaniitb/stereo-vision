"""Application-layer chat orchestration (RAG + memory + guardrails).

Why this module exists: the HTTP route stays thin; all policy and sequencing for
a single user message—sanitize, session lifecycle, retrieval, optional iterative
planning, synthesis, and persistence—lives here so it can be tested and swapped
without touching FastAPI.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Any

from copilot_iitb.agent_debug_log import agent_debug_log
from copilot_iitb.config.settings import Settings
from copilot_iitb.core.interfaces.llm import IAnswerSynthesizer
from copilot_iitb.core.interfaces.memory import IMemoryStore
from copilot_iitb.core.interfaces.query_planning import IQueryRewriter, IRetrievalPlanner
from copilot_iitb.core.interfaces.retrieval import IRetriever, RetrievedChunk
from copilot_iitb.core.interfaces.session import ISessionRepository
from copilot_iitb.domain.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    EpisodicTurn,
    RAGAnswer,
    SourceCitation,
)
from copilot_iitb.infrastructure.security.content_policy import evaluate_input_policy
from copilot_iitb.infrastructure.security.input_guard import InputGuard


class ChatService:
    """Coordinates sessions, layered memory, retrieval, guardrails, and grounded answering.

    Call graph (high level) for :meth:`handle_chat`:
        ``InputGuard`` → policy → session repo → memory → optional greeting shortcut
        → :meth:`IQueryRewriter.arewrite` → retrieval merge → optional
        :meth:`IRetrievalPlanner.aplan` loop → evidence gate →
        :meth:`IAnswerSynthesizer.asynthesize` → memory append → response.
    """

    def __init__(
        self,
        settings: Settings,
        sessions: ISessionRepository,
        memory: IMemoryStore,
        retriever: IRetriever,
        synthesizer: IAnswerSynthesizer,
        input_guard: InputGuard,
        query_rewriter: IQueryRewriter,
        retrieval_planner: IRetrievalPlanner,
    ) -> None:
        """Wire pluggable ports (session, memory, retrieval, LLM) and shared settings.

        Why: ``ChatService`` is constructed once in ``api/main.py`` so routes only
        pull a single object from the app container; dependencies stay injectable for tests.
        """
        self._settings = settings
        self._sessions = sessions
        self._memory = memory
        self._retriever = retriever
        self._synthesizer = synthesizer
        self._input_guard = input_guard
        self._query_rewriter = query_rewriter
        self._retrieval_planner = retrieval_planner

    def _recent_dialogue(self, messages: list[ChatMessage]) -> str:
        """Format the last N chat turns as plain text for LLM context (rewriter / synthesizer).

        Why: Vector search and JSON planners work better when they see recent roles
        and wording, not opaque session IDs.

        Operation: Concatenate ``role: content`` lines in chronological order.
        """
        lines: list[str] = []
        for m in messages:
            lines.append(f"{m.role.value}: {m.content}")
        return "\n".join(lines)

    def _episodic_summary(self, episodic: list[EpisodicTurn]) -> str:
        """Compress prior episodic summaries into a short string for query rewriting.

        Why: Long-term conversation themes help disambiguate pronouns and follow-ups
        without loading full message history into every model call.

        Operation: Take up to the last four non-empty summaries, join, cap length.
        """
        if not episodic:
            return ""
        parts = [e.summary.strip() for e in episodic[-4:] if e.summary.strip()]
        return "\n".join(parts)[:1600]

    def _chunk_rank_key(self, c: RetrievedChunk) -> float:
        """Scalar sort key for ordering retrieved chunks before merge / top-k.

        Why: Chroma/Pinecone may expose scores in ``metadata`` (rerank/vector
        similarity) or on ``score``; we need one comparable float for sorting.

        Operation: Prefer ``rerank_similarity``, then ``vector_similarity``, else ``score``.
        """
        md = c.metadata or {}
        for key in ("rerank_similarity", "vector_similarity"):
            v = md.get(key)
            if isinstance(v, (int, float)):
                return float(v)
        if c.score is not None and isinstance(c.score, (int, float)):
            return float(c.score)
        return 0.0

    def _merge_chunk_pair(self, a: RetrievedChunk, b: RetrievedChunk) -> RetrievedChunk:
        """Merge two :class:`~copilot_iitb.core.interfaces.retrieval.RetrievedChunk` records with the same id.

        Why: Reciprocal-rank fusion can surface the same chunk from multiple query
        lists; we keep one row but combine the strongest similarity signals.

        Operation: Union metadata (max for similarity keys), keep the better ``score``.
        """
        md = dict(a.metadata or {})
        for key, v in (b.metadata or {}).items():
            if key in ("vector_similarity", "rerank_similarity"):
                va = md.get(key)
                if isinstance(va, (int, float)) and isinstance(v, (int, float)):
                    md[key] = max(float(va), float(v))
                elif isinstance(v, (int, float)):
                    md[key] = float(v)
            elif key not in md:
                md[key] = v
        sa = float(a.score) if a.score is not None else 0.0
        sb = float(b.score) if b.score is not None else 0.0
        score = a.score if sa >= sb else b.score
        return RetrievedChunk(chunk_id=a.chunk_id, text=a.text, score=score, metadata=md)

    def _rrf_merge_ranked_lists(
        self, ranked_lists: list[list[RetrievedChunk]], *, cap: int, k: int = 60
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion across multiple already-ranked chunk lists.

        Why: Different rewritten queries retrieve different slices of the corpus;
        RRF combines them without requiring scores to be calibrated across queries.

        Operation: Score each chunk id by ``sum 1/(k+rank)``, sort ids descending,
        merge duplicate ids via :meth:`_merge_chunk_pair`, return top ``cap`` chunks.
        """
        if not ranked_lists:
            return []
        by_id: dict[str, RetrievedChunk] = {}
        id_rank_lists: list[list[str]] = []
        for lst in ranked_lists:
            ids: list[str] = []
            for c in lst:
                ids.append(c.chunk_id)
                if c.chunk_id not in by_id:
                    by_id[c.chunk_id] = c
                else:
                    by_id[c.chunk_id] = self._merge_chunk_pair(by_id[c.chunk_id], c)
            id_rank_lists.append(ids)
        scores: dict[str, float] = {}
        for ranked in id_rank_lists:
            for rank, cid in enumerate(ranked, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        ordered = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return [by_id[i] for i in ordered[:cap] if i in by_id]

    async def _retrieve_merged_queries(
        self, queries: tuple[str, ...], filters: dict[str, Any] | None
    ) -> list[RetrievedChunk]:
        """Retrieve in parallel for each query string, then RRF-merge results.

        Why: Query rewriting emits multiple search phrases; we need one fused list
        for the planner and synthesizer while honoring optional ``metadata_filters``.

        Operation: ``asyncio.gather`` on :meth:`IRetriever.aretrieve`, sort each list
        by :meth:`_chunk_rank_key`, then :meth:`_rrf_merge_ranked_lists`.
        """
        if not queries:
            return []
        lists = await asyncio.gather(*[self._retriever.aretrieve(q, filters=filters) for q in queries])
        ranked = [sorted(lst, key=self._chunk_rank_key, reverse=True) for lst in lists]
        return self._rrf_merge_ranked_lists(ranked, cap=self._settings.retrieval_merge_cap)

    def _best_evidence_signal(self, chunks: list[RetrievedChunk]) -> float | None:
        """Best scalar similarity among chunks, used to decide 'answer vs insufficient evidence'.

        Why: The LLM should not hallucinate when retrieved passages are weak; this
        value is compared to ``settings.min_evidence_similarity``.

        Operation: Max of rerank/vector similarities if present, else max chunk ``score``.
        """
        best: float | None = None
        for c in chunks:
            md = c.metadata or {}
            for key in ("rerank_similarity", "vector_similarity"):
                v = md.get(key)
                if isinstance(v, (int, float)):
                    f = float(v)
                    best = f if best is None else max(best, f)
        if best is not None:
            return best
        sims = [float(c.score) for c in chunks if c.score is not None]
        return max(sims) if sims else None

    def _chunks_to_citations(self, chunks: list[RetrievedChunk]) -> list[SourceCitation]:
        """Map internal retrieval rows to API-safe :class:`~copilot_iitb.domain.models.SourceCitation` objects.

        Why: Clients need provenance (document/title/snippet) separate from the full
        chunk text used in the prompt.

        Operation: Truncate whitespace/newlines in snippet for readability.
        """
        cites: list[SourceCitation] = []
        for c in chunks:
            snippet = c.text.strip().replace("\n", " ")
            snippet = snippet[:320]
            cites.append(
                SourceCitation(
                    chunk_id=c.chunk_id,
                    document_id=str(c.metadata.get("document_id")) if c.metadata.get("document_id") else None,
                    title=str(c.metadata.get("title")) if c.metadata.get("title") else None,
                    snippet=snippet,
                    score=c.score,
                )
            )
        return cites

    def _is_pure_greeting(self, text: str) -> bool:
        """Detect small talk so we can skip retrieval and LLM cost when configured.

        Why: greetings should not hit the vector DB or spend tokens synthesizing
        from unrelated KB chunks.

        Operation: Match ``settings.greeting_regex`` under length and feature flags.
        """
        if not self._settings.enable_greeting_short_circuit:
            return False
        if len(text) > self._settings.greeting_max_message_chars:
            return False
        pattern = (self._settings.greeting_regex or "").strip()
        if not pattern:
            return False
        try:
            return re.match(pattern, text, flags=re.IGNORECASE | re.UNICODE) is not None
        except re.error:
            return False

    async def handle_chat(self, req: ChatRequest) -> ChatResponse:
        """End-to-end handler for ``POST /v1/chat``: one user message in, one assistant turn out.

        Why: Central place for guardrails, session continuity, retrieval quality,
        and answer formatting so HTTP and future transports (CLI, workers) can share it.

        Operation (branches):
            - Blocked by policy → store user+blocked assistant messages, return canned answer.
            - Resolve/create session, append user to short-term memory.
            - Optional greeting short-circuit → canned friendly reply.
            - Else rewrite query → retrieve (+ optional planner follow-up retrievals)
              → if no/weak evidence return safe answers with citations if any
              → else call synthesizer on top-k chunks → persist assistant turn +
              episodic summary → return :class:`~copilot_iitb.domain.models.ChatResponse`.
        """
        clean = self._input_guard.sanitize_user_message(req.message).value
        policy = evaluate_input_policy(self._settings, clean)
        # #region agent log
        agent_debug_log(
            "H1",
            "chat_service.py:handle_chat",
            "after_input_policy",
            {
                "allowed": policy.allowed,
                "block_reason": getattr(policy, "block_reason", None),
                "clean_len": len(clean or ""),
            },
        )
        # #endregion
        if not policy.allowed:
            # Still bind to a session so the client can continue the conversation id.
            answer = RAGAnswer(
                answer=self._settings.guardrail_blocked_response,
                citations=[],
                confidence=0.0,
                insufficient_evidence=True,
                follow_up_question=None,
            )
            if req.session_id:
                session = await self._sessions.aget(req.session_id)
                if session is None:
                    raise ValueError(f"Unknown session_id={req.session_id}")
            else:
                session = await self._sessions.acreate(user_id=req.user_id)
            await self._sessions.atouch(session.session_id)
            await self._memory.aappend_short_term(
                session.session_id, ChatMessage(role=ChatRole.USER, content=clean)
            )
            await self._memory.aappend_short_term(
                session.session_id, ChatMessage(role=ChatRole.ASSISTANT, content=answer.answer)
            )
            summary = f"User asked: {clean[:180]} | Assistant: {answer.answer[:180]}"
            await self._memory.aappend_episodic(
                session.session_id,
                EpisodicTurn(summary=summary, user_intent=None, created_at=datetime.utcnow()),
            )
            return ChatResponse(
                session_id=session.session_id,
                result=answer,
                retrieval_debug={"blocked": policy.block_reason},
            )

        if req.session_id:
            session = await self._sessions.aget(req.session_id)
            if session is None:
                raise ValueError(f"Unknown session_id={req.session_id}")
        else:
            session = await self._sessions.acreate(user_id=req.user_id)

        await self._sessions.atouch(session.session_id)
        # User turn is recorded before retrieval so loaders include the latest question.
        # #region agent log
        agent_debug_log(
            "H2",
            "chat_service.py:handle_chat",
            "session_opened",
            {
                "used_existing_session": req.session_id is not None,
                "session_id_len": len(session.session_id or ""),
            },
        )
        # #endregion
        await self._memory.aappend_short_term(session.session_id, ChatMessage(role=ChatRole.USER, content=clean))

        short_term = await self._memory.aload_short_term(session.session_id, self._settings.short_term_max_messages)
        recent = short_term[-self._settings.llm_context_recent_turns :]
        recent_dialogue = self._recent_dialogue(recent)

        episodic = await self._memory.aload_episodic(session.session_id, limit=12)
        episodic_summary = self._episodic_summary(episodic)

        user_id = req.user_id
        memory_hints = ""
        if user_id:
            hints = await self._memory.aload_long_term_hints(user_id, clean, limit=4)
            if hints:
                memory_hints = self._settings.memory_hints_prefix + "\n- ".join(hints)

        if self._is_pure_greeting(clean):
            # Avoids vector search + synthesis; see `_is_pure_greeting`.
            # #region agent log
            agent_debug_log(
                "H5",
                "chat_service.py:handle_chat",
                "greeting_short_circuit",
                {"greeting_enabled": self._settings.enable_greeting_short_circuit},
            )
            # #endregion
            answer = RAGAnswer(
                answer=self._settings.greeting_response,
                citations=[],
                confidence=0.95,
                insufficient_evidence=False,
                follow_up_question=None,
            )
            await self._memory.aappend_short_term(
                session.session_id, ChatMessage(role=ChatRole.ASSISTANT, content=answer.answer)
            )
            summary = f"User asked: {clean[:180]} | Assistant: {answer.answer[:180]}"
            await self._memory.aappend_episodic(
                session.session_id,
                EpisodicTurn(summary=summary, user_intent="greeting", created_at=datetime.utcnow()),
            )
            return ChatResponse(
                session_id=session.session_id,
                result=answer,
                retrieval_debug={"short_circuit": "greeting"},
            )

        # Produces one or more search phrases; improves recall on ambiguous questions.
        rewrite = await self._query_rewriter.arewrite(
            user_query=clean,
            recent_dialogue=recent_dialogue,
            episodic_summary=episodic_summary,
        )
        base_queries = rewrite.queries if rewrite.queries else (clean,)
        tried_queries: list[str] = list(base_queries)

        chunks = await self._retrieve_merged_queries(base_queries, req.metadata_filters)
        planner_trace: list[dict[str, Any]] = []

        max_steps = self._settings.reasoning_max_iterations
        if max_steps > 0:
            # Optional iterative retrieval: planner may request extra `aretrieve` queries.
            for step in range(max_steps):
                excerpts = tuple(c.text[:520].strip() for c in chunks[:12] if c.text.strip())
                decision = await self._retrieval_planner.aplan(
                    user_query=clean,
                    memory_hints=memory_hints,
                    tried_queries=tuple(tried_queries),
                    evidence_excerpts=excerpts,
                )
                planner_trace.append(
                    {
                        "step": step,
                        "chain_of_thought": decision.chain_of_thought[:900],
                        "sufficient": decision.sufficient,
                        "follow_up_search_queries": list(decision.follow_up_search_queries),
                    }
                )
                if decision.sufficient or not decision.follow_up_search_queries:
                    break
                new_queries = tuple(decision.follow_up_search_queries)
                tried_queries.extend(new_queries)
                extra_lists = await asyncio.gather(
                    *[self._retriever.aretrieve(q, filters=req.metadata_filters) for q in new_queries]
                )
                ranked_prev = sorted(chunks, key=self._chunk_rank_key, reverse=True)
                ranked_new = [sorted(lst, key=self._chunk_rank_key, reverse=True) for lst in extra_lists]
                chunks = self._rrf_merge_ranked_lists(
                    [ranked_prev, *ranked_new],
                    cap=self._settings.retrieval_merge_cap,
                )

        best_sig = self._best_evidence_signal(chunks)
        # #region agent log
        branch = (
            "no_chunks"
            if not chunks
            else (
                "low_evidence"
                if best_sig is not None and best_sig < self._settings.min_evidence_similarity
                else "synthesize"
            )
        )
        agent_debug_log(
            "H3",
            "chat_service.py:handle_chat",
            "post_retrieval",
            {
                "num_chunks": len(chunks),
                "best_sig": best_sig,
                "min_evidence_similarity": float(self._settings.min_evidence_similarity),
                "branch": branch,
                "reasoning_steps": len(planner_trace),
            },
        )
        # #endregion

        if not chunks:
            # Nothing in the KB matched filters/queries — surface explicit uncertainty.
            answer = RAGAnswer(
                answer=self._settings.no_context_answer,
                citations=[],
                confidence=0.0,
                insufficient_evidence=True,
                follow_up_question=self._settings.low_evidence_follow_up,
            )
        elif best_sig is not None and best_sig < self._settings.min_evidence_similarity:
            # Chunks exist but are likely irrelevant; return low-evidence template + snippets.
            answer = RAGAnswer(
                answer=self._settings.low_evidence_answer,
                citations=self._chunks_to_citations(chunks),
                confidence=float(best_sig),
                insufficient_evidence=True,
                follow_up_question=self._settings.low_evidence_follow_up,
            )
        else:
            ordered = sorted(chunks, key=self._chunk_rank_key, reverse=True)
            top = ordered[: self._settings.retrieval_top_k]
            labels = [f"E{i}" for i in range(1, len(top) + 1)]
            blocks = [c.text for c in top]
            # Grounded JSON/text answer from the LLM using only `blocks` as evidence.
            answer = await self._synthesizer.asynthesize(
                user_query=clean,
                evidence_blocks=blocks,
                evidence_labels=labels,
                recent_dialogue=recent_dialogue,
                memory_hints=memory_hints,
            )
            if not answer.citations:
                answer.citations = self._chunks_to_citations(top)

            if best_sig is not None and answer.confidence < float(best_sig):
                answer.confidence = float(best_sig)

        await self._memory.aappend_short_term(
            session.session_id,
            ChatMessage(role=ChatRole.ASSISTANT, content=answer.answer),
        )
        summary = f"User asked: {clean[:180]} | Assistant: {answer.answer[:180]}"
        await self._memory.aappend_episodic(
            session.session_id,
            EpisodicTurn(summary=summary, user_intent=None, created_at=datetime.utcnow()),
        )

        debug = {
            "search_queries": list(base_queries),
            "rewrite_notes": rewrite.raw_model_notes,
            "tried_queries": tried_queries,
            "num_chunks": len(chunks),
            "best_evidence_signal": best_sig,
            "reasoning_trace": planner_trace,
        }
        return ChatResponse(session_id=session.session_id, result=answer, retrieval_debug=debug)
