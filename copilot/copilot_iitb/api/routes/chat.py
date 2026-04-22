"""HTTP surface for conversational RAG (``POST /v1/chat``).

Why this module exists: FastAPI should stay thin—validate transport concerns
(rate limits, HTTP errors) and delegate all chat semantics to
:class:`copilot_iitb.application.chat_service.ChatService`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from copilot_iitb.agent_debug_log import agent_debug_log
from copilot_iitb.api.container import AppContainer
from copilot_iitb.api.rate_limit import InMemoryRateLimiter, client_key
from copilot_iitb.domain.models import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Run one user turn: session + memory + retrieval + grounded answer.

    Why: Clients need a single JSON endpoint that returns ``session_id`` (for
    follow-up turns), the assistant ``result``, and optional ``retrieval_debug``.

    Operation:
        1. Enforce per-client rate limits (abuse protection).
        2. Resolve the wired :class:`~copilot_iitb.api.container.AppContainer` and
           call :meth:`copilot_iitb.application.chat_service.ChatService.handle_chat`.
        3. Map domain ``ValueError`` (e.g. unknown ``session_id``) to HTTP 400.

    Args:
        request: ASGI request carrying ``app.state`` (limiter + DI container).
        body: Validated :class:`~copilot_iitb.domain.models.ChatRequest`.

    Returns:
        :class:`~copilot_iitb.domain.models.ChatResponse` from the service layer.

    Raises:
        HTTPException: 400 when the service rejects the request with ``ValueError``.
    """
    limiter: InMemoryRateLimiter = request.app.state.limiter
    limiter.check(client_key(request))

    container: AppContainer = request.app.state.container
    # #region agent log
    agent_debug_log(
        "H0",
        "chat.py:chat",
        "chat_request",
        {
            "has_session_id": body.session_id is not None,
            "has_user_id": body.user_id is not None,
            "message_len": len(body.message or ""),
            "has_metadata_filters": body.metadata_filters is not None,
        },
    )
    # #endregion
    try:
        # All RAG orchestration (sanitize, session, retrieve, synthesize, persist).
        resp = await container.chat_service.handle_chat(body)
        # #region agent log
        agent_debug_log(
            "H0",
            "chat.py:chat",
            "chat_ok",
            {
                "session_id_len": len(resp.session_id or ""),
                "insufficient_evidence": resp.result.insufficient_evidence,
                "citations_n": len(resp.result.citations or []),
            },
        )
        # #endregion
        return resp
    except ValueError as exc:
        # #region agent log
        agent_debug_log(
            "H2",
            "chat.py:chat",
            "chat_value_error",
            {"detail_kind": "value_error", "detail_len": len(str(exc))},
        )
        # #endregion
        raise HTTPException(status_code=400, detail=str(exc)) from exc
