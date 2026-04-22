from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from copilot_iitb.api.container import AppContainer
from copilot_iitb.api.rate_limit import InMemoryRateLimiter, client_key

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str | None = Field(default=None, description="Optional stable user id for long-term memory.")


@router.post("")
async def create_session(request: Request, body: CreateSessionRequest | None = None) -> dict[str, str]:
    limiter: InMemoryRateLimiter = request.app.state.limiter
    limiter.check(client_key(request))

    container: AppContainer = request.app.state.container
    payload = body or CreateSessionRequest()
    rec = await container.sessions.acreate(user_id=payload.user_id)
    return {"session_id": rec.session_id}
