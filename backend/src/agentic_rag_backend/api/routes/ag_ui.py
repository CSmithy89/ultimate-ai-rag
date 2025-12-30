"""Universal AG-UI protocol endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...agents.orchestrator import OrchestratorAgent
from ...models.copilot import CopilotConfig, CopilotMessage, CopilotRequest
from ...protocols.ag_ui_bridge import AGUIBridge
from ...rate_limit import RateLimiter

router = APIRouter(prefix="/ag-ui", tags=["ag-ui"])


class AGUIRequest(BaseModel):
    messages: list[CopilotMessage] = Field(default_factory=list)
    tenant_id: str = Field(..., min_length=1, max_length=255)
    session_id: str | None = Field(None, max_length=255)
    actions: list[dict[str, Any]] = Field(default_factory=list)


def get_orchestrator(request: Request) -> OrchestratorAgent:
    return request.app.state.orchestrator


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


@router.post("")
async def ag_ui_handler(
    request: AGUIRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> StreamingResponse:
    """Stream AG-UI events for generic clients."""
    if not await limiter.allow(request.tenant_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    copilot_request = CopilotRequest(
        messages=request.messages,
        config=CopilotConfig(
            configurable={
                "tenant_id": request.tenant_id,
                "session_id": request.session_id,
            }
        ),
        actions=request.actions,
    )

    bridge = AGUIBridge(orchestrator)

    async def event_generator():
        async for event in bridge.process_request(copilot_request):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
