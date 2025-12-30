"""Universal AG-UI protocol endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog

from ...agents.orchestrator import OrchestratorAgent
from ...models.copilot import (
    CopilotConfig,
    CopilotMessage,
    CopilotRequest,
    RunFinishedEvent,
    TextDeltaEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ...protocols.ag_ui_bridge import AGUIBridge
from ...api.utils import rate_limit_exceeded
from ...rate_limit import RateLimiter

logger = structlog.get_logger(__name__)

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
        raise rate_limit_exceeded()

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
        try:
            async for event in bridge.process_request(copilot_request):
                yield f"data: {event.model_dump_json()}\n\n"
        except Exception as exc:
            logger.exception("ag_ui_stream_failed", error=str(exc))
            # Fallback if bridge fails before emitting error events.
            error_event = TextDeltaEvent(
                content="An error occurred while processing your request."
            )
            yield f"data: {TextMessageStartEvent().model_dump_json()}\n\n"
            yield f"data: {error_event.model_dump_json()}\n\n"
            yield f"data: {TextMessageEndEvent().model_dump_json()}\n\n"
            yield f"data: {RunFinishedEvent().model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
