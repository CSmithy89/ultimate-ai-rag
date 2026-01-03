"""Universal AG-UI protocol endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError
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
from ...validation import TENANT_ID_PATTERN

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/ag-ui", tags=["ag-ui"])


class AGUIRequest(BaseModel):
    messages: list[CopilotMessage] = Field(default_factory=list)
    tenant_id: str = Field(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN)
    session_id: str | None = Field(None, max_length=255)
    actions: list[dict[str, Any]] = Field(default_factory=list)


def get_orchestrator(request: Request) -> OrchestratorAgent:
    return request.app.state.orchestrator


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


def get_hitl_manager(request: Request | None):
    if request is None:
        return None
    return getattr(request.app.state, "hitl_manager", None)


async def ag_ui_handler(
    request: AGUIRequest,
    orchestrator: OrchestratorAgent,
    limiter: RateLimiter,
    http_request: Request | None = None,
) -> StreamingResponse:
    """Stream AG-UI events for generic clients."""
    if not await limiter.allow(request.tenant_id):
        raise rate_limit_exceeded()

    try:
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
    except ValidationError as exc:
        logger.warning("ag_ui_request_invalid", error=str(exc))
        raise HTTPException(status_code=422, detail="Invalid AG-UI request") from exc

    hitl_manager = get_hitl_manager(http_request)
    if hitl_manager is None:
        bridge = AGUIBridge(orchestrator)
    else:
        bridge = AGUIBridge(orchestrator, hitl_manager=hitl_manager)

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


@router.post("")
async def ag_ui_endpoint(
    request: AGUIRequest,
    http_request: Request,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> StreamingResponse:
    return await ag_ui_handler(
        request=request,
        orchestrator=orchestrator,
        limiter=limiter,
        http_request=http_request,
    )
