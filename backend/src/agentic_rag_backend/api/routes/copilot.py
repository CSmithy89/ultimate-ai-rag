"""CopilotKit AG-UI protocol endpoint."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import structlog

from ...agents.orchestrator import OrchestratorAgent
from ...models.copilot import CopilotRequest
from ...protocols.ag_ui_bridge import AGUIBridge
from ...rate_limit import RateLimiter

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/copilot", tags=["copilot"])


def get_orchestrator(request: Request) -> OrchestratorAgent:
    """Get orchestrator from app state."""
    return request.app.state.orchestrator


def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter from app state."""
    return request.app.state.rate_limiter


@router.post("")
async def copilot_handler(
    request: CopilotRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> StreamingResponse:
    """
    Handle AG-UI protocol requests from CopilotKit.

    Returns SSE stream with AG-UI events:
    - text_delta: Streaming text responses
    - tool_call: Agent tool invocations
    - state_snapshot: Agent state updates
    - action_request: Frontend action requests
    """
    # Extract tenant_id for rate limiting
    tenant_id = "anonymous"
    if request.config and request.config.configurable:
        tenant_id = request.config.configurable.get("tenant_id", "anonymous")

    # Check rate limit
    if not await limiter.allow(tenant_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    bridge = AGUIBridge(orchestrator)

    async def event_generator():
        async for event in bridge.process_request(request):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
