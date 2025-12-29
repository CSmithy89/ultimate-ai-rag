"""CopilotKit AG-UI protocol endpoint."""

import re
from typing import List, Optional


from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
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


# ============================================
# HITL VALIDATION ENDPOINT - Story 6-4
# ============================================

# UUID4 regex pattern for validation
UUID4_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE
)


class ValidationResponseRequest(BaseModel):
    """Request body for HITL validation response."""

    checkpoint_id: str = Field(..., description="ID of the checkpoint being responded to (UUID format)")
    approved_source_ids: List[str] = Field(
        default_factory=list,
        description="List of approved source IDs"
    )

    # Issue 8 Fix: Add UUID validation to checkpoint_id
    @field_validator('checkpoint_id')
    @classmethod
    def validate_checkpoint_id(cls, v: str) -> str:
        """Validate that checkpoint_id is a valid UUID4."""
        if not UUID4_PATTERN.match(v):
            raise ValueError('checkpoint_id must be a valid UUID4')
        return v


class ValidationResponseResult(BaseModel):
    """Response for HITL validation endpoint."""

    checkpoint_id: str
    status: str
    approved_count: int
    rejected_count: int


def get_hitl_manager(request: Request):
    """Get HITL manager from app state."""
    return getattr(request.app.state, "hitl_manager", None)


def get_tenant_id_from_header(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
) -> Optional[str]:
    """Extract tenant_id from request header."""
    return x_tenant_id


@router.post("/validation-response", response_model=ValidationResponseResult)
async def receive_validation_response(
    request_body: ValidationResponseRequest,
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
) -> ValidationResponseResult:
    """
    Receive Human-in-the-Loop validation response from frontend.

    Story 6-4: Human-in-the-Loop Source Validation

    This endpoint receives the user's approval/rejection decisions
    and signals the waiting agent to continue with approved sources.
    """
    hitl_manager = get_hitl_manager(request)

    if hitl_manager is None:
        # If no HITL manager is configured, return a mock response
        # This allows the endpoint to work for testing even without full setup
        logger.warning(
            "hitl_manager_not_configured",
            checkpoint_id=request_body.checkpoint_id,
        )
        return ValidationResponseResult(
            checkpoint_id=request_body.checkpoint_id,
            status="approved" if request_body.approved_source_ids else "rejected",
            approved_count=len(request_body.approved_source_ids),
            rejected_count=0,
        )

    try:
        # Issue 2 Fix: Verify tenant authorization
        # Get the checkpoint first to check tenant ownership
        checkpoint = hitl_manager.get_checkpoint(request_body.checkpoint_id)
        if checkpoint is None:
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint {request_body.checkpoint_id} not found"
            )

        # Verify tenant_id matches if checkpoint has tenant_id
        checkpoint_tenant = getattr(checkpoint, 'tenant_id', None)
        if checkpoint_tenant and tenant_id and checkpoint_tenant != tenant_id:
            logger.warning(
                "hitl_tenant_mismatch",
                checkpoint_id=request_body.checkpoint_id,
                checkpoint_tenant=checkpoint_tenant,
                request_tenant=tenant_id,
            )
            raise HTTPException(
                status_code=403,
                detail="Not authorized to respond to this checkpoint"
            )

        checkpoint = hitl_manager.receive_validation_response(
            checkpoint_id=request_body.checkpoint_id,
            approved_source_ids=request_body.approved_source_ids,
        )

        return ValidationResponseResult(
            checkpoint_id=checkpoint.checkpoint_id,
            status=checkpoint.status.value,
            approved_count=len(checkpoint.approved_source_ids),
            rejected_count=len(checkpoint.rejected_source_ids),
        )

    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint {request_body.checkpoint_id} not found"
        )
