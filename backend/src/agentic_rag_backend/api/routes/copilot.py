"""CopilotKit AG-UI protocol endpoint.

Story 21-C3: Wire MCP Client to CopilotRuntime
"""

import re
from typing import Any, List, Optional


from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
import structlog

from ...agents.orchestrator import OrchestratorAgent
from ...api.utils import rate_limit_exceeded
from ...mcp_client import (
    MCPClientFactory,
    discover_all_tools,
    get_mcp_factory,
    parse_namespaced_tool,
)
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
    http_request: Request,
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
        raise rate_limit_exceeded()

    bridge = AGUIBridge(orchestrator, hitl_manager=get_hitl_manager(http_request))

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
# MCP TOOLS ENDPOINTS - Story 21-C3
# ============================================


class ToolDefinition(BaseModel):
    """Tool definition returned by discovery endpoint."""

    name: str = Field(..., description="Tool name (namespaced for external tools)")
    description: str = Field(default="", description="Tool description")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, alias="inputSchema", description="JSON Schema for inputs"
    )
    source: str = Field(default="internal", description="Tool source: internal or external")
    server_name: Optional[str] = Field(
        default=None, alias="serverName", description="MCP server name for external tools"
    )

    model_config = {"populate_by_name": True}


class ToolsResponse(BaseModel):
    """Response from tools discovery endpoint."""

    tools: List[ToolDefinition] = Field(default_factory=list, description="Available tools")
    mcp_enabled: bool = Field(default=False, alias="mcpEnabled", description="Whether MCP is enabled")
    server_count: int = Field(default=0, alias="serverCount", description="Number of MCP servers")

    model_config = {"populate_by_name": True}


# Valid tool name pattern: server_name:tool_name (alphanumeric, underscores, hyphens)
TOOL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+:[a-zA-Z0-9_-]+$')


class ToolCallRequest(BaseModel):
    """Request to call an external MCP tool."""

    tool_name: str = Field(..., alias="toolName", description="Tool name (namespaced)")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

    model_config = {"populate_by_name": True}

    @field_validator('tool_name')
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Validate tool_name matches expected namespaced format."""
        if not TOOL_NAME_PATTERN.match(v):
            raise ValueError(
                'tool_name must be in format "server_name:tool_name" '
                'with alphanumeric characters, underscores, and hyphens only'
            )
        return v


class ToolCallResponse(BaseModel):
    """Response from external tool call."""

    result: Any = Field(..., description="Tool execution result")
    server_name: Optional[str] = Field(
        default=None, alias="serverName", description="MCP server that handled the call"
    )

    model_config = {"populate_by_name": True}


@router.get("/tools", response_model=ToolsResponse)
async def list_tools(
    request: Request,
    mcp_factory: Optional[MCPClientFactory] = Depends(get_mcp_factory),
) -> ToolsResponse:
    """
    List all available tools from internal agents and external MCP servers.

    Story 21-C3: Wire MCP Client to CopilotRuntime

    Returns:
        ToolsResponse with unified tool list
    """
    # Discover all tools (internal + external MCP)
    # Note: Internal tools from the orchestrator would be added here
    # For now, we only discover external MCP tools
    all_tools = await discover_all_tools(factory=mcp_factory, internal_tools=[])

    tools = [
        ToolDefinition(
            name=tool.get("name", ""),
            description=tool.get("description", ""),
            inputSchema=tool.get("inputSchema", {}),
            source=tool.get("source", "internal"),
            serverName=tool.get("serverName"),
        )
        for tool in all_tools.values()
    ]

    return ToolsResponse(
        tools=tools,
        mcpEnabled=mcp_factory is not None and mcp_factory.is_enabled,
        serverCount=len(mcp_factory.server_names) if mcp_factory else 0,
    )


@router.post("/tools/call", response_model=ToolCallResponse)
async def call_external_tool(
    request_body: ToolCallRequest,
    request: Request,
    mcp_factory: Optional[MCPClientFactory] = Depends(get_mcp_factory),
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> ToolCallResponse:
    """
    Call an external MCP tool by namespaced name.

    Story 21-C3: Wire MCP Client to CopilotRuntime

    Args:
        request_body: Tool name and arguments

    Returns:
        ToolCallResponse with execution result
    """
    if not mcp_factory or not mcp_factory.is_enabled:
        raise HTTPException(
            status_code=503,
            detail="MCP client is not enabled"
        )

    # Parse the namespaced tool name
    server_name, original_name = parse_namespaced_tool(request_body.tool_name)

    if not server_name:
        raise HTTPException(
            status_code=400,
            detail=f"Tool '{request_body.tool_name}' is not an external tool (missing server namespace)"
        )

    # Security fix: Validate server exists before attempting call (return 404 not 500)
    if server_name not in mcp_factory.server_names:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server '{server_name}' is not configured"
        )

    logger.info(
        "calling_external_tool",
        tool_name=request_body.tool_name,
        server_name=server_name,
        original_name=original_name,
        tenant_id=tenant_id,
    )

    try:
        result = await mcp_factory.call_tool(
            server_name=server_name,
            tool_name=original_name,
            arguments=request_body.arguments,
        )
        return ToolCallResponse(result=result, serverName=server_name)
    except Exception as e:
        # Log full error for debugging but don't expose to client
        logger.exception(
            "external_tool_call_failed",
            tool_name=request_body.tool_name,
            server_name=server_name,
            error=str(e),
        )
        # Security fix: Don't leak internal error details to client
        raise HTTPException(
            status_code=500,
            detail="An error occurred during external tool execution"
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


class HITLCheckpointResponse(BaseModel):
    """Response payload for HITL checkpoint queries."""

    checkpoint_id: str
    status: str
    query: str
    tenant_id: Optional[str] = None
    sources: List[dict[str, Any]]
    approved_source_ids: List[str]
    rejected_source_ids: List[str]


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

        checkpoint = await hitl_manager.receive_validation_response(
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


@router.get("/hitl/checkpoints/{checkpoint_id}", response_model=HITLCheckpointResponse)
async def get_hitl_checkpoint(
    checkpoint_id: str,
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
) -> HITLCheckpointResponse:
    """Get a HITL checkpoint by ID."""
    hitl_manager = get_hitl_manager(request)
    if hitl_manager is None:
        raise HTTPException(status_code=503, detail="HITL manager not configured")

    record = await hitl_manager.fetch_checkpoint(checkpoint_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    record_tenant = record.get("tenant_id")
    if record_tenant and tenant_id and record_tenant != tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this checkpoint")

    return HITLCheckpointResponse(**record)


@router.get("/hitl/checkpoints", response_model=List[HITLCheckpointResponse])
async def list_hitl_checkpoints(
    request: Request,
    tenant_id: Optional[str] = Depends(get_tenant_id_from_header),
    limit: int = Query(20, ge=1, le=100),
) -> List[HITLCheckpointResponse]:
    """List HITL checkpoints for a tenant."""
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")
    hitl_manager = get_hitl_manager(request)
    if hitl_manager is None:
        raise HTTPException(status_code=503, detail="HITL manager not configured")

    records = await hitl_manager.list_checkpoints(tenant_id, limit=limit)
    return [HITLCheckpointResponse(**record) for record in records]
