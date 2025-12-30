"""MCP-style tool discovery and invocation endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError
import structlog

from ...agents.orchestrator import OrchestratorAgent
from ...db.neo4j import Neo4jClient
from ...protocols.mcp import MCPToolNotFoundError, MCPToolRegistry
from ...rate_limit import RateLimiter

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])


class ToolDescriptor(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolListResponse(BaseModel):
    tools: list[ToolDescriptor]
    meta: dict[str, Any]


class ToolCallRequest(BaseModel):
    tool: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallResponse(BaseModel):
    tool: str
    result: dict[str, Any]
    meta: dict[str, Any]


def get_orchestrator(request: Request) -> OrchestratorAgent:
    """Get orchestrator from app state."""
    return request.app.state.orchestrator


def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter from app state."""
    return request.app.state.rate_limiter


def get_neo4j(request: Request) -> Neo4jClient | None:
    """Get Neo4j client from app state."""
    return getattr(request.app.state, "neo4j", None)


def _meta() -> dict[str, Any]:
    return {
        "requestId": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


@router.get("/tools", response_model=ToolListResponse)
async def list_tools(
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    neo4j: Neo4jClient | None = Depends(get_neo4j),
) -> ToolListResponse:
    """List available MCP tools."""
    registry = MCPToolRegistry(orchestrator=orchestrator, neo4j=neo4j)
    return ToolListResponse(
        tools=[ToolDescriptor(**tool) for tool in registry.list_tools()],
        meta=_meta(),
    )


@router.post("/call", response_model=ToolCallResponse)
async def call_tool(
    request_body: ToolCallRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    limiter: RateLimiter = Depends(get_rate_limiter),
    neo4j: Neo4jClient | None = Depends(get_neo4j),
) -> ToolCallResponse:
    """Invoke a tool by name with arguments."""
    tenant_id = request_body.arguments.get("tenant_id")
    if not tenant_id or not str(tenant_id).strip():
        raise HTTPException(status_code=400, detail="tenant_id is required")

    if not await limiter.allow(str(tenant_id)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    registry = MCPToolRegistry(orchestrator=orchestrator, neo4j=neo4j)

    try:
        result = await registry.call_tool(request_body.tool, request_body.arguments)
    except MCPToolNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Tool not found: {exc.args[0]}") from exc
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safeguard
        logger.exception("mcp_tool_call_failed", tool=request_body.tool, error=str(exc))
        raise HTTPException(status_code=500, detail="Tool invocation failed") from exc

    return ToolCallResponse(tool=request_body.tool, result=result, meta=_meta())
