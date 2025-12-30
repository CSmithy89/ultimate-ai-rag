"""MCP-style tool discovery and invocation endpoints."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from ...agents.orchestrator import OrchestratorAgent
from ...db.neo4j import Neo4jClient
from ...api.utils import build_meta
from ...protocols.mcp import MCPToolNotFoundError, MCPToolRegistry
from ...rate_limit import RateLimiter

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


def get_mcp_registry(
    request: Request,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    neo4j: Neo4jClient | None = Depends(get_neo4j),
) -> MCPToolRegistry:
    """Get or initialize the MCP tool registry."""
    registry = getattr(request.app.state, "mcp_registry", None)
    if registry is None:
        settings = request.app.state.settings
        registry = MCPToolRegistry(
            orchestrator=orchestrator,
            neo4j=neo4j,
            timeout_seconds=settings.mcp_tool_timeout_seconds,
            tool_timeouts=settings.mcp_tool_timeout_overrides,
        )
        request.app.state.mcp_registry = registry
    return registry


@router.get("/tools", response_model=ToolListResponse)
async def list_tools(
    registry: MCPToolRegistry = Depends(get_mcp_registry),
) -> ToolListResponse:
    """List available MCP tools."""
    return ToolListResponse(
        tools=[ToolDescriptor(**tool) for tool in registry.list_tools()],
        meta=build_meta(),
    )


@router.post("/call", response_model=ToolCallResponse)
async def call_tool(
    request_body: ToolCallRequest,
    registry: MCPToolRegistry = Depends(get_mcp_registry),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> ToolCallResponse:
    """Invoke a tool by name with arguments."""
    tenant_id = request_body.arguments.get("tenant_id")
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise HTTPException(status_code=400, detail="tenant_id is required")

    if not await limiter.allow(str(tenant_id)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        result = await registry.call_tool(request_body.tool, request_body.arguments)
    except MCPToolNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Tool not found: {exc.args[0]}") from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Tool execution timed out") from exc
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ToolCallResponse(tool=request_body.tool, result=result, meta=build_meta())
