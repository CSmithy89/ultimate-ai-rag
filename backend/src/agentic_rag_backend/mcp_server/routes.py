"""MCP Server HTTP routes for FastAPI.

Provides HTTP and SSE endpoints for MCP protocol access.

Story 14-1: Expose RAG Engine via MCP Server
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .types import MCPRequest, MCPResponse, MCPError, MCPErrorCode
from .server import MCPServer
from .auth import MCPAuthContext

router = APIRouter(prefix="/mcp/v1", tags=["mcp-server"])


class MCPToolCallRequest(BaseModel):
    """Request model for direct tool calls."""

    tool: str = Field(..., min_length=1, description="Tool name")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class MCPToolCallResponse(BaseModel):
    """Response model for tool calls."""

    tool: str
    result: dict[str, Any]
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def get_mcp_server(request: Request) -> MCPServer:
    """Get MCP server from app state."""
    server = getattr(request.app.state, "mcp_server", None)
    if server is None:
        raise HTTPException(
            status_code=503,
            detail="MCP server not initialized",
        )
    return server


async def get_auth_context(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[MCPAuthContext]:
    """Extract authentication context from request headers."""
    server = get_mcp_server(request)

    if not server._authenticator:
        return None

    # Build credentials from headers
    credentials: dict[str, str] = {}
    if authorization:
        credentials["authorization"] = authorization
    if x_api_key:
        credentials["api_key"] = x_api_key

    if not credentials:
        return None

    try:
        return await server._authenticator.authenticate(credentials)
    except MCPError as e:
        raise HTTPException(
            status_code=401 if e.code == MCPErrorCode.AUTHENTICATION_FAILED else 403,
            detail=e.message,
        )


@router.get("/tools")
async def list_tools(
    server: MCPServer = Depends(get_mcp_server),
    auth_context: Optional[MCPAuthContext] = Depends(get_auth_context),
) -> dict[str, Any]:
    """List available MCP tools.

    Returns all tools the authenticated user has access to.
    """
    tools = server.registry.list_tools(auth_context=auth_context)
    return {
        "tools": tools,
        "count": len(tools),
    }


@router.post("/tools/call", response_model=MCPToolCallResponse)
async def call_tool(
    request_body: MCPToolCallRequest,
    server: MCPServer = Depends(get_mcp_server),
    auth_context: Optional[MCPAuthContext] = Depends(get_auth_context),
) -> MCPToolCallResponse:
    """Call an MCP tool directly.

    This is a simplified endpoint for direct tool invocation.
    """
    try:
        result = await server.registry.call_tool(
            name=request_body.tool,
            arguments=request_body.arguments,
            auth_context=auth_context,
        )
        return MCPToolCallResponse(
            tool=request_body.tool,
            result=result.to_dict(),
            is_error=result.is_error,
            metadata=result.metadata,
        )
    except MCPError as e:
        status_code = 400
        if e.code == MCPErrorCode.TOOL_NOT_FOUND:
            status_code = 404
        elif e.code == MCPErrorCode.AUTHENTICATION_REQUIRED:
            status_code = 401
        elif e.code == MCPErrorCode.RATE_LIMIT_EXCEEDED:
            status_code = 429
        elif e.code == MCPErrorCode.TIMEOUT:
            status_code = 504
        elif e.code in (MCPErrorCode.INTERNAL_ERROR, MCPErrorCode.TOOL_EXECUTION_ERROR):
            status_code = 500

        raise HTTPException(status_code=status_code, detail=e.to_dict())


@router.post("/jsonrpc")
async def jsonrpc_endpoint(
    request: Request,
    server: MCPServer = Depends(get_mcp_server),
    auth_context: Optional[MCPAuthContext] = Depends(get_auth_context),
) -> dict[str, Any]:
    """JSON-RPC 2.0 endpoint for MCP protocol.

    Handles standard MCP requests via HTTP POST.
    """
    try:
        body = await request.json()
        mcp_request = MCPRequest(**body)
    except Exception as e:
        return MCPResponse.failure(
            None,
            MCPError(
                code=MCPErrorCode.INVALID_REQUEST,
                message=f"Invalid request: {e}",
            ),
        ).model_dump()

    response = await server.handle_request(mcp_request, auth_context)
    return response.model_dump()


@router.get("/sse")
async def sse_stream(
    request: Request,
    server: MCPServer = Depends(get_mcp_server),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> StreamingResponse:
    """Server-Sent Events endpoint for streaming MCP responses.

    Opens a persistent connection for receiving MCP responses.
    Send requests via the /sse/send endpoint.

    Note: Uses request.is_disconnected() to detect client disconnects
    and properly clean up resources.
    """
    import asyncio

    async def event_generator():
        # Send initial connection event
        yield "event: connected\ndata: {}\n\n"

        # Keep connection alive with periodic heartbeats
        # Check for client disconnect to avoid resource leaks
        try:
            while True:
                # Check if client has disconnected
                if await request.is_disconnected():
                    break

                await asyncio.sleep(30)
                yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            # Server is shutting down or connection was cancelled
            pass
        finally:
            # Always try to send disconnected event if possible
            try:
                yield "event: disconnected\ndata: {}\n\n"
            except Exception:
                # Client may have already disconnected
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/sse/send")
async def sse_send(
    request: Request,
    server: MCPServer = Depends(get_mcp_server),
    auth_context: Optional[MCPAuthContext] = Depends(get_auth_context),
) -> StreamingResponse:
    """Send an MCP request and receive response via SSE.

    This endpoint accepts a single MCP request and returns
    the response as an SSE stream.
    """
    body = await request.body()

    credentials: dict[str, str] = {}
    auth_header = request.headers.get("Authorization")
    if auth_header:
        credentials["authorization"] = auth_header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        credentials["api_key"] = api_key

    async def response_generator():
        async for chunk in server.handle_sse_request(body, credentials):
            yield chunk

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/info")
async def server_info(
    server: MCPServer = Depends(get_mcp_server),
) -> dict[str, Any]:
    """Get MCP server information and capabilities."""
    return {
        "name": server.name,
        "version": server.version,
        "protocol_version": "2024-11-05",
        "capabilities": server.get_capabilities().model_dump(),
        "categories": server.registry.get_categories(),
        "tools_by_category": server.registry.get_tools_by_category(),
    }


@router.get("/health")
async def health_check(
    server: MCPServer = Depends(get_mcp_server),
) -> dict[str, Any]:
    """Health check endpoint for the MCP server."""
    return {
        "status": "healthy",
        "server": server.name,
        "version": server.version,
    }
