"""MCP Server implementation.

Provides the main MCP server class with support for
HTTP/SSE and stdio transports.

Story 14-1: Expose RAG Engine via MCP Server
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, AsyncIterator, Optional, Callable, Awaitable

import structlog

from .types import (
    MCPCapabilities,
    MCPError,
    MCPErrorCode,
    MCPInitializeResult,
    MCPRequest,
    MCPResponse,
    MCPServerInfo,
)
from .registry import MCPServerRegistry
from .auth import MCPAuthContext, MCPAuthenticator, MCPRateLimiter

logger = structlog.get_logger(__name__)

# MCP Protocol version
MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPServer:
    """MCP Server with HTTP/SSE and stdio transport support.

    Implements the Model Context Protocol for exposing
    RAG engine capabilities to LLM agents and IDEs.
    """

    def __init__(
        self,
        name: str = "agentic-rag-mcp",
        version: str = "1.0.0",
        registry: Optional[MCPServerRegistry] = None,
        authenticator: Optional[MCPAuthenticator] = None,
        rate_limiter: Optional[MCPRateLimiter] = None,
    ) -> None:
        """Initialize MCP server.

        Args:
            name: Server name
            version: Server version
            registry: Tool registry (creates default if not provided)
            authenticator: Optional authenticator for auth mode
            rate_limiter: Optional rate limiter
        """
        self.name = name
        self.version = version
        self._registry = registry or MCPServerRegistry(rate_limiter=rate_limiter)
        self._authenticator = authenticator
        self._rate_limiter = rate_limiter
        self._initialized = False
        self._running = False

    @property
    def registry(self) -> MCPServerRegistry:
        """Get the tool registry."""
        return self._registry

    def get_capabilities(self) -> MCPCapabilities:
        """Get server capabilities."""
        return MCPCapabilities(
            tools={"listChanged": True},
            resources=None,
            prompts=None,
        )

    def get_server_info(self) -> MCPServerInfo:
        """Get server info."""
        return MCPServerInfo(name=self.name, version=self.version)

    async def handle_request(
        self,
        request: MCPRequest,
        auth_context: Optional[MCPAuthContext] = None,
    ) -> MCPResponse:
        """Handle an MCP request.

        Args:
            request: The MCP request
            auth_context: Optional authentication context

        Returns:
            MCP response
        """
        try:
            method = request.method
            params = request.params

            logger.debug(
                "mcp_request_received",
                method=method,
                request_id=request.id,
            )

            # Route to handler
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "initialized":
                # Client acknowledgment - no response needed
                return MCPResponse.success(request.id, {})
            elif method == "tools/list":
                result = await self._handle_tools_list(params, auth_context)
            elif method == "tools/call":
                result = await self._handle_tools_call(params, auth_context)
            elif method == "ping":
                result = {"pong": True}
            else:
                raise MCPError(
                    code=MCPErrorCode.METHOD_NOT_FOUND,
                    message=f"Unknown method: {method}",
                )

            return MCPResponse.success(request.id, result)

        except MCPError as e:
            logger.warning(
                "mcp_request_error",
                method=request.method,
                error_code=e.code.value,
                error=e.message,
            )
            return MCPResponse.failure(request.id, e)

        except Exception as e:
            logger.exception(
                "mcp_request_unexpected_error",
                method=request.method,
                error=str(e),
            )
            return MCPResponse.failure(
                request.id,
                MCPError(
                    code=MCPErrorCode.INTERNAL_ERROR,
                    message=str(e),
                ),
            )

    async def _handle_initialize(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle initialize request."""
        client_info = params.get("clientInfo", {})
        logger.info(
            "mcp_initialize",
            client_name=client_info.get("name"),
            client_version=client_info.get("version"),
        )

        self._initialized = True

        result = MCPInitializeResult(
            protocolVersion=MCP_PROTOCOL_VERSION,
            capabilities=self.get_capabilities(),
            serverInfo=self.get_server_info(),
        )
        return result.model_dump(by_alias=True)

    async def _handle_tools_list(
        self,
        params: dict[str, Any],
        auth_context: Optional[MCPAuthContext],
    ) -> dict[str, Any]:
        """Handle tools/list request."""
        cursor = params.get("cursor")
        if cursor:
            # We don't paginate - return empty for subsequent requests
            return {"tools": [], "nextCursor": None}

        tools = self._registry.list_tools(auth_context=auth_context)
        return {"tools": tools}

    async def _handle_tools_call(
        self,
        params: dict[str, Any],
        auth_context: Optional[MCPAuthContext],
    ) -> dict[str, Any]:
        """Handle tools/call request."""
        name = params.get("name")
        if not name:
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Tool name is required",
            )

        arguments = params.get("arguments", {})

        result = await self._registry.call_tool(
            name=name,
            arguments=arguments,
            auth_context=auth_context,
        )

        return result.to_dict()

    # HTTP/SSE Transport

    async def handle_sse_request(
        self,
        request_data: bytes | str,
        credentials: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[str]:
        """Handle an SSE request.

        Args:
            request_data: Raw request data
            credentials: Optional authentication credentials

        Yields:
            SSE formatted response chunks
        """
        # Authenticate if authenticator configured
        auth_context = None
        if self._authenticator and credentials:
            try:
                auth_context = await self._authenticator.authenticate(credentials)
            except MCPError as e:
                yield f"data: {json.dumps({'error': e.to_dict()})}\n\n"
                return

        # Parse request
        try:
            if isinstance(request_data, bytes):
                request_data = request_data.decode("utf-8")
            data = json.loads(request_data)
            request = MCPRequest(**data)
        except Exception as e:
            error = MCPError(
                code=MCPErrorCode.INVALID_REQUEST,
                message=f"Invalid request: {e}",
            )
            yield f"data: {json.dumps({'error': error.to_dict()})}\n\n"
            return

        # Handle request
        response = await self.handle_request(request, auth_context)
        yield f"data: {response.model_dump_json()}\n\n"

    async def create_http_handler(self) -> Callable[..., Awaitable[dict[str, Any]]]:
        """Create an HTTP handler function for FastAPI integration.

        Returns:
            Async handler function
        """

        async def handler(
            request_data: dict[str, Any],
            credentials: Optional[dict[str, str]] = None,
        ) -> dict[str, Any]:
            # Authenticate
            auth_context = None
            if self._authenticator and credentials:
                auth_context = await self._authenticator.authenticate(credentials)

            # Parse and handle
            mcp_request = MCPRequest(**request_data)
            response = await self.handle_request(mcp_request, auth_context)
            return response.model_dump()

        return handler

    # Stdio Transport

    async def run_stdio(self) -> None:
        """Run the server using stdio transport.

        Reads JSON-RPC requests from stdin and writes
        responses to stdout.
        """
        logger.info("mcp_stdio_server_starting", name=self.name)
        self._running = True

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)

        loop = asyncio.get_event_loop()
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        # Use stdout for output
        writer_transport, writer_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin,
            sys.stdout,
        )
        writer = asyncio.StreamWriter(
            writer_transport,
            writer_protocol,
            None,
            loop,
        )

        try:
            while self._running:
                try:
                    # Read line (JSON-RPC over stdio uses newline-delimited JSON)
                    line = await reader.readline()
                    if not line:
                        break

                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue

                    # Parse request
                    try:
                        data = json.loads(line_str)
                        request = MCPRequest(**data)
                    except Exception as e:
                        error = MCPError(
                            code=MCPErrorCode.INVALID_REQUEST,
                            message=str(e),
                        )
                        response = MCPResponse.failure(None, error)
                        writer.write(
                            (response.model_dump_json() + "\n").encode("utf-8")
                        )
                        await writer.drain()
                        continue

                    # Handle request (no auth for stdio - trust local process)
                    response = await self.handle_request(request, None)

                    # Write response
                    writer.write((response.model_dump_json() + "\n").encode("utf-8"))
                    await writer.drain()

                except Exception as e:
                    logger.error("mcp_stdio_error", error=str(e))
                    if not self._running:
                        break

        finally:
            self._running = False
            logger.info("mcp_stdio_server_stopped")

    def stop(self) -> None:
        """Stop the server."""
        self._running = False


class MCPServerFactory:
    """Factory for creating configured MCP servers."""

    @staticmethod
    def create_server(
        name: str = "agentic-rag-mcp",
        version: str = "1.0.0",
        enable_auth: bool = True,
        rate_limit_requests: int = 60,
        rate_limit_window: int = 60,
        default_timeout: float = 30.0,
    ) -> MCPServer:
        """Create a configured MCP server.

        Args:
            name: Server name
            version: Server version
            enable_auth: Whether to enable authentication
            rate_limit_requests: Max requests per window
            rate_limit_window: Rate limit window in seconds
            default_timeout: Default tool timeout

        Returns:
            Configured MCPServer instance
        """
        from .auth import MCPAPIKeyAuth

        rate_limiter = MCPRateLimiter(
            max_requests=rate_limit_requests,
            window_seconds=rate_limit_window,
        )

        authenticator = None
        if enable_auth:
            authenticator = MCPAPIKeyAuth()

        registry = MCPServerRegistry(
            default_timeout_seconds=default_timeout,
            rate_limiter=rate_limiter,
        )

        return MCPServer(
            name=name,
            version=version,
            registry=registry,
            authenticator=authenticator,
            rate_limiter=rate_limiter,
        )
