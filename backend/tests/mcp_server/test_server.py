"""Tests for MCP server.

Story 14-1: Expose RAG Engine via MCP Server
"""

import pytest
import json

from agentic_rag_backend.mcp_server.server import MCPServer, MCPServerFactory
from agentic_rag_backend.mcp_server.registry import MCPServerRegistry
from agentic_rag_backend.mcp_server.types import (
    MCPRequest,
    MCPToolSpec,
    MCPError,
    MCPErrorCode,
    create_tool_input_schema,
)
from agentic_rag_backend.mcp_server.auth import MCPAuthContext


async def echo_handler(args):
    """Simple echo handler for testing."""
    return {"echo": args.get("message", "hello")}


@pytest.fixture
def mcp_server():
    """Create a test MCP server."""
    registry = MCPServerRegistry()
    registry.register(MCPToolSpec(
        name="test.echo",
        description="Echo tool for testing",
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {"type": "string"},
                "message": {"type": "string"},
            },
            required=["tenant_id"],
        ),
        handler=echo_handler,
        requires_auth=False,
    ))
    return MCPServer(
        name="test-server",
        version="1.0.0",
        registry=registry,
    )


class TestMCPServer:
    """Tests for MCPServer."""

    def test_server_creation(self, mcp_server):
        """Test creating an MCP server."""
        assert mcp_server.name == "test-server"
        assert mcp_server.version == "1.0.0"

    def test_get_capabilities(self, mcp_server):
        """Test getting server capabilities."""
        caps = mcp_server.get_capabilities()
        assert caps.tools == {"listChanged": True}

    def test_get_server_info(self, mcp_server):
        """Test getting server info."""
        info = mcp_server.get_server_info()
        assert info.name == "test-server"
        assert info.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_handle_initialize(self, mcp_server):
        """Test handling initialize request."""
        request = MCPRequest(
            id="init-1",
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        )
        response = await mcp_server.handle_request(request)
        assert response.error is None
        assert response.result is not None
        assert "protocolVersion" in response.result
        assert "capabilities" in response.result
        assert "serverInfo" in response.result

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, mcp_server):
        """Test handling tools/list request."""
        request = MCPRequest(
            id="list-1",
            method="tools/list",
            params={},
        )
        response = await mcp_server.handle_request(request)
        assert response.error is None
        assert "tools" in response.result
        assert len(response.result["tools"]) == 1
        assert response.result["tools"][0]["name"] == "test.echo"

    @pytest.mark.asyncio
    async def test_handle_tools_call(self, mcp_server):
        """Test handling tools/call request."""
        request = MCPRequest(
            id="call-1",
            method="tools/call",
            params={
                "name": "test.echo",
                "arguments": {
                    "tenant_id": "00000000-0000-0000-0000-000000000001",
                    "message": "hello world",
                },
            },
        )
        response = await mcp_server.handle_request(request)
        assert response.error is None
        assert "content" in response.result

    @pytest.mark.asyncio
    async def test_handle_tools_call_missing_name(self, mcp_server):
        """Test tools/call without tool name."""
        request = MCPRequest(
            id="call-2",
            method="tools/call",
            params={"arguments": {}},
        )
        response = await mcp_server.handle_request(request)
        assert response.error is not None
        assert response.error["code"] == "invalid_params"

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, mcp_server):
        """Test handling unknown method."""
        request = MCPRequest(
            id="unknown-1",
            method="unknown/method",
            params={},
        )
        response = await mcp_server.handle_request(request)
        assert response.error is not None
        assert response.error["code"] == "method_not_found"

    @pytest.mark.asyncio
    async def test_handle_ping(self, mcp_server):
        """Test handling ping request."""
        request = MCPRequest(
            id="ping-1",
            method="ping",
            params={},
        )
        response = await mcp_server.handle_request(request)
        assert response.error is None
        assert response.result.get("pong") is True

    @pytest.mark.asyncio
    async def test_handle_initialized(self, mcp_server):
        """Test handling initialized notification."""
        request = MCPRequest(
            method="initialized",
            params={},
        )
        response = await mcp_server.handle_request(request)
        assert response.error is None

    @pytest.mark.asyncio
    async def test_handle_sse_request(self, mcp_server):
        """Test handling SSE request."""
        request_data = json.dumps({
            "jsonrpc": "2.0",
            "id": "sse-1",
            "method": "tools/list",
            "params": {},
        })

        responses = []
        async for chunk in mcp_server.handle_sse_request(request_data.encode()):
            responses.append(chunk)

        assert len(responses) == 1
        assert "data:" in responses[0]

    @pytest.mark.asyncio
    async def test_handle_sse_invalid_request(self, mcp_server):
        """Test SSE with invalid request."""
        request_data = "invalid json"

        responses = []
        async for chunk in mcp_server.handle_sse_request(request_data.encode()):
            responses.append(chunk)

        assert len(responses) == 1
        assert "error" in responses[0]


class TestMCPServerFactory:
    """Tests for MCPServerFactory."""

    def test_create_server(self):
        """Test creating server via factory."""
        server = MCPServerFactory.create_server(
            name="factory-server",
            version="2.0.0",
            enable_auth=True,
            rate_limit_requests=100,
        )
        assert server.name == "factory-server"
        assert server.version == "2.0.0"
        assert server._authenticator is not None
        assert server._rate_limiter is not None

    def test_create_server_without_auth(self):
        """Test creating server without auth."""
        server = MCPServerFactory.create_server(
            enable_auth=False,
        )
        assert server._authenticator is None


class TestMCPServerWithAuth:
    """Tests for MCP server with authentication."""

    @pytest.fixture
    def auth_server(self):
        """Create a server with authentication."""
        from agentic_rag_backend.mcp_server.auth import MCPAPIKeyAuth, generate_api_key

        auth = MCPAPIKeyAuth()
        test_key = generate_api_key()
        # Register as admin to have all scopes
        auth.register_key(
            api_key=test_key,
            tenant_id="00000000-0000-0000-0000-000000000001",
            is_admin=True,
        )

        registry = MCPServerRegistry()
        registry.register(MCPToolSpec(
            name="test.auth_tool",
            description="Tool requiring auth",
            input_schema=create_tool_input_schema(
                properties={"tenant_id": {"type": "string"}},
                required=["tenant_id"],
            ),
            handler=echo_handler,
            requires_auth=True,
        ))

        return MCPServer(
            name="auth-server",
            version="1.0.0",
            registry=registry,
            authenticator=auth,
        ), test_key

    @pytest.mark.asyncio
    async def test_tools_call_with_auth(self, auth_server):
        """Test calling tool with valid auth."""
        server, api_key = auth_server

        # Authenticate first
        auth_ctx = await server._authenticator.authenticate({"api_key": api_key})

        request = MCPRequest(
            id="auth-call-1",
            method="tools/call",
            params={
                "name": "test.auth_tool",
                "arguments": {"tenant_id": "00000000-0000-0000-0000-000000000001"},
            },
        )
        response = await server.handle_request(request, auth_ctx)
        assert response.error is None
