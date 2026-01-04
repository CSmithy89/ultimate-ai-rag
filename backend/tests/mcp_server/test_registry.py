"""Tests for MCP server registry.

Story 14-1: Expose RAG Engine via MCP Server
"""

import pytest
import asyncio

from agentic_rag_backend.mcp_server.registry import MCPServerRegistry
from agentic_rag_backend.mcp_server.types import (
    MCPToolSpec,
    MCPToolResult,
    MCPError,
    MCPErrorCode,
    create_tool_input_schema,
)
from agentic_rag_backend.mcp_server.auth import MCPAuthContext, MCPRateLimiter


async def dummy_handler(args):
    """Dummy tool handler for testing."""
    return {"echo": args.get("input", "default")}


async def slow_handler(args):
    """Handler that takes time."""
    await asyncio.sleep(0.5)
    return {"result": "done"}


async def error_handler(args):
    """Handler that raises an error."""
    raise ValueError("Test error")


class TestMCPServerRegistry:
    """Tests for MCPServerRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = MCPServerRegistry()
        tool = MCPToolSpec(
            name="test.echo",
            description="Echo tool",
            input_schema=create_tool_input_schema(
                properties={"input": {"type": "string"}},
            ),
            handler=dummy_handler,
        )
        registry.register(tool)
        assert registry.get_tool("test.echo") is not None

    def test_register_duplicate_tool(self):
        """Test registering duplicate tool fails."""
        registry = MCPServerRegistry()
        tool = MCPToolSpec(
            name="test.echo",
            description="Echo tool",
            input_schema={},
            handler=dummy_handler,
        )
        registry.register(tool)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool)

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = MCPServerRegistry()
        tool = MCPToolSpec(
            name="test.tool",
            description="Test",
            input_schema={},
            handler=dummy_handler,
        )
        registry.register(tool)
        assert registry.unregister("test.tool")
        assert registry.get_tool("test.tool") is None

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent tool."""
        registry = MCPServerRegistry()
        assert not registry.unregister("nonexistent")

    def test_list_tools(self):
        """Test listing tools."""
        registry = MCPServerRegistry()
        for i in range(3):
            tool = MCPToolSpec(
                name=f"test.tool{i}",
                description=f"Tool {i}",
                input_schema={},
                handler=dummy_handler,
                category="test",
            )
            registry.register(tool)

        tools = registry.list_tools()
        assert len(tools) == 3
        assert all("name" in t for t in tools)

    def test_list_tools_by_category(self):
        """Test listing tools filtered by category."""
        registry = MCPServerRegistry()
        registry.register(MCPToolSpec(
            name="cat1.tool",
            description="Cat1 tool",
            input_schema={},
            handler=dummy_handler,
            category="cat1",
        ))
        registry.register(MCPToolSpec(
            name="cat2.tool",
            description="Cat2 tool",
            input_schema={},
            handler=dummy_handler,
            category="cat2",
        ))

        cat1_tools = registry.list_tools(category="cat1")
        assert len(cat1_tools) == 1
        assert cat1_tools[0]["name"] == "cat1.tool"

    @pytest.mark.asyncio
    async def test_call_tool(self):
        """Test calling a tool."""
        registry = MCPServerRegistry()
        tool = MCPToolSpec(
            name="test.echo",
            description="Echo",
            input_schema={},
            handler=dummy_handler,
            requires_auth=False,
        )
        registry.register(tool)

        result = await registry.call_tool(
            "test.echo",
            {"input": "hello", "tenant_id": "00000000-0000-0000-0000-000000000001"},
        )
        assert isinstance(result, MCPToolResult)

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self):
        """Test calling nonexistent tool."""
        registry = MCPServerRegistry()
        with pytest.raises(MCPError) as exc_info:
            await registry.call_tool(
                "nonexistent",
                {"tenant_id": "00000000-0000-0000-0000-000000000001"},
            )
        assert exc_info.value.code == MCPErrorCode.TOOL_NOT_FOUND

    @pytest.mark.asyncio
    async def test_call_tool_without_tenant(self):
        """Test calling tool without tenant_id."""
        registry = MCPServerRegistry()
        tool = MCPToolSpec(
            name="test.tool",
            description="Test",
            input_schema={},
            handler=dummy_handler,
            requires_auth=False,
        )
        registry.register(tool)

        with pytest.raises(MCPError) as exc_info:
            await registry.call_tool("test.tool", {})
        assert exc_info.value.code == MCPErrorCode.TENANT_REQUIRED

    @pytest.mark.asyncio
    async def test_call_tool_timeout(self):
        """Test tool timeout handling."""
        registry = MCPServerRegistry(default_timeout_seconds=0.1)
        tool = MCPToolSpec(
            name="test.slow",
            description="Slow tool",
            input_schema={},
            handler=slow_handler,
            requires_auth=False,
        )
        registry.register(tool)

        with pytest.raises(MCPError) as exc_info:
            await registry.call_tool(
                "test.slow",
                {"tenant_id": "00000000-0000-0000-0000-000000000001"},
            )
        assert exc_info.value.code == MCPErrorCode.TIMEOUT

    @pytest.mark.asyncio
    async def test_call_tool_error(self):
        """Test tool error handling."""
        registry = MCPServerRegistry()
        tool = MCPToolSpec(
            name="test.error",
            description="Error tool",
            input_schema={},
            handler=error_handler,
            requires_auth=False,
        )
        registry.register(tool)

        with pytest.raises(MCPError) as exc_info:
            await registry.call_tool(
                "test.error",
                {"tenant_id": "00000000-0000-0000-0000-000000000001"},
            )
        assert exc_info.value.code == MCPErrorCode.TOOL_EXECUTION_ERROR

    @pytest.mark.asyncio
    async def test_call_tool_requires_auth(self):
        """Test tool requiring authentication."""
        registry = MCPServerRegistry()
        tool = MCPToolSpec(
            name="test.auth",
            description="Auth tool",
            input_schema={},
            handler=dummy_handler,
            requires_auth=True,
        )
        registry.register(tool)

        # Without auth context
        with pytest.raises(MCPError) as exc_info:
            await registry.call_tool(
                "test.auth",
                {"tenant_id": "00000000-0000-0000-0000-000000000001"},
            )
        assert exc_info.value.code == MCPErrorCode.AUTHENTICATION_REQUIRED

    @pytest.mark.asyncio
    async def test_call_tool_with_auth(self):
        """Test calling tool with auth context."""
        registry = MCPServerRegistry()
        tool = MCPToolSpec(
            name="test.auth",
            description="Auth tool",
            input_schema={},
            handler=dummy_handler,
            requires_auth=True,
        )
        registry.register(tool)

        auth_context = MCPAuthContext(
            tenant_id="00000000-0000-0000-0000-000000000001",
            scopes=None,  # Admin
        )
        result = await registry.call_tool(
            "test.auth",
            {"tenant_id": "00000000-0000-0000-0000-000000000001"},
            auth_context=auth_context,
        )
        assert isinstance(result, MCPToolResult)

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting."""
        rate_limiter = MCPRateLimiter(max_requests=2, window_seconds=60)
        registry = MCPServerRegistry(rate_limiter=rate_limiter)

        tool = MCPToolSpec(
            name="test.tool",
            description="Test",
            input_schema={},
            handler=dummy_handler,
            requires_auth=False,
        )
        registry.register(tool)

        tenant_id = "00000000-0000-0000-0000-000000000001"

        # First two calls should succeed
        await registry.call_tool("test.tool", {"tenant_id": tenant_id})
        await registry.call_tool("test.tool", {"tenant_id": tenant_id})

        # Third call should be rate limited
        with pytest.raises(MCPError) as exc_info:
            await registry.call_tool("test.tool", {"tenant_id": tenant_id})
        assert exc_info.value.code == MCPErrorCode.RATE_LIMIT_EXCEEDED

    def test_get_categories(self):
        """Test getting all categories."""
        registry = MCPServerRegistry()
        registry.register(MCPToolSpec(
            name="a.tool", description="A", input_schema={},
            handler=dummy_handler, category="alpha",
        ))
        registry.register(MCPToolSpec(
            name="b.tool", description="B", input_schema={},
            handler=dummy_handler, category="beta",
        ))

        categories = registry.get_categories()
        assert "alpha" in categories
        assert "beta" in categories

    def test_get_tools_by_category(self):
        """Test getting tools grouped by category."""
        registry = MCPServerRegistry()
        registry.register(MCPToolSpec(
            name="a.tool", description="A", input_schema={},
            handler=dummy_handler, category="cat1",
        ))
        registry.register(MCPToolSpec(
            name="b.tool", description="B", input_schema={},
            handler=dummy_handler, category="cat1",
        ))
        registry.register(MCPToolSpec(
            name="c.tool", description="C", input_schema={},
            handler=dummy_handler, category="cat2",
        ))

        by_cat = registry.get_tools_by_category()
        assert len(by_cat["cat1"]) == 2
        assert len(by_cat["cat2"]) == 1
