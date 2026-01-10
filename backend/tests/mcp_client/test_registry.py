"""Tests for MCP Client tool registry.

Story 21-C3: Wire MCP Client to CopilotRuntime
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_rag_backend.mcp_client.registry import (
    ToolInfo,
    discover_all_tools,
    merge_tool_registries,
    parse_namespaced_tool,
)


class TestMergeToolRegistries:
    """Tests for merge_tool_registries function."""

    def test_merge_internal_only(self) -> None:
        """Test merging with only internal tools."""
        internal = [
            {"name": "search", "description": "Search tool", "inputSchema": {}},
            {"name": "calculate", "description": "Calculator", "inputSchema": {}},
        ]
        external: dict = {}

        result = merge_tool_registries(internal, external)

        assert len(result) == 2
        assert "search" in result
        assert "calculate" in result
        assert result["search"]["source"] == "internal"
        assert result["search"]["serverName"] is None

    def test_merge_external_only(self) -> None:
        """Test merging with only external tools."""
        internal: list = []
        external = {
            "github": [
                {"name": "create_issue", "description": "Create GitHub issue"},
                {"name": "list_repos", "description": "List repositories"},
            ],
        }

        result = merge_tool_registries(internal, external)

        assert len(result) == 2
        assert "github:create_issue" in result
        assert "github:list_repos" in result
        assert result["github:create_issue"]["source"] == "external"
        assert result["github:create_issue"]["serverName"] == "github"
        assert result["github:create_issue"]["originalName"] == "create_issue"

    def test_merge_internal_and_external(self) -> None:
        """Test merging both internal and external tools."""
        internal = [
            {"name": "search", "description": "Search tool"},
        ]
        external = {
            "github": [
                {"name": "create_issue", "description": "Create GitHub issue"},
            ],
            "notion": [
                {"name": "create_page", "description": "Create Notion page"},
            ],
        }

        result = merge_tool_registries(internal, external)

        assert len(result) == 3
        assert "search" in result
        assert "github:create_issue" in result
        assert "notion:create_page" in result
        assert result["search"]["source"] == "internal"
        assert result["github:create_issue"]["source"] == "external"
        assert result["notion:create_page"]["source"] == "external"

    def test_skip_tools_without_name(self) -> None:
        """Test that tools without names are skipped."""
        internal = [
            {"name": "valid", "description": "Valid tool"},
            {"description": "Invalid - no name"},
            {"name": "", "description": "Empty name"},
        ]
        external: dict = {}

        result = merge_tool_registries(internal, external)

        assert len(result) == 1
        assert "valid" in result


class TestParseNamespacedTool:
    """Tests for parse_namespaced_tool function."""

    def test_parse_namespaced_tool(self) -> None:
        """Test parsing a namespaced tool name."""
        server, name = parse_namespaced_tool("github:create_issue")

        assert server == "github"
        assert name == "create_issue"

    def test_parse_internal_tool(self) -> None:
        """Test parsing an internal tool (no namespace)."""
        server, name = parse_namespaced_tool("search")

        assert server is None
        assert name == "search"

    def test_parse_tool_with_multiple_colons(self) -> None:
        """Test parsing a tool name with multiple colons."""
        server, name = parse_namespaced_tool("server:tool:with:colons")

        assert server == "server"
        assert name == "tool:with:colons"


class TestDiscoverAllTools:
    """Tests for discover_all_tools function."""

    @pytest.mark.asyncio
    async def test_discover_with_no_factory(self) -> None:
        """Test discovery when factory is None."""
        result = await discover_all_tools(factory=None, internal_tools=[
            {"name": "test", "description": "Test tool"},
        ])

        assert len(result) == 1
        assert "test" in result

    @pytest.mark.asyncio
    async def test_discover_with_disabled_factory(self) -> None:
        """Test discovery when factory is disabled."""
        mock_factory = MagicMock()
        mock_factory.is_enabled = False

        result = await discover_all_tools(factory=mock_factory, internal_tools=[
            {"name": "test", "description": "Test tool"},
        ])

        assert len(result) == 1
        # Should not call discover_all_tools on factory
        mock_factory.discover_all_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_discover_with_enabled_factory(self) -> None:
        """Test discovery when factory is enabled."""
        mock_factory = MagicMock()
        mock_factory.is_enabled = True
        mock_factory.discover_all_tools = AsyncMock(return_value={
            "github": [
                {"name": "create_issue", "description": "Create issue"},
            ],
        })

        result = await discover_all_tools(factory=mock_factory, internal_tools=[
            {"name": "search", "description": "Search"},
        ])

        assert len(result) == 2
        assert "search" in result
        assert "github:create_issue" in result
        mock_factory.discover_all_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_handles_mcp_errors(self) -> None:
        """Test that MCP errors don't prevent internal tools from being returned."""
        from agentic_rag_backend.mcp_client.errors import MCPClientError

        mock_factory = MagicMock()
        mock_factory.is_enabled = True
        mock_factory.discover_all_tools = AsyncMock(
            side_effect=MCPClientError("Connection failed")
        )

        result = await discover_all_tools(factory=mock_factory, internal_tools=[
            {"name": "search", "description": "Search"},
        ])

        # Should still return internal tools
        assert len(result) == 1
        assert "search" in result


class TestToolInfo:
    """Tests for ToolInfo class."""

    def test_tool_info_to_dict(self) -> None:
        """Test converting ToolInfo to dictionary."""
        tool = ToolInfo(
            name="github:create_issue",
            description="Create a GitHub issue",
            input_schema={"type": "object", "properties": {}},
            source="external",
            server_name="github",
        )

        result = tool.to_dict()

        assert result["name"] == "github:create_issue"
        assert result["description"] == "Create a GitHub issue"
        assert result["inputSchema"] == {"type": "object", "properties": {}}
        assert result["source"] == "external"
        assert result["serverName"] == "github"

    def test_tool_info_defaults(self) -> None:
        """Test ToolInfo default values."""
        tool = ToolInfo(
            name="search",
            description="Search tool",
            input_schema={},
        )

        result = tool.to_dict()

        assert result["source"] == "internal"
        assert result["serverName"] is None
