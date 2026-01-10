"""MCP Client module for connecting to external MCP servers.

Story 21-C2: Implement MCP Client Factory

This module provides client functionality for connecting to external MCP servers
as part of the CopilotKit integration. It enables tool discovery and execution
from MCP ecosystem servers like GitHub, Notion, etc.

Example usage:
    from agentic_rag_backend.mcp_client import (
        MCPClientFactory, MCPClientSettings, MCPServerConfig
    )

    settings = MCPClientSettings(
        enabled=True,
        servers=[
            MCPServerConfig(name="github", url="https://mcp.github.com/sse")
        ],
    )
    factory = MCPClientFactory(settings)

    async with factory.lifespan():
        tools = await factory.discover_all_tools()
        result = await factory.call_tool("github", "search", {"query": "test"})
"""

from agentic_rag_backend.mcp_client.client import MCPClient, MCPClientFactory
from agentic_rag_backend.mcp_client.config import MCPClientSettings, MCPServerConfig
from agentic_rag_backend.mcp_client.dependencies import (
    create_mcp_client_factory,
    create_mcp_client_settings,
    get_mcp_factory,
)
from agentic_rag_backend.mcp_client.errors import (
    MCPClientConnectionError,
    MCPClientError,
    MCPClientNotEnabledError,
    MCPClientTimeoutError,
    MCPProtocolError,
    MCPServerNotFoundError,
    MCPToolNotFoundError,
)
from agentic_rag_backend.mcp_client.registry import (
    ToolInfo,
    discover_all_tools,
    merge_tool_registries,
    parse_namespaced_tool,
)

__all__ = [
    # Client classes
    "MCPClient",
    "MCPClientFactory",
    # Configuration
    "MCPClientSettings",
    "MCPServerConfig",
    # Dependencies
    "create_mcp_client_factory",
    "create_mcp_client_settings",
    "get_mcp_factory",
    # Errors
    "MCPClientError",
    "MCPClientConnectionError",
    "MCPClientNotEnabledError",
    "MCPClientTimeoutError",
    "MCPProtocolError",
    "MCPServerNotFoundError",
    "MCPToolNotFoundError",
    # Registry
    "ToolInfo",
    "discover_all_tools",
    "merge_tool_registries",
    "parse_namespaced_tool",
]
