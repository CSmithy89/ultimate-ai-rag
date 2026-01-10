"""Tool registry for merging internal and external MCP tools.

Story 21-C3: Wire MCP Client to CopilotRuntime

This module provides utilities to merge internal agent tools with
external MCP tools from connected servers.
"""

from typing import Any, Optional

import structlog

from .client import MCPClientFactory
from .errors import MCPClientError

logger = structlog.get_logger(__name__)


class ToolInfo:
    """Information about a tool from any source."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        source: str = "internal",
        server_name: Optional[str] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.source = source
        self.server_name = server_name

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "source": self.source,
            "serverName": self.server_name,
        }


def merge_tool_registries(
    internal: list[dict[str, Any]],
    external: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Merge internal and external tool registries with namespacing.

    Args:
        internal: List of internal tool definitions
        external: Dict mapping server names to lists of tool definitions

    Returns:
        Dict mapping tool names to tool definitions (namespaced for external)
    """
    merged: dict[str, dict[str, Any]] = {}

    # Add internal tools (no namespace)
    for tool in internal:
        tool_name = tool.get("name", "")
        if tool_name:
            merged[tool_name] = {
                **tool,
                "source": "internal",
                "serverName": None,
            }

    # Add external tools (namespaced by server)
    for server_name, tools in external.items():
        for tool in tools:
            original_name = tool.get("name", "")
            if original_name:
                namespaced_name = f"{server_name}:{original_name}"
                merged[namespaced_name] = {
                    **tool,
                    "name": namespaced_name,
                    "originalName": original_name,
                    "source": "external",
                    "serverName": server_name,
                }

    return merged


async def discover_all_tools(
    factory: Optional[MCPClientFactory],
    internal_tools: Optional[list[dict[str, Any]]] = None,
) -> dict[str, dict[str, Any]]:
    """Discover and merge all available tools.

    Args:
        factory: MCP client factory for external tools (can be None)
        internal_tools: Optional list of internal tool definitions

    Returns:
        Dict mapping tool names to tool definitions
    """
    internal = internal_tools or []
    external: dict[str, list[dict[str, Any]]] = {}

    # Discover external tools if factory is enabled
    if factory and factory.is_enabled:
        try:
            external = await factory.discover_all_tools()
            logger.info(
                "mcp_tools_discovered",
                server_count=len(external),
                tool_count=sum(len(tools) for tools in external.values()),
            )
        except MCPClientError as e:
            logger.warning(
                "mcp_tool_discovery_failed",
                error=str(e),
            )
        except Exception as e:
            logger.exception(
                "mcp_tool_discovery_unexpected_error",
                error=str(e),
            )

    return merge_tool_registries(internal, external)


def parse_namespaced_tool(tool_name: str) -> tuple[Optional[str], str]:
    """Parse a namespaced tool name into server and original name.

    Args:
        tool_name: The tool name (possibly namespaced like "github:create_issue")

    Returns:
        Tuple of (server_name, original_name). Server is None for internal tools.
    """
    if ":" in tool_name:
        parts = tool_name.split(":", 1)
        return (parts[0], parts[1])
    return (None, tool_name)
