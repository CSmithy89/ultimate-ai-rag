"""FastAPI dependency injection for MCP client.

Story 21-C2: Implement MCP Client Factory

Provides dependency injection functions for accessing MCP client factory
from FastAPI routes.
"""

from typing import Optional

from fastapi import Request

from agentic_rag_backend.config import Settings
from agentic_rag_backend.mcp_client.client import MCPClientFactory
from agentic_rag_backend.mcp_client.config import MCPClientSettings, MCPServerConfig


def create_mcp_client_settings(settings: Settings) -> MCPClientSettings:
    """Create MCPClientSettings from application Settings.

    Args:
        settings: Application settings from config

    Returns:
        MCPClientSettings instance
    """
    # Parse server configurations from dict format
    servers: list[MCPServerConfig] = []
    for server_dict in settings.mcp_client_servers:
        try:
            servers.append(MCPServerConfig(**server_dict))
        except Exception:
            # Skip invalid server configurations
            continue

    return MCPClientSettings(
        enabled=settings.mcp_clients_enabled,
        servers=servers,
        default_timeout_ms=settings.mcp_client_timeout_ms,
        retry_count=settings.mcp_client_retry_count,
        retry_delay_ms=settings.mcp_client_retry_delay_ms,
    )


def create_mcp_client_factory(settings: Settings) -> MCPClientFactory:
    """Create MCPClientFactory from application Settings.

    Args:
        settings: Application settings from config

    Returns:
        MCPClientFactory instance
    """
    mcp_settings = create_mcp_client_settings(settings)
    return MCPClientFactory(mcp_settings)


async def get_mcp_factory(request: Request) -> Optional[MCPClientFactory]:
    """FastAPI dependency to get MCP client factory from app state.

    Args:
        request: FastAPI request object

    Returns:
        MCPClientFactory instance or None if not initialized
    """
    return getattr(request.app.state, "mcp_client_factory", None)
