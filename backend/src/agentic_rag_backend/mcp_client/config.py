"""Configuration models for MCP client connections.

Story 21-C2: Implement MCP Client Factory

These models define the configuration schema for connecting to external MCP servers.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, HttpUrl, Field


class MCPServerConfig(BaseModel):
    """Configuration for an external MCP server.

    Attributes:
        name: Unique identifier for the server
        url: MCP server endpoint URL
        api_key: Optional API key for authentication
        transport: Transport type ("sse" or "http")
        timeout_ms: Request timeout in milliseconds
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., description="Unique identifier for the server")
    url: HttpUrl = Field(..., description="MCP server endpoint URL")
    api_key: Optional[str] = Field(default=None, alias="apiKey", description="Optional API key")
    transport: str = Field(default="sse", description="Transport type: sse or http")
    timeout_ms: int = Field(default=30000, alias="timeout", ge=1000, description="Timeout in ms")


class MCPClientSettings(BaseModel):
    """MCP Client configuration settings.

    Attributes:
        enabled: Whether MCP client feature is enabled
        servers: List of MCP server configurations
        default_timeout_ms: Default timeout for requests
        retry_count: Number of retry attempts
        retry_delay_ms: Base delay between retries
    """
    enabled: bool = Field(default=False, description="Enable MCP client feature")
    servers: list[MCPServerConfig] = Field(default_factory=list, description="Server configs")
    default_timeout_ms: int = Field(default=30000, ge=1000, description="Default timeout in ms")
    retry_count: int = Field(default=3, ge=0, le=10, description="Retry attempts")
    retry_delay_ms: int = Field(default=1000, ge=100, description="Retry delay in ms")
