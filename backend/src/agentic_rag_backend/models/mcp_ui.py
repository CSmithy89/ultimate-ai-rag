"""Pydantic models for MCP-UI configuration and payloads.

This module defines models for MCP-UI iframe-based tool rendering,
including configuration, payload formats, and security settings.

Story 22-C1: Implement MCP-UI Renderer
"""

from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class MCPUIConfig(BaseModel):
    """MCP-UI configuration response model.

    Returned by the /mcp/ui/config endpoint to inform the frontend
    about allowed iframe origins and feature enablement.
    """

    enabled: bool = Field(
        default=False,
        description="Whether MCP-UI rendering is enabled",
    )
    allowed_origins: list[str] = Field(
        default_factory=list,
        description="List of allowed origins for MCP-UI iframes",
    )


class MCPUIPayload(BaseModel):
    """MCP-UI iframe payload from MCP tool response.

    This payload is emitted by MCP tools that provide interactive UIs.
    The frontend MCPUIRenderer component consumes this payload.

    Example:
        {
            "type": "mcp_ui",
            "tool_name": "calculator",
            "ui_url": "https://tools.example.com/calculator",
            "ui_type": "iframe",
            "sandbox": ["allow-scripts"],
            "size": {"width": 600, "height": 400},
            "allow": [],
            "data": {"initial_value": 0}
        }
    """

    type: str = Field(
        default="mcp_ui",
        description="Payload type identifier (always 'mcp_ui')",
    )
    tool_name: str = Field(
        ...,
        description="Name of the MCP tool providing this UI",
    )
    ui_url: HttpUrl = Field(
        ...,
        description="URL of the external tool UI to embed",
    )
    ui_type: str = Field(
        default="iframe",
        description="UI rendering type (currently only 'iframe' supported)",
    )
    sandbox: list[str] = Field(
        default_factory=lambda: ["allow-scripts"],
        description="Sandbox permissions for the iframe. Note: 'allow-same-origin' is NOT "
        "included by default as combining it with 'allow-scripts' weakens security. "
        "Add 'allow-same-origin' explicitly only if the embedded UI requires it.",
    )
    size: dict[str, int] = Field(
        default_factory=lambda: {"width": 600, "height": 400},
        description="Initial iframe dimensions",
    )
    allow: list[str] = Field(
        default_factory=list,
        description="Permissions policy for the iframe (allow attribute)",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Data to pass to the iframe via postMessage on init",
    )


class MCPUIMessage(BaseModel):
    """Base model for MCP-UI postMessage payloads.

    This is used for reference; actual validation happens in the frontend
    using Zod schemas for better TypeScript integration.
    """

    type: str = Field(
        ...,
        description="Message type: mcp_ui_resize, mcp_ui_result, or mcp_ui_error",
    )


class MCPUIResizeMessage(MCPUIMessage):
    """Message to resize the iframe container."""

    type: str = Field(default="mcp_ui_resize")
    width: int = Field(..., ge=100, le=4000, description="New width in pixels")
    height: int = Field(..., ge=50, le=4000, description="New height in pixels")


class MCPUIResultMessage(MCPUIMessage):
    """Message containing tool result from iframe."""

    type: str = Field(default="mcp_ui_result")
    result: Any = Field(..., description="Result data from the tool UI")


class MCPUIErrorMessage(MCPUIMessage):
    """Message reporting an error from iframe."""

    type: str = Field(default="mcp_ui_error")
    error: str = Field(..., description="Error message from the tool UI")
