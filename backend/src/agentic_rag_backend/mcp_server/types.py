"""MCP Server type definitions.

Defines the core types for Model Context Protocol (MCP) server operations,
following the MCP specification for tool definitions, requests, and responses.

Story 14-1: Expose RAG Engine via MCP Server
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Optional

from pydantic import BaseModel, Field, ConfigDict


class MCPErrorCode(str, Enum):
    """Standard MCP error codes."""

    INVALID_REQUEST = "invalid_request"
    METHOD_NOT_FOUND = "method_not_found"
    INVALID_PARAMS = "invalid_params"
    INTERNAL_ERROR = "internal_error"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    AUTHENTICATION_REQUIRED = "authentication_required"
    AUTHENTICATION_FAILED = "authentication_failed"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TENANT_REQUIRED = "tenant_required"
    TIMEOUT = "timeout"


class MCPError(Exception):
    """MCP protocol error with structured error information."""

    def __init__(
        self,
        code: MCPErrorCode,
        message: str,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to MCP error response format."""
        result: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
        }
        if self.data:
            result["data"] = self.data
        return result


@dataclass(frozen=True)
class MCPToolSpec:
    """Specification for an MCP tool.

    Follows the MCP tool definition format with JSON Schema input validation.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    category: str = "general"
    timeout_seconds: Optional[float] = None
    requires_auth: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP tool listing format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPToolResult:
    """Result from executing an MCP tool."""

    content: list[dict[str, Any]]
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP tool result format."""
        result: dict[str, Any] = {
            "content": self.content,
            "isError": self.is_error,
        }
        if self.metadata:
            result["_meta"] = self.metadata
        return result

    @classmethod
    def text(cls, text: str, metadata: Optional[dict[str, Any]] = None) -> MCPToolResult:
        """Create a text result."""
        return cls(
            content=[{"type": "text", "text": text}],
            is_error=False,
            metadata=metadata or {},
        )

    @classmethod
    def json(
        cls, data: dict[str, Any], metadata: Optional[dict[str, Any]] = None
    ) -> MCPToolResult:
        """Create a JSON result (as text for MCP compatibility)."""
        import json

        return cls(
            content=[{"type": "text", "text": json.dumps(data, default=str)}],
            is_error=False,
            metadata=metadata or {},
        )

    @classmethod
    def error(
        cls, message: str, code: Optional[MCPErrorCode] = None
    ) -> MCPToolResult:
        """Create an error result."""
        return cls(
            content=[{"type": "text", "text": message}],
            is_error=True,
            metadata={"error_code": code.value if code else "error"},
        )


class MCPRequest(BaseModel):
    """MCP JSON-RPC request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: str | int | None = Field(default=None, description="Request ID")
    method: str = Field(..., min_length=1, description="Method name")
    params: dict[str, Any] = Field(default_factory=dict, description="Method params")


class MCPResponse(BaseModel):
    """MCP JSON-RPC response."""

    model_config = ConfigDict(str_strip_whitespace=True)

    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: str | int | None = Field(default=None, description="Request ID")
    result: Optional[dict[str, Any]] = Field(default=None, description="Result data")
    error: Optional[dict[str, Any]] = Field(default=None, description="Error data")

    @classmethod
    def success(
        cls,
        request_id: str | int | None,
        result: dict[str, Any],
    ) -> MCPResponse:
        """Create a success response."""
        return cls(id=request_id, result=result)

    @classmethod
    def failure(
        cls,
        request_id: str | int | None,
        error: MCPError,
    ) -> MCPResponse:
        """Create an error response."""
        return cls(id=request_id, error=error.to_dict())


class MCPCapabilities(BaseModel):
    """MCP server capabilities declaration."""

    model_config = ConfigDict(str_strip_whitespace=True)

    tools: dict[str, Any] = Field(
        default_factory=lambda: {"listChanged": True},
        description="Tools capability",
    )
    resources: Optional[dict[str, Any]] = Field(
        default=None,
        description="Resources capability (not implemented)",
    )
    prompts: Optional[dict[str, Any]] = Field(
        default=None,
        description="Prompts capability (not implemented)",
    )


class MCPServerInfo(BaseModel):
    """MCP server information."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Server name")
    version: str = Field(..., description="Server version")


class MCPInitializeResult(BaseModel):
    """MCP initialize response result."""

    model_config = ConfigDict(str_strip_whitespace=True)

    protocolVersion: str = Field(default="2024-11-05", description="MCP protocol version")
    capabilities: MCPCapabilities = Field(
        default_factory=MCPCapabilities,
        description="Server capabilities",
    )
    serverInfo: MCPServerInfo = Field(..., description="Server information")


# Tool input schema helpers
def create_tool_input_schema(
    properties: dict[str, dict[str, Any]],
    required: Optional[list[str]] = None,
    additional_properties: bool = False,
) -> dict[str, Any]:
    """Create a JSON Schema for tool input validation.

    Args:
        properties: Property definitions
        required: List of required property names
        additional_properties: Whether to allow extra properties

    Returns:
        JSON Schema dict for tool input
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties,
    }
    if required:
        schema["required"] = required
    return schema
