"""MCP Client error classes.

Story 21-C2: Implement MCP Client Factory

Custom exceptions for MCP client operations following the project's error patterns.
"""

from typing import Any, Optional


class MCPClientError(Exception):
    """Base exception for MCP client errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class MCPClientTimeoutError(MCPClientError):
    """Raised when MCP server request times out after all retries."""

    def __init__(
        self,
        server_name: str,
        attempts: int,
        timeout_ms: int,
    ) -> None:
        message = f"MCP server '{server_name}' timed out after {attempts} attempts"
        super().__init__(
            message=message,
            details={
                "server_name": server_name,
                "attempts": attempts,
                "timeout_ms": timeout_ms,
            },
        )
        self.server_name = server_name
        self.attempts = attempts
        self.timeout_ms = timeout_ms


class MCPClientConnectionError(MCPClientError):
    """Raised when unable to connect to MCP server."""

    def __init__(self, server_name: str, url: str, reason: str) -> None:
        message = f"Failed to connect to MCP server '{server_name}': {reason}"
        super().__init__(
            message=message,
            details={
                "server_name": server_name,
                "url": url,
                "reason": reason,
            },
        )
        self.server_name = server_name
        self.url = url
        self.reason = reason


class MCPServerNotFoundError(MCPClientError):
    """Raised when attempting to use an unknown MCP server."""

    def __init__(self, server_name: str, available_servers: list[str]) -> None:
        message = f"Unknown MCP server: '{server_name}'"
        super().__init__(
            message=message,
            details={
                "server_name": server_name,
                "available_servers": available_servers,
            },
        )
        self.server_name = server_name
        self.available_servers = available_servers


class MCPProtocolError(MCPClientError):
    """Raised when MCP server returns a protocol error."""

    def __init__(
        self,
        server_name: str,
        error_code: Optional[int] = None,
        error_message: Optional[str] = None,
        error_data: Optional[Any] = None,
    ) -> None:
        message = f"MCP protocol error from '{server_name}': {error_message or 'Unknown error'}"
        super().__init__(
            message=message,
            details={
                "server_name": server_name,
                "error_code": error_code,
                "error_message": error_message,
                "error_data": error_data,
            },
        )
        self.server_name = server_name
        self.error_code = error_code
        self.error_message = error_message
        self.error_data = error_data


class MCPToolNotFoundError(MCPClientError):
    """Raised when attempting to call an unknown tool on an MCP server."""

    def __init__(self, server_name: str, tool_name: str) -> None:
        message = f"Tool '{tool_name}' not found on MCP server '{server_name}'"
        super().__init__(
            message=message,
            details={
                "server_name": server_name,
                "tool_name": tool_name,
            },
        )
        self.server_name = server_name
        self.tool_name = tool_name


class MCPClientNotEnabledError(MCPClientError):
    """Raised when MCP client feature is disabled but an operation is attempted."""

    def __init__(self) -> None:
        super().__init__(
            message="MCP client feature is not enabled",
            details={"hint": "Set MCP_CLIENTS_ENABLED=true to enable"},
        )
