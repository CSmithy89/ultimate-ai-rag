"""MCP Server for RAG Engine.

This module provides a Model Context Protocol (MCP) server that exposes
the RAG engine capabilities as tools for LLM agents and IDEs.

Story 14-1: Expose RAG Engine via MCP Server
"""

from .types import (
    MCPToolSpec,
    MCPToolResult,
    MCPError,
    MCPErrorCode,
    MCPRequest,
    MCPResponse,
    MCPCapabilities,
)
from .registry import MCPServerRegistry
from .server import MCPServer
from .auth import MCPAuthenticator, MCPAPIKeyAuth

__all__ = [
    # Types
    "MCPToolSpec",
    "MCPToolResult",
    "MCPError",
    "MCPErrorCode",
    "MCPRequest",
    "MCPResponse",
    "MCPCapabilities",
    # Registry
    "MCPServerRegistry",
    # Server
    "MCPServer",
    # Auth
    "MCPAuthenticator",
    "MCPAPIKeyAuth",
]
