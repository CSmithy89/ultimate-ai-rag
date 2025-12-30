"""SDK response models for MCP and A2A APIs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MCPToolDescriptor(BaseModel):
    """Descriptor for an MCP tool."""
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPToolList(BaseModel):
    """Tool discovery response."""
    tools: list[MCPToolDescriptor]
    meta: dict[str, Any] | None = None


class MCPToolCallResult(BaseModel):
    """Tool invocation response."""
    tool: str
    result: dict[str, Any]
    meta: dict[str, Any] | None = None


class A2AMessage(BaseModel):
    """A2A session message."""
    sender: str
    content: str
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ASession(BaseModel):
    """A2A session transcript."""
    session_id: str
    tenant_id: str
    created_at: str
    messages: list[A2AMessage] = Field(default_factory=list)


class A2ASessionEnvelope(BaseModel):
    """A2A session response envelope."""
    session: A2ASession
    meta: dict[str, Any] | None = None
