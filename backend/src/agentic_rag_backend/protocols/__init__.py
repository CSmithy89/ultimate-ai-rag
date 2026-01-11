"""Protocol handlers for agent communication."""

from .ag_ui_bridge import AGUIBridge
from .mcp import MCPToolRegistry
from .a2a import A2ASessionManager
from .a2a_messages import (
    A2AMessageType,
    TaskStatus,
    AgentCapability,
    AgentRegistration,
    TaskRequest,
    TaskResult,
    get_rag_capabilities,
)
from .a2a_registry import A2AAgentRegistry, RegistryConfig
from .a2a_delegation import TaskDelegationManager, DelegationConfig
from .a2a_middleware import A2AMiddlewareAgent, A2AAgentCapability, A2AAgentInfo

__all__ = [
    "AGUIBridge",
    "MCPToolRegistry",
    "A2ASessionManager",
    # A2A Protocol (Epic 14)
    "A2AMessageType",
    "TaskStatus",
    "AgentCapability",
    "AgentRegistration",
    "TaskRequest",
    "TaskResult",
    "get_rag_capabilities",
    "A2AAgentRegistry",
    "RegistryConfig",
    "TaskDelegationManager",
    "DelegationConfig",
    # A2A Middleware (Story 22-A1)
    "A2AMiddlewareAgent",
    "A2AAgentCapability",
    "A2AAgentInfo",
]
