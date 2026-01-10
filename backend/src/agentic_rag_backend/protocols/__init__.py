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
from .a2ui import (
    A2UIWidget,
    A2UIAction,
    A2UIFormField,
    create_a2ui_card,
    create_a2ui_table,
    create_a2ui_form,
    create_a2ui_chart,
    create_a2ui_image,
    create_a2ui_list,
    widgets_to_state,
)

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
    # A2UI Protocol (Epic 21)
    "A2UIWidget",
    "A2UIAction",
    "A2UIFormField",
    "create_a2ui_card",
    "create_a2ui_table",
    "create_a2ui_form",
    "create_a2ui_chart",
    "create_a2ui_image",
    "create_a2ui_list",
    "widgets_to_state",
]
