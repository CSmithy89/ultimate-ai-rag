"""Pydantic models for CopilotKit AG-UI protocol."""

import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class CopilotMessage(BaseModel):
    """A message in the CopilotKit conversation."""
    role: MessageRole
    content: str


class CopilotConfig(BaseModel):
    """Configuration for CopilotKit request."""
    configurable: dict[str, Any] = Field(default_factory=dict)


class CopilotRequest(BaseModel):
    """Request payload from CopilotKit."""
    messages: list[CopilotMessage] = Field(default_factory=list)
    config: Optional[CopilotConfig] = None
    actions: list[dict[str, Any]] = Field(default_factory=list)


class AGUIEventType(str, Enum):
    """AG-UI event types."""
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"  # Story 21-B2: Error event support
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"  # Story 21-B3: Incremental state updates
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"  # Story 21-B3: Message history sync
    CUSTOM_EVENT = "CUSTOM"  # Story 21-B3: Application-specific events
    ACTION_REQUEST = "ACTION_REQUEST"


class AGUIEvent(BaseModel):
    """Base AG-UI event."""
    event: AGUIEventType
    data: dict[str, Any] = Field(default_factory=dict)


class RunStartedEvent(AGUIEvent):
    """Event emitted when agent run starts."""
    event: AGUIEventType = AGUIEventType.RUN_STARTED


class RunFinishedEvent(AGUIEvent):
    """Event emitted when agent run finishes."""
    event: AGUIEventType = AGUIEventType.RUN_FINISHED


# ============================================
# ERROR EVENTS - Story 21-B2
# ============================================


class RunErrorCode(str, Enum):
    """Standard error codes for RUN_ERROR events.

    Story 21-B2: Add RUN_ERROR Event Support

    Error codes follow a structured naming convention:
    - AGENT_*: Errors during agent execution
    - AUTH_*: Authentication/authorization errors
    - RATE_*: Rate limiting errors
    - REQUEST_*: Invalid request errors
    """
    AGENT_EXECUTION_ERROR = "AGENT_EXECUTION_ERROR"
    TENANT_REQUIRED = "TENANT_REQUIRED"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    INVALID_REQUEST = "INVALID_REQUEST"


class RunErrorEvent(AGUIEvent):
    """Event emitted when agent run fails with error.

    Story 21-B2: Add RUN_ERROR Event Support

    Instead of embedding errors in text messages, this event provides
    structured error information that the frontend can handle gracefully.

    Attributes:
        code: Error code from RunErrorCode enum
        message: User-friendly error message
        details: Optional technical details (hidden in production)
    """
    event: AGUIEventType = AGUIEventType.RUN_ERROR

    def __init__(
        self,
        code: str | RunErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a RUN_ERROR event.

        Args:
            code: Error code (from RunErrorCode or custom string)
            message: User-friendly error message
            details: Optional technical details for debugging
        """
        # Bug fix: Only include details field when it has content
        data = {
            "code": code.value if isinstance(code, RunErrorCode) else code,
            "message": message,
        }
        if details:
            data["details"] = details
        super().__init__(data=data, **kwargs)


class TextDeltaEvent(AGUIEvent):
    """Event for streaming text content."""
    event: AGUIEventType = AGUIEventType.TEXT_MESSAGE_CONTENT

    def __init__(self, content: str, **kwargs: Any) -> None:
        super().__init__(data={"content": content}, **kwargs)


class StateSnapshotEvent(AGUIEvent):
    """Event for agent state updates."""
    event: AGUIEventType = AGUIEventType.STATE_SNAPSHOT

    def __init__(self, state: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(data={"state": state}, **kwargs)


# ============================================
# STATE DELTA EVENTS - Story 21-B3
# ============================================


class StateDeltaEvent(AGUIEvent):
    """Event for incremental state updates using JSON Patch.

    Story 21-B3: Implement STATE_DELTA and MESSAGES_SNAPSHOT Support

    Instead of replacing the entire state (STATE_SNAPSHOT), STATE_DELTA
    applies incremental JSON Patch operations (RFC 6902) for efficiency.
    Useful for real-time progress updates without full state transfer.

    Example operations:
        [{"op": "add", "path": "/steps/-", "value": {"step": "Searching...", "status": "in_progress"}}]
        [{"op": "replace", "path": "/currentStep", "value": "processing"}]
        [{"op": "remove", "path": "/steps/0"}]
    """
    event: AGUIEventType = AGUIEventType.STATE_DELTA

    def __init__(self, operations: list[dict[str, Any]], **kwargs: Any) -> None:
        """Create a STATE_DELTA event.

        Args:
            operations: JSON Patch operations (RFC 6902)
                Supported ops: add, remove, replace, move, copy, test
        """
        super().__init__(data={"delta": operations}, **kwargs)


class MessagesSnapshotEvent(AGUIEvent):
    """Event for syncing full message history.

    Story 21-B3: Implement STATE_DELTA and MESSAGES_SNAPSHOT Support

    MESSAGES_SNAPSHOT syncs the entire message history, useful for:
    - Session restoration after reconnection
    - Multi-tab synchronization
    - Chat history persistence across page reloads
    """
    event: AGUIEventType = AGUIEventType.MESSAGES_SNAPSHOT

    def __init__(self, messages: list[dict[str, Any]], **kwargs: Any) -> None:
        """Create a MESSAGES_SNAPSHOT event.

        Args:
            messages: Full message history
                [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
        """
        super().__init__(data={"messages": messages}, **kwargs)


class CustomEvent(AGUIEvent):
    """Event for application-specific custom events.

    Story 21-B3: Implement STATE_DELTA and MESSAGES_SNAPSHOT Support

    CUSTOM events enable application-specific functionality that isn't covered
    by standard AG-UI events. Use for:
    - A2UI widget updates
    - Application-specific notifications
    - Custom state synchronization
    - Third-party integration events

    Example event names:
        - a2ui_widget_update: Dynamic widget updates
        - progress_update: Long-running task progress
        - notification: User notifications
        - analytics_event: Frontend analytics triggers
    """
    event: AGUIEventType = AGUIEventType.CUSTOM_EVENT

    def __init__(
        self,
        event_name: str,
        payload: dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Create a CUSTOM event.

        Args:
            event_name: Application-specific event type (e.g., "a2ui_widget_update")
            payload: Event-specific data
        """
        super().__init__(
            data={"name": event_name, "payload": payload},
            **kwargs
        )


class ToolCallEvent(AGUIEvent):
    """Event for tool invocations."""
    event: AGUIEventType = AGUIEventType.TOOL_CALL_START

    def __init__(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(
            data={"tool_name": tool_name, "args": args},
            **kwargs
        )


class ActionRequestEvent(AGUIEvent):
    """Event requesting frontend action."""
    event: AGUIEventType = AGUIEventType.ACTION_REQUEST

    def __init__(self, action: str, args: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(
            data={"action": action, "args": args},
            **kwargs
        )


class TextMessageStartEvent(AGUIEvent):
    """Event signaling start of text message streaming."""
    event: AGUIEventType = AGUIEventType.TEXT_MESSAGE_START


class TextMessageEndEvent(AGUIEvent):
    """Event signaling end of text message streaming."""
    event: AGUIEventType = AGUIEventType.TEXT_MESSAGE_END


# ============================================
# GENERATIVE UI EVENTS - Story 6-3
# ============================================


class ToolCallStartEvent(AGUIEvent):
    """Event for triggering a tool/action call that may render UI."""

    event: Literal[AGUIEventType.TOOL_CALL_START] = AGUIEventType.TOOL_CALL_START

    def __init__(self, tool_call_id: str, tool_name: str, **kwargs: Any) -> None:
        super().__init__(
            data={"tool_call_id": tool_call_id, "tool_name": tool_name},
            **kwargs
        )


class ToolCallArgsEvent(AGUIEvent):
    """Event containing arguments for a tool call."""

    event: Literal[AGUIEventType.TOOL_CALL_ARGS] = AGUIEventType.TOOL_CALL_ARGS

    def __init__(self, tool_call_id: str, args: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(
            data={"tool_call_id": tool_call_id, "args": args},
            **kwargs
        )


class ToolCallEndEvent(AGUIEvent):
    """Event indicating tool call completion."""

    event: Literal[AGUIEventType.TOOL_CALL_END] = AGUIEventType.TOOL_CALL_END

    def __init__(self, tool_call_id: str, **kwargs: Any) -> None:
        super().__init__(
            data={"tool_call_id": tool_call_id},
            **kwargs
        )


# ============================================
# GENERATIVE UI HELPER FUNCTIONS - Story 6-3
# ============================================


def create_show_sources_events(
    sources: list[dict[str, Any]],
    title: str | None = None,
) -> list[AGUIEvent]:
    """Create AG-UI events to trigger show_sources action.

    Args:
        sources: List of source dictionaries with id, title, preview, similarity
        title: Optional title for the sources section

    Returns:
        List of AG-UI events to emit
    """
    tool_call_id = str(uuid.uuid4())
    return [
        ToolCallStartEvent(tool_call_id=tool_call_id, tool_name="show_sources"),
        ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            args={"sources": sources, "title": title},
        ),
        ToolCallEndEvent(tool_call_id=tool_call_id),
    ]


def create_show_answer_events(
    answer: str,
    sources: list[dict[str, Any]] | None = None,
    title: str | None = None,
) -> list[AGUIEvent]:
    """Create AG-UI events to trigger show_answer action.

    Args:
        answer: The answer text with optional markdown formatting
        sources: Optional sources referenced in the answer
        title: Optional title for the answer panel

    Returns:
        List of AG-UI events to emit
    """
    tool_call_id = str(uuid.uuid4())
    return [
        ToolCallStartEvent(tool_call_id=tool_call_id, tool_name="show_answer"),
        ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            args={"answer": answer, "sources": sources, "title": title},
        ),
        ToolCallEndEvent(tool_call_id=tool_call_id),
    ]


def create_show_knowledge_graph_events(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    title: str | None = None,
) -> list[AGUIEvent]:
    """Create AG-UI events to trigger show_knowledge_graph action.

    Args:
        nodes: List of node dictionaries with id, label, and optional type
        edges: List of edge dictionaries with id, source, target, and optional label
        title: Optional title for the graph

    Returns:
        List of AG-UI events to emit
    """
    tool_call_id = str(uuid.uuid4())
    return [
        ToolCallStartEvent(tool_call_id=tool_call_id, tool_name="show_knowledge_graph"),
        ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            args={"nodes": nodes, "edges": edges, "title": title},
        ),
        ToolCallEndEvent(tool_call_id=tool_call_id),
    ]
