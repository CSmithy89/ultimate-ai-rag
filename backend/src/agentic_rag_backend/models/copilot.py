"""Pydantic models for CopilotKit AG-UI protocol.

AG-UI Protocol Event Format:
- Events use 'type' as the discriminator field (not 'event')
- Event-specific data is at the top level (not nested under 'data')
- Example: {"type": "TEXT_MESSAGE_START", "messageId": "...", "role": "assistant"}
"""

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
    RUN_ERROR = "RUN_ERROR"  # Story 22-B2: Extended error events
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    ACTION_REQUEST = "ACTION_REQUEST"


def _generate_message_id() -> str:
    """Generate a unique message ID for AG-UI events."""
    return f"msg-{uuid.uuid4().hex[:12]}"


class AGUIEvent(BaseModel):
    """Base AG-UI event.

    AG-UI Protocol uses 'type' as the discriminator field.
    All event-specific fields are at the top level.
    """
    type: AGUIEventType = Field(..., description="Event type discriminator")

    class Config:
        # Use 'type' as the field name in JSON serialization
        populate_by_name = True


class RunStartedEvent(AGUIEvent):
    """Event emitted when agent run starts."""
    type: Literal[AGUIEventType.RUN_STARTED] = AGUIEventType.RUN_STARTED
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")

    def __init__(self, threadId: Optional[str] = None, runId: Optional[str] = None, **kwargs: Any) -> None:
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(**kwargs)


class RunFinishedEvent(AGUIEvent):
    """Event emitted when agent run finishes."""
    type: Literal[AGUIEventType.RUN_FINISHED] = AGUIEventType.RUN_FINISHED
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")

    def __init__(self, threadId: Optional[str] = None, runId: Optional[str] = None, **kwargs: Any) -> None:
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(**kwargs)


class TextDeltaEvent(AGUIEvent):
    """Event for streaming text content (TEXT_MESSAGE_CONTENT)."""
    type: Literal[AGUIEventType.TEXT_MESSAGE_CONTENT] = AGUIEventType.TEXT_MESSAGE_CONTENT
    messageId: str = Field(default_factory=_generate_message_id)
    delta: str = ""

    def __init__(self, content: str = "", messageId: Optional[str] = None, **kwargs: Any) -> None:
        # AG-UI protocol uses 'delta' field for text content chunks at top level
        if messageId is not None:
            kwargs["messageId"] = messageId
        super().__init__(delta=content, **kwargs)


class StateSnapshotEvent(AGUIEvent):
    """Event for agent state updates."""
    type: Literal[AGUIEventType.STATE_SNAPSHOT] = AGUIEventType.STATE_SNAPSHOT
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    snapshot: dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        state: Optional[dict[str, Any]] = None,
        threadId: Optional[str] = None,
        runId: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # AG-UI protocol: state data goes in 'snapshot' field at top level
        # AG-UI protocol: threadId and runId are required on all events
        if state is not None:
            kwargs["snapshot"] = state
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(**kwargs)


class ToolCallEvent(AGUIEvent):
    """Event for tool invocations."""
    type: Literal[AGUIEventType.TOOL_CALL_START] = AGUIEventType.TOOL_CALL_START
    toolCallId: str = Field(default_factory=lambda: f"call-{uuid.uuid4().hex[:12]}")
    toolCallName: str = ""  # AG-UI protocol uses toolCallName, not toolName
    args: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, tool_name: str = "", args: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(toolCallName=tool_name, args=args or {}, **kwargs)


class ActionRequestEvent(AGUIEvent):
    """Event requesting frontend action."""
    type: Literal[AGUIEventType.ACTION_REQUEST] = AGUIEventType.ACTION_REQUEST
    action: str = ""
    args: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, action: str = "", args: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(action=action, args=args or {}, **kwargs)


class TextMessageStartEvent(AGUIEvent):
    """Event signaling start of text message streaming."""
    type: Literal[AGUIEventType.TEXT_MESSAGE_START] = AGUIEventType.TEXT_MESSAGE_START
    messageId: str = Field(default_factory=_generate_message_id)
    role: str = "assistant"

    def __init__(self, role: str = "assistant", messageId: Optional[str] = None, **kwargs: Any) -> None:
        # AG-UI protocol: role specifies the sender (defaults to 'assistant')
        if messageId is not None:
            kwargs["messageId"] = messageId
        super().__init__(role=role, **kwargs)


class TextMessageEndEvent(AGUIEvent):
    """Event signaling end of text message streaming."""
    type: Literal[AGUIEventType.TEXT_MESSAGE_END] = AGUIEventType.TEXT_MESSAGE_END
    messageId: str = Field(default_factory=_generate_message_id)

    def __init__(self, messageId: Optional[str] = None, **kwargs: Any) -> None:
        if messageId is not None:
            kwargs["messageId"] = messageId
        super().__init__(**kwargs)


# ============================================
# GENERATIVE UI EVENTS - Story 6-3
# ============================================


class ToolCallStartEvent(AGUIEvent):
    """Event for triggering a tool/action call that may render UI."""
    type: Literal[AGUIEventType.TOOL_CALL_START] = AGUIEventType.TOOL_CALL_START
    toolCallId: str = Field(default_factory=lambda: f"call-{uuid.uuid4().hex[:12]}")
    toolCallName: str = ""  # AG-UI protocol uses toolCallName, not toolName

    def __init__(self, tool_call_id: Optional[str] = None, tool_name: str = "", **kwargs: Any) -> None:
        if tool_call_id is not None:
            kwargs["toolCallId"] = tool_call_id
        super().__init__(toolCallName=tool_name, **kwargs)


class ToolCallArgsEvent(AGUIEvent):
    """Event containing arguments for a tool call.

    AG-UI protocol uses 'delta' (JSON string), not 'args' (dict).
    The delta field contains a JSON-serialized string of the arguments.
    """
    type: Literal[AGUIEventType.TOOL_CALL_ARGS] = AGUIEventType.TOOL_CALL_ARGS
    toolCallId: str = ""
    delta: str = ""  # AG-UI protocol: delta is a JSON string

    def __init__(self, tool_call_id: str = "", args: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        import json
        # AG-UI protocol expects delta as a JSON string
        delta_str = json.dumps(args or {})
        super().__init__(toolCallId=tool_call_id, delta=delta_str, **kwargs)


class ToolCallEndEvent(AGUIEvent):
    """Event indicating tool call completion."""
    type: Literal[AGUIEventType.TOOL_CALL_END] = AGUIEventType.TOOL_CALL_END
    toolCallId: str = ""

    def __init__(self, tool_call_id: str = "", **kwargs: Any) -> None:
        super().__init__(toolCallId=tool_call_id, **kwargs)


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
