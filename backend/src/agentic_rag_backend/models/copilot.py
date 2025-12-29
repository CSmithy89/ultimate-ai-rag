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
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
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
