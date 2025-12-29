"""Pydantic models for CopilotKit AG-UI protocol."""

from enum import Enum
from typing import Any, Optional
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
