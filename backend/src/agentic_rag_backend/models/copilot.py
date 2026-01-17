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


# ============================================
# MULTIMODAL CONTENT TYPES - Phase 7.2
# For handling files, images, audio, and mixed content
# ============================================


class TextInputContent(BaseModel):
    """Text content in a multimodal message.

    Used for text portions of messages that may also contain
    binary content like images or audio.
    """
    type: Literal["text"] = "text"
    content: str = Field(..., description="The text content")


class BinaryInputContent(BaseModel):
    """Binary content (files, images, audio) in a multimodal message.

    AG-UI Protocol: Binary content is base64-encoded and includes
    a media type for proper handling.

    Supported media types:
    - Images: image/png, image/jpeg, image/gif, image/webp
    - Audio: audio/wav, audio/mp3, audio/ogg
    - Documents: application/pdf, text/plain
    """
    type: Literal["binary"] = "binary"
    media_type: str = Field(..., description="MIME type (e.g., 'image/png', 'audio/wav')")
    data: str = Field(..., description="Base64-encoded binary data")
    filename: Optional[str] = Field(None, description="Original filename if available")

    def get_size_bytes(self) -> int:
        """Calculate the approximate size of the binary data in bytes."""
        import base64
        # Base64 encoding increases size by ~33%
        return len(base64.b64decode(self.data))


class MultimodalMessage(BaseModel):
    """A message containing mixed text and binary content.

    AG-UI Protocol: Multimodal messages support rich content types
    for use cases like image analysis, audio transcription, and
    document processing.

    Example:
        message = MultimodalMessage(
            role=MessageRole.USER,
            content=[
                TextInputContent(content="What's in this image?"),
                BinaryInputContent(
                    media_type="image/png",
                    data="base64encodeddata...",
                    filename="screenshot.png"
                ),
            ]
        )
    """
    role: MessageRole
    content: list[TextInputContent | BinaryInputContent] = Field(
        ..., description="List of content parts (text and/or binary)"
    )

    def get_text_content(self) -> str:
        """Extract all text content from the message."""
        return " ".join(
            part.content for part in self.content
            if isinstance(part, TextInputContent)
        )

    def get_binary_content(self) -> list[BinaryInputContent]:
        """Extract all binary content from the message."""
        return [
            part for part in self.content
            if isinstance(part, BinaryInputContent)
        ]

    def has_images(self) -> bool:
        """Check if message contains any image content."""
        return any(
            isinstance(part, BinaryInputContent) and part.media_type.startswith("image/")
            for part in self.content
        )

    def has_audio(self) -> bool:
        """Check if message contains any audio content."""
        return any(
            isinstance(part, BinaryInputContent) and part.media_type.startswith("audio/")
            for part in self.content
        )


class CopilotConfig(BaseModel):
    """Configuration for CopilotKit request."""
    configurable: dict[str, Any] = Field(default_factory=dict)


class CopilotRequest(BaseModel):
    """Request payload from CopilotKit."""
    messages: list[CopilotMessage] = Field(default_factory=list)
    config: Optional[CopilotConfig] = None
    actions: list[dict[str, Any]] = Field(default_factory=list)


class AGUIEventType(str, Enum):
    """AG-UI event types.

    Full AG-UI Protocol Event Types:
    - Run lifecycle: RUN_STARTED, RUN_FINISHED, RUN_ERROR
    - Text messages: TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, TEXT_MESSAGE_END
    - Tool calls: TOOL_CALL_START, TOOL_CALL_ARGS, TOOL_CALL_RESULT, TOOL_CALL_END
    - State management: STATE_SNAPSHOT, STATE_DELTA
    - Message history: MESSAGES_SNAPSHOT
    - Activities: ACTIVITY_SNAPSHOT, ACTIVITY_DELTA
    - Agent reasoning: THINKING_START, THINKING_TEXT_MESSAGE_CONTENT, THINKING_END
    - Custom events: CUSTOM
    - Actions: ACTION_REQUEST
    """
    # Run lifecycle events
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"  # Story 22-B2: Extended error events

    # Text message events
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"

    # Tool call events
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    TOOL_CALL_END = "TOOL_CALL_END"

    # State management events
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"

    # Message history events
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"

    # Activity tracking events (for long-running operations)
    ACTIVITY_SNAPSHOT = "ACTIVITY_SNAPSHOT"
    ACTIVITY_DELTA = "ACTIVITY_DELTA"

    # Agent reasoning/thinking events
    THINKING_START = "THINKING_START"
    THINKING_TEXT_MESSAGE_CONTENT = "THINKING_TEXT_MESSAGE_CONTENT"
    THINKING_END = "THINKING_END"

    # Custom events for application-specific data
    CUSTOM = "CUSTOM"

    # Action request events
    ACTION_REQUEST = "ACTION_REQUEST"

    # RAW events for external protocol wrapping
    RAW = "RAW"


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


class ToolCallResultEvent(AGUIEvent):
    """Event containing the result of a tool call.

    AG-UI Protocol: TOOL_CALL_RESULT should be emitted after tool execution
    completes and before TOOL_CALL_END. The result field contains the
    JSON-serialized tool output.
    """
    type: Literal[AGUIEventType.TOOL_CALL_RESULT] = AGUIEventType.TOOL_CALL_RESULT
    toolCallId: str = ""
    result: str = ""  # JSON-serialized result string

    def __init__(
        self,
        tool_call_id: str = "",
        result: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(toolCallId=tool_call_id, result=result or "", **kwargs)


# ============================================
# STATE DELTA EVENTS - AG-UI Protocol Enhancement
# ============================================


class StateDeltaEvent(AGUIEvent):
    """Event for incremental state updates using RFC 6902 JSON Patch format.

    AG-UI Protocol: STATE_DELTA provides efficient incremental state updates
    instead of full STATE_SNAPSHOT events. Use for frequent small updates
    like step status changes.

    JSON Patch operations:
    - {"op": "add", "path": "/steps/-", "value": {...}}
    - {"op": "replace", "path": "/steps/0/status", "value": "completed"}
    - {"op": "remove", "path": "/steps/0"}

    Example:
        yield StateDeltaEvent(delta=[
            {"op": "replace", "path": "/steps/0/status", "value": "completed"},
            {"op": "replace", "path": "/currentStep", "value": 1},
        ])
    """
    type: Literal[AGUIEventType.STATE_DELTA] = AGUIEventType.STATE_DELTA
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    delta: list[dict[str, Any]] = Field(default_factory=list)

    def __init__(
        self,
        delta: Optional[list[dict[str, Any]]] = None,
        threadId: Optional[str] = None,
        runId: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if delta is not None:
            kwargs["delta"] = delta
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(**kwargs)


# ============================================
# MESSAGES SNAPSHOT EVENT - AG-UI Protocol Enhancement
# ============================================


class MessagesSnapshotEvent(AGUIEvent):
    """Event for syncing full conversation history.

    AG-UI Protocol: MESSAGES_SNAPSHOT provides the full conversation history
    to the frontend for synchronization. Useful after reconnection or
    when frontend needs to restore state.
    """
    type: Literal[AGUIEventType.MESSAGES_SNAPSHOT] = AGUIEventType.MESSAGES_SNAPSHOT
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    messages: list[dict[str, Any]] = Field(default_factory=list)

    def __init__(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        threadId: Optional[str] = None,
        runId: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if messages is not None:
            kwargs["messages"] = messages
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(**kwargs)


# ============================================
# ACTIVITY EVENTS - AG-UI Protocol Enhancement
# For long-running operations like document ingestion
# ============================================


class ActivitySnapshotEvent(AGUIEvent):
    """Event for tracking long-running operation progress.

    AG-UI Protocol: ACTIVITY_SNAPSHOT provides full activity state for
    long-running operations. Use for initial activity state or major changes.

    Activity structure:
        {
            "id": "activity-uuid",
            "type": "indexing" | "search" | "processing",
            "progress": 0.0 to 1.0,
            "message": "Processing page 3/7...",
            "metadata": {...}
        }
    """
    type: Literal[AGUIEventType.ACTIVITY_SNAPSHOT] = AGUIEventType.ACTIVITY_SNAPSHOT
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    activity: dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        activity: Optional[dict[str, Any]] = None,
        threadId: Optional[str] = None,
        runId: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if activity is not None:
            kwargs["activity"] = activity
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(**kwargs)


class ActivityDeltaEvent(AGUIEvent):
    """Event for incremental activity progress updates.

    AG-UI Protocol: ACTIVITY_DELTA provides efficient incremental updates
    using RFC 6902 JSON Patch format.

    Example:
        yield ActivityDeltaEvent(delta=[
            {"op": "replace", "path": "/progress", "value": 0.45},
            {"op": "replace", "path": "/message", "value": "Processing page 3/7..."},
        ])
    """
    type: Literal[AGUIEventType.ACTIVITY_DELTA] = AGUIEventType.ACTIVITY_DELTA
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    delta: list[dict[str, Any]] = Field(default_factory=list)

    def __init__(
        self,
        delta: Optional[list[dict[str, Any]]] = None,
        threadId: Optional[str] = None,
        runId: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if delta is not None:
            kwargs["delta"] = delta
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(**kwargs)


# ============================================
# THINKING EVENTS - AG-UI Protocol Enhancement
# For exposing agent reasoning to the frontend
# ============================================


class ThinkingStartEvent(AGUIEvent):
    """Event signaling start of agent reasoning/thinking phase.

    AG-UI Protocol: THINKING_START marks the beginning of an agent's
    internal reasoning process that will be exposed to the user.
    """
    type: Literal[AGUIEventType.THINKING_START] = AGUIEventType.THINKING_START
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")

    def __init__(
        self,
        threadId: Optional[str] = None,
        runId: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(**kwargs)


class ThinkingTextMessageContentEvent(AGUIEvent):
    """Event containing agent reasoning content.

    AG-UI Protocol: THINKING_TEXT_MESSAGE_CONTENT streams the agent's
    reasoning/thinking content to the frontend for display.
    """
    type: Literal[AGUIEventType.THINKING_TEXT_MESSAGE_CONTENT] = AGUIEventType.THINKING_TEXT_MESSAGE_CONTENT
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    content: str = ""

    def __init__(
        self,
        content: str = "",
        threadId: Optional[str] = None,
        runId: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(content=content, **kwargs)


class ThinkingEndEvent(AGUIEvent):
    """Event signaling end of agent reasoning/thinking phase.

    AG-UI Protocol: THINKING_END marks the completion of an agent's
    internal reasoning process.
    """
    type: Literal[AGUIEventType.THINKING_END] = AGUIEventType.THINKING_END
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")

    def __init__(
        self,
        threadId: Optional[str] = None,
        runId: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(**kwargs)


# ============================================
# CUSTOM EVENTS - AG-UI Protocol Enhancement
# For application-specific events
# ============================================


class CustomEvent(AGUIEvent):
    """Event for application-specific custom data.

    AG-UI Protocol: CUSTOM events allow sending arbitrary application-specific
    data through the AG-UI stream. Can be used for declarative UI rendering,
    custom widgets, or any domain-specific events.

    Example:
        yield CustomEvent(
            name="render_ui",
            value={
                "type": "approval_dialog",
                "props": {"title": "Approve Sources", "items": sources}
            }
        )
    """
    type: Literal[AGUIEventType.CUSTOM] = AGUIEventType.CUSTOM
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    name: str = ""
    value: dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        name: str = "",
        value: Optional[dict[str, Any]] = None,
        threadId: Optional[str] = None,
        runId: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if value is not None:
            kwargs["value"] = value
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(name=name, **kwargs)


# ============================================
# RAW EVENTS - Phase 7.3
# For wrapping external protocol events (MCP, A2A)
# ============================================


class RawEvent(AGUIEvent):
    """Event for wrapping external protocol events in AG-UI stream.

    AG-UI Protocol: RAW events allow passing through events from external
    protocols (like MCP or A2A) without modification, while maintaining
    the AG-UI event stream format.

    Use cases:
    - Wrapping MCP (Model Context Protocol) tool responses
    - Forwarding A2A (Agent-to-Agent) delegation events
    - Passing through events from external agent frameworks

    Example:
        # Wrap an MCP tool response
        yield RawEvent(
            event=mcp_response.model_dump(),
            source="mcp",
            protocol_version="1.0"
        )

        # Wrap an A2A delegation event
        yield RawEvent(
            event=a2a_event.to_dict(),
            source="a2a",
            metadata={"delegated_agent": "research_agent"}
        )
    """
    type: Literal[AGUIEventType.RAW] = AGUIEventType.RAW
    threadId: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:12]}")
    runId: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    event: dict[str, Any] = Field(default_factory=dict, description="The wrapped external event")
    source: Optional[str] = Field(None, description="Source protocol identifier (e.g., 'mcp', 'a2a')")
    protocol_version: Optional[str] = Field(None, description="Version of the source protocol")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def __init__(
        self,
        event: Optional[dict[str, Any]] = None,
        source: Optional[str] = None,
        protocol_version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        threadId: Optional[str] = None,
        runId: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if event is not None:
            kwargs["event"] = event
        if source is not None:
            kwargs["source"] = source
        if protocol_version is not None:
            kwargs["protocol_version"] = protocol_version
        if metadata is not None:
            kwargs["metadata"] = metadata
        if threadId is not None:
            kwargs["threadId"] = threadId
        if runId is not None:
            kwargs["runId"] = runId
        super().__init__(**kwargs)


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
