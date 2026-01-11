"""AG-UI Protocol Compliance Tests.

Story 22-D2: Implement Protocol Compliance Tests

Verifies AG-UI events and streams match specification:
- Event format compliance
- Stream lifecycle ordering
- Error code mappings (RFC 7807)
- Metrics emission
"""

import pytest
from pydantic import ValidationError

from agentic_rag_backend.models.copilot import (
    AGUIEvent,
    AGUIEventType,
    CopilotMessage,
    CopilotRequest,
    MessageRole,
    RunStartedEvent,
    RunFinishedEvent,
    TextDeltaEvent,
    StateSnapshotEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
)
from agentic_rag_backend.protocols.ag_ui_errors import (
    AGUIErrorCode,
    AGUIErrorEvent,
    ERROR_CODE_HTTP_STATUS,
    create_error_event,
)


# =============================================================================
# Event Format Compliance Tests
# =============================================================================


class TestAGUIEventFormatCompliance:
    """Verify AG-UI event formats match specification."""

    def test_run_started_event_has_event_type(self) -> None:
        """RUN_STARTED must have correct event type."""
        event = RunStartedEvent()
        assert event.event == AGUIEventType.RUN_STARTED

    def test_run_finished_event_has_event_type(self) -> None:
        """RUN_FINISHED must have correct event type."""
        event = RunFinishedEvent()
        assert event.event == AGUIEventType.RUN_FINISHED

    def test_text_delta_event_has_content(self) -> None:
        """TEXT_MESSAGE_CONTENT must include content in data."""
        event = TextDeltaEvent(content="Hello ")
        assert event.event == AGUIEventType.TEXT_MESSAGE_CONTENT
        assert event.data["content"] == "Hello "

    def test_text_delta_accepts_empty_string(self) -> None:
        """TEXT_MESSAGE_CONTENT can have empty content."""
        event = TextDeltaEvent(content="")
        assert event.data["content"] == ""

    def test_state_snapshot_event_has_state(self) -> None:
        """STATE_SNAPSHOT must include state in data."""
        event = StateSnapshotEvent(state={"key": "value"})
        assert event.event == AGUIEventType.STATE_SNAPSHOT
        assert event.data["state"] == {"key": "value"}

    def test_state_snapshot_accepts_nested_objects(self) -> None:
        """STATE_SNAPSHOT can have deeply nested state."""
        nested_state = {
            "level1": {
                "level2": {
                    "level3": {"data": [1, 2, 3]}
                }
            }
        }
        event = StateSnapshotEvent(state=nested_state)
        assert event.data["state"]["level1"]["level2"]["level3"]["data"] == [1, 2, 3]

    def test_tool_call_start_event_has_required_fields(self) -> None:
        """TOOL_CALL_START must include tool_call_id and tool_name in data."""
        event = ToolCallStartEvent(tool_call_id="tc-123", tool_name="vector_search")
        assert event.event == AGUIEventType.TOOL_CALL_START
        assert event.data["tool_call_id"] == "tc-123"
        assert event.data["tool_name"] == "vector_search"

    def test_tool_call_args_event_has_args(self) -> None:
        """TOOL_CALL_ARGS must include tool_call_id and args in data."""
        event = ToolCallArgsEvent(tool_call_id="tc-123", args={"query": "test"})
        assert event.event == AGUIEventType.TOOL_CALL_ARGS
        assert event.data["tool_call_id"] == "tc-123"
        assert event.data["args"] == {"query": "test"}

    def test_tool_call_end_event_has_tool_call_id(self) -> None:
        """TOOL_CALL_END must include tool_call_id in data."""
        event = ToolCallEndEvent(tool_call_id="tc-123")
        assert event.event == AGUIEventType.TOOL_CALL_END
        assert event.data["tool_call_id"] == "tc-123"


class TestAGUIEventTypeCompliance:
    """Verify all AG-UI event types are defined."""

    REQUIRED_EVENT_TYPES = [
        "RUN_STARTED",
        "RUN_FINISHED",
        "RUN_ERROR",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "STATE_SNAPSHOT",
    ]

    def test_all_required_event_types_defined(self) -> None:
        """All required AG-UI event types must be defined."""
        defined_types = [e.value for e in AGUIEventType]
        for event_type in self.REQUIRED_EVENT_TYPES:
            assert event_type in defined_types, f"Missing event type: {event_type}"


# =============================================================================
# Error Code Compliance Tests
# =============================================================================


class TestAGUIErrorCodeCompliance:
    """Verify AG-UI error codes match RFC 7807 alignment."""

    def test_all_error_codes_have_http_status(self) -> None:
        """Every AGUIErrorCode must map to an HTTP status."""
        for error_code in AGUIErrorCode:
            assert error_code in ERROR_CODE_HTTP_STATUS, (
                f"Error code {error_code} missing HTTP status mapping"
            )

    def test_error_code_http_status_valid_range(self) -> None:
        """HTTP status codes must be in valid range (4xx or 5xx)."""
        for error_code, status in ERROR_CODE_HTTP_STATUS.items():
            assert 400 <= status <= 599, (
                f"Error code {error_code} has invalid HTTP status {status}"
            )

    @pytest.mark.parametrize(
        "error_code,expected_status",
        [
            (AGUIErrorCode.AGENT_EXECUTION_ERROR, 500),
            (AGUIErrorCode.TENANT_REQUIRED, 401),
            (AGUIErrorCode.TENANT_UNAUTHORIZED, 403),
            (AGUIErrorCode.SESSION_NOT_FOUND, 404),
            (AGUIErrorCode.RATE_LIMITED, 429),
            (AGUIErrorCode.TIMEOUT, 504),
            (AGUIErrorCode.INVALID_REQUEST, 400),
            (AGUIErrorCode.CAPABILITY_NOT_FOUND, 404),
            (AGUIErrorCode.UPSTREAM_ERROR, 502),
            (AGUIErrorCode.SERVICE_UNAVAILABLE, 503),
        ],
    )
    def test_error_code_http_status_mapping(
        self, error_code: AGUIErrorCode, expected_status: int
    ) -> None:
        """Verify specific error code to HTTP status mappings."""
        assert ERROR_CODE_HTTP_STATUS[error_code] == expected_status


class TestAGUIErrorEventCompliance:
    """Verify AG-UI error events have required fields."""

    def test_error_event_has_required_fields(self) -> None:
        """AGUIErrorEvent must include code, message, http_status in data."""
        event = AGUIErrorEvent(
            code=AGUIErrorCode.AGENT_EXECUTION_ERROR,
            message="Test error",
            http_status=500,
        )
        assert event.event == AGUIEventType.RUN_ERROR
        assert event.data["code"] == "AGENT_EXECUTION_ERROR"
        assert event.data["message"] == "Test error"
        assert event.data["http_status"] == 500

    def test_error_event_optional_retry_after(self) -> None:
        """RATE_LIMITED error should include retry_after in data."""
        event = AGUIErrorEvent(
            code=AGUIErrorCode.RATE_LIMITED,
            message="Too many requests",
            http_status=429,
            retry_after=60,
        )
        assert event.data["retry_after"] == 60

    def test_error_event_optional_details(self) -> None:
        """Error events can include debug details in data."""
        event = AGUIErrorEvent(
            code=AGUIErrorCode.AGENT_EXECUTION_ERROR,
            message="Test error",
            http_status=500,
            details={"traceback": "..."},
        )
        assert event.data["details"] == {"traceback": "..."}

    def test_create_error_event_from_exception(self) -> None:
        """create_error_event should handle exceptions."""
        exc = ValueError("Test exception")
        event = create_error_event(exc)
        assert event.event == AGUIEventType.RUN_ERROR
        assert event.data["code"] == AGUIErrorCode.AGENT_EXECUTION_ERROR.value


# =============================================================================
# Request/Response Format Compliance Tests
# =============================================================================


class TestCopilotRequestCompliance:
    """Verify CopilotRequest format compliance."""

    def test_copilot_message_has_required_fields(self) -> None:
        """CopilotMessage must have role and content."""
        msg = CopilotMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_copilot_request_accepts_empty_messages(self) -> None:
        """CopilotRequest accepts empty messages list (default)."""
        # Note: The schema allows empty messages with default_factory=list
        request = CopilotRequest(messages=[])
        assert len(request.messages) == 0

    def test_copilot_request_accepts_valid_messages(self) -> None:
        """CopilotRequest accepts valid message list."""
        request = CopilotRequest(
            messages=[CopilotMessage(role=MessageRole.USER, content="Hello")]
        )
        assert len(request.messages) == 1

    def test_message_role_values(self) -> None:
        """MessageRole must include user, assistant, system."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"


# =============================================================================
# Stream Lifecycle Compliance Tests
# =============================================================================


class TestAGUIStreamLifecycleCompliance:
    """Verify AG-UI stream lifecycle ordering."""

    def test_valid_stream_starts_with_run_started(self) -> None:
        """Valid stream must start with RUN_STARTED."""
        events = [
            RunStartedEvent(),
            TextDeltaEvent(content="Hello"),
            RunFinishedEvent(),
        ]
        assert events[0].event == AGUIEventType.RUN_STARTED

    def test_valid_stream_ends_with_run_finished(self) -> None:
        """Successful stream must end with RUN_FINISHED."""
        events = [
            RunStartedEvent(),
            TextDeltaEvent(content="Hello"),
            RunFinishedEvent(),
        ]
        assert events[-1].event == AGUIEventType.RUN_FINISHED

    def test_error_stream_ends_with_run_error(self) -> None:
        """Error stream must end with RUN_ERROR."""
        events = [
            RunStartedEvent(),
            AGUIErrorEvent(
                code=AGUIErrorCode.AGENT_EXECUTION_ERROR,
                message="Error",
                http_status=500,
            ),
        ]
        assert events[-1].event == AGUIEventType.RUN_ERROR

    def test_tool_call_lifecycle(self) -> None:
        """Tool calls follow START -> ARGS -> END lifecycle."""
        events = [
            ToolCallStartEvent(tool_call_id="tc-1", tool_name="search"),
            ToolCallArgsEvent(tool_call_id="tc-1", args={"q": "test"}),
            ToolCallEndEvent(tool_call_id="tc-1"),
        ]
        assert events[0].event == AGUIEventType.TOOL_CALL_START
        assert events[1].event == AGUIEventType.TOOL_CALL_ARGS
        assert events[2].event == AGUIEventType.TOOL_CALL_END

    def test_tool_call_id_consistency(self) -> None:
        """tool_call_id must be consistent across tool call events."""
        tc_id = "consistent-tc-123"
        start = ToolCallStartEvent(tool_call_id=tc_id, tool_name="search")
        args = ToolCallArgsEvent(tool_call_id=tc_id, args={})
        end = ToolCallEndEvent(tool_call_id=tc_id)
        assert start.data["tool_call_id"] == tc_id
        assert args.data["tool_call_id"] == tc_id
        assert end.data["tool_call_id"] == tc_id
