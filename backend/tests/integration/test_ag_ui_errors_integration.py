"""Integration tests for AG-UI Error Events (Story 22-B2).

This module tests:
- AG-UI bridge emits correct error events on exception
- Error events appear in SSE stream
- Error events have correct structure

Story: 22-B2 - Implement Extended AG-UI Error Events
Epic: 22 - Advanced Protocol Integration
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag_backend.models.copilot import (
    AGUIEventType,
    CopilotConfig,
    CopilotMessage,
    CopilotRequest,
    MessageRole,
)
from agentic_rag_backend.protocols.ag_ui_bridge import AGUIBridge
from agentic_rag_backend.protocols.ag_ui_errors import (
    AGUIErrorCode,
    AGUIErrorEvent,
    create_error_event,
)


class TestAGUIBridgeErrorEvents:
    """Tests for AG-UI bridge error event emission."""

    @pytest.fixture
    def mock_orchestrator(self) -> MagicMock:
        """Create a mock orchestrator that raises an exception."""
        orchestrator = MagicMock()
        return orchestrator

    @pytest.fixture
    def copilot_request(self) -> CopilotRequest:
        """Create a test copilot request with tenant_id."""
        return CopilotRequest(
            messages=[
                CopilotMessage(role=MessageRole.USER, content="test query"),
            ],
            config=CopilotConfig(
                configurable={
                    "tenant_id": "test-tenant",
                    "session_id": "test-session",
                }
            ),
        )

    @pytest.mark.asyncio
    async def test_ag_ui_bridge_emits_error_event_on_exception(
        self,
        mock_orchestrator: MagicMock,
        copilot_request: CopilotRequest,
    ) -> None:
        """Test bridge emits AGUIErrorEvent on exception (AC: 11)."""
        # Configure orchestrator to raise an exception
        mock_orchestrator.run = AsyncMock(
            side_effect=TimeoutError("Request timed out")
        )

        bridge = AGUIBridge(orchestrator=mock_orchestrator)

        events = []
        async for event in bridge.process_request(copilot_request):
            events.append(event)

        # Find the error event
        error_events = [
            e for e in events
            if isinstance(e, AGUIErrorEvent)
            or (hasattr(e, "event") and e.event == AGUIEventType.RUN_ERROR)
        ]

        assert len(error_events) >= 1, "Expected at least one error event"

        # Verify the error event structure
        error_event = error_events[0]
        assert error_event.event == AGUIEventType.RUN_ERROR
        assert error_event.data["code"] == "TIMEOUT"
        assert error_event.data["http_status"] == 504

    @pytest.mark.asyncio
    async def test_ag_ui_bridge_emits_run_error_before_stream_ends(
        self,
        mock_orchestrator: MagicMock,
        copilot_request: CopilotRequest,
    ) -> None:
        """Test error event is emitted before RUN_FINISHED (AC: 11)."""
        # Configure orchestrator to raise an exception
        mock_orchestrator.run = AsyncMock(
            side_effect=ValueError("Internal error")
        )

        bridge = AGUIBridge(orchestrator=mock_orchestrator)

        events = []
        async for event in bridge.process_request(copilot_request):
            events.append(event)

        # RUN_ERROR should appear before RUN_FINISHED
        run_error_indices = [
            i for i, e in enumerate(events)
            if e.event == AGUIEventType.RUN_ERROR
        ]
        run_finished_indices = [
            i for i, e in enumerate(events)
            if e.event == AGUIEventType.RUN_FINISHED
        ]

        assert len(run_error_indices) >= 1, "Expected RUN_ERROR event"
        assert len(run_finished_indices) >= 1, "Expected RUN_FINISHED event"

        # Error should come before finished
        assert min(run_error_indices) < max(run_finished_indices)

    @pytest.mark.asyncio
    async def test_ag_ui_bridge_rate_limit_error_includes_retry_after(
        self,
        mock_orchestrator: MagicMock,
        copilot_request: CopilotRequest,
    ) -> None:
        """Test rate limit error includes retry_after field."""
        from agentic_rag_backend.core.errors import A2ARateLimitExceededError

        mock_orchestrator.run = AsyncMock(
            side_effect=A2ARateLimitExceededError(
                session_id="test-session",
                limit=60,
                retry_after=120,
            )
        )

        bridge = AGUIBridge(orchestrator=mock_orchestrator)

        events = []
        async for event in bridge.process_request(copilot_request):
            events.append(event)

        # Find the error event
        error_events = [
            e for e in events
            if e.event == AGUIEventType.RUN_ERROR
        ]

        assert len(error_events) >= 1
        error_event = error_events[0]
        assert error_event.data["code"] == "RATE_LIMITED"
        assert error_event.data["retry_after"] == 120


class TestErrorEventSSEStream:
    """Tests for error events in SSE stream format."""

    def test_error_event_json_serializable(self) -> None:
        """Test error event can be serialized for SSE (AC: 8)."""
        event = AGUIErrorEvent(
            code=AGUIErrorCode.RATE_LIMITED,
            message="Rate limit exceeded",
            http_status=429,
            retry_after=60,
        )

        # Serialize to JSON
        json_str = event.model_dump_json()

        # Parse back
        parsed = json.loads(json_str)

        assert parsed["event"] == "RUN_ERROR"
        assert parsed["data"]["code"] == "RATE_LIMITED"
        assert parsed["data"]["retry_after"] == 60

    def test_error_event_sse_format(self) -> None:
        """Test error event matches expected SSE format."""
        event = AGUIErrorEvent(
            code=AGUIErrorCode.AGENT_EXECUTION_ERROR,
            message="An error occurred",
            http_status=500,
            details={"error_type": "ValueError"},
        )

        # Build SSE line
        data = json.dumps(event.model_dump()["data"])
        sse_line = f"event: {event.event.value}\ndata: {data}\n\n"

        # Verify format
        assert "event: RUN_ERROR" in sse_line
        assert '"code": "AGENT_EXECUTION_ERROR"' in sse_line or '"code":"AGENT_EXECUTION_ERROR"' in sse_line


class TestErrorEventDebugMode:
    """Tests for debug mode behavior in error events."""

    def test_debug_mode_includes_error_type(self) -> None:
        """Test debug mode includes exception type name (AC: 7)."""
        exc = RuntimeError("Unexpected failure")

        event = create_error_event(exc, is_debug=True)

        assert event.data["details"]["error_type"] == "RuntimeError"

    def test_non_debug_mode_excludes_error_type(self) -> None:
        """Test non-debug mode excludes exception type (AC: 6)."""
        exc = RuntimeError("Unexpected failure")

        event = create_error_event(exc, is_debug=False)

        # Details should be empty or not contain error_type
        details = event.data.get("details", {})
        assert "error_type" not in details

    @pytest.mark.asyncio
    async def test_bridge_uses_development_mode_for_debug(
        self,
    ) -> None:
        """Test bridge passes is_debug based on app environment."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            side_effect=ValueError("Test error")
        )

        request = CopilotRequest(
            messages=[
                CopilotMessage(role=MessageRole.USER, content="test"),
            ],
            config=CopilotConfig(
                configurable={"tenant_id": "test-tenant"}
            ),
        )

        bridge = AGUIBridge(orchestrator=mock_orchestrator)

        # Test with development environment
        # The import happens inside the exception handler, so we patch at the config module level
        with patch(
            "agentic_rag_backend.config.get_settings"
        ) as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.app_env = "development"
            mock_get_settings.return_value = mock_settings

            events = []
            async for event in bridge.process_request(request):
                events.append(event)

            # Find error event
            error_events = [
                e for e in events
                if e.event == AGUIEventType.RUN_ERROR
            ]

            # In development, should include error details
            assert len(error_events) >= 1
            error_event = error_events[0]
            # The error should have details with error_type
            assert error_event.data["details"].get("error_type") == "ValueError"


class TestErrorEventMapping:
    """Tests for complete exception-to-error mapping coverage."""

    @pytest.mark.parametrize(
        "exception_class,exception_args,expected_code,expected_status",
        [
            # Rate limit exceptions
            (
                "A2ARateLimitExceededError",
                {"session_id": "s1", "limit": 60, "retry_after": 60},
                "RATE_LIMITED",
                429,
            ),
            (
                "A2ASessionLimitExceededError",
                {"tenant_id": "t1", "limit": 100},
                "RATE_LIMITED",
                429,
            ),
            (
                "A2AMessageLimitExceededError",
                {"session_id": "s1", "limit": 1000},
                "RATE_LIMITED",
                429,
            ),
            # Authentication/Authorization
            (
                "TenantRequiredError",
                {},
                "TENANT_REQUIRED",
                401,
            ),
            (
                "A2APermissionError",
                {"reason": "Not allowed"},
                "TENANT_UNAUTHORIZED",
                403,
            ),
            # Not found errors
            (
                "A2AAgentNotFoundError",
                {"agent_id": "agent1"},
                "CAPABILITY_NOT_FOUND",
                404,
            ),
            (
                "A2ACapabilityNotFoundError",
                {"capability_name": "search"},
                "CAPABILITY_NOT_FOUND",
                404,
            ),
            # Service errors
            (
                "A2AServiceUnavailableError",
                {"service": "orchestrator", "reason": "overloaded"},
                "SERVICE_UNAVAILABLE",
                503,
            ),
        ],
    )
    def test_exception_mapping_complete(
        self,
        exception_class: str,
        exception_args: dict[str, Any],
        expected_code: str,
        expected_status: int,
    ) -> None:
        """Test each exception type maps to correct error code (AC: 12)."""
        from agentic_rag_backend import core

        # Get the exception class
        exc_cls = getattr(core.errors, exception_class)

        # Create the exception
        exc = exc_cls(**exception_args)

        # Map to error event
        event = create_error_event(exc)

        assert event.data["code"] == expected_code
        assert event.data["http_status"] == expected_status
