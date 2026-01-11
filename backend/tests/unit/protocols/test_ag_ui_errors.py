"""Unit tests for AG-UI Error Events (Story 22-B2).

This module tests:
- AGUIErrorCode enum completeness
- AGUIErrorEvent initialization and serialization
- create_error_event() exception mapping
- Debug mode behavior for error details
- SSE format output

Story: 22-B2 - Implement Extended AG-UI Error Events
Epic: 22 - Advanced Protocol Integration
"""

import asyncio
import json
from unittest.mock import MagicMock

import httpx

from agentic_rag_backend.protocols.ag_ui_errors import (
    AGUIErrorCode,
    AGUIErrorEvent,
    ERROR_CODE_HTTP_STATUS,
    create_error_event,
)
from agentic_rag_backend.models.copilot import AGUIEventType


class TestAGUIErrorCodeEnum:
    """Tests for AGUIErrorCode enum."""

    def test_agui_error_code_enum_has_all_10_values(self) -> None:
        """Verify enum contains all 10 error codes (AC: 1)."""
        expected_codes = {
            "AGENT_EXECUTION_ERROR",
            "TENANT_REQUIRED",
            "TENANT_UNAUTHORIZED",
            "SESSION_NOT_FOUND",
            "RATE_LIMITED",
            "TIMEOUT",
            "INVALID_REQUEST",
            "CAPABILITY_NOT_FOUND",
            "UPSTREAM_ERROR",
            "SERVICE_UNAVAILABLE",
        }
        actual_codes = {code.value for code in AGUIErrorCode}
        assert actual_codes == expected_codes, f"Missing codes: {expected_codes - actual_codes}"

    def test_agui_error_code_is_string_enum(self) -> None:
        """Verify error codes are string values."""
        for code in AGUIErrorCode:
            assert isinstance(code.value, str)
            assert code.value == code.name  # Name matches value

    def test_error_code_http_status_mapping_complete(self) -> None:
        """Verify all error codes have HTTP status mappings."""
        for code in AGUIErrorCode:
            assert code in ERROR_CODE_HTTP_STATUS
            status = ERROR_CODE_HTTP_STATUS[code]
            assert isinstance(status, int)
            assert 400 <= status < 600


class TestAGUIErrorEvent:
    """Tests for AGUIErrorEvent class."""

    def test_agui_error_event_initialization_basic(self) -> None:
        """Test basic error event creation (AC: 2)."""
        event = AGUIErrorEvent(
            code=AGUIErrorCode.AGENT_EXECUTION_ERROR,
            message="Test error message",
        )

        assert event.event == AGUIEventType.RUN_ERROR
        assert event.data["code"] == "AGENT_EXECUTION_ERROR"
        assert event.data["message"] == "Test error message"
        assert event.data["http_status"] == 500
        assert event.data["details"] == {}

    def test_agui_error_event_initialization_with_all_fields(self) -> None:
        """Test error event with all fields populated (AC: 2)."""
        event = AGUIErrorEvent(
            code=AGUIErrorCode.RATE_LIMITED,
            message="Rate limit exceeded",
            http_status=429,
            details={"limit": 60, "window": "minute"},
            retry_after=60,
        )

        assert event.data["code"] == "RATE_LIMITED"
        assert event.data["message"] == "Rate limit exceeded"
        assert event.data["http_status"] == 429
        assert event.data["details"] == {"limit": 60, "window": "minute"}
        assert event.data["retry_after"] == 60

    def test_agui_error_event_uses_default_http_status(self) -> None:
        """Test that error events use default HTTP status from mapping."""
        test_cases = [
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
        ]

        for code, expected_status in test_cases:
            event = AGUIErrorEvent(code=code, message="Test")
            assert event.data["http_status"] == expected_status, f"Failed for {code}"

    def test_agui_error_event_serialization_json(self) -> None:
        """Test error event serializes to valid JSON (AC: 8)."""
        event = AGUIErrorEvent(
            code=AGUIErrorCode.RATE_LIMITED,
            message="Request rate limit exceeded.",
            http_status=429,
            retry_after=60,
        )

        # Serialize to JSON (using model_dump)
        json_data = event.model_dump_json()
        parsed = json.loads(json_data)

        assert parsed["event"] == "RUN_ERROR"
        assert parsed["data"]["code"] == "RATE_LIMITED"
        assert parsed["data"]["message"] == "Request rate limit exceeded."
        assert parsed["data"]["http_status"] == 429
        assert parsed["data"]["retry_after"] == 60

    def test_agui_error_event_retry_after_optional(self) -> None:
        """Test retry_after is not included when not specified."""
        event = AGUIErrorEvent(
            code=AGUIErrorCode.AGENT_EXECUTION_ERROR,
            message="Test error",
        )

        assert "retry_after" not in event.data


class TestCreateErrorEvent:
    """Tests for create_error_event() function."""

    def test_create_error_event_rate_limit(self) -> None:
        """Test A2ARateLimitExceeded maps to RATE_LIMITED (AC: 3)."""
        from agentic_rag_backend.core.errors import A2ARateLimitExceededError

        exc = A2ARateLimitExceededError(
            session_id="test-session",
            limit=60,
            retry_after=120,
        )

        event = create_error_event(exc)

        assert event.data["code"] == "RATE_LIMITED"
        assert event.data["http_status"] == 429
        assert event.data["retry_after"] == 120  # From exception details

    def test_create_error_event_session_limit(self) -> None:
        """Test A2ASessionLimitExceeded maps to RATE_LIMITED (AC: 4)."""
        from agentic_rag_backend.core.errors import A2ASessionLimitExceededError

        exc = A2ASessionLimitExceededError(tenant_id="test-tenant", limit=100)

        event = create_error_event(exc)

        assert event.data["code"] == "RATE_LIMITED"
        assert event.data["http_status"] == 429

    def test_create_error_event_message_limit(self) -> None:
        """Test A2AMessageLimitExceeded maps to RATE_LIMITED (AC: 4)."""
        from agentic_rag_backend.core.errors import A2AMessageLimitExceededError

        exc = A2AMessageLimitExceededError(session_id="test-session", limit=1000)

        event = create_error_event(exc)

        assert event.data["code"] == "RATE_LIMITED"
        assert event.data["http_status"] == 429

    def test_create_error_event_timeout(self) -> None:
        """Test TimeoutError maps to TIMEOUT (AC: 5)."""
        exc = TimeoutError("Connection timed out")

        event = create_error_event(exc)

        assert event.data["code"] == "TIMEOUT"
        assert event.data["http_status"] == 504
        assert "timed out" in event.data["message"].lower()

    def test_create_error_event_asyncio_timeout(self) -> None:
        """Test asyncio.TimeoutError maps to TIMEOUT (AC: 5)."""
        exc = asyncio.TimeoutError()

        event = create_error_event(exc)

        assert event.data["code"] == "TIMEOUT"
        assert event.data["http_status"] == 504

    def test_create_error_event_unknown_no_debug(self) -> None:
        """Test unknown exception without debug details (AC: 6)."""
        exc = ValueError("Some internal error")

        event = create_error_event(exc, is_debug=False)

        assert event.data["code"] == "AGENT_EXECUTION_ERROR"
        assert event.data["http_status"] == 500
        assert event.data["details"] == {} or event.data["details"] is None
        # Should not expose error type
        assert "ValueError" not in str(event.data.get("details", {}))

    def test_create_error_event_unknown_with_debug(self) -> None:
        """Test unknown exception with debug details (AC: 7)."""
        exc = ValueError("Some internal error")

        event = create_error_event(exc, is_debug=True)

        assert event.data["code"] == "AGENT_EXECUTION_ERROR"
        assert event.data["http_status"] == 500
        assert event.data["details"]["error_type"] == "ValueError"

    def test_create_error_event_agent_not_found(self) -> None:
        """Test A2AAgentNotFoundError maps to CAPABILITY_NOT_FOUND (AC: 3)."""
        from agentic_rag_backend.core.errors import A2AAgentNotFoundError

        exc = A2AAgentNotFoundError("test-agent")

        event = create_error_event(exc)

        assert event.data["code"] == "CAPABILITY_NOT_FOUND"
        assert event.data["http_status"] == 404

    def test_create_error_event_capability_not_found(self) -> None:
        """Test A2ACapabilityNotFoundError maps to CAPABILITY_NOT_FOUND (AC: 3)."""
        from agentic_rag_backend.core.errors import A2ACapabilityNotFoundError

        exc = A2ACapabilityNotFoundError("vector_search")

        event = create_error_event(exc)

        assert event.data["code"] == "CAPABILITY_NOT_FOUND"
        assert event.data["http_status"] == 404

    def test_create_error_event_httpx_timeout(self) -> None:
        """Test httpx.TimeoutException maps to TIMEOUT (AC: 5)."""
        exc = httpx.TimeoutException("Request timed out")

        event = create_error_event(exc)

        assert event.data["code"] == "TIMEOUT"
        assert event.data["http_status"] == 504

    def test_create_error_event_httpx_status_error(self) -> None:
        """Test httpx.HTTPStatusError maps to UPSTREAM_ERROR (AC: 3)."""
        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_request = MagicMock()

        exc = httpx.HTTPStatusError(
            "Service unavailable",
            request=mock_request,
            response=mock_response,
        )

        event = create_error_event(exc)

        assert event.data["code"] == "UPSTREAM_ERROR"
        assert event.data["http_status"] == 502

    def test_create_error_event_tenant_required(self) -> None:
        """Test TenantRequiredError maps to TENANT_REQUIRED."""
        from agentic_rag_backend.core.errors import TenantRequiredError

        exc = TenantRequiredError()

        event = create_error_event(exc)

        assert event.data["code"] == "TENANT_REQUIRED"
        assert event.data["http_status"] == 401

    def test_create_error_event_permission_error(self) -> None:
        """Test A2APermissionError maps to TENANT_UNAUTHORIZED."""
        from agentic_rag_backend.core.errors import A2APermissionError

        exc = A2APermissionError("Not authorized", resource_id="resource-123")

        event = create_error_event(exc)

        assert event.data["code"] == "TENANT_UNAUTHORIZED"
        assert event.data["http_status"] == 403

    def test_create_error_event_validation_error(self) -> None:
        """Test ValidationError maps to INVALID_REQUEST."""
        from agentic_rag_backend.core.errors import ValidationError

        exc = ValidationError("Invalid input data")

        event = create_error_event(exc)

        assert event.data["code"] == "INVALID_REQUEST"
        assert event.data["http_status"] == 400

    def test_create_error_event_key_error(self) -> None:
        """Test KeyError maps to SESSION_NOT_FOUND."""
        exc = KeyError("session-123")

        event = create_error_event(exc)

        assert event.data["code"] == "SESSION_NOT_FOUND"
        assert event.data["http_status"] == 404

    def test_create_error_event_service_unavailable(self) -> None:
        """Test A2AServiceUnavailableError maps to SERVICE_UNAVAILABLE."""
        from agentic_rag_backend.core.errors import A2AServiceUnavailableError

        exc = A2AServiceUnavailableError("orchestrator", "System overloaded")

        event = create_error_event(exc)

        assert event.data["code"] == "SERVICE_UNAVAILABLE"
        assert event.data["http_status"] == 503


class TestAGUIErrorEventSerialization:
    """Tests for SSE serialization of error events."""

    def test_error_event_sse_format(self) -> None:
        """Test error event SSE output format (AC: 8)."""
        event = AGUIErrorEvent(
            code=AGUIErrorCode.RATE_LIMITED,
            message="Request rate limit exceeded.",
            http_status=429,
            details={},
            retry_after=60,
        )

        # Simulate SSE format
        event_type = event.event.value
        event_data = event.model_dump()["data"]

        # Verify structure matches expected SSE format
        assert event_type == "RUN_ERROR"
        assert event_data["code"] == "RATE_LIMITED"
        assert event_data["message"] == "Request rate limit exceeded."
        assert event_data["http_status"] == 429
        assert event_data["retry_after"] == 60

    def test_all_error_codes_serialize_correctly(self) -> None:
        """Test all error codes produce valid JSON (AC: 12)."""
        for code in AGUIErrorCode:
            event = AGUIErrorEvent(
                code=code,
                message=f"Test message for {code.value}",
            )

            # Should serialize without error
            json_str = event.model_dump_json()
            parsed = json.loads(json_str)

            assert parsed["event"] == "RUN_ERROR"
            assert parsed["data"]["code"] == code.value


class TestAGUIErrorEventFromMiddleware:
    """Tests for error events from A2A middleware exceptions."""

    def test_middleware_agent_not_found_error(self) -> None:
        """Test middleware A2AAgentNotFoundError (not AppError)."""
        from agentic_rag_backend.protocols.a2a_middleware import (
            A2AAgentNotFoundError as MiddlewareAgentNotFoundError,
        )

        # This is an Exception, not AppError
        exc = MiddlewareAgentNotFoundError("Agent not found: test-agent")

        event = create_error_event(exc)

        assert event.data["code"] == "CAPABILITY_NOT_FOUND"
        assert event.data["http_status"] == 404

    def test_middleware_capability_not_found_error(self) -> None:
        """Test middleware A2ACapabilityNotFoundError (not AppError)."""
        from agentic_rag_backend.protocols.a2a_middleware import (
            A2ACapabilityNotFoundError as MiddlewareCapabilityNotFoundError,
        )

        exc = MiddlewareCapabilityNotFoundError("Capability not found: search")

        event = create_error_event(exc)

        assert event.data["code"] == "CAPABILITY_NOT_FOUND"
        assert event.data["http_status"] == 404
