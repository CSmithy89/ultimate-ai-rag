"""Extended AG-UI Error Events for CopilotKit integration.

This module provides a comprehensive error event taxonomy for AG-UI streams,
including standardized error codes, HTTP status mappings, and retry guidance.

Features:
- AGUIErrorCode enum with 10 standardized error codes
- AGUIErrorEvent class extending AGUIEvent with error metadata
- create_error_event() function for exception-to-error mapping
- Debug mode support for detailed error information

RFC 7807 Alignment:
All AG-UI error codes map to RFC 7807 AppError patterns from core/errors.py:
- AGENT_EXECUTION_ERROR -> /errors/agent-execution (500)
- TENANT_REQUIRED -> /errors/tenant-required (401)
- TENANT_UNAUTHORIZED -> /errors/tenant-unauthorized (403)
- RATE_LIMITED -> /errors/rate-limited (429)
- TIMEOUT -> /errors/timeout (504)

Story: 22-B2 - Implement Extended AG-UI Error Events
Epic: 22 - Advanced Protocol Integration
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

import httpx
import structlog

from ..models.copilot import AGUIEvent, AGUIEventType

logger = structlog.get_logger(__name__)


class AGUIErrorCode(str, Enum):
    """Standardized AG-UI error codes aligned with RFC 7807.

    Each code maps to a specific HTTP status and error type:

    - AGENT_EXECUTION_ERROR (500): Unhandled agent exception
    - TENANT_REQUIRED (401): Missing tenant_id header
    - TENANT_UNAUTHORIZED (403): Invalid or unauthorized tenant_id
    - SESSION_NOT_FOUND (404): Invalid session reference
    - RATE_LIMITED (429): Request/session/message limit exceeded
    - TIMEOUT (504): Request processing timeout
    - INVALID_REQUEST (400): Malformed or invalid request
    - CAPABILITY_NOT_FOUND (404): Requested capability unavailable
    - UPSTREAM_ERROR (502): External service failure
    - SERVICE_UNAVAILABLE (503): System overloaded/unavailable
    """

    AGENT_EXECUTION_ERROR = "AGENT_EXECUTION_ERROR"
    TENANT_REQUIRED = "TENANT_REQUIRED"
    TENANT_UNAUTHORIZED = "TENANT_UNAUTHORIZED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    INVALID_REQUEST = "INVALID_REQUEST"
    CAPABILITY_NOT_FOUND = "CAPABILITY_NOT_FOUND"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


# HTTP status code mapping for each error code
ERROR_CODE_HTTP_STATUS: dict[AGUIErrorCode, int] = {
    AGUIErrorCode.AGENT_EXECUTION_ERROR: 500,
    AGUIErrorCode.TENANT_REQUIRED: 401,
    AGUIErrorCode.TENANT_UNAUTHORIZED: 403,
    AGUIErrorCode.SESSION_NOT_FOUND: 404,
    AGUIErrorCode.RATE_LIMITED: 429,
    AGUIErrorCode.TIMEOUT: 504,
    AGUIErrorCode.INVALID_REQUEST: 400,
    AGUIErrorCode.CAPABILITY_NOT_FOUND: 404,
    AGUIErrorCode.UPSTREAM_ERROR: 502,
    AGUIErrorCode.SERVICE_UNAVAILABLE: 503,
}


class AGUIErrorEvent(AGUIEvent):
    """Extended error event with standardized codes and retry guidance.

    This event extends AGUIEvent with structured error information following
    the AG-UI protocol RUN_ERROR event type. It includes:
    - Standardized error code from AGUIErrorCode enum
    - Human-readable error message
    - HTTP status code for client-side handling
    - Optional details dict for debugging (only in debug mode)
    - Optional retry_after hint for rate-limited requests

    Example SSE output:
        event: RUN_ERROR
        data: {"code": "RATE_LIMITED", "message": "Request rate limit exceeded.",
               "http_status": 429, "details": {}, "retry_after": 60}

    Attributes:
        event: Always RUN_ERROR for error events
        data: Dictionary containing error details (code, message, http_status, etc.)
    """

    event: AGUIEventType = AGUIEventType.RUN_ERROR

    def __init__(
        self,
        code: AGUIErrorCode,
        message: str,
        http_status: int | None = None,
        details: dict[str, Any] | None = None,
        retry_after: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an AG-UI error event.

        Args:
            code: Standardized error code from AGUIErrorCode enum
            message: Human-readable error message for the user
            http_status: HTTP status code (defaults to ERROR_CODE_HTTP_STATUS mapping)
            details: Optional additional details (only included if provided)
            retry_after: Optional retry hint in seconds (for rate limiting)
            **kwargs: Additional arguments passed to AGUIEvent
        """
        # Use the mapping if http_status not explicitly provided
        if http_status is None:
            http_status = ERROR_CODE_HTTP_STATUS.get(code, 500)

        # Build the data payload
        data: dict[str, Any] = {
            "code": code.value,
            "message": message,
            "http_status": http_status,
            "details": details or {},
        }

        # Only include retry_after if specified
        if retry_after is not None:
            data["retry_after"] = retry_after

        super().__init__(data=data, **kwargs)

        logger.debug(
            "agui_error_event_created",
            code=code.value,
            http_status=http_status,
            has_retry_after=retry_after is not None,
        )


def create_error_event(
    exception: Exception,
    is_debug: bool = False,
) -> AGUIErrorEvent:
    """Create appropriate error event from an exception.

    Maps Python exceptions to structured AG-UI error events with appropriate
    error codes, HTTP status codes, and user-facing messages.

    Mapping logic:
    - A2ARateLimitExceeded -> RATE_LIMITED (429) with retry_after=60
    - A2ASessionLimitExceeded/A2AMessageLimitExceeded -> RATE_LIMITED (429)
    - TimeoutError/asyncio.TimeoutError -> TIMEOUT (504)
    - httpx.TimeoutException -> TIMEOUT (504)
    - httpx.HTTPStatusError -> UPSTREAM_ERROR (502)
    - A2AAgentNotFoundError -> CAPABILITY_NOT_FOUND (404)
    - A2ACapabilityNotFoundError -> CAPABILITY_NOT_FOUND (404)
    - TenantRequiredError -> TENANT_REQUIRED (401)
    - A2APermissionError -> TENANT_UNAUTHORIZED (403)
    - ValidationError -> INVALID_REQUEST (400)
    - KeyError (for session lookups) -> SESSION_NOT_FOUND (404)
    - A2AServiceUnavailableError -> SERVICE_UNAVAILABLE (503)
    - Other exceptions -> AGENT_EXECUTION_ERROR (500)

    Args:
        exception: The exception to map to an error event
        is_debug: If True, include exception type name in details

    Returns:
        AGUIErrorEvent with appropriate code, status, and message

    Example:
        >>> try:
        ...     await some_operation()
        ... except Exception as e:
        ...     error_event = create_error_event(e, is_debug=settings.debug)
        ...     yield error_event
    """
    # Import here to avoid circular imports
    from ..core.errors import (
        A2AAgentNotFoundError as CoreA2AAgentNotFoundError,
        A2ACapabilityNotFoundError as CoreA2ACapabilityNotFoundError,
        A2ARateLimitExceededError,
        A2ASessionLimitExceededError,
        A2AMessageLimitExceededError,
        A2APermissionError,
        A2AServiceUnavailableError,
        TenantRequiredError,
        ValidationError,
    )

    # Also check for protocol-level errors from a2a_middleware
    from .a2a_middleware import (
        A2AAgentNotFoundError as MiddlewareAgentNotFoundError,
        A2ACapabilityNotFoundError as MiddlewareCapabilityNotFoundError,
    )

    # =========================================================================
    # Rate limit exceptions (429)
    # =========================================================================
    if isinstance(exception, A2ARateLimitExceededError):
        # Extract retry_after from the AppError details if available
        retry_after = exception.details.get("retry_after", 60)
        return AGUIErrorEvent(
            code=AGUIErrorCode.RATE_LIMITED,
            message="Request rate limit exceeded. Please wait before retrying.",
            http_status=429,
            retry_after=retry_after,
        )

    if isinstance(exception, (A2ASessionLimitExceededError, A2AMessageLimitExceededError)):
        return AGUIErrorEvent(
            code=AGUIErrorCode.RATE_LIMITED,
            message="Resource limit exceeded.",
            http_status=429,
        )

    # =========================================================================
    # Timeout exceptions (504)
    # =========================================================================
    if isinstance(exception, (TimeoutError, asyncio.TimeoutError)):
        return AGUIErrorEvent(
            code=AGUIErrorCode.TIMEOUT,
            message="Request timed out. Please try again.",
            http_status=504,
        )

    if isinstance(exception, httpx.TimeoutException):
        return AGUIErrorEvent(
            code=AGUIErrorCode.TIMEOUT,
            message="Upstream service timed out.",
            http_status=504,
        )

    # =========================================================================
    # Capability/Agent not found (404)
    # =========================================================================
    if isinstance(exception, (CoreA2AAgentNotFoundError, MiddlewareAgentNotFoundError)):
        return AGUIErrorEvent(
            code=AGUIErrorCode.CAPABILITY_NOT_FOUND,
            message="Requested agent not found.",
            http_status=404,
        )

    if isinstance(exception, (CoreA2ACapabilityNotFoundError, MiddlewareCapabilityNotFoundError)):
        return AGUIErrorEvent(
            code=AGUIErrorCode.CAPABILITY_NOT_FOUND,
            message="Requested capability not available.",
            http_status=404,
        )

    # =========================================================================
    # Upstream errors (502)
    # =========================================================================
    if isinstance(exception, httpx.HTTPStatusError):
        return AGUIErrorEvent(
            code=AGUIErrorCode.UPSTREAM_ERROR,
            message="Upstream service error.",
            http_status=502,
        )

    # =========================================================================
    # Authentication/Authorization errors (401/403)
    # =========================================================================
    if isinstance(exception, TenantRequiredError):
        return AGUIErrorEvent(
            code=AGUIErrorCode.TENANT_REQUIRED,
            message="Authentication required. Please provide tenant credentials.",
            http_status=401,
        )

    if isinstance(exception, A2APermissionError):
        return AGUIErrorEvent(
            code=AGUIErrorCode.TENANT_UNAUTHORIZED,
            message="Access denied. You do not have permission for this resource.",
            http_status=403,
        )

    # =========================================================================
    # Validation errors (400)
    # =========================================================================
    if isinstance(exception, ValidationError):
        return AGUIErrorEvent(
            code=AGUIErrorCode.INVALID_REQUEST,
            message="Invalid request. Please check your input.",
            http_status=400,
        )

    # =========================================================================
    # Session not found (404)
    # =========================================================================
    if isinstance(exception, KeyError):
        # KeyError is typically raised when a session ID is not found
        return AGUIErrorEvent(
            code=AGUIErrorCode.SESSION_NOT_FOUND,
            message="Session not found. Please start a new session.",
            http_status=404,
        )

    # =========================================================================
    # Service unavailable (503)
    # =========================================================================
    if isinstance(exception, A2AServiceUnavailableError):
        return AGUIErrorEvent(
            code=AGUIErrorCode.SERVICE_UNAVAILABLE,
            message="Service temporarily unavailable. Please try again later.",
            http_status=503,
        )

    # =========================================================================
    # Default: Agent execution error (500)
    # =========================================================================
    details: dict[str, Any] | None = None
    if is_debug:
        details = {"error_type": type(exception).__name__}

    logger.warning(
        "agui_unmapped_exception",
        exception_type=type(exception).__name__,
        is_debug=is_debug,
    )

    return AGUIErrorEvent(
        code=AGUIErrorCode.AGENT_EXECUTION_ERROR,
        message="An error occurred processing your request.",
        http_status=500,
        details=details,
    )


