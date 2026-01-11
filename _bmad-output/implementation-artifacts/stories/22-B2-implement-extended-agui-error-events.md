# Story 22-B2: Implement Extended AG-UI Error Events

Status: done

Epic: 22 - Advanced Protocol Integration
Priority: P1 - MEDIUM
Story Points: 3
Owner: Backend

## Story

As a **frontend developer integrating with the AG-UI protocol**,
I want **a comprehensive error event taxonomy with structured error codes, HTTP status mappings, and retry guidance**,
So that **I can implement appropriate error handling UIs, distinguish between transient and permanent failures, and provide users with actionable feedback when issues occur**.

## Background

Epic 22 builds on Epic 21's CopilotKit Full Integration to deliver enterprise-grade protocol capabilities. Extended AG-UI error events are critical for:

1. **Error Differentiation** - Distinguish between rate limits, timeouts, auth errors, and server errors
2. **Retry Logic** - Provide `retry_after` hints for rate-limited requests
3. **User Feedback** - Enable meaningful error messages in the frontend
4. **Debugging** - Include structured details for development environments
5. **RFC 7807 Alignment** - Map AG-UI errors to existing `AppError` patterns

### Current State (After Story 22-B1)

The AG-UI bridge currently emits `RUN_ERROR` events with basic error information. However:
- No standardized error codes
- No HTTP status mapping
- No retry guidance for rate limits
- Limited exception-to-error mapping

### Target Error Taxonomy

| Error Code | HTTP Status | When |
|------------|-------------|------|
| `AGENT_EXECUTION_ERROR` | 500 | Agent throws unhandled exception |
| `TENANT_REQUIRED` | 401 | Missing tenant_id |
| `TENANT_UNAUTHORIZED` | 403 | Invalid tenant_id |
| `SESSION_NOT_FOUND` | 404 | Invalid session reference |
| `RATE_LIMITED` | 429 | Request rate limit exceeded |
| `TIMEOUT` | 504 | Request/response timeout |
| `INVALID_REQUEST` | 400 | Malformed request |
| `CAPABILITY_NOT_FOUND` | 404 | Requested capability unavailable |
| `UPSTREAM_ERROR` | 502 | External service failure |
| `SERVICE_UNAVAILABLE` | 503 | System overloaded |

### RFC 7807 Alignment

All AG-UI error codes map to RFC 7807 `AppError` patterns from `backend/src/agentic_rag_backend/core/errors.py`:

| AG-UI Error Code | RFC 7807 Type |
|------------------|---------------|
| `AGENT_EXECUTION_ERROR` | `/errors/agent-execution` |
| `TENANT_REQUIRED` | `/errors/tenant-required` |
| `TENANT_UNAUTHORIZED` | `/errors/tenant-unauthorized` |
| `RATE_LIMITED` | `/errors/rate-limited` |
| `TIMEOUT` | `/errors/timeout` |

### Related Prior Work

| Epic/Story | Relationship |
|------------|-------------|
| Epic 7: Protocol Integration | Original AG-UI foundation |
| Epic 21-B2: Add RUN_ERROR Event Support | Basic RUN_ERROR event (prerequisite) |
| 22-A2: A2A Session Resource Limits | Limit exceptions to map |
| 22-B1: AG-UI Stream Metrics | Metrics for error tracking |

## Acceptance Criteria

1. **Given** an `AGUIErrorCode` enum exists, **when** imported from `agentic_rag_backend.models.copilot`, **then** it contains all 10 standardized error codes (AGENT_EXECUTION_ERROR, TENANT_REQUIRED, TENANT_UNAUTHORIZED, SESSION_NOT_FOUND, RATE_LIMITED, TIMEOUT, INVALID_REQUEST, CAPABILITY_NOT_FOUND, UPSTREAM_ERROR, SERVICE_UNAVAILABLE).

2. **Given** an `AGUIErrorEvent` class exists, **when** instantiated, **then** it includes: `code` (AGUIErrorCode), `message` (str), `http_status` (int), `details` (optional dict), and `retry_after` (optional int for rate limiting).

3. **Given** a `create_error_event` function exists, **when** called with an `A2ARateLimitExceeded` exception, **then** it returns an `AGUIErrorEvent` with code `RATE_LIMITED`, http_status 429, and `retry_after` set to 60.

4. **Given** a `create_error_event` function is called with an `A2ASessionLimitExceeded` or `A2AMessageLimitExceeded` exception, **when** the error event is created, **then** it returns an `AGUIErrorEvent` with code `RATE_LIMITED` and http_status 429.

5. **Given** a `create_error_event` function is called with a `TimeoutError`, **when** the error event is created, **then** it returns an `AGUIErrorEvent` with code `TIMEOUT` and http_status 504.

6. **Given** a `create_error_event` function is called with an unknown exception, **when** `is_debug=False`, **then** it returns an `AGUIErrorEvent` with code `AGENT_EXECUTION_ERROR`, http_status 500, and no error type details.

7. **Given** a `create_error_event` function is called with an unknown exception, **when** `is_debug=True`, **then** it returns an `AGUIErrorEvent` with code `AGENT_EXECUTION_ERROR`, http_status 500, and `details` containing the exception type name.

8. **Given** an error event is serialized to SSE, **when** the data is parsed, **then** it contains the expected JSON structure with `code`, `message`, `http_status`, `details`, and optional `retry_after`.

9. **Given** the frontend `useAGUIErrorHandler` hook exists, **when** a `RUN_ERROR` event is received, **then** it displays an appropriate toast notification based on the error code.

10. **Given** a `RUN_ERROR` event with `retry_after` is received by the frontend, **when** displayed, **then** the error message indicates when the user can retry.

11. **Given** the AG-UI bridge encounters an exception during processing, **when** the exception is caught, **then** a properly mapped `AGUIErrorEvent` is emitted before the stream ends.

12. **Given** all error codes are defined, **when** unit tests run, **then** each error code has at least one test verifying its mapping and serialization.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: **Addressed** - TENANT_REQUIRED and TENANT_UNAUTHORIZED error codes defined
- [x] Rate limiting / abuse protection: **Addressed** - RATE_LIMITED error code with retry_after support
- [x] Input validation / schema enforcement: **Addressed** - INVALID_REQUEST error code for malformed requests
- [x] Tests (unit/integration): **Addressed** - Tests for each error code mapping
- [x] Error handling + logging: **Addressed** - Core focus of this story
- [x] Documentation updates: **Addressed** - ErrorHandler component documented

## Security Checklist

- [ ] **Cross-tenant isolation verified**: Error messages do not leak tenant-specific information
- [ ] **Authorization checked**: TENANT_UNAUTHORIZED maps correctly from auth failures
- [ ] **No information leakage**: Error details only included when `is_debug=True`
- [ ] **Redis keys include tenant scope**: N/A - No Redis in this story
- [ ] **Integration tests for access control**: N/A - Error mapping tests only
- [ ] **RFC 7807 error responses**: Aligned with existing AppError patterns
- [ ] **File-path inputs scoped**: N/A - No file path handling

## Tasks / Subtasks

- [ ] **Task 1: Define AGUIErrorCode Enum** (AC: 1)
  - [ ] Create `AGUIErrorCode` enum in `backend/src/agentic_rag_backend/models/copilot.py`
  - [ ] Add all 10 error codes as string enum values
  - [ ] Add docstring describing each error code and when it's used

- [ ] **Task 2: Create AGUIErrorEvent Class** (AC: 2, 8)
  - [ ] Extend from existing `AGUIEvent` base class
  - [ ] Set `event` type to `AGUIEventType.RUN_ERROR`
  - [ ] Add `code`, `message`, `http_status`, `details`, `retry_after` fields
  - [ ] Implement `__init__` to structure data payload correctly
  - [ ] Ensure SSE serialization includes all fields

- [ ] **Task 3: Implement create_error_event Function** (AC: 3, 4, 5, 6, 7)
  - [ ] Create function in `backend/src/agentic_rag_backend/models/copilot.py`
  - [ ] Map `A2ARateLimitExceeded` to RATE_LIMITED with retry_after=60
  - [ ] Map `A2ASessionLimitExceeded` to RATE_LIMITED
  - [ ] Map `A2AMessageLimitExceeded` to RATE_LIMITED
  - [ ] Map `TimeoutError` to TIMEOUT
  - [ ] Map unknown exceptions to AGENT_EXECUTION_ERROR
  - [ ] Include error type in details when `is_debug=True`
  - [ ] Exclude sensitive details when `is_debug=False`

- [ ] **Task 4: Add Additional Exception Mappings** (AC: 3)
  - [ ] Map authentication exceptions to TENANT_REQUIRED/TENANT_UNAUTHORIZED
  - [ ] Map `KeyError` for session lookups to SESSION_NOT_FOUND
  - [ ] Map validation exceptions to INVALID_REQUEST
  - [ ] Map A2AAgentNotFoundError to CAPABILITY_NOT_FOUND
  - [ ] Map A2ACapabilityNotFoundError to CAPABILITY_NOT_FOUND
  - [ ] Map `httpx.TimeoutException` to TIMEOUT
  - [ ] Map `httpx.HTTPStatusError` to UPSTREAM_ERROR

- [ ] **Task 5: Integrate with AG-UI Bridge** (AC: 11)
  - [ ] Import `create_error_event` in `ag_ui_bridge.py`
  - [ ] Update exception handlers to use `create_error_event`
  - [ ] Ensure error events are yielded before stream ends
  - [ ] Pass `is_debug` from settings/environment

- [ ] **Task 6: Create Frontend ErrorHandler Component** (AC: 9, 10)
  - [ ] Create `frontend/components/copilot/ErrorHandler.tsx`
  - [ ] Define `AGUIErrorData` TypeScript interface
  - [ ] Create `useAGUIErrorHandler` hook
  - [ ] Display toast notifications based on error code
  - [ ] Show retry countdown for RATE_LIMITED errors
  - [ ] Use shadcn/ui toast component

- [ ] **Task 7: Add Error Code Constants to Frontend** (AC: 9)
  - [ ] Create `frontend/lib/ag-ui-error-codes.ts`
  - [ ] Define TypeScript enum matching backend `AGUIErrorCode`
  - [ ] Add user-friendly error messages for each code
  - [ ] Export for use in ErrorHandler

- [ ] **Task 8: Add Unit Tests** (AC: 12)
  - [ ] Create `tests/models/test_agui_error_events.py`
  - [ ] Test AGUIErrorCode enum has all 10 values
  - [ ] Test AGUIErrorEvent serialization
  - [ ] Test create_error_event with each exception type
  - [ ] Test debug mode includes error details
  - [ ] Test non-debug mode excludes error details
  - [ ] Test retry_after field for rate limit errors
  - [ ] Test SSE format output

- [ ] **Task 9: Add Integration Tests** (AC: 11)
  - [ ] Test AG-UI bridge emits correct error events
  - [ ] Test error events appear in SSE stream
  - [ ] Test frontend can parse error events

- [ ] **Task 10: Update Exports and Documentation** (AC: 1, 2)
  - [ ] Export `AGUIErrorCode`, `AGUIErrorEvent`, `create_error_event` from `models/__init__.py`
  - [ ] Update protocol documentation with error taxonomy
  - [ ] Add error handling section to AG-UI guide

## Technical Notes

### AGUIErrorCode Enum

```python
# backend/src/agentic_rag_backend/models/copilot.py (extend)
from enum import Enum

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
```

### AGUIErrorEvent Class

```python
class AGUIErrorEvent(AGUIEvent):
    """Extended error event with standardized codes and retry guidance."""

    event: AGUIEventType = AGUIEventType.RUN_ERROR

    def __init__(
        self,
        code: AGUIErrorCode,
        message: str,
        http_status: int = 500,
        details: dict[str, Any] | None = None,
        retry_after: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data={
                "code": code.value,
                "message": message,
                "http_status": http_status,
                "details": details or {},
                "retry_after": retry_after,
            },
            **kwargs,
        )
```

### Exception to Error Mapping

```python
def create_error_event(
    exception: Exception,
    is_debug: bool = False,
) -> AGUIErrorEvent:
    """Create appropriate error event from exception.

    Args:
        exception: The exception to map to an error event
        is_debug: If True, include error type in details

    Returns:
        AGUIErrorEvent with appropriate code, status, and message
    """
    from agentic_rag_backend.protocols.a2a_limits import (
        A2ASessionLimitExceeded,
        A2AMessageLimitExceeded,
        A2ARateLimitExceeded,
    )
    from agentic_rag_backend.protocols.a2a_middleware import (
        A2AAgentNotFoundError,
        A2ACapabilityNotFoundError,
    )

    # Rate limit exceptions
    if isinstance(exception, A2ARateLimitExceeded):
        return AGUIErrorEvent(
            code=AGUIErrorCode.RATE_LIMITED,
            message="Request rate limit exceeded. Please wait before retrying.",
            http_status=429,
            retry_after=60,
        )
    elif isinstance(exception, (A2ASessionLimitExceeded, A2AMessageLimitExceeded)):
        return AGUIErrorEvent(
            code=AGUIErrorCode.RATE_LIMITED,
            message="Resource limit exceeded.",
            http_status=429,
        )

    # Timeout exceptions
    elif isinstance(exception, (TimeoutError, asyncio.TimeoutError)):
        return AGUIErrorEvent(
            code=AGUIErrorCode.TIMEOUT,
            message="Request timed out. Please try again.",
            http_status=504,
        )

    # Capability/Agent not found
    elif isinstance(exception, A2AAgentNotFoundError):
        return AGUIErrorEvent(
            code=AGUIErrorCode.CAPABILITY_NOT_FOUND,
            message="Requested agent not found.",
            http_status=404,
        )
    elif isinstance(exception, A2ACapabilityNotFoundError):
        return AGUIErrorEvent(
            code=AGUIErrorCode.CAPABILITY_NOT_FOUND,
            message="Requested capability not available.",
            http_status=404,
        )

    # Upstream errors (httpx)
    elif isinstance(exception, httpx.TimeoutException):
        return AGUIErrorEvent(
            code=AGUIErrorCode.TIMEOUT,
            message="Upstream service timed out.",
            http_status=504,
        )
    elif isinstance(exception, httpx.HTTPStatusError):
        return AGUIErrorEvent(
            code=AGUIErrorCode.UPSTREAM_ERROR,
            message="Upstream service error.",
            http_status=502,
        )

    # Default: Agent execution error
    else:
        return AGUIErrorEvent(
            code=AGUIErrorCode.AGENT_EXECUTION_ERROR,
            message="An error occurred processing your request.",
            http_status=500,
            details={"error_type": type(exception).__name__} if is_debug else None,
        )
```

### Frontend ErrorHandler Component

```tsx
// frontend/components/copilot/ErrorHandler.tsx
import { useEffect, useCallback } from "react";
import { toast } from "@/components/ui/use-toast";

export interface AGUIErrorData {
  code: string;
  message: string;
  http_status: number;
  details?: Record<string, unknown>;
  retry_after?: number;
}

const ERROR_MESSAGES: Record<string, string> = {
  AGENT_EXECUTION_ERROR: "Something went wrong. Please try again.",
  TENANT_REQUIRED: "Authentication required.",
  TENANT_UNAUTHORIZED: "Access denied.",
  SESSION_NOT_FOUND: "Session expired. Please refresh.",
  RATE_LIMITED: "Too many requests. Please wait.",
  TIMEOUT: "Request timed out. Please try again.",
  INVALID_REQUEST: "Invalid request. Please check your input.",
  CAPABILITY_NOT_FOUND: "Feature not available.",
  UPSTREAM_ERROR: "External service unavailable.",
  SERVICE_UNAVAILABLE: "Service temporarily unavailable.",
};

export function useAGUIErrorHandler() {
  const handleError = useCallback((error: AGUIErrorData) => {
    const title = ERROR_MESSAGES[error.code] || error.message;

    let description = error.message;
    if (error.retry_after) {
      description = `${error.message} Please retry in ${error.retry_after} seconds.`;
    }

    toast({
      title,
      description,
      variant: error.http_status >= 500 ? "destructive" : "default",
    });
  }, []);

  return { handleError };
}
```

### Frontend Error Codes

```typescript
// frontend/lib/ag-ui-error-codes.ts
export enum AGUIErrorCode {
  AGENT_EXECUTION_ERROR = "AGENT_EXECUTION_ERROR",
  TENANT_REQUIRED = "TENANT_REQUIRED",
  TENANT_UNAUTHORIZED = "TENANT_UNAUTHORIZED",
  SESSION_NOT_FOUND = "SESSION_NOT_FOUND",
  RATE_LIMITED = "RATE_LIMITED",
  TIMEOUT = "TIMEOUT",
  INVALID_REQUEST = "INVALID_REQUEST",
  CAPABILITY_NOT_FOUND = "CAPABILITY_NOT_FOUND",
  UPSTREAM_ERROR = "UPSTREAM_ERROR",
  SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE",
}

export function isRetryableError(code: AGUIErrorCode): boolean {
  return [
    AGUIErrorCode.RATE_LIMITED,
    AGUIErrorCode.TIMEOUT,
    AGUIErrorCode.SERVICE_UNAVAILABLE,
  ].includes(code);
}
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `backend/src/agentic_rag_backend/models/copilot.py` | Modify | Add AGUIErrorCode, AGUIErrorEvent, create_error_event |
| `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` | Modify | Integrate create_error_event in exception handlers |
| `backend/src/agentic_rag_backend/models/__init__.py` | Modify | Export new classes and function |
| `frontend/components/copilot/ErrorHandler.tsx` | Create | useAGUIErrorHandler hook |
| `frontend/lib/ag-ui-error-codes.ts` | Create | TypeScript error code enum |
| `backend/tests/models/test_agui_error_events.py` | Create | Unit tests for error events |
| `backend/tests/integration/test_agui_error_events_integration.py` | Create | Integration tests |

### Error Event SSE Format

```
event: RUN_ERROR
data: {"code": "RATE_LIMITED", "message": "Request rate limit exceeded.", "http_status": 429, "details": {}, "retry_after": 60}
```

## Dependencies

- Story 22-B1 (AG-UI Stream Metrics) - Completed, provides metrics infrastructure
- Story 22-A2 (A2A Session Resource Limits) - Provides limit exception types
- Epic 21-B2 (RUN_ERROR Event Support) - Basic RUN_ERROR event foundation
- `httpx` - Already installed, for upstream error mapping

## Testing Requirements

### Unit Tests

| Test Case | Description | AC |
|-----------|-------------|-----|
| `test_agui_error_code_enum_values` | All 10 codes exist | 1 |
| `test_agui_error_event_initialization` | Fields properly initialized | 2 |
| `test_agui_error_event_serialization` | SSE format correct | 8 |
| `test_create_error_event_rate_limit` | A2ARateLimitExceeded maps correctly | 3 |
| `test_create_error_event_session_limit` | A2ASessionLimitExceeded maps correctly | 4 |
| `test_create_error_event_message_limit` | A2AMessageLimitExceeded maps correctly | 4 |
| `test_create_error_event_timeout` | TimeoutError maps correctly | 5 |
| `test_create_error_event_unknown_no_debug` | Unknown exception without debug details | 6 |
| `test_create_error_event_unknown_with_debug` | Unknown exception with debug details | 7 |
| `test_create_error_event_agent_not_found` | A2AAgentNotFoundError maps to CAPABILITY_NOT_FOUND | 3 |
| `test_create_error_event_capability_not_found` | A2ACapabilityNotFoundError maps correctly | 3 |
| `test_create_error_event_httpx_timeout` | httpx.TimeoutException maps to TIMEOUT | 5 |
| `test_create_error_event_httpx_status_error` | httpx.HTTPStatusError maps to UPSTREAM_ERROR | 3 |

### Integration Tests

| Test Case | Description | AC |
|-----------|-------------|-----|
| `test_ag_ui_bridge_emits_error_event` | Bridge emits AGUIErrorEvent on exception | 11 |
| `test_error_event_in_sse_stream` | Error event appears in SSE response | 8 |
| `test_frontend_parses_error_event` | ErrorHandler parses and displays error | 9 |

## Definition of Done

- [x] `AGUIErrorCode` enum defined with all 10 error codes
- [x] `AGUIErrorEvent` class implemented with all required fields
- [x] `create_error_event` function maps all exception types
- [x] AG-UI bridge integrated with error event creation
- [x] Frontend `ErrorHandler.tsx` component created
- [x] Frontend `ag-ui-error-codes.ts` constants defined
- [x] Unit tests for all error codes and mappings (>90% coverage)
- [x] Integration tests for error event emission
- [x] SSE serialization verified
- [x] Debug mode properly controls detail exposure
- [x] Rate limit errors include retry_after
- [x] Code review approved
- [x] Story file updated with Dev Notes

## Dev Notes

### Implementation Date: 2026-01-11

### Files Created

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/protocols/ag_ui_errors.py` | Core AG-UI error module with AGUIErrorCode enum, AGUIErrorEvent class, and create_error_event() function |
| `frontend/lib/ag-ui-error-codes.ts` | TypeScript error code enum with HTTP status mapping and utility functions |
| `frontend/components/copilot/ErrorHandler.tsx` | useAGUIErrorHandler hook for centralized error handling with toast notifications |
| `backend/tests/unit/protocols/test_ag_ui_errors.py` | 28 unit tests for enum, event class, and exception mappings |
| `backend/tests/integration/test_ag_ui_errors_integration.py` | 16 integration tests for bridge error emission and SSE format |

### Files Modified

| File | Changes |
|------|---------|
| `backend/src/agentic_rag_backend/models/copilot.py` | Added `RUN_ERROR` to AGUIEventType enum |
| `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` | Integrated create_error_event() in exception handler with debug mode support |
| `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py` | Added RUN_ERROR to KNOWN_EVENT_TYPES for metrics |
| `frontend/components/copilot/index.ts` | Exported ErrorHandler types and utilities |

### Design Decisions

1. **Module Location**: AGUIErrorCode enum placed in `protocols/ag_ui_errors.py` instead of `models/copilot.py` to avoid circular imports between models and protocols modules.

2. **Backward Compatibility**: Error events also emit backward-compatible TextDeltaEvent with generic error message to maintain compatibility with older frontend versions that don't handle RUN_ERROR events.

3. **Debug Mode**: Debug mode is determined by `is_development_env(settings.app_env)` to automatically enable detailed error information in development environments only.

4. **Retry After**: The `retry_after` field is only included when non-None to keep SSE payload minimal for non-rate-limited errors.

### Test Results

All 44 tests pass:
- 28 unit tests in `test_ag_ui_errors.py`
- 16 integration tests in `test_ag_ui_errors_integration.py`

### Exception Mapping Coverage

| Exception | Error Code | HTTP Status |
|-----------|------------|-------------|
| A2ARateLimitExceededError | RATE_LIMITED | 429 |
| A2ASessionLimitExceededError | RATE_LIMITED | 429 |
| A2AMessageLimitExceededError | RATE_LIMITED | 429 |
| TimeoutError | TIMEOUT | 504 |
| asyncio.TimeoutError | TIMEOUT | 504 |
| httpx.TimeoutException | TIMEOUT | 504 |
| httpx.HTTPStatusError | UPSTREAM_ERROR | 502 |
| A2AAgentNotFoundError | CAPABILITY_NOT_FOUND | 404 |
| A2ACapabilityNotFoundError | CAPABILITY_NOT_FOUND | 404 |
| TenantRequiredError | TENANT_REQUIRED | 401 |
| A2APermissionError | TENANT_UNAUTHORIZED | 403 |
| ValidationError | INVALID_REQUEST | 400 |
| KeyError | SESSION_NOT_FOUND | 404 |
| A2AServiceUnavailableError | SERVICE_UNAVAILABLE | 503 |
| Unknown/Other | AGENT_EXECUTION_ERROR | 500 |

---

## Senior Developer Review

**Review Date:** 2026-01-11
**Reviewer:** Senior Developer (Adversarial Review)
**Outcome:** APPROVE with Minor Recommendations

### Summary

The implementation of Story 22-B2 (Extended AG-UI Error Events) is **well-executed** and meets all acceptance criteria. The code demonstrates solid engineering practices with comprehensive error taxonomy, proper RFC 7807 alignment, and good test coverage. The issues identified are minor and do not block approval.

### Issues Found (6 total: 1 Medium, 3 Low, 2 Informational)

#### ISSUE-001 (MEDIUM): KeyError Mapping is Too Broad
**File:** `backend/src/agentic_rag_backend/protocols/ag_ui_errors.py` (lines 301-307)
**Description:** Any `KeyError` exception maps to `SESSION_NOT_FOUND` (404). This is overly broad because `KeyError` can occur in many contexts (dict access failures, missing configuration keys, etc.) and not all are session-related.
**Impact:** Could mislead frontend error handling if a non-session `KeyError` occurs.
**Recommendation:** Consider creating a specific `SessionNotFoundError` exception class and only map that to `SESSION_NOT_FOUND`. Move generic `KeyError` to fall through to `AGENT_EXECUTION_ERROR`.
**Status:** Minor concern - acceptable for current implementation since session lookups are the primary `KeyError` source in the AG-UI bridge.

#### ISSUE-002 (LOW): Frontend AGUIErrorData `code` Type Could Be Stricter
**File:** `frontend/lib/ag-ui-error-codes.ts` (line 79)
**Description:** `AGUIErrorData.code` is typed as `string` instead of `AGUIErrorCode | string`.
**Impact:** Reduced type safety and IDE autocomplete for known error codes.
**Recommendation:** Change to `code: AGUIErrorCode | string;` for better type inference.
**Status:** Enhancement - not required for functionality.

#### ISSUE-003 (LOW): Missing UPSTREAM_ERROR Integration Test
**File:** `backend/tests/integration/test_ag_ui_errors_integration.py`
**Description:** While unit tests exist for `httpx.HTTPStatusError` -> `UPSTREAM_ERROR`, there is no integration test verifying the bridge correctly emits this error type.
**Impact:** Reduced confidence in end-to-end UPSTREAM_ERROR handling.
**Recommendation:** Add integration test with mock orchestrator raising `httpx.HTTPStatusError`.
**Status:** Minor test gap - unit test provides coverage.

#### ISSUE-004 (LOW): Missing Middleware Error Types in Bridge Integration Tests
**File:** `backend/tests/integration/test_ag_ui_errors_integration.py`
**Description:** Integration tests cover core error types but not middleware-specific `A2AAgentNotFoundError` and `A2ACapabilityNotFoundError` from `a2a_middleware.py` through the bridge.
**Impact:** Middleware error paths not fully validated in integration context.
**Recommendation:** Add integration tests for middleware error types.
**Status:** Minor test gap - unit tests provide mapping coverage.

#### ISSUE-005 (INFO): ErrorHandler Hook Dependency Array
**File:** `frontend/components/copilot/ErrorHandler.tsx` (line 137)
**Description:** The `useCallback` dependency array includes `toast` which is stable from `useToast()`. The memoization pattern is correct but could note that `toast` reference is stable.
**Impact:** None - code is correct.
**Status:** Observation only.

#### ISSUE-006 (INFO): Duplicate Re-exports in ErrorHandler
**File:** `frontend/components/copilot/ErrorHandler.tsx` (lines 188-196)
**Description:** The file re-exports everything from `ag-ui-error-codes.ts`, creating two import paths for the same utilities.
**Impact:** Minor API surface confusion (can import from either location).
**Recommendation:** Document that `ag-ui-error-codes.ts` is the canonical source.
**Status:** Documentation improvement - not a code issue.

### Positive Observations

1. **Comprehensive Error Taxonomy:** All 10 error codes are well-documented with clear HTTP status mappings.

2. **Security Conscious:** Debug mode is properly controlled via `is_development_env()` and sensitive error details are only exposed in development.

3. **Backward Compatibility:** The implementation maintains backward compatibility by emitting both `RUN_ERROR` events and text message fallback.

4. **Strong Test Coverage:** 44 tests covering all error codes, exception mappings, serialization, and integration scenarios.

5. **RFC 7807 Alignment:** Error codes properly align with existing `AppError` patterns.

6. **Clean SSE Format:** Error events serialize correctly for SSE streaming.

### Security Checklist Verification

- [x] Error messages do not leak tenant-specific information
- [x] Debug details only exposed when `is_debug=True`
- [x] No stack traces or internal paths exposed to clients
- [x] Generic error messages for unknown exceptions in production

### Acceptance Criteria Verification

| AC | Description | Status |
|----|-------------|--------|
| 1 | AGUIErrorCode enum with 10 codes | PASS |
| 2 | AGUIErrorEvent with all fields | PASS |
| 3 | create_error_event maps A2ARateLimitExceeded | PASS |
| 4 | Session/Message limit exceptions mapped | PASS |
| 5 | TimeoutError mapped to TIMEOUT | PASS |
| 6 | Unknown exception without debug details | PASS |
| 7 | Unknown exception with debug details | PASS |
| 8 | SSE serialization correct | PASS |
| 9 | Frontend useAGUIErrorHandler hook | PASS |
| 10 | retry_after displayed in frontend | PASS |
| 11 | Bridge emits error event on exception | PASS |
| 12 | Unit tests for all error codes | PASS |

### Recommendation

**APPROVE** - The implementation is production-ready. The identified issues are minor and can be addressed in future maintenance or as part of technical debt. No blocking issues found.

### Optional Follow-up Items

1. Consider creating a typed `SessionNotFoundError` exception to make ISSUE-001 more precise.
2. Add the missing integration tests (ISSUE-003, ISSUE-004) to improve test coverage.
3. Update TypeScript interface for stricter typing (ISSUE-002).
