# Story 21-B2: Add RUN_ERROR Event Support

## Status: Review

## Story

As a developer, I need structured error events instead of embedded error text, so I can handle errors gracefully with toast notifications and analytics tracking.

## Implementation Summary

### Backend Changes

1. **Added RUN_ERROR to AGUIEventType enum** (`models/copilot.py`)
   - New event type for structured error reporting

2. **Created RunErrorCode enum** (`models/copilot.py`)
   - Standard error codes: AGENT_EXECUTION_ERROR, TENANT_REQUIRED, RATE_LIMITED, TIMEOUT, INVALID_REQUEST
   - Follows naming convention: CATEGORY_SPECIFIC_ERROR

3. **Created RunErrorEvent class** (`models/copilot.py`)
   - Structured event with code, message, and optional details
   - Details only populated in development mode for security

4. **Updated AG-UI Bridge** (`protocols/ag_ui_bridge.py`)
   - Tenant missing error now emits RunErrorEvent instead of text
   - Exception handler emits RunErrorEvent with details in dev mode only
   - Imported is_development_env for environment detection

### Frontend Changes

1. **Created CopilotErrorHandler component** (`components/copilot/CopilotErrorHandler.tsx`)
   - Exports RunErrorCode constants matching backend
   - handleRunError callback for processing RUN_ERROR events
   - Toast notifications with user-friendly messages
   - Analytics tracking for errors
   - Development mode console logging with details
   - Window hook for manual testing in dev

### Dev Notes

- RUN_ERROR events follow AG-UI protocol specification
- Error details are only included in development mode to prevent information leakage
- Frontend component is prepared for future CopilotKit event subscription API
- Currently exposes handler via window object for testing until native event subscription is available

## Acceptance Criteria

- [x] Backend emits RUN_ERROR events instead of embedding errors in text messages
- [x] Error events include structured code, message, and optional details
- [x] Frontend has error handler component with toast notifications
- [x] Error codes are consistent between frontend and backend
- [x] Details are only shown in development mode

## Files Changed

- `backend/src/agentic_rag_backend/models/copilot.py` - Added RunErrorCode, RunErrorEvent
- `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` - Updated error handling
- `frontend/components/copilot/CopilotErrorHandler.tsx` - New component
