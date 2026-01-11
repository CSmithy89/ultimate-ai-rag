# Story 22-C1: Implement MCP-UI Renderer

Status: in-progress

Epic: 22 - Advanced Protocol Integration
Priority: P1 - MEDIUM
Story Points: 8
Owner: Frontend + Backend

## Story

As a **developer integrating MCP tools**,
I want **to render external MCP tool UIs in sandboxed iframes with proper security controls**,
So that **users can interact with rich tool interfaces while maintaining application security through origin validation and CSP policies**.

## Background

Epic 22 extends protocol integration with cross-platform UI capabilities. MCP-UI enables embedding external interactive tool UIs within the CopilotKit chat interface via sandboxed iframes. This requires:

1. **Backend Configuration** - Allowed origins management and signed URL generation
2. **Frontend Renderer** - Sandboxed iframe component with proper security attributes
3. **PostMessage Bridge** - Secure communication between iframe and parent application
4. **Security Controls** - Origin allowlist validation, CSP headers, and signature verification

### MCP-UI Protocol Context

The MCP-UI protocol allows MCP tools to return UI payloads that render interactive interfaces:
- External tool providers host web UIs at designated endpoints
- Iframes are sandboxed with minimal permissions (allow-scripts, allow-same-origin)
- PostMessage API enables resize events, results, and error reporting
- Origins must be pre-approved via allowlist configuration

### Related Prior Work

| Epic/Story | Relationship |
|------------|-------------|
| Epic 7-1: MCP Tool Server | Original MCP server implementation (completed) |
| Epic 14-1: RAG Engine via MCP | Extended MCP capabilities (completed) |
| Epic 21-C: MCP Client Integration | MCP client for external tools (completed) |
| Epic 21-D: A2UI Widget Rendering | UI rendering patterns (completed) |

## Acceptance Criteria

1. **Given** an `MCP_UI_ALLOWED_ORIGINS` environment variable is set, **when** the backend starts, **then** the origins are parsed into a validated allowlist.

2. **Given** a `GET /mcp/ui/config` request, **when** called with valid tenant_id, **then** the endpoint returns the allowed origins list for that tenant.

3. **Given** an MCP-UI payload with a `ui_url`, **when** rendered in MCPUIRenderer, **then** an iframe is created with `sandbox="allow-scripts allow-same-origin"` attributes.

4. **Given** an MCP-UI payload with a URL not in the allowlist, **when** MCPUIRenderer attempts to render, **then** the component displays a security warning instead of the iframe.

5. **Given** a postMessage from the iframe, **when** `event.origin` is not in the allowed origins, **then** the message is silently ignored with a console warning.

6. **Given** a valid `mcp_ui_resize` postMessage, **when** received by MCPUIBridge, **then** the iframe dimensions are updated to match the specified width/height.

7. **Given** a valid `mcp_ui_result` postMessage, **when** received by MCPUIBridge, **then** the `onResult` callback is invoked with the result data.

8. **Given** a valid `mcp_ui_error` postMessage, **when** received by MCPUIBridge, **then** the error is logged and optionally displayed to the user.

9. **Given** the frontend security utilities, **when** `isAllowedOrigin(origin)` is called, **then** it correctly validates against the loaded allowlist.

10. **Given** all components, **when** tests are run, **then** unit tests pass with >85% coverage.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: **Addressed** - Origins can be tenant-specific via config endpoint
- [x] Input validation / schema enforcement: **Addressed** - Zod schema for postMessage validation
- [x] Tests (unit/integration): **Addressed** - Unit tests for renderer, bridge, and security utilities
- [x] Error handling + logging: **Addressed** - Console warnings for blocked origins, error callbacks
- [x] Documentation updates: **Addressed** - Docstrings on all public components

## Security Checklist

- [ ] **Origin allowlist enforced**: All iframe URLs validated against MCP_UI_ALLOWED_ORIGINS
- [ ] **Sandbox attributes applied**: iframe uses minimal sandbox permissions
- [ ] **PostMessage origin validated**: All incoming messages checked against allowlist
- [ ] **CSP headers configured**: frame-src includes only allowed origins
- [ ] **No parent DOM access**: Iframe cannot access parent document
- [ ] **Zod schema validation**: PostMessage data validated before processing

## Tasks / Subtasks

- [ ] **Task 1: Create Backend MCP-UI Configuration Models** (AC: 1, 2)
  - [ ] Create `backend/src/agentic_rag_backend/models/mcp_ui.py`
  - [ ] Define `MCPUIConfig` Pydantic model (allowed_origins, signing_secret, enabled)
  - [ ] Add `mcp_ui_enabled` and `mcp_ui_allowed_origins` to Settings dataclass
  - [ ] Add `mcp_ui_signing_secret` to Settings dataclass

- [ ] **Task 2: Create MCP-UI Config API Endpoint** (AC: 2)
  - [ ] Create `backend/src/agentic_rag_backend/api/routes/mcp_ui.py`
  - [ ] Implement `GET /mcp/ui/config` endpoint returning allowed origins
  - [ ] Add router to main.py

- [ ] **Task 3: Create Frontend Security Utilities** (AC: 4, 5, 9)
  - [ ] Create `frontend/lib/mcp-ui-security.ts`
  - [ ] Implement `loadAllowedOrigins()` function to fetch from backend
  - [ ] Implement `isAllowedOrigin(origin)` validation function
  - [ ] Implement Zod schema for postMessage validation

- [ ] **Task 4: Create MCPUIRenderer Component** (AC: 3, 4)
  - [ ] Create `frontend/components/mcp-ui/MCPUIRenderer.tsx`
  - [ ] Implement sandboxed iframe with proper attributes
  - [ ] Implement origin validation before rendering
  - [ ] Display security warning for blocked origins

- [ ] **Task 5: Create MCPUIBridge Component** (AC: 5, 6, 7, 8)
  - [ ] Create `frontend/components/mcp-ui/MCPUIBridge.tsx`
  - [ ] Implement postMessage listener with origin validation
  - [ ] Handle `mcp_ui_resize` messages
  - [ ] Handle `mcp_ui_result` messages
  - [ ] Handle `mcp_ui_error` messages

- [ ] **Task 6: Add Backend Unit Tests** (AC: 10)
  - [ ] Create `backend/tests/unit/models/test_mcp_ui.py`
  - [ ] Test MCPUIConfig model validation
  - [ ] Test origin parsing logic

- [ ] **Task 7: Add Backend Integration Tests** (AC: 2)
  - [ ] Create `backend/tests/integration/test_mcp_ui_api.py`
  - [ ] Test `/mcp/ui/config` endpoint

- [ ] **Task 8: Add Frontend Tests** (AC: 10)
  - [ ] Create `frontend/__tests__/components/mcp-ui/MCPUIRenderer.test.tsx`
  - [ ] Create `frontend/__tests__/components/mcp-ui/MCPUIBridge.test.tsx`
  - [ ] Create `frontend/__tests__/lib/mcp-ui-security.test.ts`
  - [ ] Test origin validation
  - [ ] Test postMessage handling
  - [ ] Test renderer blocking for invalid origins

## Technical Notes

### Backend Configuration

```python
# backend/src/agentic_rag_backend/config.py
# Add to Settings dataclass:
mcp_ui_enabled: bool
mcp_ui_allowed_origins: list[str]
mcp_ui_signing_secret: str
```

```bash
# .env
MCP_UI_ENABLED=true
MCP_UI_ALLOWED_ORIGINS=https://mcp-ui.example.com,https://tools.copilotkit.ai
MCP_UI_SIGNING_SECRET=your-secret-key-here
```

### Frontend Component Structure

```tsx
// MCPUIRenderer.tsx
interface MCPUIRendererProps {
  payload: MCPUIPayload;
  onResult?: (result: unknown) => void;
  onError?: (error: string) => void;
}

// MCPUIBridge.tsx (hook)
function useMCPUIBridge(
  iframeRef: RefObject<HTMLIFrameElement>,
  allowedOrigins: Set<string>,
  callbacks: MCPUIBridgeCallbacks
): void;
```

### PostMessage Protocol

```typescript
// Outgoing from iframe:
{ type: "mcp_ui_resize", width: number, height: number }
{ type: "mcp_ui_result", result: unknown }
{ type: "mcp_ui_error", error: string }

// Incoming to iframe:
{ type: "mcp_ui_init", data: Record<string, unknown> }
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `backend/src/agentic_rag_backend/models/mcp_ui.py` | Create | MCP-UI Pydantic models |
| `backend/src/agentic_rag_backend/api/routes/mcp_ui.py` | Create | Config endpoint |
| `backend/src/agentic_rag_backend/config.py` | Modify | Add MCP-UI settings |
| `frontend/lib/mcp-ui-security.ts` | Create | Security utilities |
| `frontend/components/mcp-ui/MCPUIRenderer.tsx` | Create | Iframe renderer |
| `frontend/components/mcp-ui/MCPUIBridge.tsx` | Create | PostMessage bridge |
| `.env.example` | Modify | Add MCP-UI variables |

### Dependencies

- `zod` - Already installed, for postMessage schema validation
- `pydantic` - Already installed, for backend models

## Dependencies

- **Epic 21-C completed** - MCP client infrastructure available
- **Epic 21-D completed** - UI rendering patterns established

## Definition of Done

- [x] `MCPUIConfig` Pydantic model created
- [x] `GET /mcp/ui/config` endpoint functional
- [x] MCP-UI settings added to Settings dataclass
- [x] `MCPUIRenderer` component renders sandboxed iframes
- [x] `MCPUIBridge` hook handles postMessage communication
- [x] Origin validation blocks untrusted URLs
- [x] PostMessage origin validation implemented
- [x] Zod schema validates incoming messages
- [x] Backend unit tests pass (>85% coverage)
- [x] Frontend tests pass (>85% coverage)
- [x] .env.example updated with new variables
- [ ] Code review approved
- [x] Story file updated with Dev Notes

## Dev Notes

### Implementation Summary

Implemented MCP-UI renderer with sandboxed iframes and secure postMessage communication. Key components:

1. **Backend Configuration**: Added `mcp_ui_enabled`, `mcp_ui_allowed_origins`, and `mcp_ui_signing_secret` to Settings dataclass. Origins are parsed from comma-separated env var.

2. **API Endpoint**: Added `/mcp/ui/config` endpoint to the existing MCP router. Returns allowed origins for frontend validation.

3. **Security Utilities** (`frontend/lib/mcp-ui-security.ts`):
   - `MCPUIMessageSchema` - Zod discriminated union for type-safe postMessage validation
   - `loadAllowedOrigins()` - Fetches config from backend with caching
   - `validateMCPUIMessage()` - Origin + schema validation in one call

4. **MCPUIBridge Hook**: Listens for window messages, validates origin and schema, dispatches to callbacks for resize/result/error.

5. **MCPUIRenderer Component**:
   - Shows loading state while fetching config
   - Blocks untrusted origins with security warning
   - Renders sandboxed iframe with `allow-scripts allow-same-origin`
   - Sends `mcp_ui_init` message on iframe load

### Security Controls
- Origin allowlist validated on both backend config and frontend rendering
- Sandbox attributes prevent form submission, popups, and parent DOM access
- PostMessage origin checked before processing any message
- Zod schema prevents malformed message processing

### Test Coverage
- Backend: 22 unit tests for models, 9 integration tests for API
- Frontend: 51 tests covering security utilities, bridge hook, and renderer component

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Completion Notes List

1. Created story file and context file with comprehensive implementation details
2. Implemented backend models at `backend/src/agentic_rag_backend/models/mcp_ui.py`
3. Added MCP-UI settings to config.py with env var parsing
4. Extended existing `/mcp` router with `/ui/config` endpoint (no new router needed)
5. Created frontend security utilities with Zod schema validation
6. Created MCPUIBridge hook for postMessage handling
7. Created MCPUIRenderer component with loading, blocked, and render states
8. Added comprehensive backend and frontend tests
9. Updated .env.example with new MCP-UI variables

### File List

**Created:**
- `_bmad-output/implementation-artifacts/stories/22-C1-implement-mcp-ui-renderer.md`
- `_bmad-output/implementation-artifacts/stories/22-C1-implement-mcp-ui-renderer.context.xml`
- `backend/src/agentic_rag_backend/models/mcp_ui.py`
- `frontend/lib/mcp-ui-security.ts`
- `frontend/components/mcp-ui/MCPUIRenderer.tsx`
- `frontend/components/mcp-ui/MCPUIBridge.tsx`
- `frontend/components/mcp-ui/index.ts`
- `backend/tests/unit/models/__init__.py`
- `backend/tests/unit/models/test_mcp_ui.py`
- `backend/tests/integration/test_mcp_ui_api.py`
- `frontend/__tests__/lib/mcp-ui-security.test.ts`
- `frontend/__tests__/components/mcp-ui/MCPUIRenderer.test.tsx`
- `frontend/__tests__/components/mcp-ui/MCPUIBridge.test.tsx`

**Modified:**
- `backend/src/agentic_rag_backend/config.py` - Added MCP-UI settings to Settings dataclass
- `backend/src/agentic_rag_backend/models/__init__.py` - Export MCP-UI models
- `backend/src/agentic_rag_backend/api/routes/mcp.py` - Added /ui/config endpoint
- `.env.example` - Added MCP-UI environment variables
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated to review status

## Test Outcomes

### Backend Tests
```
tests/unit/models/test_mcp_ui.py: 22 passed
tests/integration/test_mcp_ui_api.py: 9 passed
Total: 31 passed
```

### Frontend Tests
```
__tests__/lib/mcp-ui-security.test.ts: 23 passed
__tests__/components/mcp-ui/MCPUIBridge.test.tsx: 12 passed
__tests__/components/mcp-ui/MCPUIRenderer.test.tsx: 16 passed
Total: 51 passed
```
