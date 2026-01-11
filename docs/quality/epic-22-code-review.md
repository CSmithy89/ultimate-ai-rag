# Epic 22 Code Review

Scope: A2A middleware + resource limits, AG-UI metrics/errors, MCP-UI, Open-JSON-UI, protocol docs, and Epic 22 tech debt items.

## Critical Findings (Must Fix Before Completion)
- [x] A2A resource limits are not enforced in session or message endpoints; the manager is only used by `/a2a/metrics/{tenant_id}`. `backend/src/agentic_rag_backend/api/routes/a2a.py`, `backend/src/agentic_rag_backend/main.py`
- [x] A2A middleware registration contract diverges from the story: `/a2a/agents/register` (registry) accepts tenant_id in body and skips header/API-key/prefix validation, while the middleware uses `/a2a/middleware/*`. `backend/src/agentic_rag_backend/api/routes/a2a.py`
- [x] MCP-UI and Open-JSON-UI renderers are not wired into the CopilotKit rendering pipeline; tool calls only render MCPToolCallCard/VectorSearchCard. `frontend/components/copilot/tool-renderers.tsx`, `frontend/components/mcp-ui/MCPUIRenderer.tsx`, `frontend/components/open-json-ui/OpenJSONUIRenderer.tsx`
- [x] Backend Open-JSON-UI models do not validate component schemas or URL safety; `OpenJSONUIPayload.components` is `list[dict]`. `backend/src/agentic_rag_backend/protocols/open_json_ui.py`
- [x] Missing tenant_id in AG-UI bridge emits only text; it never emits a structured RUN_ERROR event, and the frontend error handler hook is not integrated anywhere. `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`, `frontend/components/copilot/ErrorHandler.tsx`
- [x] A2A middleware auth key is captured at import time; if `A2A_API_KEY` is unset or rotated later, auth can be disabled or stale for the process lifetime. `backend/src/agentic_rag_backend/api/routes/a2a.py`
- [x] MCP-UI signing secret accepts empty/whitespace values, resulting in an effectively blank signing key. `backend/src/agentic_rag_backend/config.py`
- [x] Middleware delegation buffers up to 1000 events instead of streaming, risking high memory usage and truncated/incomplete UI state. `backend/src/agentic_rag_backend/api/routes/a2a.py`
- [x] Redis tenant keys are built directly from tenant_id; validate tenant_id before Redis key construction to prevent keyspace injection. `backend/src/agentic_rag_backend/protocols/a2a_resource_limits.py`
- [x] `/mcp/ui/config` lacks explicit CORS handling for cross-origin fetches using X-Tenant-ID. `backend/src/agentic_rag_backend/api/routes/mcp.py`

## High Findings
- [x] MCP-UI signing secret is configured but unused, CSP headers are not enforced, and docs show sandbox attributes that do not match defaults. `backend/src/agentic_rag_backend/config.py`, `backend/src/agentic_rag_backend/models/mcp_ui.py`, `docs/guides/protocol-integration/mcp-ui-rendering.md`
- [x] A2A limits config drift: new `A2A_*_LIMITS` env vars exist in config but are missing from `.env.example`, while older session limit vars still exist. `backend/src/agentic_rag_backend/config.py`, `.env.example`
- [x] Telemetry metrics label cardinality control is missing; event names are unbounded and the metric name differs from the story. `backend/src/agentic_rag_backend/observability/metrics.py`, `backend/src/agentic_rag_backend/api/routes/telemetry.py`
- [x] Protocol docs list env vars and APIs that do not exist in code (AGUI_STREAM_TIMEOUT_SECONDS, A2A_DEFAULT_TIMEOUT_SECONDS, A2A_MAX_DELEGATION_DEPTH, A2A_REDIS_URL, OPEN_JSON_UI_ENABLED). `docs/guides/protocol-integration/overview.md`, `docs/guides/protocol-integration/a2a-protocol.md`
- [x] In-memory A2A rate-limit tracking may grow large for high configured limits; cleanup is time-based but unbounded by count. Consider bounded deque or maxlen. `backend/src/agentic_rag_backend/protocols/a2a_resource_limits.py`
- [x] Redis tenant usage hashes never expire (TTL set only on session keys), leaving stale tenant keys indefinitely. `backend/src/agentic_rag_backend/protocols/a2a_resource_limits.py`
- [x] Redis active session counts can drift when session keys expire via TTL without `close_session`; cleanup only clamps negatives, not reconciles counts. `backend/src/agentic_rag_backend/protocols/a2a_resource_limits.py`
- [x] `/mcp/ui/config` has no per-tenant rate limiting. `backend/src/agentic_rag_backend/api/routes/mcp.py`
- [x] SSRF checks do not resolve hostnames; DNS rebinding/CNAME to private IPs can bypass checks. `backend/src/agentic_rag_backend/protocols/a2a_middleware.py`
- [x] Warn when A2A endpoints are unauthenticated because `A2A_API_KEY` is unset. `backend/src/agentic_rag_backend/main.py`
- [x] Ensure 429 responses include `Retry-After` guidance for A2A session/message limit errors. `backend/src/agentic_rag_backend/api/routes/a2a.py`, `backend/src/agentic_rag_backend/core/errors.py`
- [x] MCP-UI/Open-JSON-UI renderers are wrapped in a Copilot error boundary to prevent UI crashes from propagating. `frontend/components/copilot/tool-renderers.tsx`
- [x] Emit a warning when `METRICS_TENANT_LABEL_MODE=full` in production to avoid unbounded tenant label cardinality. `backend/src/agentic_rag_backend/main.py`

## Medium Findings
- [x] Epic 22 status tracking is inconsistent (epic tech spec says Backlog; sprint status says done; story 22-C1 still in-progress). `_bmad-output/epics/epic-22-tech-spec.md`, `_bmad-output/implementation-artifacts/sprint-status.yaml`, `_bmad-output/implementation-artifacts/stories/22-C1-implement-mcp-ui-renderer.md`
- [x] Compliance tests validate schemas but not runtime wiring; MCP-UI/Open-JSON-UI can pass tests while never rendering in UI. `frontend/__tests__/protocols/mcp-ui-compliance.test.ts`, `frontend/__tests__/protocols/open-json-ui-compliance.test.ts`
- [x] A2A middleware HTTP client read timeout is fixed at 30s; for long SSE streams it should be configurable or disable read timeout while keeping connect timeout. `backend/src/agentic_rag_backend/protocols/a2a_middleware.py`
- [x] A2A resource manager initialization errors surface as RuntimeError (500) instead of structured 503. `backend/src/agentic_rag_backend/api/routes/a2a.py`
- [x] `Retry-After` header is only set when `retry_after` is truthy, so valid `0` is dropped. `backend/src/agentic_rag_backend/core/errors.py`
- [x] AG-UI bridge emits RUN_FINISHED without RUN_STARTED when no user message; violates protocol ordering. `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`
- [x] MCP-UI allowed-origin fetch effect lacks abort/cleanup on unmount. `frontend/components/mcp-ui/MCPUIRenderer.tsx`
- [x] `record_stream_completed` helper decrements active gauge even if no matching start; can underflow. `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`
- [x] KeyError is mapped to SESSION_NOT_FOUND broadly; should be a specific session lookup error to avoid masking unrelated failures. `backend/src/agentic_rag_backend/protocols/ag_ui_errors.py`
- [x] Redis scan in cleanup uses default count; add `count=100` to avoid latency spikes. `backend/src/agentic_rag_backend/protocols/a2a_resource_limits.py`

## Low Findings
- [x] Use Pydantic discriminated unions for Open-JSON-UI component validation (`Field(discriminator="type")`). `backend/src/agentic_rag_backend/protocols/open_json_ui.py`
- [x] Prefer `exception.retry_after` when available instead of hardcoded 60s in error mapping. `backend/src/agentic_rag_backend/protocols/ag_ui_errors.py`
- [x] Remove unused Redis Lua SHA attributes or implement script caching (EVALSHA). `backend/src/agentic_rag_backend/protocols/a2a_resource_limits.py`

## Testing & Quality Notes
- [x] Replace real sleeps in tests with time mocking to avoid flakiness.
- [x] Isolate Prometheus CollectorRegistry per test to prevent cross-test pollution.
- [x] Avoid asserting on Prometheus internal attributes (`_name`, `_labelnames`).
- [x] Add E2E test for A2A delegation → MCP-UI rendering.
- [x] Add frontend edge-case tests for network failures and malformed UI payloads.

## Performance & Operational Watchlist
- [x] Monitor SSE stream memory usage for long-running streams.
- [x] Redis Lua scripts are atomic but block; monitor execution time under load.
- [x] Prometheus tenant_id label cardinality should be monitored in production.

## Security Review Recommendations
- [x] Add API gateway rate limiting in addition to in-app limits.
- [x] Add CSP headers for MCP-UI iframe responses.
- [x] Add request signing for A2A agent-to-agent communication.
- [x] Log security events (blocked origins, SSRF attempts) to SIEM.

## Investigation Notes (Resolved Questions)
### Canonical A2A Middleware API
Current canonical middleware endpoints are `/a2a/middleware/*`, not `/a2a/agents/*`. The latter targets the A2A registry and skips middleware tenant-prefix + API-key enforcement. Evidence: integration tests target `/a2a/middleware/*`, and middleware dependencies are only wired on those routes. `backend/tests/integration/test_a2a_middleware_api.py`, `backend/src/agentic_rag_backend/api/routes/a2a.py`

### A2A Resource Limits Integration
A2AResourceManager is now integrated into session create/message/close so rate limits and usage metrics share the same source of truth. Evidence: `register_session`, `record_message`, and `delete_session` are called from A2A routes and errors map to RFC 7807. `backend/src/agentic_rag_backend/api/routes/a2a.py`, `backend/src/agentic_rag_backend/protocols/a2a_resource_limits.py`

### MCP-UI / Open-JSON-UI Rendering Path
Tool-call results now dispatch `mcp_ui` and `open_json_ui` payloads to MCPUIRenderer/OpenJSONUIRenderer via the CopilotKit tool renderer hook, with error boundaries and tests validating the wiring. `frontend/components/copilot/tool-renderers.tsx`, `frontend/__tests__/components/copilot/tool-renderers.test.tsx`

## Implementation Checklist
- [x] Decide canonical A2A middleware endpoint and align docs + story ACs (either move middleware to `/a2a/agents/*` or document `/a2a/middleware/*` as canonical).
- [x] Integrate A2AResourceManager into session lifecycle (create/message/close) and map its exceptions to RFC 7807 errors (or remove duplicate A2ASessionManager limits).
- [x] Wire MCP-UI and Open-JSON-UI renderers into the CopilotKit tool rendering pipeline (or add explicit AG-UI UI payload rendering).
- [x] Tighten backend Open-JSON-UI validation (typed component union, URL validation) to match frontend Zod schemas.
- [x] Emit structured AG-UI RUN_ERROR for missing tenant_id and integrate `useAGUIErrorHandler`/`parseAGUIError` in the chat UI path.
- [x] Implement MCP-UI signing verification (or remove secret), and document actual sandbox/CSP behavior.
- [x] Add telemetry event allowlist/normalization to prevent label cardinality explosion.
- [x] Update `.env.example` and protocol docs to match actual config names and behavior.

## PR-ready Work Items
### PR 1: Enforce A2A Resource Limits in Session Lifecycle
- Scope: wire `A2AResourceManager` into session create/message/close and map limit errors to RFC 7807 429 responses.
- Files: `backend/src/agentic_rag_backend/api/routes/a2a.py`, `backend/src/agentic_rag_backend/protocols/a2a_resource_limits.py`, `backend/src/agentic_rag_backend/core/errors.py`, `.env.example`
- Acceptance: session/message limits enforced and metrics reflect actual usage; retry_after included for rate limits.
- Tests: update `backend/tests/integration/test_a2a_resource_limits_api.py` and `backend/tests/protocols/compliance/test_a2a_compliance.py`.

### PR 2: Align A2A Middleware Endpoint Contract
- Scope: choose canonical middleware route and enforce X-Tenant-ID + API key + tenant prefix checks with correct status codes.
- Files: `backend/src/agentic_rag_backend/api/routes/a2a.py`, `docs/guides/protocol-integration/a2a-protocol.md`, `.env.example`
- Acceptance: middleware registration requires tenant header and returns 403 on tenant-prefix mismatch; docs match API.
- Tests: update `backend/tests/integration/test_a2a_middleware_api.py`.

### PR 3: Render MCP-UI and Open-JSON-UI in Tool Call UI
- Scope: dispatch tool results with `type: "mcp_ui"` or `type: "open_json_ui"` to the appropriate renderer in the tool-call pipeline.
- Files: `frontend/components/copilot/tool-renderers.tsx`, `frontend/components/mcp-ui/MCPUIRenderer.tsx`, `frontend/components/open-json-ui/OpenJSONUIRenderer.tsx`
- Acceptance: MCP-UI iframe and Open-JSON-UI components render from tool results; fallback card remains for unknown payloads.
- Tests: add tool-renderer tests and update `frontend/__tests__/components/mcp-ui/MCPUIRenderer.test.tsx`.

### PR 4: Harden Open-JSON-UI Backend Validation
- Scope: replace `components: list[dict]` with a discriminated union and validate URLs for image/link components.
- Files: `backend/src/agentic_rag_backend/protocols/open_json_ui.py`, `backend/tests/unit/protocols/test_open_json_ui.py`
- Acceptance: invalid component types and unsafe URLs are rejected server-side.
- Tests: expand unit tests for invalid component schemas and URL validation.

### PR 5: AG-UI Structured Errors + Frontend Handling
- Scope: emit `AGUIErrorEvent` for missing tenant_id and wire `useAGUIErrorHandler` into the chat UI event flow.
- Files: `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`, `frontend/components/copilot/CopilotProvider.tsx` (or a new hook), `frontend/components/copilot/ErrorHandler.tsx`
- Acceptance: RUN_ERROR is always structured and surfaced to users via toasts/handlers.
- Tests: add frontend tests for error handling and update backend integration tests for tenant-missing errors.

### PR 6: MCP-UI Security Enforcement + Docs Alignment
- Scope: implement signed URL verification (or remove secret), add CSP `frame-src` enforcement, align docs with runtime sandbox defaults.
- Files: `backend/src/agentic_rag_backend/api/routes/mcp.py` (or middleware), `backend/src/agentic_rag_backend/models/mcp_ui.py`, `docs/guides/protocol-integration/mcp-ui-rendering.md`
- Acceptance: CSP is enforced and signing behavior is explicit and tested; docs match code.
- Tests: add backend tests for config/signing/CSP where applicable.

### PR 7: Telemetry Metrics Cardinality Guardrails
- Scope: normalize/allowlist telemetry event labels and align metric naming with docs or story decision.
- Files: `backend/src/agentic_rag_backend/observability/metrics.py`, `backend/src/agentic_rag_backend/api/routes/telemetry.py`
- Acceptance: unknown events map to a bounded label (e.g., `other`) and metrics remain stable.
- Tests: extend `backend/tests/api/test_telemetry.py` for allowlist behavior.

### PR 8: A2A Middleware Streaming + Error Mapping Precision
- Scope: implement StreamingResponse for middleware delegation; tighten SSE parsing and error mapping for session lookup.
- Files: `backend/src/agentic_rag_backend/api/routes/a2a.py`, `backend/src/agentic_rag_backend/protocols/ag_ui_errors.py`, `backend/src/agentic_rag_backend/protocols/a2a_middleware.py`
- Acceptance: true streaming behavior, no truncated streams, and accurate SESSION_NOT_FOUND mapping.
- Tests: add integration test for streaming and specific session-not-found error mapping.

### PR 9: Redis Resource Manager Hygiene
- Scope: set TTL on tenant hashes, reconcile active session counts, add scan count, and optionally add Lua script caching.
- Files: `backend/src/agentic_rag_backend/protocols/a2a_resource_limits.py`
- Acceptance: no stale tenant keys, counts match live sessions, scan is bounded.
- Tests: add Redis integration test for TTL + count reconciliation.

### PR 10: AG-UI Metrics & Error Handling Polishing
- Scope: fix record_stream_completed gauge adjustment, add RUN_STARTED for empty requests, and ensure Retry-After header presence check.
- Files: `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`, `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`, `backend/src/agentic_rag_backend/core/errors.py`
- Acceptance: protocol ordering preserved, gauge never underflows, Retry-After always emitted when present.
- Tests: expand metrics unit tests and AG-UI ordering tests.

### PR 11: Security Hardening
- Scope: validate MCP-UI signing secret, resolve hostnames for SSRF checks, and add rate limiting for `/mcp/ui/config`.
- Files: `backend/src/agentic_rag_backend/config.py`, `backend/src/agentic_rag_backend/protocols/a2a_middleware.py`, `backend/src/agentic_rag_backend/api/routes/mcp.py`
- Acceptance: secure secret handling, SSRF protection covers DNS rebinding, MCP-UI config is rate-limited.
- Tests: add SSRF hostname resolution tests and MCP-UI config rate-limit tests.

### PR 12: Docs + Status Alignment
- Scope: update protocol docs to match real env vars and reconcile epic/story status artifacts with actual implementation.
- Files: `docs/guides/protocol-integration/*.md`, `.env.example`, `_bmad-output/epics/epic-22-tech-spec.md`, `_bmad-output/implementation-artifacts/sprint-status.yaml`, `_bmad-output/implementation-artifacts/stories/22-C1-implement-mcp-ui-renderer.md`
- Acceptance: docs reflect current configuration and status tracking is consistent.
- Tests: n/a.

## Open Decisions (Need Alignment)
- A2A limit ownership: Should A2ASessionManager be the only limiter, or should A2AResourceManager replace it entirely?
- MCP-UI/Open-JSON-UI trigger: Should tool-call results dispatch to renderers by payload `type`, or should AG-UI emit explicit UI events?
