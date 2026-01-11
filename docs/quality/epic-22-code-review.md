# Epic 22 Code Review

Scope: A2A middleware + resource limits, AG-UI metrics/errors, MCP-UI, Open-JSON-UI, protocol docs, and Epic 22 tech debt items.

## Critical Findings (Must Fix Before Completion)
- [ ] A2A resource limits are not enforced in session or message endpoints; the manager is only used by `/a2a/metrics/{tenant_id}`. `backend/src/agentic_rag_backend/api/routes/a2a.py`, `backend/src/agentic_rag_backend/main.py`
- [ ] A2A middleware registration contract diverges from the story: `/a2a/agents/register` (registry) accepts tenant_id in body and skips header/API-key/prefix validation, while the middleware uses `/a2a/middleware/*`. `backend/src/agentic_rag_backend/api/routes/a2a.py`
- [ ] MCP-UI and Open-JSON-UI renderers are not wired into the CopilotKit rendering pipeline; tool calls only render MCPToolCallCard/VectorSearchCard. `frontend/components/copilot/tool-renderers.tsx`, `frontend/components/mcp-ui/MCPUIRenderer.tsx`, `frontend/components/open-json-ui/OpenJSONUIRenderer.tsx`
- [ ] Backend Open-JSON-UI models do not validate component schemas or URL safety; `OpenJSONUIPayload.components` is `list[dict]`. `backend/src/agentic_rag_backend/protocols/open_json_ui.py`
- [ ] Missing tenant_id in AG-UI bridge emits only text; it never emits a structured RUN_ERROR event, and the frontend error handler hook is not integrated anywhere. `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`, `frontend/components/copilot/ErrorHandler.tsx`

## High Findings
- [ ] MCP-UI signing secret is configured but unused, CSP headers are not enforced, and docs show sandbox attributes that do not match defaults. `backend/src/agentic_rag_backend/config.py`, `backend/src/agentic_rag_backend/models/mcp_ui.py`, `docs/guides/protocol-integration/mcp-ui-rendering.md`
- [ ] A2A limits config drift: new `A2A_*_LIMITS` env vars exist in config but are missing from `.env.example`, while older session limit vars still exist. `backend/src/agentic_rag_backend/config.py`, `.env.example`
- [ ] Telemetry metrics label cardinality control is missing; event names are unbounded and the metric name differs from the story. `backend/src/agentic_rag_backend/observability/metrics.py`, `backend/src/agentic_rag_backend/api/routes/telemetry.py`
- [ ] Protocol docs list env vars and APIs that do not exist in code (AGUI_STREAM_TIMEOUT_SECONDS, A2A_DEFAULT_TIMEOUT_SECONDS, A2A_MAX_DELEGATION_DEPTH, A2A_REDIS_URL, OPEN_JSON_UI_ENABLED). `docs/guides/protocol-integration/overview.md`, `docs/guides/protocol-integration/a2a-protocol.md`

## Medium Findings
- [ ] Epic 22 status tracking is inconsistent (epic tech spec says Backlog; sprint status says done; story 22-C1 still in-progress). `_bmad-output/epics/epic-22-tech-spec.md`, `_bmad-output/implementation-artifacts/sprint-status.yaml`, `_bmad-output/implementation-artifacts/stories/22-C1-implement-mcp-ui-renderer.md`
- [ ] Compliance tests validate schemas but not runtime wiring; MCP-UI/Open-JSON-UI can pass tests while never rendering in UI. `frontend/__tests__/protocols/mcp-ui-compliance.test.ts`, `frontend/__tests__/protocols/open-json-ui-compliance.test.ts`

## Investigation Notes (Resolved Questions)
### Canonical A2A Middleware API
Current canonical middleware endpoints are `/a2a/middleware/*`, not `/a2a/agents/*`. The latter targets the A2A registry and skips middleware tenant-prefix + API-key enforcement. Evidence: integration tests target `/a2a/middleware/*`, and middleware dependencies are only wired on those routes. `backend/tests/integration/test_a2a_middleware_api.py`, `backend/src/agentic_rag_backend/api/routes/a2a.py`

### A2A Resource Limits Integration
A2ASessionManager already enforces per-tenant and per-session limits, while A2AResourceManager is unused outside metrics and background cleanup. This creates parallel limit systems with no shared state. Evidence: no calls to `register_session`/`record_message` in A2A routes, and no session close endpoint to decrement counts. `backend/src/agentic_rag_backend/api/routes/a2a.py`, `backend/src/agentic_rag_backend/protocols/a2a_resource_limits.py`

### MCP-UI / Open-JSON-UI Rendering Path
Neither renderer is registered in the tool-call rendering hook, and MCPToolCallCard only stringifies results. There is no AG-UI event or tool result handler that dispatches `mcp_ui` or `open_json_ui` payloads. `frontend/components/copilot/tool-renderers.tsx`, `frontend/components/copilot/MCPToolCallCard.tsx`

## Implementation Checklist
- [ ] Decide canonical A2A middleware endpoint and align docs + story ACs (either move middleware to `/a2a/agents/*` or document `/a2a/middleware/*` as canonical).
- [ ] Integrate A2AResourceManager into session lifecycle (create/message/close) and map its exceptions to RFC 7807 errors (or remove duplicate A2ASessionManager limits).
- [ ] Wire MCP-UI and Open-JSON-UI renderers into the CopilotKit tool rendering pipeline (or add explicit AG-UI UI payload rendering).
- [ ] Tighten backend Open-JSON-UI validation (typed component union, URL validation) to match frontend Zod schemas.
- [ ] Emit structured AG-UI RUN_ERROR for missing tenant_id and integrate `useAGUIErrorHandler`/`parseAGUIError` in the chat UI path.
- [ ] Implement MCP-UI signing verification (or remove secret), and document actual sandbox/CSP behavior.
- [ ] Add telemetry event allowlist/normalization to prevent label cardinality explosion.
- [ ] Update `.env.example` and protocol docs to match actual config names and behavior.

## Open Decisions (Need Alignment)
- A2A limit ownership: Should A2ASessionManager be the only limiter, or should A2AResourceManager replace it entirely?
- MCP-UI/Open-JSON-UI trigger: Should tool-call results dispatch to renderers by payload `type`, or should AG-UI emit explicit UI events?
