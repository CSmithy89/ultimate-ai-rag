# CopilotKit Feature Gap Roadmap

This guide catalogs CopilotKit features referenced earlier (A2UI, AG-UI, A2A, MCP, generative UI specs, MCP client integration, and observability) and maps what is implemented today versus what is missing. Each gap includes a concrete implementation approach and the system improvements it enables. This document is intended to evolve into an epic plan.

## Current Coverage (Baseline)
- **AG-UI transport**: CopilotKit AG-UI endpoint and generic AG-UI endpoint exist (`backend/src/agentic_rag_backend/api/routes/copilot.py`, `backend/src/agentic_rag_backend/api/routes/ag_ui.py`).
- **A2A**: Session lifecycle and message APIs are implemented (`backend/src/agentic_rag_backend/api/routes/a2a.py`).
- **MCP**: Tool registry and call APIs exist, plus a dedicated MCP server (`backend/src/agentic_rag_backend/api/routes/mcp.py`, `backend/src/agentic_rag_backend/mcp_server/`).
- **Frontend**: CopilotKit React dependencies are present and custom UI components exist in `frontend/components/`.

## Feature Coverage Matrix (Implemented vs Missing)
Status labels: **Implemented**, **Partial**, **Missing**.

| Feature | Status | Wiring (current or needed) | Improvement |
| --- | --- | --- | --- |
| AG-UI transport (SSE events) | Implemented | Backend routes: `backend/src/agentic_rag_backend/api/routes/copilot.py`, `backend/src/agentic_rag_backend/api/routes/ag_ui.py` | Baseline interoperability with CopilotKit clients |
| Prebuilt Copilot UI (CopilotSidebar) | Implemented | `frontend/components/copilot/ChatSidebar.tsx` wraps `@copilotkit/react-ui` | Faster UI integration and consistent chat UX |
| Custom Generative UI actions | Implemented | `frontend/components/copilot/GenerativeUIRenderer.tsx`, hooks in `frontend/hooks/` | Rich, app-specific UI for sources/answers/graphs |
| A2A protocol | Implemented | `backend/src/agentic_rag_backend/api/routes/a2a.py` + SDK (`backend/src/agentic_rag_backend/sdk/`) | Agent collaboration and session persistence |
| MCP server + tool registry | Implemented | `backend/src/agentic_rag_backend/mcp_server/`, `backend/src/agentic_rag_backend/api/routes/mcp.py` | Tool discovery and invocation for clients |
| MCP client integration (outbound) | Missing | Add MCP client factory + config, merge into registry | Access external tool ecosystems |
| A2UI spec support | Missing | Add A2UI schema validation + renderer | Portable, declarative UI output |
| MCP-UI spec support | Missing | Add iframe renderer + CSP + allowlist | Embed external interactive tools safely |
| Open-JSON-UI support | Missing | Add schema validation + renderer | Interop with OpenAI-style UI payloads |
| Tool call visualization | Missing | Add tool call UI component | Debuggability and user transparency |
| Observability + Inspector | Missing | Add hooks + Inspector UI + event pipeline | Faster debugging and product insight |
| Headless UI mode | Partial | Hooks exist via `@copilotkit/react-core` but not fully documented | Full UI control for advanced clients |
| BYO LLM adapters | Implemented (backend) | Provider adapters in backend config | Flexible model/provider selection |

## Gaps and Implementation Plan

### 1) A2UI (Agent-to-UI) Support
**What’s missing:** No A2UI spec handling or rendering. No A2UI payload validation or rendering pipeline.

**Implementation:**
- **Backend**: Allow A2UI payloads through AG-UI events. Add a schema validator module (e.g., `backend/src/agentic_rag_backend/protocols/a2ui.py`) to validate A2UI JSON before streaming to clients. Add a feature flag (`A2UI_ENABLED=true`) in `backend/src/agentic_rag_backend/config.py`.
- **Frontend**: Implement an A2UI renderer in `frontend/components/copilot/` that maps A2UI widgets to React components. Start with a safe allowlist of widgets (cards, tables, forms, charts) and a fallback to text rendering when unsupported.
- **Prompting**: Update orchestrator prompts to emit A2UI when `A2UI_ENABLED` is on and a widget is requested or useful.
- **Security**: Enforce strict schema validation and allowlisted component types to avoid UI injection.
- **Testing**: Add schema validation tests and rendering snapshot tests in `frontend/__tests__/`.

**Improvements:** Faster UI iteration, portable agent output across clients, richer UX (structured results instead of plain text).

### 2) MCP-UI Support (iframe-based)
**What’s missing:** No MCP-UI renderer or iframe sandbox handling.

**Implementation:**
- **Backend**: Accept MCP-UI payloads over AG-UI and pass through metadata needed for iframe sources. Add signed URLs or proxy endpoints if remote UI needs hosting.
- **Frontend**: Build an MCP-UI renderer that uses sandboxed iframes (`sandbox` + CSP) and a postMessage bridge for resizing and event handling.
- **Security**: Strict origin allowlist and content-security-policy headers.
- **Testing**: Add unit tests for iframe rendering and postMessage handling.

**Improvements:** Enables embedding external interactive tools without custom UI builds.

### 3) Open-JSON-UI Support
**What’s missing:** No renderer or schema support for OpenAI-style declarative UI.

**Implementation:**
- **Backend**: Add schema validation and pass Open-JSON-UI payloads in AG-UI events.
- **Frontend**: Implement a renderer that maps Open-JSON-UI components to existing UI primitives, with fallback to text.
- **Testing**: Contract tests using canonical Open-JSON-UI payloads.

**Improvements:** Interoperability with agents that emit Open-JSON-UI, reducing integration friction.

### 4) MCP Client Integration (Outbound)
**What’s missing:** The system exposes an MCP server, but doesn’t connect to external MCP servers.

**Implementation:**
- **Backend**: Add MCP client configuration (e.g., `MCP_CLIENTS=[{name,url,auth}]`) and implement a client factory. Wire it into tool execution so the orchestrator can call external MCP tools.
- **Routing**: Merge internal and external MCP tools into a unified registry with namespacing.
- **Resilience**: Timeout controls (reuse existing `MCP_TOOL_TIMEOUT_*` settings), retries, and circuit breakers.
- **Testing**: Integration tests against a mock MCP server.

**Improvements:** Immediate access to third-party tool ecosystems and broader agent capabilities.

### 5) AG-UI Enhancements
**What’s missing:** The transport exists, but richer telemetry and event auditing are limited.

**Implementation:**
- **Backend**: Add AG-UI stream metrics (event count, latency, drop rate). Emit standardized error events for stream failures.
- **Frontend**: Expose stream status indicators for user trust and debugging.
- **Testing**: Load tests for sustained event streams.

**Improvements:** Higher reliability, clearer error handling, and improved developer visibility.

### 6) A2A Enhancements
**What’s missing:** Per-tenant caps and observability are noted as TODOs in `docs/todo.md`.

**Implementation:**
- **Backend**: Implement per-tenant message/session caps and emit metrics for session churn and message throughput. Add periodic cleanup metrics.
- **Testing**: Concurrency and TTL tests; verify cap enforcement.

**Improvements:** Predictable resource usage and safer multi-tenant scaling.

### 7) Observability + Inspector
**What’s missing:** No Inspector UI or observability hooks wired to UI events.

**Implementation:**
- **Frontend**: Wire CopilotKit Inspector (license-key gated) and add observability hooks to Copilot UI components.
- **Backend**: Add endpoints for event ingestion and dashboards (if using external observability, emit structured logs).
- **Testing**: Validate event emission and basic dashboards.

**Improvements:** Faster debugging, lower MTTR, and evidence-based UX iteration.

### 8) Tool Call Visualization
**What’s missing:** No UI component showing MCP tool calls.

**Implementation:**
- **Frontend**: Add an MCP tool call component (collapsed JSON viewer) and render it for tool call events.
- **Backend**: Ensure tool call events include status, args, and results.

**Improvements:** Transparency for users and better debugging for developers.

## Sequencing (Suggested)
1. A2UI support (foundation for richer UI)
2. Observability + tool call visualization (debugging leverage)
3. MCP client integration (tool ecosystem expansion)
4. MCP-UI + Open-JSON-UI (interop breadth)
5. A2A caps and AG-UI telemetry polish

## Implementation Backlog (Epic-Ready)
Ownership tags: **Backend**, **Frontend**, **Platform/DevOps**, **Product/UX**.

### A2UI Support
- **Owner**: Backend + Frontend
- **Dependencies**: AG-UI stream pipeline; existing CopilotKit UI shell
- **Tasks**:
  - Backend: add A2UI schema validator and `A2UI_ENABLED` flag; reject invalid payloads with RFC 7807 errors.
  - Backend: expose A2UI payloads through AG-UI events (new event type or payload envelope).
  - Frontend: build A2UI renderer with allowlisted widgets and safe fallbacks.
  - Prompts: enable A2UI output when enabled and beneficial.
  - Tests: schema validation tests + rendering snapshot tests.
- **Acceptance**: A2UI widget renders in UI and falls back cleanly on unsupported components.

### Observability + Tool Call Visualization
- **Owner**: Frontend + Backend
- **Dependencies**: AG-UI event flow and MCP tool call metadata
- **Tasks**:
  - Frontend: add tool call UI component for args/results with status badges.
  - Backend: emit standardized tool call events (start/args/end) with timing.
  - Frontend: wire CopilotKit Inspector + observability hooks (if license used).
  - Backend: emit AG-UI and MCP metrics (latency, counts, failures).
  - Tests: verify tool call render + event emission.
- **Acceptance**: Tool calls are visible in UI and metrics are emitted per request.

### MCP Client Integration (Outbound)
- **Owner**: Backend
- **Dependencies**: MCP tool registry, existing timeout settings
- **Tasks**:
  - Add MCP client config (env or settings) and client factory.
  - Merge external MCP tools into registry with namespacing.
  - Add retries/timeouts and error normalization.
  - Integration tests against a mock MCP server.
- **Acceptance**: External MCP tool calls succeed with retries and timeouts enforced.

### MCP-UI Support
- **Owner**: Frontend + Backend
- **Dependencies**: AG-UI payload envelope, CSP configuration
- **Tasks**:
  - Backend: accept MCP-UI payloads and provide signed/proxied iframe URLs.
  - Frontend: sandboxed iframe renderer with postMessage bridge.
  - Security: CSP + allowlist for iframe origins.
  - Tests: iframe render + postMessage resize handling.
- **Acceptance**: MCP-UI widgets render safely with sandboxing.

### Open-JSON-UI Support
- **Owner**: Frontend + Backend
- **Dependencies**: Schema validator framework
- **Tasks**:
  - Backend: validate Open-JSON-UI payloads and pass through AG-UI.
  - Frontend: map Open-JSON-UI components to existing primitives.
  - Tests: contract payload fixtures for validation and rendering.
- **Acceptance**: Canonical Open-JSON-UI payloads render without errors.

### AG-UI Telemetry Enhancements
- **Owner**: Backend
- **Dependencies**: Observability pipeline
- **Tasks**:
  - Emit stream metrics (latency, event count, drop rate).
  - Standardize error event payloads for client consumption.
  - Add load tests for sustained streams.
- **Acceptance**: Metrics and standardized errors appear in logs/telemetry.

### A2A Limits + Metrics
- **Owner**: Backend
- **Dependencies**: A2A session manager
- **Tasks**:
  - Implement per-tenant session/message caps with config.
  - Emit metrics for session churn and message throughput.
  - Extend concurrency and TTL tests.
- **Acceptance**: Cap enforcement verified; metrics available.

## Dependencies & Risks
- **UI spec changes**: A2UI/MCP-UI/Open-JSON-UI schemas may evolve; pin versions and document.
- **Security**: iframe and declarative UI require strict allowlists and CSP.
- **Observability**: Ensure metrics are sampled to avoid performance regressions.

## Success Metrics
- A2UI render success rate and fallback frequency
- MCP tool latency percentiles
- AG-UI stream error rate
- A2A session churn and cap enforcement
- Time-to-debug (Inspector adoption)
