# Review Follow-ups

## Implemented
- [x] Fix backend app export so `from agentic_rag_backend import app` works.
- [x] Harden config loading with BACKEND_PORT validation and clearer missing-var errors.
- [x] Prevent frontend volume mount from clobbering node_modules in Docker Compose.
- [x] Make frontend Docker build reproducible with root context + pnpm lockfile.
- [x] Add development-only credential warnings and env defaults.
- [x] Pin core runtime dependencies for reproducible installs.
- [x] Update frontend healthcheck to use Node http instead of fetch.
- [x] Clarify README env setup instructions.
- [x] Add RootLayout return type annotation.
- [x] Gemini Code Assist bot is active on the repo (GitHub App).
- [x] Add shared API meta helper to avoid duplication across routes.
- [x] Initialize A2A session manager and MCP registry during app startup.
- [x] Add type hints for AG-UI dependencies and SDK context manager exit.
- [x] Enforce MCP tenant_id as non-empty string and add tool timeout handling.
- [x] Cache MCP registry in app state to avoid per-request recreation.
- [x] Add A2A session limits, TTL pruning, and thread safety guard.
- [x] Add AG-UI error event fallback for failed streams.
- [x] Align orchestrator logging with structlog key/value style.
- [x] Add tests for MCP graph_stats without Neo4j and AG-UI empty messages.
- [x] Make MCP tool timeouts configurable per tool or tool class.
- [x] Narrow broad exception handling in MCP/A2A routes where practical.
- [x] Add SDK retry/backoff for 429/503 and custom SDK exception types.
- [x] Add integration tests that cover MCP → orchestrator → response flow.
- [x] Add AG-UI validation tests for invalid actions payloads.

## Optional / Future
- [ ] Add Dependabot for npm + pip updates.
- [ ] Add CodeQL security scanning.
- [ ] Consider switching backend healthcheck to curl/wget (requires adding it to the image).
- [ ] Add TODOs/notes for deferred service wiring (DB/Redis clients) if needed.
- [ ] Add persistent A2A session storage (e.g., Redis) or clearly document in-memory limits.
- [ ] Add per-tenant A2A message limits and session caps for tiered plans.
- [ ] Replace MCP tool arg validation with Pydantic models for all tools.
- [x] Replace A2A threading.Lock with asyncio.Lock and make session ops async-safe.
- [x] Add RFC 7807 Problem Details responses for HTTPException-based errors (incl. AG-UI rate limits).
- [x] Log AG-UI stream fallback exceptions before emitting error events.
- [x] Add A2A TTL expiration test with controlled time.
- [x] Add A2A concurrent access tests (session creation + message add).
- [x] Add MCP timeout path test using a slow orchestrator mock.
- [x] Expand AG-UI error handling test to assert fallback event structure.
- [x] Document rate limit behavior (429 + retry-after) and A2A session lifecycle.
- [x] Add SDK usage examples in docstrings or README.
- [x] Remove redundant MCP registry lazy init in routes (use app state only).
- [ ] Add observability metrics for A2A sessions, MCP tool latency, AG-UI streams.
