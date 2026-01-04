# Repository Guidelines

## Project Structure & Module Organization
- `backend/src/agentic_rag_backend/`: FastAPI service, protocols, retrieval, ingestion, and MCP/A2A/AG-UI code.
- `backend/tests/`: Python unit + integration tests (pytest).
- `frontend/`: Next.js app with UI, hooks, and CopilotKit integration.
- `frontend/__tests__/`: Jest + React Testing Library tests.
- `docs/`: Architecture notes, runbooks, epics, and quality checklists.
- `scripts/` and `backend/scripts/`: maintenance utilities and migrations.
- `docker-compose.yml`: local stack for Postgres/Neo4j/Redis.

## Build, Test, and Development Commands
- `cd backend && uv sync`: install backend dependencies.
- `cd backend && uv run alembic upgrade head`: apply DB migrations.
- `cd backend && uv run agentic-rag-backend`: run the API server.
- `cd frontend && pnpm install`: install frontend dependencies.
- `cd frontend && pnpm dev`: run the Next.js dev server at `localhost:3000`.
- `docker compose up -d`: start the full stack locally.
- `pnpm lint | pnpm test | pnpm type-check`: run monorepo checks via Turbo.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `ruff` enforced with line length 100 (`backend/pyproject.toml`).
- TypeScript/React: follow ESLint Next.js defaults (`frontend/package.json`).
- Keep module names descriptive (e.g., `protocols/`, `retrieval/`, `indexing/`).

## Testing Guidelines
- Backend uses `pytest`, `pytest-asyncio`, and `pytest-cov`.
- Integration tests require running services and `INTEGRATION_TESTS=1`:
  - `docker compose up -d postgres neo4j redis`
  - `cd backend && uv run pytest -m integration`
- Frontend uses Jest + React Testing Library:
  - `cd frontend && pnpm test`
- See skip inventory in `docs/testing/README.md`.

## Commit & Pull Request Guidelines
- Recent commits follow a short prefix format like `Fix: ...` or `Refactor: ...`; mirror that style.
- Use the PR template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Complete the pre-review checklist and the protocol compliance checklist when touching API routes (`docs/quality/pre-review-checklist.md`, `docs/quality/protocol-compliance-checklist.md`).

## Security & Configuration Tips
- Copy `.env.example` to `.env` and set provider keys (see `README.md`).
- `TRACE_ENCRYPTION_KEY` is required outside dev/test environments.

## Error Code Conventions

All API errors follow RFC 7807 Problem Details format. Error codes use these naming patterns:

| Domain | Pattern | Example |
|--------|---------|---------|
| General | `SCREAMING_SNAKE` | `VALIDATION_ERROR`, `NOT_FOUND` |
| A2A Protocol | `A2A_*` | `A2A_AGENT_NOT_FOUND`, `A2A_TASK_TIMEOUT` |
| MCP Protocol | `MCP_*` | `MCP_TOOL_NOT_FOUND`, `MCP_AUTH_FAILED` |
| Retrieval | `RETRIEVAL_*` | `RETRIEVAL_FAILED`, `RETRIEVAL_TIMEOUT` |
| Ingestion | `INGESTION_*` | `INGESTION_FAILED`, `INGESTION_QUOTA_EXCEEDED` |

Error classes extend `AppError` from `core/errors.py` and include:
- `code`: Error code enum value
- `message`: Human-readable description
- `status`: HTTP status code (400, 401, 403, 404, 500, 503)
- `details`: Optional dict with contextual data

When adding new error codes:
1. Add to `ErrorCode` enum in `core/errors.py`
2. Create corresponding `*Error` class extending `AppError`
3. Add to `ERROR_CODE_TO_STATUS` mapping if non-standard status
4. Document in this section if protocol-specific
