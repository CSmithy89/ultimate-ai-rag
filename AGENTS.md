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
