# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic RAG + GraphRAG infrastructure combining Agno agent orchestration, hybrid retrieval (Neo4j graph + pgvector), and CopilotKit for AI copilot UIs. This is a full-stack AI platform with Python backend and Next.js frontend.

## Development Commands

```bash
# Full stack (Docker Compose)
docker compose up -d

# Backend development (hot reload)
cd backend && uv sync && uv run uvicorn agentic_rag_backend.main:app --reload

# Frontend development
cd frontend && pnpm install && pnpm dev

# Run tests
cd backend && uv run pytest
pnpm turbo test          # Frontend via turbo

# Lint
cd backend && uv run ruff check src/
pnpm turbo lint          # Frontend via turbo

# Type check
pnpm turbo type-check    # Frontend
```

## Architecture

**Monorepo Structure:**
- `backend/` - Python/FastAPI/Agno service (port 8000)
- `frontend/` - Next.js 15 + CopilotKit (port 3000)
- Turborepo orchestrates frontend tasks

**Technology Stack:**
- Backend: Python 3.11+, FastAPI, Agno v2.3.21, uv package manager
- Frontend: Next.js 15 (App Router), TypeScript, CopilotKit, Tailwind CSS
- Databases: PostgreSQL 16 + pgvector, Neo4j 5 Community, Redis 7

**Protocol Compliance:**
- MCP (Model Context Protocol) for tool execution
- A2A for agent-to-agent delegation
- AG-UI for frontend state sync via CopilotKit

## Critical Conventions

**Naming:**
- Python: `snake_case` functions, `PascalCase` classes, `SCREAMING_SNAKE` constants
- TypeScript: `camelCase` functions, `PascalCase` components (file: `ComponentName.tsx`)
- Hooks: `use-hook-name.ts` with `useHookName` export
- Database tables: `snake_case`, plural
- Neo4j labels: `PascalCase` singular, relationships: `SCREAMING_SNAKE_CASE`

**API Responses:**
```json
// Success
{"data": {...}, "meta": {"requestId": "uuid", "timestamp": "ISO8601"}}

// Error (RFC 7807)
{"type": "url", "title": "Error", "status": 400, "detail": "message", "instance": "/path"}
```

**Multi-Tenancy:** Every database query MUST include `tenant_id` filtering.

**Agent Logging:** All agent decisions must use trajectory logging:
```python
agent.log_thought("...")
agent.log_action("tool_call", {...})
agent.log_observation("...")
```

**Frontend Data Fetching:** Always use TanStack Query, never raw `fetch()`.

**Validation:** Pydantic (backend), Zod (frontend).

## Project Structure

```
backend/src/agentic_rag_backend/
├── agents/       # Agno agent definitions
├── tools/        # MCP tool implementations
├── retrieval/    # RAG retrieval logic
├── indexing/     # Document ingestion
├── models/       # Pydantic models
├── api/routes/   # FastAPI endpoints
├── db/           # Database clients
├── core/         # Config, errors, logging
└── protocols/    # MCP, A2A, AG-UI

frontend/
├── app/          # Next.js App Router
├── components/   # UI components (shadcn/ui, copilot/, graphs/)
├── hooks/        # Custom React hooks
├── lib/          # API client, utilities
└── types/        # TypeScript types
```

## Planning Artifacts

The `_bmad-output/` directory contains project planning documents:
- `architecture.md` - Complete architectural decisions
- `project-context.md` - Implementation rules summary
- `project-planning-artifacts/epics.md` - All user stories
- `implementation-artifacts/sprint-status.yaml` - Development tracking
