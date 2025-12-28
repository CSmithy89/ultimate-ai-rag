# Epic 1 Tech Spec: Foundation & Developer Quick Start

Date: 2025-12-28

## Overview
Epic 1 delivers a zero-friction developer setup for the Agentic RAG + GraphRAG system.
The goal is to get a first response in under 15 minutes using Docker Compose with no
initial code modifications. This epic establishes the baseline backend and frontend
projects plus a local multi-service development environment.

## Goals
- Provide a working backend using the Agno agent-api starter (FastAPI + agents).
- Provide a working frontend using Next.js 15+ with CopilotKit integration.
- Provide a Docker Compose stack for all required services with health checks.
- Enable environment-based configuration via `.env` and `.env.example`.

## Non-goals
- Retrieval logic (vector or graph) beyond service scaffolding.
- Production deployment hardening.
- UI feature work beyond getting the app running.

## Architecture Summary
### Backend
- Stack: Python 3.11+, Agno agent-api starter, FastAPI.
- Package management: `uv`.
- Key dependencies: Agno v2.3.21, FastAPI, PostgreSQL client.
- Expected path: `backend/`.

### Frontend
- Stack: Next.js 15+ (App Router), TypeScript 5.x, CopilotKit UI + core.
- Package management: `pnpm`.
- Styling: Tailwind CSS (per starter defaults).
- Expected path: `frontend/`.

### Data/Infra Services
- PostgreSQL + pgvector (port 5432).
- Neo4j Community (ports 7474, 7687).
- Redis (port 6379).

### Local Orchestration
- Docker Compose orchestrates all services.
- Hot reload for backend and frontend via bind mounts.
- Health checks for all services.

## Key Decisions
- **Backend starter**: Agno agent-api provides agent scaffolding and FastAPI.
- **Frontend starter**: Next.js + CopilotKit enables AG-UI integration.
- **Graph database**: Neo4j Community for graph traversal and tooling support.
- **Vector store**: pgvector via PostgreSQL for similarity search.

## Configuration
Provide `.env.example` with the minimum required variables and point both frontend
and backend to read from `.env`.

Required variables (baseline):
- `OPENAI_API_KEY`
- `DATABASE_URL` (Postgres/pgvector)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `REDIS_URL`

## Story Breakdown
### Story 1.1: Backend Project Initialization
- Initialize backend via Agno agent-api starter.
- Ensure `uv sync` installs dependencies.
- Validate `pyproject.toml` includes Agno v2.3.21 and FastAPI.
- Confirm backend project structure aligns with Agno starter conventions.

### Story 1.2: Frontend Project Initialization
- Initialize frontend via Next.js 15+ with App Router.
- Install CopilotKit packages.
- Ensure `pnpm install` completes and app starts.

### Story 1.3: Docker Compose Development Environment
- Compose file starts backend, frontend, Postgres, Neo4j, Redis.
- Health checks for all services.
- Hot reload enabled for backend and frontend.

### Story 1.4: Environment Configuration System
- `.env.example` included with required variables.
- Backend validates required vars at startup with clear errors.
- Connection strings configurable via environment variables.

## Risks and Mitigations
- **Starter template drift**: Pin specific versions in `pyproject.toml` and `package.json`.
- **Cross-platform Docker issues**: Document known Docker Desktop requirements.
- **Developer setup friction**: Provide clear README instructions and defaults.

## Validation
- Smoke test: `uv sync` and `uv run` for backend.
- Smoke test: `pnpm install` and `pnpm dev` for frontend.
- Stack test: `docker compose up -d` brings up all services with healthy status.
