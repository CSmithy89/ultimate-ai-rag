# Ultimate AI RAG

## Quick Start

### Environment

```bash
cp .env.example .env
```

Edit `.env` and set at least `OPENAI_API_KEY` and database credentials.

### Backend

```bash
cd backend
uv sync
uv run agentic-rag-backend
```

### Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

### Full Stack (Docker Compose)

```bash
docker compose up -d
```

## Epic Progress

### Epic 1: Foundation & Developer Quick Start
- Status: Complete
- Stories: 4/4 completed
- Key Features:
  - Backend scaffold using Agno agent-api + FastAPI
  - Frontend scaffold using Next.js App Router + CopilotKit deps
  - Docker Compose dev stack with Postgres/pgvector, Neo4j, Redis
  - Environment configuration via .env validation

### Epic 2: Agentic Query & Reasoning
- Status: Complete
- Stories: 4/4 completed
- Key Features:
  - Orchestrator agent API with `POST /query`
  - Multi-step planning with visible plan and thought list
  - Dynamic retrieval strategy selection (vector/graph/hybrid)
  - Persistent trajectory logging to Postgres with trajectory IDs
