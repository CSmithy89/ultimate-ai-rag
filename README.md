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
uv run alembic upgrade head
uv run agentic-rag-backend
```

Notes:
- Rate limiting supports `RATE_LIMIT_BACKEND=redis` for multi-worker deployments; the in-memory limiter is per-process.

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

### Epic 3: Hybrid Knowledge Retrieval
- Status: Complete
- Stories: 4/4 completed
- Key Features:
  - Vector semantic search over pgvector embeddings
  - Neo4j relationship traversal with tenant-scoped queries
  - Hybrid answer synthesis combining vector + graph evidence
  - Graph explainability artifacts (nodes, edges, paths, explanations)
