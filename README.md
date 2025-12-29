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
  - Vector semantic search with pgvector embeddings
  - Graph relationship traversal via Neo4j
  - Hybrid answer synthesis combining vector and graph results
  - Graph-based explainability with source attribution

### Epic 4: Knowledge Ingestion Pipeline
- Status: Complete
- Stories: 4/4 completed
- Key Features:
  - URL documentation crawling with Crawl4AI
  - PDF document parsing with Docling
  - Agentic entity extraction with LLM-powered NER
  - Knowledge graph visualization endpoints

### Epic 5: Graphiti Temporal Knowledge Graph Integration
- Status: Complete
- Stories: 6/6 completed
- Key Features:
  - Graphiti integration for temporal knowledge graphs
  - Episode-based document ingestion with automatic entity/edge extraction
  - Hybrid retrieval with Graphiti search + vector fallback
  - Temporal query capabilities (point-in-time search, knowledge changes)
  - Custom entity types (TechnicalConcept, CodePattern, APIEndpoint, ConfigurationOption)
  - Feature flags for backend selection (`INGESTION_BACKEND`, `RETRIEVAL_BACKEND`)
  - Legacy module deprecation with migration path
  - Comprehensive test suite with 263 tests (86%+ Graphiti module coverage)
