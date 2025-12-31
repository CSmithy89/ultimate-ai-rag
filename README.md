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
- Rate-limited endpoints return `429` with RFC 7807 Problem Details and `Retry-After`, configurable via `RATE_LIMIT_RETRY_AFTER_SECONDS`.
- `TRACE_ENCRYPTION_KEY` is required in non-dev environments; dev/test auto-generates a key per run (existing encrypted traces cannot be decrypted after restart).
- To rotate `TRACE_ENCRYPTION_KEY`, decrypt existing traces with the old key and re-encrypt with the new key before switching.
- Cost estimates are token-based and may vary for non-OpenAI models; treat them as directional until reconciled with provider billing.

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

### SDK (Python)

```python
from agentic_rag_backend.sdk.client import AgenticRagClient

async def example() -> None:
    async with AgenticRagClient(
        base_url="http://localhost:8000",
        max_retries=2,
        backoff_factor=0.5,
    ) as client:
        tools = await client.list_tools()
        result = await client.call_tool(
            "knowledge.query",
            {"tenant_id": "11111111-1111-1111-1111-111111111111", "query": "hello"},
        )
        session = await client.create_a2a_session("t1")
        await client.add_a2a_message(session.session.session_id, "t1", "agent", "hello")
```

### A2A Session Lifecycle

- Sessions are stored in memory and cleared on service restart.
- Sessions expire after `A2A_SESSION_TTL_SECONDS` (default 6 hours).
- Expired sessions are pruned every `A2A_CLEANUP_INTERVAL_SECONDS` (default 1 hour).
- Limits are enforced via `A2A_MAX_SESSIONS_PER_TENANT`, `A2A_MAX_SESSIONS_TOTAL`, and `A2A_MAX_MESSAGES_PER_SESSION`.
- At defaults, worst-case memory usage can exceed ~10GB; tune limits or use a persistent backend for production.

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

### Epic 4: Knowledge Ingestion Pipeline
- Status: Complete
- Stories: 4/4 completed
- Key Features:
  - URL documentation crawling via Crawl4AI
  - PDF parsing with Docling for structured ingestion
  - Agentic entity extraction and graph construction
  - Knowledge graph visualization in the UI

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

### Epic 6: Interactive Copilot Experience
- Status: Complete
- Stories: 5/5 completed
- Key Features:
  - CopilotKit React integration with provider and API route
  - Chat sidebar interface with "Thought Trace" stepper showing agent progress
  - Generative UI components (SourceCard, AnswerPanel, GraphPreview)
  - Human-in-the-Loop source validation before answer synthesis
  - Frontend actions (Save, Export, Share, Bookmark, Follow-up)
  - Toast notification system with success/error states
  - Action history panel for tracking user interactions
  - Backend workspace API endpoints for content management
  - 315+ frontend tests with comprehensive component coverage

### Epic 7: Protocol Integration & Extensibility
- Status: Complete
- Stories: 4/4 completed
- Key Features:
  - MCP tool discovery and invocation endpoints
  - A2A collaboration sessions with tenant-scoped message exchange
  - Python SDK for MCP and A2A integrations
  - Universal AG-UI endpoint for non-Copilot clients

### Epic 8: Operations & Observability
- Status: Complete
- Stories: 4/4 completed
- Key Features:
  - LLM cost monitoring with per-request usage events and alert thresholds
  - Intelligent model routing based on query complexity with configurable settings
  - Trajectory debugging interface with filters and timeline inspection
  - AES-256-GCM encrypted trajectory storage with controlled decryption

### Epic 9: Process & Quality Foundation
- Status: Complete
- Stories: 6/6 completed
- Key Features:
  - Retrospective action item tracking with template + overdue check script
  - Story template standards coverage section and authoring guide
  - PR pre-review checklist with protocol compliance prompts
  - Protocol compliance checklist for API routes
  - Story status validation script with CI enforcement
  - Mandatory dev notes, test outcomes, and challenges for story completion

### Epic 10: Testing Infrastructure
- Status: Complete
- Stories: 7/7 completed
- Key Features:
  - Integration test framework with real Postgres/Neo4j/Redis services
  - Hybrid retrieval, ingestion, Graphiti, and protocol integration tests
  - PDF fixtures for ingestion coverage
  - Skipped test inventory and follow-up tracking
  - Benchmarks for ingestion speed and query latency with CI gating
