# Ultimate AI RAG

## Quick Start

### Environment

```bash
cp .env.example .env
```

Edit `.env` and set at least `LLM_PROVIDER` (defaults to `openai`), the matching
provider API key, and database credentials.

Provider keys:
- `openai`: `OPENAI_API_KEY`
- `openrouter`: `OPENROUTER_API_KEY`
- `ollama`: `OLLAMA_BASE_URL` (no API key by default)
- `anthropic`: `ANTHROPIC_API_KEY`
- `gemini`: `GEMINI_API_KEY`

Use `LLM_MODEL_ID` to set the primary model (defaults to `OPENAI_MODEL_ID`).
OpenAI-compatible providers can override base URLs via `OPENAI_BASE_URL`,
`OPENROUTER_BASE_URL`, or `OLLAMA_BASE_URL`.
Runtime adapters support `openai`, `openrouter`, `ollama`, `anthropic`, and `gemini`
for both LLM orchestration and embeddings. Set `EMBEDDING_PROVIDER` to override the
default (matches `LLM_PROVIDER` when that provider supports embeddings). Voyage AI
(`voyage`) is available for Anthropic users since Anthropic has no native embeddings.

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

- Sessions are cached in memory and persisted to Redis when available.
- Sessions expire after `A2A_SESSION_TTL_SECONDS` (default 6 hours).
- Expired sessions are pruned every `A2A_CLEANUP_INTERVAL_SECONDS` (default 1 hour).
- Limits are enforced via `A2A_MAX_SESSIONS_PER_TENANT`, `A2A_MAX_SESSIONS_TOTAL`, and `A2A_MAX_MESSAGES_PER_SESSION`.
- At defaults, worst-case memory usage can exceed ~10GB; tune limits in production.

### Runbooks

- Graphiti migration: `docs/runbooks/graphiti-migration.md`
- Persistence + usage: `docs/runbooks/persistence-and-usage.md`

## Documentation Map

- `docs/` houses user/developer guidance (guides, runbooks, testing, observability, quality checklists, retrospective templates, roadmap/todo).
- `_bmad-output/` houses BMAD method artifacts (PRD/architecture/project-context, project planning artifacts, epics, implementation artifacts such as stories, retrospectives, reviews, and sprint status).

## Epic Progress

Detailed epic specs and reports live in `_bmad-output/epics/`.

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
  - Graphiti-only ingestion and retrieval (legacy backend flags removed)
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

### Epic 11: Code Cleanup & Migration
- Status: Complete
- Stories: 11/11 completed
- Key Features:
  - Deprecated datetime cleanup and legacy module removal
  - Graphiti migration finalization with HITL validation wiring
  - Workspace persistence for save/share/bookmark flows
  - Parser-based HTML-to-Markdown conversion
  - Neo4j pooling configuration and A2A session persistence
  - Embedding token usage tracking for cost monitoring
  - Multi-provider configuration with LLM_PROVIDER and base URL overrides
  - Provider adapters for OpenAI-compatible LLM clients

### Epic 12: Advanced Retrieval (Archon Upgrade)
- Status: Complete
- Stories: 3/3 completed
- Key Features:
  - Cross-encoder reranking with Cohere and FlashRank providers
  - Contextual retrieval chunking with prompt caching for 90% cost reduction
  - Corrective RAG (CRAG) grader agent with web search fallback (Tavily)
  - Configurable via `RERANKER_ENABLED`, `CONTEXTUAL_RETRIEVAL_ENABLED`, `GRADER_ENABLED`
- Documentation: `docs/guides/advanced-retrieval-configuration.md`

### Epic 13: Enterprise Ingestion
- Status: Complete
- Stories: 4/4 completed
- Key Features:
  - Crawl4AI library migration for JS rendering and parallel crawling
  - YouTube transcript API ingestion for video content
  - Apify/Brightdata fallback for anti-bot protected sites
  - Crawl configuration profiles (fast/thorough/stealth)

### Epic 14: Connectivity (MCP Wrapper Architecture)
- Status: Complete
- Stories: 2/2 completed
- Key Features:
  - MCP server wrapper over Graphiti with RAG-specific tools
  - A2A protocol enhancements for discovery, delegation, and sessions
  - External client integrations via MCP stdio/HTTP transports

### Epic 15: Codebase Intelligence
- Status: Complete
- Stories: 2/2 completed
- Key Features:
  - AST-based hallucination detection with symbol/path/import/API validation
  - Codebase RAG indexing with pgvector embeddings and Neo4j relationships
  - New endpoints: `/api/v1/codebase/validate-response`, `/api/v1/codebase/index`, `/api/v1/codebase/search`
  - Configurable CODEBASE_* settings for detection and indexing

### Epic 19: Quality Foundation & Tech Debt Resolution
- Status: Complete
- Stories: 26/26 completed
- Key Features:
  - Externalized crawl profile mappings with YAML config and tuning guides
  - Configurable user-agent rotation strategies for crawling
  - Deprecated crawler aliases with migration guidance
  - Async HTML parsing for large documents and crawl-many error recovery
  - Bloom filter visited sets and enforced crawl rate limiting
  - Strict crawl config and fallback credential validation
  - Docling-based MCP PDF ingestion with page-level chunking
  - New crawler operational guides (profiles, memory, parsing)

### Epic 20: Advanced Retrieval Intelligence
- Status: Complete
- Stories: 18/18 completed
- Key Features:
  - Memory scopes (user/session/agent/global) with hierarchical inheritance
  - Memory consolidation with deduplication, decay, and cleanup
  - Community detection with Louvain algorithm for graph summaries
  - Query routing (global vs local retrieval based on query type)
  - Graph rerankers (episode recency, distance-based, hybrid)
  - Hierarchical chunking with small-to-big retrieval
  - Dual-level retrieval combining chunk and document levels
  - Prometheus metrics for retrieval quality monitoring

### Epic 21: CopilotKit Full Integration
- Status: Complete
- Stories: 8/8 completed
- Key Features:
  - A2UI widget rendering (card, table, form, chart, image, list)
  - MCP client tool bridge with retry logic
  - CopilotKit backend action handlers for tool execution
  - Source validation frontend tools with Zod schemas
  - Voice I/O: Speech-to-text (Whisper) and text-to-speech (OpenAI/ElevenLabs/pyttsx3)
  - Telemetry endpoint with PII sanitization
  - Prometheus metrics for telemetry events
- Documentation: `docs/guides/voice-io-configuration.md`

### Epic 22: Advanced Protocol Integration
- Status: In progress
- Stories: 9/12 done
- Key Features:
  - A2A middleware agent for agent-to-agent collaboration
  - AG-UI telemetry and error event handling
  - Multi-tenant A2A resource limits
  - MCP-UI and Open-JSON-UI renderers

## Roadmap

### Backlog Epics

| Epic | Focus | Status |
|------|-------|--------|
| **Epic 17** | Developer Experience, CLI & Framework Integration | Backlog |
| **Epic 18** | Enhanced Documentation & DevOps | Backlog |

### Deprecated Epics
- Epic 16: Framework Agnosticism (merged into Epic 17)
