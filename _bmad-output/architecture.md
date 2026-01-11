---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
status: 'complete'
completedAt: '2025-12-28'
inputDocuments:
  - '_bmad-output/prd.md'
  - '_bmad-output/project-planning-artifacts/ux-design-specification.md'
  - '_bmad-output/project-planning-artifacts/research/technical-Agentic-RAG-and-GraphRAG-System-research-2025-12-24.md'
  - '_bmad-output/project-planning-artifacts/epics.md'
  - 'docs/recommendations_2025.md'
documentCounts:
  prd: 1
  ux: 1
  research: 1
  epics: 1
  projectDocs: 1
hasProjectContext: false
workflowType: 'architecture'
project_name: 'Agentic Rag and Graphrag with copilot'
user_name: 'Chris'
date: '2025-12-28'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements (25 FRs across 5 domains):**

| Domain | FRs | Architectural Significance |
|--------|-----|---------------------------|
| Developer Experience | FR1-FR5 | Container delivery, SDK packaging, protocol-based integration |
| Agentic Orchestration | FR6-FR10 | Multi-agent coordination, MCP/A2A protocols, trajectory persistence |
| Hybrid Retrieval | FR11-FR14 | Dual-database queries, synthesis layer, explainability |
| Ingestion Pipeline | FR15-FR18 | Async processing, entity extraction, graph construction |
| Copilot Interface | FR19-FR22 | Generative UI, HITL workflows, frontend actions |
| Operations | FR23-FR25 | Cost monitoring, intelligent routing, debugging |

**Non-Functional Requirements (8 NFRs):**

| NFR | Target | Architectural Impact |
|-----|--------|---------------------|
| Response Time | < 10 sec E2E | Async patterns, semantic caching, model routing |
| Ingestion Speed | < 5 min / 50 pages | Background workers, queue-based processing |
| Multi-tenancy | Strict isolation | Namespace patterns, tenant-aware queries |
| Security | Encrypted traces | At-rest encryption, secure storage layer |
| Graph Scale | 1M nodes/edges | Graph optimization, indexing strategy |
| Concurrency | 50+ agent runs | Stateless services, horizontal scaling |
| Protocol Compliance | 100% MCP/AG-UI | Strict interface contracts |
| Recovery | Stateless | External state, session reconstruction |

**Scale & Complexity:**

- Primary domain: Full-stack AI platform (Backend-heavy)
- Complexity level: High/Enterprise
- Estimated architectural components: 8-12 major services

### Technical Constraints & Dependencies

**Hard Constraints:**
- Protocol compliance: MCP, A2A, AG-UI specifications are non-negotiable
- Delivery mechanism: Docker container as primary distribution
- Language split: Python (backend/agents) + TypeScript (frontend/React)

**External Dependencies:**
- LLM Provider APIs (OpenAI, Anthropic, etc.)
- Graph Database (Neo4j or Memgraph)
- Vector Database (pgvector via PostgreSQL)
- Document processing (Docling service)

**Developer Experience Constraints:**
- < 15 min time-to-first-response for new developers
- Environment variable configuration only (no deep knowledge required)

### Cross-Cutting Concerns Identified

1. **Observability & Trajectory Logging** - Every agent decision must be traceable for debugging
2. **Cost Management** - Intelligent model routing affects all LLM interactions
3. **State Management** - Stateless recovery requirement affects every stateful component
4. **Protocol Abstraction** - MCP/A2A/AG-UI interfaces span frontend, backend, and external tools
5. **Multi-tenancy Isolation** - Affects data storage, agent sessions, and knowledge graphs
6. **Error Handling & Resilience** - Distributed agent systems need consistent failure patterns

## Starter Template Evaluation

### Primary Technology Domain

**Full-stack AI Platform** (Backend-heavy) based on project requirements analysis.

This is an "infrastructure-as-a-product" build requiring:
- Python backend with agent orchestration (Agno + FastAPI)
- React frontend with AI copilot integration (Next.js + CopilotKit)
- Polyglot persistence (Neo4j + pgvector)

### Starter Options Considered

**Backend Starters:**

| Option | Verdict |
|--------|---------|
| Agno agent-api | ✅ Selected - FastAPI + PostgreSQL + Agent scaffolding built-in |
| Plain FastAPI | ❌ Would need to build agent infrastructure from scratch |
| LangServe | ❌ LangChain-specific, doesn't align with Agno choice |

**Frontend Starters:**

| Option | Verdict |
|--------|---------|
| Next.js + CopilotKit | ✅ Selected - First-party integration, AG-UI protocol support |
| Vite + React | ❌ Missing SSR, API routes for CopilotKit backend |
| T3 Stack | ❌ Overkill for this use case, adds tRPC complexity |

### Selected Starters

#### Backend: Agno Agent-API

**Initialization Command:**
```bash
# Clone the official Agno agent-api starter
git clone https://github.com/agno-agi/agent-api.git backend
cd backend
uv sync
```

**What it provides:**
- FastAPI server with agent routes
- PostgreSQL for sessions, knowledge, memories
- Pre-built agent scaffolding
- Docker Compose configuration
- Agno v2.3.21 (latest stable)

#### Frontend: Next.js + CopilotKit

**Initialization Command:**
```bash
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir
cd frontend
npm install @copilotkit/react-core @copilotkit/react-ui
```

**What it provides:**
- Next.js 15+ with App Router
- TypeScript configuration
- Tailwind CSS (matches CopilotKit styling)
- ESLint configuration
- CopilotKit React components

### Architectural Decisions Provided by Starters

**Language & Runtime:**
- Backend: Python 3.11+ with uv package management
- Frontend: TypeScript 5.x with Node.js 20+

**Styling Solution:**
- Tailwind CSS 3.x (CopilotKit component compatible)

**Build Tooling:**
- Backend: uv for dependency management, Docker for containerization
- Frontend: Next.js built-in bundler (Turbopack)

**Testing Framework:**
- Backend: pytest (to be configured)
- Frontend: Vitest + Playwright (to be configured)

**Code Organization:**
- Backend: Agno's agent-centric structure (agents/, tools/, knowledge/)
- Frontend: Next.js App Router conventions (app/, components/, lib/)

**Development Experience:**
- Hot reload on both frontend and backend
- Docker Compose for local multi-service development
- Environment variable configuration (.env files)

### Graph Database: Neo4j

**Rationale:** Neo4j selected over Memgraph for:
- Integrated vector search (critical for hybrid RAG)
- Handles graphs exceeding available memory (1M+ nodes NFR)
- Mature ecosystem with Python drivers
- Better documentation and community support

**Docker Configuration:**
```yaml
services:
  neo4j:
    image: neo4j:5-community
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
    ports:
      - "7474:7474"
      - "7687:7687"
```

### Python Package Management: uv

**Rationale:** uv is production-stable as of 2025:
- 10-100x faster than pip
- Vercel's default Python package manager
- Single tool replacing pip, pip-tools, virtualenv, pyenv
- Rust-based, extremely fast CI/CD builds

**Note:** Project initialization using these commands should be Story 1 in the first epic.

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
- ✅ Data layer: Neo4j + pgvector + PostgreSQL + Redis
- ✅ Auth: API Key + namespace isolation
- ✅ Protocols: MCP + A2A + AG-UI (non-negotiable per PRD)
- ✅ Observability: Trajectory logging for debugging

**Important Decisions (Shape Architecture):**
- ✅ Semantic caching via Redis
- ✅ Async ingestion via Redis Streams
- ✅ LLM-based entity extraction

**Deferred Decisions (Post-MVP):**
- OAuth/SSO integration (Growth feature)
- Multi-cloud deployment (Phase 2)
- Self-healing knowledge graph (Vision)

### Data Architecture

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Graph Store | Neo4j | 5.x Community | Entity relationships, GraphRAG |
| Vector Store | pgvector | Latest | Semantic embeddings |
| Relational | PostgreSQL | 16.x | Sessions, app state, Agno storage |
| Cache/Queue | Redis | 7.x | Semantic cache + async message queue |

**Data Flow:**
1. Documents → Docling → Chunks → pgvector (embeddings)
2. Chunks → Agno Indexer → Neo4j (entities/relationships)
3. Queries → Hybrid retrieval (vector + graph) → Synthesis

### Authentication & Security

| Aspect | Approach |
|--------|----------|
| Developer Access | API Key authentication |
| Multi-tenancy | Namespace-based isolation (schema per tenant) |
| Trace Security | AES-256 encryption at rest |
| API Protection | Redis-backed rate limiting |
| Input Validation | Pydantic models (backend), Zod (frontend) |

### API & Communication Patterns

| Pattern | Implementation |
|---------|---------------|
| Agent-to-Frontend | AG-UI protocol via CopilotKit |
| Agent-to-Tools | MCP (JSON-RPC 2.0 over HTTP/SSE) |
| Agent-to-Agent | A2A protocol for delegation |
| Client-to-Backend | REST + SSE streaming |
| Documentation | OpenAPI 3.1 (auto-generated) |
| Error Handling | RFC 7807 Problem Details |

### Frontend Architecture

| Concern | Solution |
|---------|----------|
| Agent State | CopilotKit (AG-UI built-in) |
| Server State | TanStack Query v5 |
| UI Components | CopilotKit + shadcn/ui |
| Graph Visualization | React Flow |
| Forms | React Hook Form + Zod |
| Styling | Tailwind CSS 3.x |

### Infrastructure & Deployment

| Environment | Approach |
|-------------|----------|
| Development | Docker Compose (all services local) |
| Production | Kubernetes-ready (Helm charts provided) |
| Observability | Agno trajectory + LangSmith integration |
| Logging | Structured JSON (structlog) |
| Metrics | Token counting for LLM cost tracking |

### Decision Impact Analysis

**Implementation Sequence:**
1. Project scaffolding (Epic 1, Story 1)
2. Database setup (Neo4j + PostgreSQL + Redis)
3. Agno agent infrastructure
4. CopilotKit frontend integration
5. Ingestion pipeline (Docling + indexing)
6. Hybrid retrieval implementation
7. HITL UI components
8. Observability + cost tracking

**Cross-Component Dependencies:**
- Redis serves dual purpose (cache + queue) to minimize services
- PostgreSQL is shared between Agno (sessions) and pgvector (embeddings)
- All agents must implement MCP for tool access
- AG-UI protocol couples frontend state to agent state

## Implementation Patterns & Consistency Rules

These patterns ensure multiple AI agents write compatible, consistent code that works together seamlessly.

### Critical Conflict Points Identified

**15 areas** where AI agents could make different choices if not specified.

### Naming Patterns

#### Database Naming (PostgreSQL)

| Element | Convention | Example |
|---------|------------|---------|
| Tables | snake_case, plural | `agent_sessions`, `document_chunks` |
| Columns | snake_case | `created_at`, `tenant_id`, `embedding_vector` |
| Primary Keys | `id` (UUID) | `id UUID PRIMARY KEY DEFAULT gen_random_uuid()` |
| Foreign Keys | `{table_singular}_id` | `user_id`, `session_id` |
| Indexes | `idx_{table}_{column}` | `idx_documents_tenant_id` |
| Constraints | `{table}_{type}_{column}` | `users_unique_email` |

#### Graph Database Naming (Neo4j)

| Element | Convention | Example |
|---------|------------|---------|
| Node Labels | PascalCase, singular | `Document`, `Entity`, `Chunk` |
| Relationship Types | SCREAMING_SNAKE_CASE | `MENTIONS`, `AUTHORED_BY`, `CONTAINS` |
| Properties | camelCase | `createdAt`, `sourceUrl`, `entityType` |

#### API Naming

| Element | Convention | Example |
|---------|------------|---------|
| Endpoints | snake_case, plural, REST | `/api/v1/documents`, `/api/v1/agent_sessions` |
| Path Parameters | snake_case | `/documents/{document_id}` |
| Query Parameters | snake_case | `?page_size=10&tenant_id=abc` |
| Headers | Kebab-Case | `X-Tenant-Id`, `X-Request-Id` |

#### Code Naming

**Python (Backend):**

| Element | Convention | Example |
|---------|------------|---------|
| Functions | snake_case | `get_document_chunks()`, `run_hybrid_query()` |
| Classes | PascalCase | `DocumentIndexer`, `HybridRetriever` |
| Constants | SCREAMING_SNAKE | `MAX_CHUNK_SIZE`, `DEFAULT_MODEL` |
| Variables | snake_case | `chunk_count`, `embedding_dim` |
| Files | snake_case | `document_indexer.py`, `hybrid_retriever.py` |

**TypeScript (Frontend):**

| Element | Convention | Example |
|---------|------------|---------|
| Functions | camelCase | `getDocumentChunks()`, `useAgentState()` |
| Components | PascalCase | `DocumentViewer`, `SourceValidation` |
| Interfaces/Types | PascalCase | `DocumentChunk`, `AgentResponse` |
| Constants | SCREAMING_SNAKE | `MAX_RETRY_COUNT`, `API_BASE_URL` |
| Files | kebab-case | `document-viewer.tsx`, `use-agent-state.ts` |

### Structure Patterns

#### Backend Structure (Python/Agno)

```
backend/
├── src/
│   ├── agents/              # Agno agent definitions
│   │   ├── __init__.py
│   │   ├── orchestrator.py  # Master orchestrator agent
│   │   ├── retriever.py     # Hybrid RAG agent
│   │   └── indexer.py       # Document indexing agent
│   ├── tools/               # MCP tool implementations
│   │   ├── __init__.py
│   │   ├── graph_query.py
│   │   └── vector_search.py
│   ├── retrieval/           # Retrieval logic
│   │   ├── __init__.py
│   │   ├── hybrid.py
│   │   └── graph_rag.py
│   ├── indexing/            # Ingestion pipeline
│   │   ├── __init__.py
│   │   ├── docling_parser.py
│   │   └── entity_extractor.py
│   ├── models/              # Pydantic models
│   │   ├── __init__.py
│   │   ├── documents.py
│   │   └── agents.py
│   ├── api/                 # FastAPI routes
│   │   ├── __init__.py
│   │   ├── routes/
│   │   └── middleware/
│   └── core/                # Core utilities
│       ├── __init__.py
│       ├── config.py
│       └── errors.py
├── tests/                   # Co-located test pattern
│   ├── agents/
│   ├── tools/
│   └── conftest.py
├── pyproject.toml
└── Dockerfile
```

#### Frontend Structure (Next.js/CopilotKit)

```
frontend/
├── src/
│   ├── app/                 # Next.js App Router
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── api/             # API routes for CopilotKit
│   │   │   └── copilot/
│   │   └── (features)/      # Route groups
│   │       ├── chat/
│   │       └── knowledge/
│   ├── components/
│   │   ├── ui/              # shadcn/ui components
│   │   ├── copilot/         # CopilotKit wrappers
│   │   │   ├── chat-sidebar.tsx
│   │   │   └── source-validation.tsx
│   │   └── graphs/          # React Flow visualizations
│   ├── lib/
│   │   ├── api.ts           # API client (TanStack Query)
│   │   ├── utils.ts
│   │   └── constants.ts
│   ├── hooks/
│   │   ├── use-agent-state.ts
│   │   └── use-knowledge-graph.ts
│   └── types/
│       ├── api.ts
│       └── agents.ts
├── tests/                   # Playwright E2E tests
├── package.json
└── Dockerfile
```

### Format Patterns

#### API Response Format

**Success Response:**
```json
{
  "data": { },
  "meta": {
    "requestId": "uuid",
    "timestamp": "2025-12-28T10:00:00Z",
    "tokensUsed": 150
  }
}
```

**Error Response (RFC 7807):**
```json
{
  "type": "https://api.example.com/errors/validation-error",
  "title": "Validation Error",
  "status": 400,
  "detail": "The 'document_id' field is required",
  "instance": "/api/v1/documents",
  "errors": [
    { "field": "document_id", "message": "Required field missing" }
  ]
}
```

#### Data Exchange

| Context | Convention |
|---------|------------|
| JSON fields (API) | camelCase in responses, auto-convert from snake_case |
| Dates | ISO 8601 strings: `"2025-12-28T10:00:00Z"` |
| UUIDs | Lowercase with hyphens: `"a1b2c3d4-e5f6-..."` |
| Nulls | Explicit `null`, never omit or use empty string |
| Booleans | `true`/`false`, never `1`/`0` or strings |

### Communication Patterns

#### Event Naming (Redis Streams / Internal)

```
{domain}.{action}

Examples:
- document.uploaded
- document.indexed
- agent.started
- agent.completed
- retrieval.executed
- source.validated
```

#### Event Payload Structure

```json
{
  "eventId": "uuid",
  "eventType": "document.indexed",
  "timestamp": "ISO-8601",
  "tenantId": "uuid",
  "payload": {
    "documentId": "uuid",
    "chunkCount": 42,
    "entityCount": 15
  },
  "metadata": {
    "agentId": "indexer",
    "traceId": "uuid"
  }
}
```

#### Agent Trajectory Logging

All agents MUST log trajectory using Agno's built-in patterns:

```python
# Every agent decision is logged
agent.log_thought("Analyzing query complexity...")
agent.log_action("tool_call", {"tool": "graph_query", "params": {...}})
agent.log_observation("Found 5 relevant entities")
```

### Process Patterns

#### Error Handling

**Backend (Python):**
```python
class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 500):
        self.code = code
        self.message = message
        self.status = status

# Usage
raise AppError("DOCUMENT_NOT_FOUND", "Document does not exist", 404)
```

**Frontend (TypeScript):**
```typescript
// Use TanStack Query error handling
const { data, error, isLoading } = useQuery({
  queryKey: ['documents', documentId],
  queryFn: () => api.getDocument(documentId),
  retry: 3,
  retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 30000)
});

// Display with error boundary or inline
if (error) return <ErrorDisplay error={error} />;
```

#### Loading States

**Frontend Pattern:**
```typescript
// Use TanStack Query states consistently
if (isLoading) return <Skeleton />;
if (isError) return <ErrorDisplay error={error} />;
return <Content data={data} />;
```

#### Retry Pattern

```python
# Backend: Exponential backoff with jitter
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=60)
)
async def call_llm(prompt: str) -> str:
    ...
```

### Enforcement Guidelines

**All AI Agents MUST:**

1. ✅ Follow naming conventions exactly as specified (no variations)
2. ✅ Use the project structure patterns (no custom organization)
3. ✅ Return API responses in the specified format (data/error wrapper)
4. ✅ Log all agent decisions using Agno trajectory patterns
5. ✅ Handle errors using the structured error types
6. ✅ Use Pydantic models for all data validation (backend)
7. ✅ Use Zod schemas for all data validation (frontend)

**Pattern Verification:**

- Pre-commit hooks enforce linting (ruff for Python, ESLint for TypeScript)
- CI pipeline includes pattern compliance checks
- Architecture Decision Records (ADRs) document any pattern exceptions

### Anti-Patterns to Avoid

| Don't | Do |
|-------|-----|
| `getUserData()` in Python | `get_user_data()` |
| `user-card.tsx` for components | `UserCard.tsx` |
| Direct `fetch()` in React | TanStack Query hooks |
| `console.log()` for debugging | Structured logger |
| Inline error messages | Error codes + i18n |
| Mixed case in DB columns | Consistent snake_case |
| Custom event formats | Standard event structure |

## Project Structure & Boundaries

### Complete Project Directory Structure

```
agentic-rag-copilot/
├── README.md
├── docker-compose.yml               # Full stack orchestration
├── docker-compose.dev.yml           # Development overrides
├── .env.example                      # Environment template
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Lint + test + build
│       └── release.yml               # Container publishing
│
├── backend/                          # Python/FastAPI/Agno service
│   ├── pyproject.toml                # uv-managed dependencies
│   ├── uv.lock
│   ├── Dockerfile
│   ├── .env.example
│   ├── ruff.toml                     # Python linting
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entry point
│   │   │
│   │   ├── agents/                   # Agno agent definitions
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py       # FR6: Multi-step planning
│   │   │   ├── retriever.py          # FR11-14: Hybrid RAG
│   │   │   ├── indexer.py            # FR15-18: Agentic indexing
│   │   │   └── router.py             # FR24: Intelligent model routing
│   │   │
│   │   ├── tools/                    # MCP tool implementations
│   │   │   ├── __init__.py
│   │   │   ├── graph_query.py        # FR12: Relationship traversal
│   │   │   ├── vector_search.py      # FR11: Semantic similarity
│   │   │   ├── docling_parse.py      # FR16: Document parsing
│   │   │   └── external_mcp.py       # FR8: External MCP integration
│   │   │
│   │   ├── retrieval/                # Retrieval logic
│   │   │   ├── __init__.py
│   │   │   ├── hybrid.py             # FR13: Combined synthesis
│   │   │   ├── graph_rag.py          # GraphRAG implementation
│   │   │   ├── vector_rag.py         # Vector retrieval
│   │   │   └── cache.py              # Semantic caching (Redis)
│   │   │
│   │   ├── indexing/                 # Ingestion pipeline
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py           # FR15: Crawl4AI integration
│   │   │   ├── chunker.py            # Document chunking
│   │   │   ├── entity_extractor.py   # FR17: Entity extraction
│   │   │   └── graph_builder.py      # FR18: Graph construction
│   │   │
│   │   ├── models/                   # Pydantic models
│   │   │   ├── __init__.py
│   │   │   ├── documents.py          # Document schemas
│   │   │   ├── agents.py             # Agent request/response
│   │   │   ├── graphs.py             # Neo4j node/relationship
│   │   │   └── events.py             # Event payload schemas
│   │   │
│   │   ├── api/                      # FastAPI routes
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agents.py         # /api/v1/agents
│   │   │   │   ├── documents.py      # /api/v1/documents
│   │   │   │   ├── knowledge.py      # /api/v1/knowledge
│   │   │   │   └── health.py         # /health, /ready
│   │   │   └── middleware/
│   │   │       ├── __init__.py
│   │   │       ├── auth.py           # API key validation
│   │   │       ├── tenant.py         # Multi-tenancy isolation
│   │   │       ├── rate_limit.py     # Redis-backed limiting
│   │   │       └── trajectory.py     # Request tracing
│   │   │
│   │   ├── db/                       # Database clients
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py           # PostgreSQL + pgvector
│   │   │   ├── neo4j.py              # Neo4j driver
│   │   │   └── redis.py              # Redis client
│   │   │
│   │   ├── core/                     # Core utilities
│   │   │   ├── __init__.py
│   │   │   ├── config.py             # Settings from env
│   │   │   ├── errors.py             # AppError, error codes
│   │   │   ├── logging.py            # structlog setup
│   │   │   └── cost_tracker.py       # FR23: Token counting
│   │   │
│   │   └── protocols/                # Protocol implementations
│   │       ├── __init__.py
│   │       ├── mcp_server.py         # MCP tool server
│   │       ├── a2a_handler.py        # FR9: A2A delegation
│   │       └── ag_ui_bridge.py       # AG-UI protocol bridge
│   │
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py               # pytest fixtures
│       ├── agents/
│       │   ├── test_orchestrator.py
│       │   └── test_retriever.py
│       ├── tools/
│       │   ├── test_graph_query.py
│       │   └── test_vector_search.py
│       ├── retrieval/
│       │   └── test_hybrid.py
│       └── api/
│           └── test_routes.py
│
├── frontend/                         # Next.js/CopilotKit
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── Dockerfile
│   ├── .env.example
│   ├── .eslintrc.json
│   ├── components.json               # shadcn/ui config
│   │
│   ├── src/
│   │   ├── app/
│   │   │   ├── globals.css
│   │   │   ├── layout.tsx            # Root layout + providers
│   │   │   ├── page.tsx              # Landing/dashboard
│   │   │   │
│   │   │   ├── api/                  # API routes
│   │   │   │   └── copilot/
│   │   │   │       └── route.ts      # CopilotKit backend
│   │   │   │
│   │   │   └── (features)/           # Route groups
│   │   │       ├── chat/
│   │   │       │   └── page.tsx      # FR19: Chat sidebar
│   │   │       ├── knowledge/
│   │   │       │   ├── page.tsx      # FR18: Graph visualization
│   │   │       │   └── [id]/
│   │   │       │       └── page.tsx  # Document detail
│   │   │       └── settings/
│   │   │           └── page.tsx      # Configuration
│   │   │
│   │   ├── components/
│   │   │   ├── ui/                   # shadcn/ui components
│   │   │   │   ├── button.tsx
│   │   │   │   ├── card.tsx
│   │   │   │   ├── dialog.tsx
│   │   │   │   └── skeleton.tsx
│   │   │   │
│   │   │   ├── copilot/              # CopilotKit wrappers
│   │   │   │   ├── chat-sidebar.tsx  # FR19: Chat interface
│   │   │   │   ├── source-validation.tsx  # FR21: HITL
│   │   │   │   ├── generative-ui.tsx # FR20: Dynamic UI
│   │   │   │   └── action-panel.tsx  # FR22: Frontend actions
│   │   │   │
│   │   │   ├── graphs/               # React Flow visualizations
│   │   │   │   ├── knowledge-graph.tsx
│   │   │   │   ├── entity-node.tsx
│   │   │   │   └── relationship-edge.tsx
│   │   │   │
│   │   │   └── features/             # Feature components
│   │   │       ├── document-uploader.tsx
│   │   │       ├── trajectory-viewer.tsx  # FR25: Debug
│   │   │       └── cost-dashboard.tsx     # FR23: Monitoring
│   │   │
│   │   ├── lib/
│   │   │   ├── api.ts                # TanStack Query client
│   │   │   ├── utils.ts              # Utility functions
│   │   │   ├── constants.ts          # App constants
│   │   │   └── schemas.ts            # Zod schemas
│   │   │
│   │   ├── hooks/
│   │   │   ├── use-agent-state.ts    # AG-UI state hook
│   │   │   ├── use-knowledge-graph.ts
│   │   │   ├── use-document-upload.ts
│   │   │   └── use-trajectory.ts
│   │   │
│   │   ├── types/
│   │   │   ├── api.ts                # API response types
│   │   │   ├── agents.ts             # Agent types
│   │   │   └── graphs.ts             # Graph types
│   │   │
│   │   └── providers/
│   │       ├── copilot-provider.tsx  # CopilotKit setup
│   │       ├── query-provider.tsx    # TanStack Query
│   │       └── theme-provider.tsx
│   │
│   ├── tests/
│   │   ├── components/
│   │   │   └── chat-sidebar.test.tsx
│   │   └── e2e/
│   │       ├── chat.spec.ts
│   │       └── knowledge.spec.ts
│   │
│   └── public/
│       └── favicon.ico
│
├── infra/                            # Infrastructure config
│   ├── docker/
│   │   └── docling/
│   │       └── Dockerfile            # Custom Docling build
│   ├── k8s/                          # Kubernetes manifests
│   │   ├── namespace.yaml
│   │   ├── backend/
│   │   ├── frontend/
│   │   └── databases/
│   └── helm/                         # Helm charts
│       └── agentic-rag/
│           ├── Chart.yaml
│           ├── values.yaml
│           └── templates/
│
└── docs/                             # Documentation
    ├── architecture.md               # This document (copied)
    ├── api.md                        # OpenAPI reference
    ├── deployment.md                 # Deployment guide
    └── development.md                # Dev setup guide
```

### Architectural Boundaries

#### API Boundaries

| Boundary | Endpoint Pattern | Purpose |
|----------|-----------------|---------|
| External API | `/api/v1/*` | Client-facing REST endpoints |
| CopilotKit Bridge | `/api/copilot/*` | AG-UI protocol handling |
| Health | `/health`, `/ready` | Container orchestration |
| Internal | Direct function calls | Service-to-service within container |

#### Service Boundaries

| Service | Responsibility | Protocol |
|---------|---------------|----------|
| Orchestrator Agent | Multi-step reasoning, delegation | Agno internal |
| Retriever Agent | Hybrid RAG execution | Agno internal |
| Indexer Agent | Document processing | Agno internal |
| MCP Tools | External tool execution | MCP (JSON-RPC 2.0) |
| A2A Handler | Inter-agent delegation | A2A protocol |
| AG-UI Bridge | Frontend state sync | AG-UI via SSE |

#### Data Boundaries

| Store | Access Layer | Isolation |
|-------|-------------|-----------|
| PostgreSQL | `src/db/postgres.py` | Tenant schema prefix |
| Neo4j | `src/db/neo4j.py` | Tenant property filter |
| Redis | `src/db/redis.py` | Tenant key prefix |
| pgvector | Via PostgreSQL client | Same as PostgreSQL |

### Requirements to Structure Mapping

#### FR Categories to Directories

| Category | Backend Location | Frontend Location |
|----------|-----------------|-------------------|
| FR1-5: Developer Experience | `docker-compose.yml`, `infra/` | - |
| FR6-10: Agentic Orchestration | `src/agents/`, `src/protocols/` | - |
| FR11-14: Hybrid Retrieval | `src/retrieval/`, `src/tools/` | - |
| FR15-18: Ingestion Pipeline | `src/indexing/`, `src/tools/docling_parse.py` | `components/features/document-uploader.tsx` |
| FR19-22: Copilot Interface | `src/protocols/ag_ui_bridge.py` | `src/components/copilot/` |
| FR23-25: Operations | `src/core/cost_tracker.py`, `src/api/middleware/trajectory.py` | `components/features/cost-dashboard.tsx`, `trajectory-viewer.tsx` |

#### Cross-Cutting Concerns Mapping

| Concern | Backend Files | Frontend Files |
|---------|--------------|----------------|
| Multi-tenancy | `middleware/tenant.py`, all DB clients | N/A (backend enforced) |
| Observability | `core/logging.py`, `middleware/trajectory.py` | `hooks/use-trajectory.ts` |
| Error Handling | `core/errors.py`, `api/middleware/` | `lib/api.ts` (query error handling) |
| Authentication | `middleware/auth.py` | Via API headers |
| Rate Limiting | `middleware/rate_limit.py` | N/A (backend enforced) |

### Integration Points

#### Internal Communication

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ CopilotKit   │◄──►│ TanStack     │◄──►│ React Flow   │  │
│  │ Components   │    │ Query        │    │ Graphs       │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│           │                  │                              │
│           ▼                  ▼                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               Next.js API Routes                      │  │
│  │            /api/copilot/* (AG-UI SSE)                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │ SSE
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        Backend                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 FastAPI + Agno                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │  │
│  │  │Orchestrator│◄►│ Retriever  │◄►│    Indexer     │  │  │
│  │  │   Agent    │  │   Agent    │  │     Agent      │  │  │
│  │  └────────────┘  └────────────┘  └────────────────┘  │  │
│  │         │              │               │              │  │
│  │         ▼              ▼               ▼              │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │              MCP Tool Layer                      │ │  │
│  │  │  graph_query │ vector_search │ docling_parse    │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│           ┌───────────────┼───────────────┐                 │
│           ▼               ▼               ▼                 │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│     │ Neo4j    │    │PostgreSQL│    │  Redis   │           │
│     │ (Graph)  │    │(pgvector)│    │ (Cache)  │           │
│     └──────────┘    └──────────┘    └──────────┘           │
└─────────────────────────────────────────────────────────────┘
```

#### External Integrations

| Integration | Protocol | Location |
|-------------|----------|----------|
| LLM APIs (OpenAI, Anthropic) | HTTPS | Via Agno model abstraction |
| External MCP Tools | MCP JSON-RPC | `src/tools/external_mcp.py` |
| LangSmith (observability) | HTTPS | Via Agno integration |
| Docling Service | gRPC/REST | `src/tools/docling_parse.py` |

### Development Workflow

#### Docker Compose Services

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./backend/src:/app/src  # Hot reload
    depends_on: [postgres, neo4j, redis]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    volumes:
      - ./frontend/src:/app/src  # Hot reload
    depends_on: [backend]

  postgres:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]
    volumes: [postgres_data:/var/lib/postgresql/data]

  neo4j:
    image: neo4j:5-community
    ports: ["7474:7474", "7687:7687"]
    volumes: [neo4j_data:/data]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  docling:
    build: ./infra/docker/docling
    ports: ["8080:8080"]
```

#### Development Commands

```bash
# Start full stack
docker compose up -d

# Backend development (hot reload)
cd backend && uv run uvicorn src.main:app --reload

# Frontend development
cd frontend && pnpm dev

# Run all tests
docker compose exec backend pytest
docker compose exec frontend pnpm test

# Lint
cd backend && uv run ruff check src/
cd frontend && pnpm lint
```

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:**
All architectural decisions work together seamlessly:
- Agno v2.3.21 (Python 3.12) + FastAPI creates consistent async backend
- Next.js 15+ with CopilotKit provides native AG-UI protocol support
- Neo4j 5.x + PostgreSQL 16 + pgvector + Redis 7.x form compatible polyglot persistence layer
- uv package management aligns with Python 3.12+ requirements
- All selected technologies have verified 2025 stable releases

**Pattern Consistency:**
- Naming conventions align across Python (snake_case) and TypeScript (camelCase) with clear API boundary translation
- Database naming (snake_case tables, PascalCase Neo4j labels) follows established conventions
- API response format (RFC 7807 errors, data wrapper) consistently applied
- Event patterns (domain.action naming) uniform across Redis Streams

**Structure Alignment:**
- Backend structure follows Agno agent-centric organization
- Frontend structure uses Next.js App Router conventions with CopilotKit integration
- Test organization (co-located in tests/ directories) consistent across both services
- Infrastructure organized separately for deployment concerns

### Requirements Coverage Validation ✅

**Functional Requirements Coverage:**
All 25 FRs are architecturally supported:

| FR Group | Coverage | Implementation Location |
|----------|----------|------------------------|
| FR1-5: Developer Experience | ✅ Full | Docker Compose, infra/, SDK packaging |
| FR6-10: Agentic Orchestration | ✅ Full | src/agents/, src/protocols/ |
| FR11-14: Hybrid Retrieval | ✅ Full | src/retrieval/, src/tools/ |
| FR15-18: Ingestion Pipeline | ✅ Full | src/indexing/, Docling service |
| FR19-22: Copilot Interface | ✅ Full | src/components/copilot/, AG-UI bridge |
| FR23-25: Operations | ✅ Full | cost_tracker.py, trajectory middleware |

**Non-Functional Requirements Coverage:**

| NFR | Architectural Support |
|-----|----------------------|
| NFR1: <10s E2E response | Semantic caching (Redis), async patterns, model routing |
| NFR2: <5 min ingestion | Background workers, Redis Streams queue |
| NFR3: Multi-tenancy | Namespace isolation in all DB clients |
| NFR4: Encrypted traces | AES-256 at-rest encryption pattern |
| NFR5: 1M+ graph nodes | Neo4j 5.x with disk-based storage |
| NFR6: 50+ concurrent | Stateless services, horizontal scaling |
| NFR7: 100% protocol | MCP/A2A/AG-UI implementations specified |
| NFR8: Stateless recovery | External state storage pattern |

### Implementation Readiness Validation ✅

**Decision Completeness:**
- All critical decisions include specific version numbers
- Implementation patterns comprehensive for 15 conflict points identified
- Enforcement guidelines specify mandatory compliance rules
- Examples provided for all major patterns

**Structure Completeness:**
- 100+ files explicitly defined in project tree
- All directories and their purposes documented
- Integration points mapped with clear boundaries
- Component responsibilities clearly specified

**Pattern Completeness:**
- Naming conventions cover database, API, and code
- Communication patterns specify event formats and state management
- Error handling patterns include both backend and frontend
- Process patterns address loading states, retries, and logging

### Gap Analysis Results

**Critical Gaps:** None identified ✅

**Important Gaps (for future enhancement):**
- OAuth/SSO integration patterns (post-MVP feature)
- Multi-cloud deployment specifics (Phase 2)
- Performance benchmarking setup (operational concern)

**Nice-to-Have Gaps:**
- Detailed migration strategy documentation
- Advanced monitoring dashboard specifications
- Load testing framework selection

### Architecture Completeness Checklist

**✅ Requirements Analysis**
- [x] Project context thoroughly analyzed (25 FRs, 8 NFRs)
- [x] Scale and complexity assessed (High/Enterprise)
- [x] Technical constraints identified (MCP/A2A/AG-UI protocols)
- [x] Cross-cutting concerns mapped (6 identified)

**✅ Architectural Decisions**
- [x] Critical decisions documented with versions
- [x] Technology stack fully specified (13 major technologies)
- [x] Integration patterns defined (MCP, A2A, AG-UI)
- [x] Performance considerations addressed (caching, async)

**✅ Implementation Patterns**
- [x] Naming conventions established (database, API, code)
- [x] Structure patterns defined (backend, frontend, infra)
- [x] Communication patterns specified (events, state)
- [x] Process patterns documented (errors, loading, retries)

**✅ Project Structure**
- [x] Complete directory structure defined (100+ files)
- [x] Component boundaries established
- [x] Integration points mapped
- [x] Requirements to structure mapping complete

### Architecture Readiness Assessment

**Overall Status:** READY FOR IMPLEMENTATION ✅

**Confidence Level:** HIGH based on validation results

**Key Strengths:**
- Modern, production-stable technology stack (all 2025 versions)
- Comprehensive pattern coverage preventing agent conflicts
- Clear FR-to-implementation mapping
- Protocol-first design ensuring ecosystem compatibility
- Polyglot persistence optimized for hybrid RAG workloads

**Areas for Future Enhancement:**
- OAuth/SSO patterns when Growth features added
- Multi-cloud deployment manifests for Phase 2
- Self-healing knowledge graph patterns (Vision feature)

### Implementation Handoff

**AI Agent Guidelines:**
- Follow all architectural decisions exactly as documented
- Use implementation patterns consistently across all components
- Respect project structure and boundaries
- Refer to this document for all architectural questions

**First Implementation Priority:**
```bash
# Backend initialization
git clone https://github.com/agno-agi/agent-api.git backend
cd backend && uv sync

# Frontend initialization
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir
cd frontend && npm install @copilotkit/react-core @copilotkit/react-ui
```

## Architecture Completion Summary

### Workflow Completion

**Architecture Decision Workflow:** COMPLETED ✅
**Total Steps Completed:** 8
**Date Completed:** 2025-12-28
**Document Location:** _bmad-output/architecture.md

### Final Architecture Deliverables

**📋 Complete Architecture Document**
- All architectural decisions documented with specific versions
- Implementation patterns ensuring AI agent consistency
- Complete project structure with all files and directories
- Requirements to architecture mapping
- Validation confirming coherence and completeness

**🏗️ Implementation Ready Foundation**
- 13+ architectural decisions made
- 15 implementation pattern categories defined
- 8+ architectural components specified
- 25 FRs + 8 NFRs fully supported

**📚 AI Agent Implementation Guide**
- Technology stack with verified 2025 versions
- Consistency rules that prevent implementation conflicts
- Project structure with clear boundaries
- Integration patterns and communication standards

### Development Sequence

1. Initialize project using documented starter template commands
2. Set up Docker Compose development environment
3. Implement database schemas (PostgreSQL, Neo4j, Redis)
4. Build Agno agent infrastructure (Orchestrator, Retriever, Indexer)
5. Integrate CopilotKit frontend with AG-UI protocol
6. Implement ingestion pipeline with Docling
7. Build hybrid retrieval system
8. Add HITL UI components
9. Implement observability and cost tracking

### Quality Assurance Checklist

**✅ Architecture Coherence**
- [x] All decisions work together without conflicts
- [x] Technology choices are compatible
- [x] Patterns support the architectural decisions
- [x] Structure aligns with all choices

**✅ Requirements Coverage**
- [x] All 25 functional requirements are supported
- [x] All 8 non-functional requirements are addressed
- [x] 6 cross-cutting concerns are handled
- [x] Integration points are defined

**✅ Implementation Readiness**
- [x] Decisions are specific and actionable
- [x] Patterns prevent agent conflicts
- [x] Structure is complete and unambiguous
- [x] Examples are provided for clarity

---

**Architecture Status:** READY FOR IMPLEMENTATION ✅

**Next Phase:** Begin implementation using the architectural decisions and patterns documented herein.

**Document Maintenance:** Update this architecture when major technical decisions are made during implementation.

---

## Architecture Addendum: Graphiti Integration (Epic 5)

**Date Added:** 2025-12-29
**Status:** Approved

### Overview

Epic 5 introduces Graphiti, Zep's temporal knowledge graph framework, replacing our custom entity extraction and graph building pipeline. This is a significant architectural enhancement that simplifies the codebase while adding temporal query capabilities.

### Technology Change

| Component | Before (Epic 4) | After (Epic 5) |
|-----------|-----------------|----------------|
| Entity Extraction | Custom OpenAI prompts | Graphiti SDK |
| Graph Building | Custom Neo4j queries | Graphiti episode ingestion |
| Embeddings | Custom pgvector integration | Graphiti built-in (BGE-m3) |
| Relationship Management | Manual deduplication | Automatic temporal edges |
| Search | Custom hybrid logic | Graphiti hybrid retrieval |

### New Dependencies

```toml
# backend/pyproject.toml
dependencies = [
  "graphiti-core>=0.5.0",  # Temporal knowledge graph SDK
]
```

### Architecture Impact

**Data Flow Change:**

```
BEFORE:
Document → Chunker → EntityExtractor → GraphBuilder → Neo4j (custom schema)

AFTER:
Document → Parser → Graphiti.add_episode() → Neo4j (Graphiti-managed temporal schema)
```

**New Capabilities:**
- **Bi-temporal tracking**: Know when facts were true AND when ingested
- **Point-in-time queries**: Query knowledge graph at specific historical dates
- **Automatic contradiction resolution**: Temporal edge invalidation
- **Agent memory**: Episode-based ingestion optimized for agent workflows

### Code Reduction

| Module | Status | Lines |
|--------|--------|-------|
| `indexing/entity_extractor.py` | DELETED | -352 |
| `indexing/graph_builder.py` | DELETED | -295 |
| `indexing/embeddings.py` | DELETED | -228 |
| `agents/indexer.py` | SIMPLIFIED | -200 |
| **Total Reduction** | | **~1,075 lines** |

### New Modules

| Module | Purpose |
|--------|---------|
| `db/graphiti.py` | Graphiti client wrapper with custom entity types |
| `models/entity_types.py` | Pydantic entity type definitions |
| `indexing/graphiti_ingestion.py` | Episode-based document ingestion |
| `retrieval/graphiti_retrieval.py` | Temporal-aware hybrid search |

### Custom Entity Types

```python
from graphiti_core.models import EntityModel
from pydantic import Field

class TechnicalConcept(EntityModel):
    domain: str = Field(description="Technical domain")
    complexity: str = Field(description="Complexity level")

class CodePattern(EntityModel):
    language: str = Field(description="Programming language")
    pattern_type: str = Field(description="Pattern type")

class APIEndpoint(EntityModel):
    method: str = Field(description="HTTP method")
    path: str = Field(description="Endpoint path")

class ConfigurationOption(EntityModel):
    config_type: str = Field(description="Configuration type")
    default_value: str = Field(description="Default value")
```

### New API Endpoints

```
POST /api/v1/knowledge/temporal-query
  - Query knowledge graph at specific point in time

GET /api/v1/knowledge/changes
  - Get knowledge changes over time period

GET /api/v1/knowledge/entity/{id}/history
  - Get all temporal versions of an entity
```

### Migration Strategy

1. **Phase 1**: Parallel installation (Graphiti + existing)
2. **Phase 2**: Feature flag routing (new docs → Graphiti)
3. **Phase 3**: Migration of existing knowledge graph
4. **Phase 4**: Legacy code removal
5. **Phase 5**: Test suite adaptation

### References

- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [Zep: Temporal Knowledge Graph Architecture (arXiv)](https://arxiv.org/abs/2501.13956)
- [Epic 5 Tech Spec](_bmad-output/epics/epic-5-tech-spec.md)

---

**Architecture Addendum Status:** APPROVED ✅

---

## Architecture Addendum: 2025 Recommendations (Epics 11-18)

**Status:** Draft (Recommendation-Aligned)
**Source:** `docs/recommendations_2025.md`

### Architectural Impact Summary

- **Multi-Provider LLM Adapter:** Introduce provider factory + adapter interface to decouple OpenAI-only flows (Epic 11).
- **Advanced Retrieval Pipeline:** Add reranking, contextual chunking, and corrective grading stages (Epic 12).
- **Enterprise Ingestion Adapters:** Add Apify/BrightData fallbacks and transcript-first YouTube ingestion (Epic 13).
- **Connectivity Layer:** MCP server hardening and robust A2A protocol with RFC 7807 errors (Epic 14).
- **Multimodal Processing:** Video/image ingestion pipelines and codebase hallucination detection (Epic 15).
- **Framework Agnosticism:** Headless Agent Protocol and framework adapters (Epic 16).
- **Install/Deployment CLI:** Guided setup, hardware detection, and startup validation (Epic 17).
- **Documentation & DevOps:** Provider guides, retrieval tuning, MCP usage, and security automation (Epic 18).

### New/Updated Components

| Component | Purpose | Epics |
|-----------|---------|-------|
| `providers/*` | LLM provider adapters + factory | 11, 16 |
| `retrieval/reranker.py` | Cross-encoder reranking stage | 12 |
| `retrieval/grader.py` | Corrective RAG grading + fallback | 12 |
| `ingestion/adapters/*` | Apify/BrightData ingestion adapters | 13 |
| `ingestion/youtube.py` | Transcript-first YouTube ingestion | 13 |
| `protocols/mcp_server.py` | External MCP tool server | 14 |
| `protocols/a2a.py` | Robust A2A session handling | 14 |
| `agents/headless_protocol.py` | Framework-agnostic agent interface | 16 |
| `cli/rag_install.py` | Interactive CLI installer | 17 |

### Config Additions

- `LLM_PROVIDER` (openai, anthropic, gemini, ollama, openrouter)
- `RERANK_PROVIDER`, `RERANK_TOP_N`
- `GRADER_THRESHOLD`, `CONTEXTUAL_CHUNKING_ENABLED`
- `INGESTION_PROVIDER` (crawl4ai, apify, brightdata)
- `AGENT_FRAMEWORK` (agno, pydanticai, crewai, anthropic, langgraph)

### References

- `docs/recommendations_2025.md`
- `_bmad-output/project-planning-artifacts/epics.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`

---

## Architecture Addendum: Advanced Intelligence (Epics 19-22)

**Date Added:** 2026-01-11
**Status:** Approved
**Scope:** Quality foundation, memory platform, graph intelligence, CopilotKit full integration, advanced protocol integration

### Overview

Epics 19-22 deliver advanced intelligence capabilities that position the platform competitively against Mem0, Zep, MS GraphRAG, LightRAG, Cognee, RAGFlow, and Qdrant. These epics add 64 stories covering quality observability, memory scopes, graph intelligence, and advanced protocol integration.

---

### Epic 19: Quality Foundation & Tech Debt (26 stories)

**Architectural Focus:** Establish quality gates and observability infrastructure before competitive features.

#### New Components

```
backend/src/agentic_rag_backend/
├── observability/
│   ├── metrics.py              # Prometheus metrics collection
│   ├── benchmarks.py           # Retrieval quality benchmarking
│   └── dashboards/             # Grafana dashboard templates
├── tests/
│   ├── integration/            # Full pipeline integration tests
│   ├── security/
│   │   ├── test_tenant_isolation.py
│   │   └── test_tenant_isolation_attacks.py
│   ├── compliance/
│   │   └── test_endpoint_spec.py
│   └── benchmarks/
│       └── results/            # Benchmark result storage
└── config/
    └── crawl-profiles.yaml     # Externalized crawler configuration
```

#### Configuration Options

| Config | Purpose | Default |
|--------|---------|---------|
| `PROMETHEUS_ENABLED` | Enable metrics export | `false` |
| `PROMETHEUS_PORT` | Metrics endpoint port | `9090` |
| `GRADER_MODEL` | Cross-encoder model selection | `L-6-v2` |
| `GRADER_NORMALIZATION_STRATEGY` | Score normalization | `min_max` |
| `RERANKER_CACHE_ENABLED` | Cache reranking results | `false` |
| `RERANKER_CACHE_TTL_SECONDS` | Cache TTL | `300` |
| `CRAWLER_USER_AGENT_STRATEGY` | User-agent rotation | `rotate` |
| `CRAWLER_BLOOM_FILTER_THRESHOLD` | Large crawl optimization | `10000` |

#### Quality Metrics Exposed

- `retrieval_mrr_at_k` - Mean Reciprocal Rank
- `retrieval_ndcg_at_k` - Normalized Discounted Cumulative Gain
- `retrieval_precision_at_k` - Precision @ K
- `retrieval_recall_at_k` - Recall @ K

---

### Epic 20: Advanced Retrieval Intelligence (18 stories)

**Architectural Focus:** Competitive feature parity with market leaders via opt-in capabilities.

#### New Module Structure

```
backend/src/agentic_rag_backend/
├── memory/
│   ├── __init__.py
│   ├── scopes.py               # 4-level hierarchy (user/session/agent/global)
│   ├── consolidation.py        # Dedup, merge, decay algorithms
│   └── store.py                # Redis + Graphiti hybrid storage
├── graph/
│   ├── __init__.py
│   ├── community.py            # Louvain/Leiden detection
│   ├── routing.py              # Global vs local query router
│   └── rerankers.py            # Episode, distance, hybrid scoring
├── retrieval/
│   ├── sparse_vectors.py       # BM42 + fastembed
│   ├── colbert.py              # Late-interaction reranking
│   └── multilingual.py         # Cross-language support
└── features/
    ├── voice_io.py             # Whisper STT + TTS
    ├── ontology.py             # OWL support (owlready2)
    └── external_sync/
        ├── s3.py
        ├── notion.py
        └── gdrive.py
```

#### Memory Scope Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GLOBAL SCOPE                          │
│  (Cross-tenant shared knowledge, read-only to tenants)   │
├─────────────────────────────────────────────────────────┤
│                    AGENT SCOPE                           │
│  (Per-agent memories, accessible by all user sessions)   │
├─────────────────────────────────────────────────────────┤
│                    USER SCOPE                            │
│  (Per-user memories, persisted across sessions)          │
├─────────────────────────────────────────────────────────┤
│                   SESSION SCOPE                          │
│  (Ephemeral, cleared on session end)                     │
└─────────────────────────────────────────────────────────┘
```

#### Graph Intelligence Flow

```
Query → Router (global/local) → Community Detection → Graph Reranker → Results
                                      │
                                      ↓
                              LazyRAG Summarization
                              (query-time, not pre-computed)
```

#### New Dependencies

```toml
networkx>=3.0          # Community detection
owlready2>=0.45        # OWL ontology
fastembed>=0.3.0       # BM42 sparse vectors
colbert-ai>=0.2.0      # Late interaction (optional)
openai-whisper>=20231117  # Voice I/O (optional)
pyttsx3>=2.90          # TTS fallback (optional)
notion-client>=2.0.0   # Notion sync (optional)
aioboto3>=12.0.0       # S3 sync (optional)
```

#### Configuration Options

| Config | Purpose | Default |
|--------|---------|---------|
| `MEMORY_CONSOLIDATION_ENABLED` | Auto-consolidate memories | `true` |
| `MEMORY_CONSOLIDATION_INTERVAL_HOURS` | Consolidation frequency | `24` |
| `MEMORY_IMPORTANCE_DECAY_RATE` | Decay multiplier | `0.95` |
| `COMMUNITY_DETECTION_ALGORITHM` | Louvain or Leiden | `louvain` |
| `GRAPH_RERANKER_ENABLED` | Use graph-based reranking | `false` |
| `GLOBAL_LOCAL_ROUTER_ENABLED` | Query routing | `false` |
| `SPARSE_VECTOR_ENABLED` | BM42 hybrid search | `false` |
| `COLBERT_ENABLED` | Late-interaction reranking | `false` |
| `VOICE_IO_ENABLED` | Speech features | `false` |

---

### Epic 21: CopilotKit Full Integration (8 stories)

**Architectural Focus:** Unlock full CopilotKit capabilities with modern patterns.

#### Frontend Architecture Changes

```
frontend/
├── components/copilot/
│   ├── ToolCallRenderer.tsx    # Modern tool visualization
│   ├── A2UIWidgetRenderer.tsx  # Declarative widget rendering
│   ├── ErrorHandler.tsx        # AG-UI error display
│   └── DevConsole.tsx          # Debug overlay
├── hooks/
│   ├── use-frontend-tool.ts    # Replaces useCopilotAction
│   ├── use-human-in-the-loop.ts # HITL approval flow
│   ├── use-copilot-readable.ts # Context provision
│   └── use-telemetry.ts        # Observability hook
├── lib/
│   ├── schemas/
│   │   └── tools.ts            # Shared Zod schemas (DRY)
│   └── utils/
│       └── redact.ts           # Sensitive data masking
└── api/routes/
    └── telemetry/
        └── route.ts            # Telemetry endpoint
```

#### Hook Migration Pattern

```typescript
// DEPRECATED (useCopilotAction)
useCopilotAction({
  name: "validate_sources",
  handler: async () => { ... }
});

// MODERN (useFrontendTool)
useFrontendTool({
  name: "validate_sources",
  description: "Approve or reject retrieved sources",
  parameters: z.object({
    sourceIds: z.array(z.string()),
    decision: z.enum(["approve", "reject"])
  }),
  handler: async ({ sourceIds, decision }) => { ... }
});
```

#### A2UI Widget Types

| Widget | Purpose | Props |
|--------|---------|-------|
| `card` | Information display | title, content, actions |
| `table` | Tabular data | headers, rows, sortable |
| `form` | User input | fields, onSubmit |
| `chart` | Data visualization | type, data, options |
| `image` | Media display | src, alt, caption |
| `list` | Enumerated items | items, ordered |

#### AG-UI Event Flow

```
RUN_STARTED (run_id) → STATE_SNAPSHOT → STATE_DELTA* → RUN_FINISHED
                              │
                              ↓
                       MESSAGES_SNAPSHOT
                              │
                              ↓
                       A2UI_WIDGETS (optional)
```

---

### Epic 22: Advanced Protocol Integration (12 stories)

**Architectural Focus:** Complete protocol stack with A2A middleware and multi-tenant safety.

#### New Protocol Components

```
backend/src/agentic_rag_backend/protocols/
├── a2a_middleware.py           # Agent delegation handler
├── a2a_resource_limits.py      # Per-tenant session/message caps
├── ag_ui_metrics.py            # Prometheus stream metrics
├── ag_ui_errors.py             # RFC 7807 error taxonomy
├── mcp_ui.py                   # Iframe signing and validation
└── open_json_ui.py             # Component schema definitions

frontend/components/
├── mcp-ui/
│   └── MCPUIRenderer.tsx       # Iframe sandbox renderer
├── open-json-ui/
│   └── OpenJSONUIRenderer.tsx  # Declarative component renderer
└── copilot/
    └── ErrorHandler.tsx        # AG-UI error display
```

#### A2A Middleware Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   A2A Middleware Agent                        │
├─────────────────────────────────────────────────────────────┤
│  POST /a2a/agents/register                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  {                                                       │ │
│  │    "agent_id": "specialized-analyzer",                   │ │
│  │    "capabilities": ["analyze", "summarize"],             │ │
│  │    "endpoint": "https://agent.example.com/a2a"          │ │
│  │  }                                                       │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Delegation Flow:                                             │
│  Orchestrator → A2AMiddleware → Target Agent → Response       │
│                      │                                        │
│                      ↓                                        │
│              Resource Limit Check                             │
│              Rate Limit Check                                 │
│              Context Preservation                             │
└─────────────────────────────────────────────────────────────┘
```

#### Resource Limits Backend Options

| Backend | Use Case | Config |
|---------|----------|--------|
| `memory` | Development | Default, single-process |
| `redis` | Production | Distributed state |
| `postgres` | Audit trail | Persistent with history |

#### AG-UI Error Taxonomy (RFC 7807)

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `AGENT_EXECUTION_ERROR` | 500 | Internal agent failure |
| `TENANT_REQUIRED` | 401 | Missing X-Tenant-ID |
| `TENANT_UNAUTHORIZED` | 403 | Tenant access denied |
| `SESSION_NOT_FOUND` | 404 | Invalid session ID |
| `RATE_LIMITED` | 429 | Request throttled |
| `TIMEOUT` | 504 | Operation timeout |
| `INVALID_REQUEST` | 400 | Malformed request |
| `CAPABILITY_NOT_FOUND` | 404 | Unknown capability |
| `UPSTREAM_ERROR` | 502 | External service failure |
| `SERVICE_UNAVAILABLE` | 503 | System overloaded |

#### MCP-UI Security Model

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP-UI Security                            │
├─────────────────────────────────────────────────────────────┤
│  1. Origin Validation                                         │
│     - MCP_UI_ALLOWED_ORIGINS whitelist                       │
│     - Reject unknown origins                                  │
│                                                               │
│  2. Signed URLs                                               │
│     - MCP_UI_SIGNING_SECRET HMAC                             │
│     - TTL expiration                                          │
│                                                               │
│  3. CSP Headers                                               │
│     - frame-src restriction                                   │
│     - script-src 'self'                                       │
│                                                               │
│  4. PostMessage Validation                                    │
│     - Origin check on message receipt                         │
│     - Schema validation via Zod                               │
└─────────────────────────────────────────────────────────────┘
```

#### Open-JSON-UI Component Schema

```typescript
type OpenJSONUIComponent =
  | { type: "text"; content: string }
  | { type: "heading"; level: 1|2|3|4|5|6; content: string }
  | { type: "code"; language: string; content: string }
  | { type: "list"; items: string[]; ordered: boolean }
  | { type: "table"; headers: string[]; rows: string[][] }
  | { type: "image"; src: string; alt: string }
  | { type: "link"; href: string; text: string }
  | { type: "button"; label: string; action: string }
  | { type: "card"; title: string; content: string }
  | { type: "divider" };
```

#### Configuration Options

| Config | Purpose | Default |
|--------|---------|---------|
| `A2A_LIMITS_BACKEND` | Limits storage | `memory` |
| `A2A_SESSION_LIMIT_PER_TENANT` | Max sessions | `100` |
| `A2A_MESSAGE_LIMIT_PER_SESSION` | Max messages | `1000` |
| `A2A_SESSION_TTL_HOURS` | Session expiry | `24` |
| `A2A_MESSAGE_RATE_LIMIT` | Per-minute limit | `60` |
| `METRICS_TENANT_SAMPLING_ENABLED` | Cardinality control | `true` |
| `MCP_UI_ENABLED` | Enable iframe embedding | `false` |
| `MCP_UI_SIGNING_SECRET` | URL signing key | (required) |
| `MCP_UI_ALLOWED_ORIGINS` | Origin whitelist | (required) |
| `OPEN_JSON_UI_ENABLED` | Declarative UI | `true` |

---

### Cross-Epic Integration Points

#### Protocol Stack Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Protocol Layer                            │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│    MCP      │    A2A      │   AG-UI     │   MCP-UI    │O-J-UI│
│  (Tools)    │ (Delegate)  │  (Stream)   │  (Embed)    │(Decl)│
├─────────────┴─────────────┴─────────────┴─────────────┴─────┤
│                   Observability Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Prometheus  │  │  Telemetry  │  │  Benchmarks │          │
│  │   Metrics   │  │   Endpoint  │  │   Suite     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                    Security Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Tenant    │  │   Origin    │  │    Rate     │          │
│  │  Isolation  │  │ Validation  │  │   Limits    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

### Architecture Addendum Status: APPROVED ✅

**Date:** 2026-01-11
**Epics Covered:** 19, 20, 21, 22
**Total Stories:** 64
**New Components:** 30+
**Configuration Options:** 40+

**Next Phase:** Epic 17 and Epic 18 stories are in backlog status, covering CLI implementation and comprehensive documentation.
