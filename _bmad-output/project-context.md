---
project_name: 'Agentic Rag and Graphrag with copilot'
user_name: 'Chris'
date: '2025-12-28'
architecture_doc: '_bmad-output/architecture.md'
---

# Project Context for AI Agents

_Critical rules and patterns for implementing code in this project. Read before implementing any feature._

---

## Technology Stack (Exact Versions)

| Component | Technology | Version |
|-----------|------------|---------|
| Backend Runtime | Python | 3.12+ |
| Backend Framework | FastAPI | Latest |
| Agent Framework | Agno | v2.3.21 |
| Package Manager (Python) | uv | Latest |
| Frontend Framework | Next.js | 15+ (App Router) |
| Frontend Language | TypeScript | 5.x |
| AI Copilot | CopilotKit | Latest |
| Server State | TanStack Query | v5 |
| UI Components | shadcn/ui | Latest |
| Graph Visualization | React Flow | Latest |
| Graph Database | Neo4j | 5.x Community |
| Vector Database | PostgreSQL + pgvector | 16.x |
| Cache/Queue | Redis | 7.x |
| Document Processing | Docling | 2.66.0 |

---

## Critical Implementation Rules

### 1. Naming Conventions (MANDATORY)

**Python (Backend):**
```python
# Functions: snake_case
def get_document_chunks(): ...

# Classes: PascalCase
class HybridRetriever: ...

# Constants: SCREAMING_SNAKE
MAX_CHUNK_SIZE = 1000

# Files: snake_case.py
document_indexer.py
```

**TypeScript (Frontend):**
```typescript
// Functions: camelCase
function getDocumentChunks() {}

// Components: PascalCase (file and component)
// File: DocumentViewer.tsx
export function DocumentViewer() {}

// Hooks: camelCase with use prefix
// File: use-agent-state.ts
export function useAgentState() {}
```

**Database (PostgreSQL):**
```sql
-- Tables: snake_case, plural
CREATE TABLE document_chunks (...);

-- Columns: snake_case
tenant_id, created_at, embedding_vector

-- Foreign keys: {singular}_id
user_id, session_id
```

**Graph Database (Neo4j):**
```cypher
// Node labels: PascalCase, singular
(:Document), (:Entity), (:Chunk)

// Relationships: SCREAMING_SNAKE_CASE
-[:MENTIONS]->
-[:AUTHORED_BY]->

// Properties: camelCase
{createdAt: datetime(), sourceUrl: "..."}
```

### 2. API Response Format (MANDATORY)

**Success Response:**
```json
{
  "data": { ... },
  "meta": {
    "requestId": "uuid",
    "timestamp": "2025-12-28T10:00:00Z"
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
  "instance": "/api/v1/documents"
}
```

### 3. API Endpoints

```
/api/v1/documents       # REST, snake_case, plural
/api/v1/agent_sessions
/api/copilot/*          # CopilotKit AG-UI routes
/health, /ready         # Health checks
```

### 4. Event Naming (Redis Streams)

```
{domain}.{action}

document.uploaded
document.indexed
agent.started
agent.completed
retrieval.executed
```

---

## Project Structure Rules

### Backend (`backend/src/`)
```
agents/          # Agno agent definitions ONLY
tools/           # MCP tool implementations ONLY
retrieval/       # RAG retrieval logic
indexing/        # Document ingestion pipeline
models/          # Pydantic models
api/routes/      # FastAPI endpoints
api/middleware/  # Auth, tenant, rate limiting
db/              # Database clients (postgres.py, neo4j.py, redis.py)
core/            # Config, errors, logging
protocols/       # MCP, A2A, AG-UI implementations
```

### Frontend (`frontend/src/`)
```
app/             # Next.js App Router pages
components/ui/   # shadcn/ui components
components/copilot/  # CopilotKit wrappers
components/graphs/   # React Flow visualizations
hooks/           # Custom React hooks
lib/             # API client, utilities
types/           # TypeScript types
providers/       # Context providers
```

---

## Code Patterns

### Backend Error Handling
```python
from src.core.errors import AppError

# Always use structured errors
raise AppError("DOCUMENT_NOT_FOUND", "Document does not exist", 404)

# Never use bare exceptions
# BAD: raise Exception("Not found")
```

### Backend Async Retry
```python
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=60)
)
async def call_llm(prompt: str) -> str:
    ...
```

### Agent Trajectory Logging (MANDATORY)
```python
# Every agent decision MUST be logged
agent.log_thought("Analyzing query complexity...")
agent.log_action("tool_call", {"tool": "graph_query", "params": {...}})
agent.log_observation("Found 5 relevant entities")
```

### Frontend Data Fetching
```typescript
// ALWAYS use TanStack Query, NEVER raw fetch
const { data, error, isLoading } = useQuery({
  queryKey: ['documents', documentId],
  queryFn: () => api.getDocument(documentId),
  retry: 3,
});

// Loading state pattern
if (isLoading) return <Skeleton />;
if (error) return <ErrorDisplay error={error} />;
return <Content data={data} />;
```

### Validation
```python
# Backend: Pydantic ONLY
from pydantic import BaseModel

class DocumentCreate(BaseModel):
    title: str
    content: str
    tenant_id: str
```

```typescript
// Frontend: Zod ONLY
import { z } from 'zod';

const documentSchema = z.object({
  title: z.string().min(1),
  content: z.string(),
});
```

---

## Multi-Tenancy Rules

**Every database query MUST include tenant isolation:**

```python
# PostgreSQL
WHERE tenant_id = :tenant_id

# Neo4j
MATCH (n:Document {tenantId: $tenantId})

# Redis key prefix
f"{tenant_id}:cache:{key}"
```

---

## Protocol Compliance

| Protocol | Purpose | Implementation |
|----------|---------|----------------|
| MCP | Tool execution | `src/protocols/mcp_server.py` |
| A2A | Agent delegation | `src/protocols/a2a_handler.py` |
| AG-UI | Frontend state sync | `src/protocols/ag_ui_bridge.py` via CopilotKit |

---

## Anti-Patterns (NEVER DO)

| Don't | Do |
|-------|-----|
| `getUserData()` in Python | `get_user_data()` |
| `user-card.tsx` for components | `UserCard.tsx` |
| Direct `fetch()` in React | TanStack Query hooks |
| `console.log()` | Structured logger |
| Mixed case in DB columns | Consistent `snake_case` |
| Raw SQL strings | Parameterized queries |
| Hardcoded secrets | Environment variables |
| Skip tenant checks | Always filter by `tenant_id` |

---

## Quick Reference Commands

```bash
# Start full stack
docker compose up -d

# Backend dev (hot reload)
cd backend && uv run uvicorn src.main:app --reload

# Frontend dev
cd frontend && pnpm dev

# Run tests
cd backend && uv run pytest
cd frontend && pnpm test

# Lint
cd backend && uv run ruff check src/
cd frontend && pnpm lint
```

---

_For complete architectural decisions, see: `_bmad-output/architecture.md`_
