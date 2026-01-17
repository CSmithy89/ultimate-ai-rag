# Comprehensive User Testing Plan - All Epics

**Date:** 2026-01-17
**Tester:** Claude Code
**Scope:** End-to-end testing of all implemented features (Epics 1-22)
**Approach:** Test as a new user would experience the system

---

## Test Environment

### Services Required
| Service | Port | Status |
|---------|------|--------|
| Backend API | 8000 | Running |
| Frontend | 3000 | Starting |
| PostgreSQL + pgvector | 5432 | Running |
| Neo4j | 7474, 7687 | Running |
| Redis | 6379 | Running |
| Crawl Worker | - | Running |
| Index Worker | - | Running |
| Parse Worker | - | Running |

### Test Tenant ID
`550e8400-e29b-41d4-a716-446655440000`

---

## Phase 1: Environment & Setup Verification (Epic 1)

### 1.1 Docker Compose Services
| Test | Command | Expected | Status |
|------|---------|----------|--------|
| All services running | `docker compose ps` | All containers healthy | |
| Backend health | `curl localhost:8000/health` | `{"status": "healthy"}` | |
| Frontend health | `curl localhost:3000` | 200 OK | |
| PostgreSQL connection | `docker compose exec postgres pg_isready` | accepting connections | |
| Neo4j connection | `curl localhost:7474` | 200 OK | |
| Redis connection | `docker compose exec redis redis-cli ping` | PONG | |

### 1.2 API Documentation
| Test | URL | Expected | Status |
|------|-----|----------|--------|
| OpenAPI docs | http://localhost:8000/docs | Swagger UI loads | |
| ReDoc | http://localhost:8000/redoc | ReDoc loads | |

---

## Phase 2: Ingestion Pipeline - URL Crawling (Epic 4, 13)

### 2.1 Basic URL Ingestion
```bash
curl -X POST http://localhost:8000/api/v1/ingest/url \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "url": "https://docs.python.org/3/tutorial/index.html",
    "max_depth": 0
  }'
```

| Test | Steps | Expected | Status |
|------|-------|----------|--------|
| Submit URL | POST to /api/v1/ingest/url | Returns job_id | |
| Job queued | Check job status | Status: pending/processing | |
| Job completes | Poll status endpoint | Status: completed | |
| Chunks created | Query chunks for tenant | Chunks exist in DB | |

### 2.2 Crawl Profile Auto-Detection
| URL Type | Expected Profile | Status |
|----------|-----------------|--------|
| docs.*.org | fast | |
| github.io | fast | |
| github.com | fast | |
| linkedin.com | stealth | |
| SPA site | thorough | |

### 2.3 GitHub Repository Crawling
```bash
curl -X POST http://localhost:8000/api/v1/ingest/url \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "url": "https://github.com/anthropics/anthropic-cookbook",
    "max_depth": 0
  }'
```

| Test | Expected | Status |
|------|----------|--------|
| GitHub README extracted | Markdown content preserved | |
| Code blocks preserved | Syntax highlighting markers | |

---

## Phase 3: Ingestion Pipeline - PDF Upload (Epic 4)

### 3.1 PDF Document Upload via API
```bash
curl -X POST http://localhost:8000/api/v1/ingest/document \
  -H "X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000" \
  -F "file=@test.pdf"
```

### 3.2 PDF Upload via Frontend
| Test | Steps | Expected | Status |
|------|-------|----------|--------|
| Navigate to Ingest page | Go to /ingest | Page loads | |
| PDF upload form visible | Check UI | File input present | |
| Upload PDF | Select and submit | Upload succeeds | |
| Job tracking | View job status | Progress shown | |
| Tables extracted | Check chunk content | Table data preserved | |
| Sections preserved | Check structure | Headers/sections intact | |

### 3.3 PDF Processing Features (Docling)
| Feature | Expected | Status |
|---------|----------|--------|
| Table extraction | Markdown tables | |
| Section headers | H1-H6 preserved | |
| Code blocks | Preserved in markdown | |
| Images | Placeholder noted | |

---

## Phase 4: Ingestion Pipeline - Codebase Indexing (Epic 15)

### 4.1 Index Local Repository
```bash
curl -X POST http://localhost:8000/api/v1/codebase/index \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000" \
  -d '{
    "repo_path": "/app/backend/src",
    "languages": ["python"]
  }'
```

| Test | Expected | Status |
|------|----------|--------|
| Files discovered | List of Python files | |
| Symbols extracted | Classes, functions, methods | |
| Relationships captured | Call graph edges | |
| Chunks created | Code chunks indexed | |

### 4.2 Symbol Table Stats
```bash
curl -H "X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000" \
  http://localhost:8000/api/v1/codebase/symbol-table/stats
```

---

## Phase 5: Retrieval - Vector Search (Epic 3)

### 5.1 Basic Vector Search
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I create a function in Python?",
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "retrieval_strategy": "vector"
  }'
```

| Test | Expected | Status |
|------|----------|--------|
| Query processed | Response returned | |
| Relevant chunks | Top-k results | |
| Similarity scores | Scores included | |
| Source references | File/URL sources | |

### 5.2 Search with Reranking (Epic 12)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database connection pooling",
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "rerank": true
  }'
```

| Test | Expected | Status |
|------|----------|--------|
| Reranker applied | Results reordered | |
| Better relevance | Top results more accurate | |

---

## Phase 6: Retrieval - Hybrid Search (Epic 3, 5)

### 6.1 Hybrid Vector + Graph Search
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the relationship between FastAPI and Pydantic?",
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "retrieval_strategy": "hybrid"
  }'
```

| Test | Expected | Status |
|------|----------|--------|
| Vector results | Semantic matches | |
| Graph results | Relationship paths | |
| Combined synthesis | Unified answer | |
| Source citations | Both vector + graph | |

### 6.2 Knowledge Graph Stats
```bash
curl -H "X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000" \
  http://localhost:8000/api/v1/knowledge/graph-stats
```

| Test | Expected | Status |
|------|----------|--------|
| Entity count | Number > 0 | |
| Relationship count | Edges exist | |
| Node types | Multiple labels | |

---

## Phase 7: Chat UI - Full Interaction Flow (Epic 6, 21)

### 7.1 Chat Page Load
| Test | Steps | Expected | Status |
|------|-------|----------|--------|
| Navigate to /chat | Browser to localhost:3000/chat | Page loads | |
| CopilotKit initialized | Check for chat UI | Sidebar visible | |
| Suggestions displayed | Check for suggestions | Contextual suggestions | |
| Input available | Check text input | Can type message | |

### 7.2 Send Message & Response
| Test | Steps | Expected | Status |
|------|-------|----------|--------|
| Type message | Enter "What is RAG?" | Text appears | |
| Send message | Click send / press Enter | Message sent | |
| Response streaming | Watch response | Streams incrementally | |
| Agent Progress | Check thought trace | Shows steps | |
| Tool calls | If tools used | Tool cards render | |
| Response complete | Wait for completion | Full response shown | |
| No console errors | Check DevTools | No React errors | |

### 7.3 Advanced Chat Features
| Test | Expected | Status |
|------|----------|--------|
| Regenerate button | Available on responses | |
| Copy button | Copies text | |
| Thumbs up/down | Feedback buttons | |
| Chat history | Previous messages visible | |

### 7.4 Human-in-the-Loop (Epic 6)
| Test | Expected | Status |
|------|----------|--------|
| Source cards shown | Sources displayed | |
| Approve/Reject | Toggle functionality | |
| Rejected excluded | Not used in synthesis | |

---

## Phase 8: Operations Dashboard (Epic 8)

### 8.1 Ops Page Load
| Test | Steps | Expected | Status |
|------|-------|----------|--------|
| Navigate to /ops | Browser to localhost:3000/ops | Page loads | |
| Cost summary | Check cost section | Total costs shown | |
| Model breakdown | Check by-model costs | Per-model data | |
| Token usage | Check token counts | Usage displayed | |

### 8.2 Cost Monitoring
| Test | Expected | Status |
|------|----------|--------|
| Real-time updates | Data refreshes | |
| Cost per request | Individual costs | |
| Tenant aggregation | By-tenant view | |

### 8.3 Trajectory Debugging
| Test | Steps | Expected | Status |
|------|-------|----------|--------|
| Trajectories list | View list | Multiple entries | |
| Select trajectory | Click one | Details shown | |
| Timeline events | View events | Thought/Action/Observation | |
| Tool calls visible | Check tool events | Tool info shown | |
| Timing info | Check durations | Timestamps present | |

---

## Phase 9: Knowledge Graph Visualization (Epic 4)

### 9.1 Knowledge Page Load
| Test | Steps | Expected | Status |
|------|-------|----------|--------|
| Navigate to /knowledge | Browser to localhost:3000/knowledge | Page loads | |
| Graph stats | Check stats panel | Entity counts | |
| Visualization | Check graph view | Nodes/edges render | |

### 9.2 Graph Interaction
| Test | Expected | Status |
|------|----------|--------|
| Zoom | Mouse wheel works | |
| Pan | Drag works | |
| Select node | Click selects | |
| Node details | Info panel shows | |
| Filter by type | Type filtering | |

---

## Phase 10: Protocol Features (Epic 7, 14, 21, 22)

### 10.1 MCP Tools
```bash
curl http://localhost:8000/mcp/v1/tools \
  -H "X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000"
```

| Test | Expected | Status |
|------|----------|--------|
| Tools listed | List of available tools | |
| knowledge.query | Present | |
| knowledge.graph_stats | Present | |
| Tool schemas | JSON schemas included | |

### 10.2 A2A Middleware (Epic 22)
```bash
# List agents
curl http://localhost:8000/a2a/middleware/agents \
  -H "X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000"

# List capabilities
curl http://localhost:8000/a2a/middleware/capabilities \
  -H "X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000"
```

| Test | Expected | Status |
|------|----------|--------|
| Agents listed | Registered agents | |
| Capabilities | hybrid_retrieve, etc. | |
| Tenant isolation | Only own tenant's agents | |

### 10.3 AG-UI Metrics (Epic 22)
```bash
curl http://localhost:8000/metrics
```

| Test | Expected | Status |
|------|----------|--------|
| agui_stream_started_total | Counter present | |
| agui_stream_completed_total | Counter present | |
| agui_event_emitted_total | Counter by type | |
| agui_stream_duration_seconds | Histogram present | |

---

## Phase 11: Codebase Intelligence (Epic 15)

### 11.1 Hallucination Detection
```bash
curl -X POST http://localhost:8000/api/v1/codebase/validate-response \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000" \
  -d '{
    "response": "The VectorSearchService class handles similarity search.",
    "repo_path": "/app/backend/src"
  }'
```

| Test | Expected | Status |
|------|----------|--------|
| Valid symbols | Marked as valid | |
| Invalid symbols | Flagged as hallucinated | |
| File paths | Validated | |
| Confidence score | 0-1 score | |

### 11.2 Codebase Search
```bash
curl -X POST http://localhost:8000/api/v1/codebase/search \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000" \
  -d '{"query": "authentication middleware"}'
```

---

## Phase 12: Memory & Graph Intelligence (Epic 20)

### 12.1 Memory Scopes
```bash
curl http://localhost:8000/api/v1/memories/scopes \
  -H "X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000"
```

| Test | Expected | Status |
|------|----------|--------|
| Scopes listed | session, user, agent | |
| Memory storage | Memories persist | |
| Scope isolation | Properly scoped | |

### 12.2 Community Detection
| Test | Expected | Status |
|------|----------|--------|
| Communities detected | Clusters formed | |
| Algorithm working | Louvain/Leiden | |

---

## Test Execution Log

### Session Info
- **Start Time:** 2026-01-17
- **Tester:** Claude Code
- **Browser:** Playwright Chromium

### Results Summary
| Phase | Pass | Fail | Skip | Notes |
|-------|------|------|------|-------|
| Phase 1: Environment | | | | |
| Phase 2: URL Ingestion | | | | |
| Phase 3: PDF Upload | | | | |
| Phase 4: Codebase Index | | | | |
| Phase 5: Vector Search | | | | |
| Phase 6: Hybrid Search | | | | |
| Phase 7: Chat UI | | | | |
| Phase 8: Ops Dashboard | | | | |
| Phase 9: Knowledge Graph | | | | |
| Phase 10: Protocols | | | | |
| Phase 11: Codebase Intel | | | | |
| Phase 12: Memory/Graph | | | | |
| **Total** | | | | |

---

## Issues Found

| # | Severity | Phase | Description | Status |
|---|----------|-------|-------------|--------|
| | | | | |

---

## UI/UX Evaluation

### Design System Compliance
| Element | Expected | Actual | Status |
|---------|----------|--------|--------|
| Primary color | Indigo-600 | | |
| Success color | Emerald-500 | | |
| Attention color | Amber-400 | | |
| Neutral colors | Slate | | |
| Font (headings) | Inter | | |
| Font (code) | JetBrains Mono | | |

### Usability Assessment
| Criteria | Score (1-5) | Notes |
|----------|-------------|-------|
| Intuitive navigation | | |
| Clear error messages | | |
| Loading states | | |
| Responsive design | | |
| Accessibility | | |

---

## Notes

- All tests should be run with the test tenant ID
- Screenshot any UI issues found
- Log all console errors
- Document any workarounds needed
