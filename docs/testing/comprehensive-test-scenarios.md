# Comprehensive Test Scenarios - Ultimate AI RAG

This document outlines all test scenarios derived from the README, CHANGELOG, and Epic specifications. Each scenario includes manual testing steps and expected outcomes.

---

## Table of Contents

1. [Epic 1: Foundation & Developer Quick Start](#epic-1-foundation--developer-quick-start)
2. [Epic 2: Agentic Query & Reasoning](#epic-2-agentic-query--reasoning)
3. [Epic 3: Hybrid Knowledge Retrieval](#epic-3-hybrid-knowledge-retrieval)
4. [Epic 4: Knowledge Ingestion Pipeline](#epic-4-knowledge-ingestion-pipeline)
5. [Epic 5: Graphiti Temporal Knowledge Graph](#epic-5-graphiti-temporal-knowledge-graph)
6. [Epic 6: Interactive Copilot Experience](#epic-6-interactive-copilot-experience)
7. [Epic 7: Protocol Integration & Extensibility](#epic-7-protocol-integration--extensibility)
8. [Epic 8: Operations & Observability](#epic-8-operations--observability)
9. [Epic 12: Advanced Retrieval](#epic-12-advanced-retrieval)
10. [Epic 13: Enterprise Ingestion](#epic-13-enterprise-ingestion)
11. [Epic 14: Connectivity (MCP Wrapper)](#epic-14-connectivity-mcp-wrapper)
12. [Epic 15: Codebase Intelligence](#epic-15-codebase-intelligence)
13. [Epic 17: Developer Experience CLI](#epic-17-developer-experience-cli)
14. [Epic 20: Advanced Retrieval Intelligence](#epic-20-advanced-retrieval-intelligence)
15. [Epic 21: CopilotKit Full Integration](#epic-21-copilotkit-full-integration)
16. [Epic 22: Advanced Protocol Integration](#epic-22-advanced-protocol-integration)

---

## Epic 1: Foundation & Developer Quick Start

### TS-1.1: Backend Startup
**Preconditions:** .env file configured with valid API keys
**Steps:**
1. Navigate to `backend/` directory
2. Run `uv sync` to install dependencies
3. Run `uv run alembic upgrade head` to run migrations
4. Run `uv run agentic-rag-backend` to start the server

**Expected Results:**
- [ ] Server starts without errors on port 8000
- [ ] Health endpoint `GET /health` returns 200 OK
- [ ] OpenAPI docs available at `/docs`
- [ ] Missing required env vars produce clear error messages

### TS-1.2: Frontend Startup
**Preconditions:** Node.js 18+ and pnpm installed
**Steps:**
1. Navigate to `frontend/` directory
2. Run `pnpm install`
3. Run `pnpm dev`

**Expected Results:**
- [ ] Server starts without errors on port 3000
- [ ] Homepage loads successfully
- [ ] No console errors in browser dev tools

### TS-1.3: Docker Compose Full Stack
**Preconditions:** Docker Desktop running with WSL2 integration
**Steps:**
1. Copy `.env.example` to `.env` and configure API keys
2. Run `docker compose up -d`
3. Wait for all services to become healthy

**Expected Results:**
- [ ] PostgreSQL (pgvector) running on port 5432
- [ ] Neo4j running on ports 7474 (web), 7687 (bolt)
- [ ] Redis running on port 6379
- [ ] Backend running on port 8000
- [ ] Frontend running on port 3000
- [ ] All health checks pass

### TS-1.4: Environment Variable Validation
**Steps:**
1. Start backend with missing `OPENAI_API_KEY`
2. Start backend with invalid database URL

**Expected Results:**
- [ ] Clear error message for missing API key
- [ ] Clear error message for invalid DB URL
- [ ] Application does not start with invalid config

---

## Epic 2: Agentic Query & Reasoning

### TS-2.1: Basic Query Processing
**Preconditions:** Backend running, LLM API key configured
**Steps:**
1. Send POST request to `/api/v1/query`
   ```json
   {
     "tenant_id": "test-tenant-001",
     "query": "What is Python?"
   }
   ```

**Expected Results:**
- [ ] Response received within 10 seconds (NFR1)
- [ ] Response contains `answer` field
- [ ] Response contains `trajectory_id` for debugging
- [ ] Response includes `request_id` and `timestamp` metadata

### TS-2.2: Multi-Step Query Planning
**Steps:**
1. Send a complex query requiring multiple steps:
   ```json
   {
     "tenant_id": "test-tenant-001",
     "query": "Compare the authentication mechanisms in Express.js and FastAPI, then recommend which is better for a microservices architecture"
   }
   ```

**Expected Results:**
- [ ] Agent creates visible execution plan
- [ ] Each step is logged as a "thought"
- [ ] Steps execute in logical sequence
- [ ] Final answer synthesizes all steps

### TS-2.3: Dynamic Retrieval Method Selection
**Steps:**
1. Submit semantic query: "What is machine learning?"
2. Submit relationship query: "What functions call the UserService?"
3. Submit complex query: "How does authentication flow from login to session creation?"

**Expected Results:**
- [ ] Semantic query uses Vector RAG
- [ ] Relationship query uses GraphRAG
- [ ] Complex query uses hybrid approach
- [ ] Selection decision logged in trajectory

### TS-2.4: Trajectory Logging
**Steps:**
1. Submit any query
2. Retrieve trajectory via `/api/v1/trajectories/{trajectory_id}`

**Expected Results:**
- [ ] Trajectory contains thoughts, actions, observations
- [ ] Trajectory persists across container restarts (NFR8)
- [ ] Trajectory includes timing information

---

## Epic 3: Hybrid Knowledge Retrieval

### TS-3.1: Vector Semantic Search
**Preconditions:** Documents indexed with embeddings in pgvector
**Steps:**
1. Submit query to vector search endpoint
2. Use a query with no exact keyword matches

**Expected Results:**
- [ ] Returns semantically similar results
- [ ] Each result includes similarity score
- [ ] Results include source references

### TS-3.2: Graph Relationship Traversal
**Preconditions:** Entities and relationships exist in Neo4j
**Steps:**
1. Query for relationships between entities:
   ```json
   {
     "tenant_id": "test-tenant-001",
     "query": "What entities are connected to FastAPI?"
   }
   ```

**Expected Results:**
- [ ] Returns connected entities
- [ ] Shows relationship paths
- [ ] Respects tenant isolation (only returns tenant's data)

### TS-3.3: Hybrid Answer Synthesis
**Steps:**
1. Submit query that benefits from both methods
2. Check response for combined sources

**Expected Results:**
- [ ] Response merges vector and graph results
- [ ] Citations from both sources included
- [ ] Unified, coherent answer generated

### TS-3.4: Graph-Based Explainability
**Steps:**
1. Submit query and request explanation
2. Check response for graph artifacts

**Expected Results:**
- [ ] Response includes nodes referenced
- [ ] Shows edges that connected nodes
- [ ] Human-readable explanation of reasoning path

---

## Epic 4: Knowledge Ingestion Pipeline

### TS-4.1: URL Documentation Crawling
**Steps:**
1. Trigger crawl via API:
   ```json
   {
     "tenant_id": "test-tenant-001",
     "url": "https://docs.python.org/3/"
   }
   ```

**Expected Results:**
- [ ] Crawl respects robots.txt
- [ ] Extracts content from linked pages
- [ ] Reports crawl progress
- [ ] Content queued for processing

### TS-4.2: PDF Document Parsing
**Steps:**
1. Upload PDF via ingestion API
2. Check extracted content

**Expected Results:**
- [ ] Text extracted preserving structure
- [ ] Tables parsed into structured data
- [ ] Headers and sections identified
- [ ] 50-page document processes in <5 minutes (NFR2)

### TS-4.3: Entity Extraction
**Steps:**
1. Ingest a document with named entities
2. Query the knowledge graph

**Expected Results:**
- [ ] Entities identified (people, organizations, concepts)
- [ ] Relationships extracted
- [ ] Nodes created in Neo4j
- [ ] Embeddings stored in pgvector

### TS-4.4: Knowledge Graph Visualization
**Preconditions:** Frontend running, entities exist
**Steps:**
1. Open graph visualization view in UI
2. Interact with the graph

**Expected Results:**
- [ ] Graph renders with React Flow
- [ ] Nodes show labels and types
- [ ] Edges show relationship types
- [ ] Zoom, pan, and selection work
- [ ] Can filter by entity type

---

## Epic 5: Graphiti Temporal Knowledge Graph

### TS-5.1: Episode-Based Ingestion
**Steps:**
1. Ingest document as Graphiti episode
2. Check temporal metadata

**Expected Results:**
- [ ] Document ingested as episode
- [ ] Entities/relationships have temporal metadata
- [ ] Tenant isolation preserved

### TS-5.2: Temporal Query
**Steps:**
1. Query with time filter:
   ```json
   {
     "tenant_id": "test-tenant-001",
     "query": "What was known about X on date Y?",
     "as_of": "2025-12-01T00:00:00Z"
   }
   ```

**Expected Results:**
- [ ] Results reflect state at specified time
- [ ] Without time filter, returns latest data

### TS-5.3: Knowledge Changes API
**Steps:**
1. Call `/api/v1/knowledge/temporal/changes`
2. Specify date range

**Expected Results:**
- [ ] Returns changes in knowledge over time
- [ ] Shows what was added/modified/removed

---

## Epic 6: Interactive Copilot Experience

### TS-6.1: CopilotKit Integration
**Steps (Playwright):**
1. Navigate to `http://localhost:3000`
2. Check for CopilotProvider in DOM
3. Open chat sidebar

**Expected Results:**
- [ ] CopilotProvider renders
- [ ] Chat sidebar opens
- [ ] Connection to backend established

### TS-6.2: Chat Sidebar Interface
**Steps (Playwright):**
1. Open chat sidebar
2. Type a message and submit
3. Observe response

**Expected Results:**
- [ ] Chat interface renders with shadcn/ui styling
- [ ] Messages can be typed and submitted
- [ ] Streaming responses display
- [ ] Thought Trace stepper shows progress
- [ ] Colors match design system (Indigo-600, Slate)

### TS-6.3: Generative UI Components
**Steps (Playwright):**
1. Ask query that triggers visualization
2. Check rendered component

**Expected Results:**
- [ ] Graph visualizer renders (React Flow)
- [ ] Data tables render for structured info
- [ ] Components are interactive
- [ ] Follows "Professional Forge" design

### TS-6.4: Human-in-the-Loop Source Validation
**Steps (Playwright):**
1. Submit query that retrieves sources
2. Check for HITL validation panel
3. Approve/reject sources
4. Submit final answer

**Expected Results:**
- [ ] Side-panel displays source cards
- [ ] Each card shows title, snippet, metadata
- [ ] Approve/Reject toggles work
- [ ] Rejected sources excluded from answer
- [ ] Amber-400 used for attention items

### TS-6.5: Frontend Actions
**Steps (Playwright):**
1. Trigger action buttons (Save, Export, Share, Bookmark)
2. Test Export dropdown (Markdown, PDF, JSON)

**Expected Results:**
- [ ] Action buttons render and function
- [ ] Export produces correct format
- [ ] Toast notifications appear
- [ ] Action history panel shows interactions

### TS-6.6: Workspace API Endpoints
**Steps:**
1. Test `POST /api/v1/workspace/save`
2. Test `POST /api/v1/workspace/export`
3. Test `POST /api/v1/workspace/share`
4. Test `POST /api/v1/workspace/bookmark`
5. Test `GET /api/v1/workspace/bookmarks`

**Expected Results:**
- [ ] Content saved successfully
- [ ] Export returns correct format
- [ ] Share returns shareable link
- [ ] Bookmarks can be created and listed

---

## Epic 7: Protocol Integration & Extensibility

### TS-7.1: MCP Tool Discovery
**Steps:**
1. GET `/api/v1/mcp/tools`

**Expected Results:**
- [ ] Returns list of available tools
- [ ] Includes `knowledge.query` and `knowledge.graph_stats`
- [ ] JSON-RPC 2.0 format compliance

### TS-7.2: MCP Tool Invocation
**Steps:**
1. POST `/api/v1/mcp/call` with:
   ```json
   {
     "method": "knowledge.query",
     "params": {
       "tenant_id": "test-tenant-001",
       "query": "What is RAG?"
     }
   }
   ```

**Expected Results:**
- [ ] Tool executes successfully
- [ ] Response follows MCP spec
- [ ] Tool call logged in trajectory

### TS-7.3: A2A Session Management
**Steps:**
1. Create session: `POST /api/v1/a2a/sessions`
2. Add message: `POST /api/v1/a2a/sessions/{id}/messages`
3. Get session: `GET /api/v1/a2a/sessions/{id}`

**Expected Results:**
- [ ] Session created with ID
- [ ] Messages added successfully
- [ ] Session retrievable
- [ ] Tenant isolation enforced

### TS-7.4: Python SDK
**Steps:**
1. Install SDK
2. Run example:
   ```python
   from agentic_rag_backend.sdk.client import AgenticRagClient
   async with AgenticRagClient(base_url="http://localhost:8000") as client:
       tools = await client.list_tools()
   ```

**Expected Results:**
- [ ] SDK connects to backend
- [ ] List tools returns results
- [ ] Tool calls work
- [ ] A2A session operations work

### TS-7.5: AG-UI Universal Endpoint
**Steps:**
1. Connect to `/api/v1/ag-ui` via HTTP/WebSocket
2. Send query following AG-UI spec
3. Receive streaming response

**Expected Results:**
- [ ] Connection established
- [ ] Query processed
- [ ] SSE streaming works
- [ ] State updates received

---

## Epic 8: Operations & Observability

### TS-8.1: LLM Cost Monitoring
**Steps:**
1. Make several queries
2. Check cost tracking endpoint
3. Check usage events

**Expected Results:**
- [ ] Token usage tracked per request
- [ ] Costs calculated based on model pricing
- [ ] Usage aggregated by tenant

### TS-8.2: Intelligent Model Routing
**Steps:**
1. Submit simple query (expect cheaper model)
2. Submit complex query (expect premium model)
3. Check routing decisions in trajectory

**Expected Results:**
- [ ] Simple queries routed to `ROUTING_SIMPLE_MODEL`
- [ ] Complex queries routed to `ROUTING_COMPLEX_MODEL`
- [ ] Routing decisions logged
- [ ] Cost savings tracked

### TS-8.3: Trajectory Debugging Interface
**Steps (Playwright):**
1. Navigate to trajectory viewer
2. Select a past session
3. Drill into trajectory details

**Expected Results:**
- [ ] Past sessions listed
- [ ] Thoughts, actions, observations visible
- [ ] Tool calls and results shown
- [ ] Timing information displayed
- [ ] Filter by error status works

### TS-8.4: Encrypted Trace Storage
**Preconditions:** `TRACE_ENCRYPTION_KEY` configured
**Steps:**
1. Make a query
2. Check database for trajectory
3. Verify encryption

**Expected Results:**
- [ ] Trajectory data encrypted at rest (AES-256)
- [ ] Decryption works for authorized access
- [ ] Multi-tenant data isolated

---

## Epic 12: Advanced Retrieval

### TS-12.1: Cross-Encoder Reranking
**Preconditions:** `RERANKER_ENABLED=true`
**Steps:**
1. Submit retrieval query
2. Check reranked results

**Expected Results:**
- [ ] Results rescored and reordered
- [ ] Top results more relevant
- [ ] Reranker provider used (flashrank/cohere)

### TS-12.2: Contextual Retrieval Chunking
**Preconditions:** `CONTEXTUAL_RETRIEVAL_ENABLED=true`
**Steps:**
1. Ingest document
2. Check chunk embeddings

**Expected Results:**
- [ ] Chunks include title/summary context
- [ ] Embeddings reflect enriched content

### TS-12.3: Corrective RAG (CRAG) Grader
**Preconditions:** `GRADER_ENABLED=true`, `TAVILY_API_KEY` set
**Steps:**
1. Submit query with low-quality matches
2. Observe fallback behavior

**Expected Results:**
- [ ] Results scored for relevance
- [ ] Low scores trigger fallback
- [ ] Web search fallback works (Tavily)
- [ ] Decisions logged in trajectory

---

## Epic 13: Enterprise Ingestion

### TS-13.1: Crawl4AI Integration
**Steps:**
1. Trigger crawl with JavaScript-heavy site
2. Check content extraction

**Expected Results:**
- [ ] JS-rendered content captured
- [ ] Parallel crawling works
- [ ] Caching enabled

### TS-13.2: YouTube Transcript Ingestion
**Steps:**
1. Ingest YouTube URL:
   ```json
   {
     "tenant_id": "test-tenant-001",
     "url": "https://www.youtube.com/watch?v=VIDEO_ID"
   }
   ```

**Expected Results:**
- [ ] Transcript fetched successfully
- [ ] Chunks include source metadata
- [ ] Completes in <30 seconds

### TS-13.3: Crawl Configuration Profiles
**Steps:**
1. Test with `CRAWL4AI_PROFILE=fast`
2. Test with `CRAWL4AI_PROFILE=thorough`
3. Test with `CRAWL4AI_PROFILE=stealth`

**Expected Results:**
- [ ] Fast: High concurrency, headless
- [ ] Thorough: JS wait, screenshots
- [ ] Stealth: Proxy, random delays

### TS-13.4: Fallback Providers
**Preconditions:** Apify/Brightdata keys configured
**Steps:**
1. Crawl a site that blocks standard crawlers
2. Check fallback behavior

**Expected Results:**
- [ ] Fallback to Apify/Brightdata triggered
- [ ] Content successfully retrieved
- [ ] Fallback reason logged

---

## Epic 14: Connectivity (MCP Wrapper)

### TS-14.1: MCP Server RAG Tools
**Steps:**
1. List available MCP tools
2. Call `vector_search`
3. Call `hybrid_retrieve`
4. Call `ingest_url`

**Expected Results:**
- [ ] Graphiti tools proxied (add_memory, search_nodes, search_facts)
- [ ] RAG extension tools available
- [ ] Tenant isolation enforced

### TS-14.2: A2A Protocol Robustness
**Steps:**
1. Send invalid A2A request
2. Simulate timeout
3. Check error responses

**Expected Results:**
- [ ] Errors use RFC 7807 format
- [ ] Retries handled correctly
- [ ] Timeouts reported properly

---

## Epic 15: Codebase Intelligence

### TS-15.1: Hallucination Detection
**Preconditions:** `CODEBASE_DETECTOR_MODE=warn`
**Steps:**
1. Call `/api/v1/codebase/validate-response` with response containing fake symbols
2. Check validation results

**Expected Results:**
- [ ] AST parsing detects unknown symbols
- [ ] Warnings recorded with missing symbol list
- [ ] Python, TypeScript, JavaScript supported

### TS-15.2: Codebase Indexing
**Steps:**
1. Call `/api/v1/codebase/index` with repo path
2. Query `/api/v1/codebase/search`

**Expected Results:**
- [ ] Symbols extracted (functions, classes, methods)
- [ ] Relationships captured (call graphs)
- [ ] .gitignore respected
- [ ] Incremental indexing works

---

## Epic 17: Developer Experience CLI

### TS-17.1: rag-install CLI
**Steps:**
1. Run `rag-install` interactively
2. Select options for provider, framework, profile

**Expected Results:**
- [ ] Interactive prompts work
- [ ] Valid configuration produced
- [ ] API key validation works
- [ ] Non-interactive mode works: `rag-install --profile standard --yes`

### TS-17.2: CLI Doctor Command
**Steps:**
1. Run `rag-cli doctor`

**Expected Results:**
- [ ] Checks Docker availability
- [ ] Validates .env configuration
- [ ] Reports service health
- [ ] Provides actionable recommendations

### TS-17.3: Profile-Based Configuration
**Steps:**
1. Set `CONFIG_PROFILE=minimal`
2. Set `CONFIG_PROFILE=standard`
3. Set `CONFIG_PROFILE=enterprise`

**Expected Results:**
- [ ] Each profile applies appropriate defaults
- [ ] Profile settings documented

---

## Epic 20: Advanced Retrieval Intelligence

### TS-20.1: Memory Scopes
**Preconditions:** `MEMORY_SCOPES_ENABLED=true`
**Steps:**
1. Create memories in different scopes (user/session/agent/global)
2. Query with different scope contexts

**Expected Results:**
- [ ] Memories stored in correct scope
- [ ] Hierarchical inheritance works
- [ ] Scope isolation maintained

### TS-20.2: Memory Consolidation
**Preconditions:** `MEMORY_CONSOLIDATION_ENABLED=true`
**Steps:**
1. Create duplicate memories
2. Trigger consolidation
3. Check results

**Expected Results:**
- [ ] Duplicates deduplicated
- [ ] Decay applied to old memories
- [ ] Low-importance memories cleaned up

### TS-20.3: Community Detection
**Steps:**
1. Ingest related documents
2. Check for community summaries

**Expected Results:**
- [ ] Louvain algorithm detects communities
- [ ] Graph summaries generated

### TS-20.4: Query Routing
**Preconditions:** `QUERY_ROUTING_ENABLED=true`
**Steps:**
1. Submit global query: "What are the main topics in this knowledge base?"
2. Submit local query: "What is FastAPI?"

**Expected Results:**
- [ ] Global queries route to community-level retrieval
- [ ] Local queries route to entity-level retrieval

### TS-20.5: Graph Rerankers
**Preconditions:** `GRAPH_RERANKER_ENABLED=true`
**Steps:**
1. Submit query with graph results
2. Check reranking applied

**Expected Results:**
- [ ] Episode recency affects ranking
- [ ] Distance-based scoring works
- [ ] Hybrid combines both

### TS-20.6: Hierarchical Chunking
**Preconditions:** `HIERARCHICAL_CHUNKS_ENABLED=true`
**Steps:**
1. Ingest document
2. Query with small-to-big retrieval

**Expected Results:**
- [ ] Multiple chunk levels created
- [ ] Small chunks match, big chunks returned for context

### TS-20.7: Prometheus Metrics
**Steps:**
1. GET `/metrics`
2. Check retrieval metrics

**Expected Results:**
- [ ] Retrieval quality metrics exposed
- [ ] Latency histograms available

---

## Epic 21: CopilotKit Full Integration

### TS-21.1: A2UI Widget Rendering (Playwright)
**Steps:**
1. Trigger responses that render widgets
2. Check rendered widgets

**Expected Results:**
- [ ] Card widget renders
- [ ] Table widget renders
- [ ] Form widget renders
- [ ] Chart widget renders
- [ ] Image widget renders
- [ ] List widget renders

### TS-21.2: MCP Client Tool Bridge
**Steps:**
1. Execute MCP tool through CopilotKit
2. Check retry logic on failure

**Expected Results:**
- [ ] Tool execution works
- [ ] Retry logic handles failures
- [ ] Results rendered correctly

### TS-21.3: Voice I/O - Speech-to-Text
**Preconditions:** `VOICE_IO_ENABLED=true`
**Steps:**
1. Record audio input
2. Submit for transcription

**Expected Results:**
- [ ] Whisper transcription works
- [ ] Transcription accurate

### TS-21.4: Voice I/O - Text-to-Speech
**Preconditions:** `VOICE_IO_ENABLED=true`, `TTS_PROVIDER` configured
**Steps:**
1. Submit text for speech synthesis
2. Check audio output

**Expected Results:**
- [ ] TTS generates audio
- [ ] Voice matches configured setting

### TS-21.5: Telemetry Endpoint
**Steps:**
1. POST `/api/v1/telemetry/events`
2. Check Prometheus metrics

**Expected Results:**
- [ ] Events received
- [ ] PII sanitized
- [ ] Metrics incremented

---

## Epic 22: Advanced Protocol Integration

### TS-22.1: A2A Middleware Agent
**Steps:**
1. Register middleware agent
2. Test agent-to-agent delegation

**Expected Results:**
- [ ] Middleware handles delegation
- [ ] Discovery works
- [ ] Context preserved

### TS-22.2: A2A Resource Limits
**Preconditions:** `A2A_LIMITS_BACKEND=redis`
**Steps:**
1. Create sessions up to limit
2. Attempt to exceed limit

**Expected Results:**
- [ ] Per-tenant limits enforced
- [ ] Rate limiting works
- [ ] Appropriate errors returned

### TS-22.3: AG-UI Stream Metrics
**Steps:**
1. Start AG-UI stream
2. Check Prometheus metrics

**Expected Results:**
- [ ] `agui_stream_started_total` incremented
- [ ] `agui_stream_completed_total` incremented
- [ ] Duration and latency recorded

### TS-22.4: AG-UI Extended Error Taxonomy
**Steps:**
1. Trigger rate limit error
2. Trigger timeout error
3. Trigger capability not found error

**Expected Results:**
- [ ] RATE_LIMITED error returned correctly
- [ ] TIMEOUT error returned correctly
- [ ] CAPABILITY_NOT_FOUND error returned correctly

### TS-22.5: MCP-UI Renderer (Playwright)
**Preconditions:** `MCP_UI_ENABLED=true`
**Steps:**
1. Trigger MCP tool with UI
2. Check iframe rendering

**Expected Results:**
- [ ] Iframe sandboxed correctly
- [ ] Origin allowlist enforced
- [ ] postMessage bridge works

### TS-22.6: Open-JSON-UI Renderer (Playwright)
**Steps:**
1. Receive JSON-UI payload
2. Check rendered components

**Expected Results:**
- [ ] Text renders
- [ ] Heading renders
- [ ] Code block renders
- [ ] Table renders
- [ ] Image renders
- [ ] Button renders

### TS-22.7: Composite Tenant Rate Limiting
**Steps:**
1. Send telemetry requests rapidly
2. Check rate limit enforcement

**Expected Results:**
- [ ] Rate limit key: `telemetry:{tenant_id}:{ip}`
- [ ] Limit enforced correctly

---

## Test Execution Checklist

### Prerequisites
- [ ] Docker Desktop running with WSL2 integration
- [ ] .env file configured from .env.example
- [ ] Valid API keys for LLM provider
- [ ] All services started via `docker compose up -d`

### Backend Tests
```bash
cd backend
uv run pytest                           # Unit tests
INTEGRATION_TESTS=1 uv run pytest       # Integration tests (requires services)
GRAPHITI_E2E=1 uv run pytest           # Graphiti E2E tests
```

### Frontend Tests
```bash
cd frontend
pnpm test                              # Unit tests
pnpm turbo test                        # Via turbo
```

### Manual Testing with Playwright MCP
Use the Playwright MCP tools to test frontend interactions:
- `mcp__playwright__browser_navigate` - Navigate to URLs
- `mcp__playwright__browser_snapshot` - Capture accessibility snapshots
- `mcp__playwright__browser_click` - Click elements
- `mcp__playwright__browser_type` - Type text
- `mcp__playwright__browser_take_screenshot` - Capture screenshots

---

## Coverage Requirements

As per CI configuration:
- Backend: 80% minimum coverage
- Frontend: 80% minimum coverage

---

## Related Documentation

- `docs/guides/advanced-retrieval-configuration.md` - Retrieval features
- `docs/guides/voice-io-configuration.md` - Voice I/O setup
- `docs/guides/protocol-integration.md` - Protocol details
- `docs/testing/integration-tests.md` - Integration test guide
- `docs/testing/benchmark-suite.md` - Benchmark documentation
