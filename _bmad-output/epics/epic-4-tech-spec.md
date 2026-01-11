# Epic 4 Tech Spec: Knowledge Ingestion Pipeline

**Date:** 2025-12-28
**Status:** Complete
**Epic Owner:** Development Team

---

## 1. Overview

Epic 4 delivers the Knowledge Ingestion Pipeline, enabling users to ingest documents from URLs or PDFs and autonomously build knowledge graphs with entity extraction, relationship mapping, and graph visualization.

### 1.1 Epic Goals

- Enable autonomous crawling of documentation websites using Crawl4AI
- Parse complex PDF documents with tables and structured layouts using Docling
- Autonomously extract entities and relationships to build the knowledge graph
- Provide visualization tools for data engineers to inspect graph quality

### 1.2 Functional Requirements Covered

| FR ID | Description |
|-------|-------------|
| FR15 | Users can trigger autonomous crawl of documentation websites using Crawl4AI |
| FR16 | System can parse complex document layouts (tables, headers, footnotes) from PDFs using Docling |
| FR17 | Agentic Indexer can autonomously extract entities and relationships from parsed text |
| FR18 | Data Engineers can visualize the current state of the knowledge graph |

### 1.3 Non-Functional Requirements Addressed

| NFR ID | Target | Epic 4 Impact |
|--------|--------|---------------|
| NFR2 | < 5 min ingestion for 50-page doc | Pipeline must be async with background workers |
| NFR5 | 1M+ nodes/edges | Schema design must support scale |

### 1.4 Non-Goals

- Hybrid retrieval logic (Epic 3)
- Query-time agent orchestration (Epic 2)
- Frontend HITL validation UI (Epic 5)
- Cost monitoring and trajectory debugging (Epic 7)

---

## 2. Architecture Decisions

### 2.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │  URL /   │───▶│ Crawl4AI │───▶│ Raw HTML │    │                      │  │
│  │  Website │    │ Crawler  │    │ Content  │    │                      │  │
│  └──────────┘    └──────────┘    └────┬─────┘    │                      │  │
│                                       │          │                      │  │
│  ┌──────────┐    ┌──────────┐    ┌────▼─────┐    │   ┌──────────────┐   │  │
│  │   PDF    │───▶│ Docling  │───▶│ Unified  │───▶│──▶│   Chunker    │   │  │
│  │   File   │    │ Parser   │    │ Document │    │   │  (semantic)  │   │  │
│  └──────────┘    └──────────┘    └──────────┘    │   └──────┬───────┘   │  │
│                                                   │          │           │  │
│                                                   │   ┌──────▼───────┐   │  │
│                                                   │   │   Embedding  │   │  │
│                                                   │   │   Generator  │   │  │
│                                                   │   └──────┬───────┘   │  │
│                                                   │          │           │  │
│                                        ┌──────────┴──────────┼───────────┘  │
│                                        │                     │              │
│                                        ▼                     ▼              │
│                                  ┌──────────┐         ┌──────────┐          │
│                                  │ pgvector │         │  Entity  │          │
│                                  │ (chunks) │         │Extractor │          │
│                                  └──────────┘         └────┬─────┘          │
│                                                            │                │
│                                                     ┌──────▼───────┐        │
│                                                     │    Neo4j     │        │
│                                                     │ (entities/   │        │
│                                                     │  relations)  │        │
│                                                     └──────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack for Epic 4

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Web Crawling | Crawl4AI | Latest | Autonomous documentation site crawling |
| Document Parsing | Docling | 2.66.0 | PDF/complex document parsing |
| Entity Extraction | OpenAI GPT-4o | - | LLM-based entity/relationship extraction |
| Vector Store | pgvector | Latest | Chunk embeddings storage |
| Graph Store | Neo4j | 5.x Community | Entity relationships storage |
| Queue | Redis Streams | 7.x | Async job processing |
| Backend | FastAPI + Agno | 2.3.21 | API and agent orchestration |

### 2.3 Key Architectural Decisions

1. **Async Processing via Redis Streams**: Ingestion jobs run asynchronously to avoid blocking API responses. Redis Streams provides reliable message delivery with consumer groups.

2. **Unified Document Model**: Both Crawl4AI output and Docling output are normalized to a common `UnifiedDocument` Pydantic model before processing.

3. **Semantic Chunking**: Use overlapping semantic chunks (512 tokens, 64 token overlap) to preserve context boundaries.

4. **LLM-Based Entity Extraction**: Use GPT-4o with structured output (JSON mode) for entity/relationship extraction. This is the "Agentic Indexer" approach.

5. **Idempotent Ingestion**: Documents are identified by content hash to enable re-ingestion without duplicates.

---

## 3. Stories Breakdown with Technical Approach

### 3.1 Story 4.1: URL Documentation Crawling

**As a** data engineer,
**I want** to trigger autonomous crawling of documentation websites,
**So that** I can ingest external knowledge sources without manual downloads.

#### Technical Approach

1. **API Endpoint**: `POST /api/v1/ingest/url`
   - Request body: `{ "url": "https://...", "tenant_id": "uuid", "max_depth": 3, "options": {...} }`
   - Returns: `{ "job_id": "uuid", "status": "queued" }`

2. **Crawl4AI Integration**:
   - Install via: `pip install crawl4ai`
   - Configure browser-based crawling for JavaScript-rendered sites
   - Respect `robots.txt` and implement rate limiting (default: 1 req/sec)
   - Extract content from all linked pages up to `max_depth`

3. **Job Queue Flow**:
   ```
   API Request → Redis Stream (crawl.jobs) → Crawler Worker → Redis Stream (parse.jobs)
   ```

4. **Output Format**: Raw HTML/Markdown content with metadata (source URL, crawl timestamp, page title)

#### Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/api/routes/ingest.py` | Ingestion API endpoints |
| `backend/src/agentic_rag_backend/indexing/crawler.py` | Crawl4AI wrapper |
| `backend/src/agentic_rag_backend/indexing/workers/crawl_worker.py` | Async crawl worker |
| `backend/src/agentic_rag_backend/models/ingest.py` | Pydantic models for ingestion |

#### Acceptance Criteria

- [ ] Valid documentation URL triggers crawling via API
- [ ] Crawler respects robots.txt and rate limits
- [ ] All linked pages extracted up to max_depth
- [ ] Content queued for parsing pipeline
- [ ] Progress and statistics available via status endpoint

---

### 3.2 Story 4.2: PDF Document Parsing

**As a** data engineer,
**I want** to parse complex PDF documents with tables and structured layouts,
**So that** information is accurately extracted regardless of document format.

#### Technical Approach

1. **API Endpoint**: `POST /api/v1/ingest/document`
   - Multipart form: PDF file upload
   - Request: `{ "tenant_id": "uuid", "metadata": {...} }`
   - Returns: `{ "job_id": "uuid", "status": "queued" }`

2. **Docling Integration**:
   - Install via: `pip install docling==2.66.0`
   - Use `DocumentConverter` for PDF parsing
   - Enable table extraction with `TableMode.ACCURATE`
   - Preserve document structure (headers, sections, footnotes)

3. **Processing Flow**:
   ```python
   from docling.document_converter import DocumentConverter
   from docling.datamodel.base_models import InputFormat

   converter = DocumentConverter()
   result = converter.convert(pdf_path, input_format=InputFormat.PDF)
   ```

4. **Output**: `UnifiedDocument` with structured sections, tables as markdown, and metadata

#### Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/indexing/parser.py` | Docling parser wrapper |
| `backend/src/agentic_rag_backend/indexing/workers/parse_worker.py` | Async parse worker |
| `backend/src/agentic_rag_backend/models/documents.py` | Document models |

#### Acceptance Criteria

- [ ] PDF upload triggers parsing via API
- [ ] Tables extracted and converted to structured data
- [ ] Headers, sections, footnotes preserved
- [ ] 50-page document processed in < 5 minutes
- [ ] Output normalized to UnifiedDocument format

---

### 3.3 Story 4.3: Agentic Entity Extraction

**As a** data engineer,
**I want** an agent to autonomously extract entities and relationships from text,
**So that** the knowledge graph is built without manual schema mapping.

#### Technical Approach

1. **Agentic Indexer Agent** (Agno-based):
   ```python
   from agno.agent import Agent
   from agno.models.openai import OpenAIChat

   indexer_agent = Agent(
       name="IndexerAgent",
       model=OpenAIChat(id="gpt-4o"),
       instructions=[
           "Extract named entities (people, organizations, concepts, technologies)",
           "Identify relationships between entities",
           "Output structured JSON following the EntityGraph schema"
       ]
   )
   ```

2. **Entity Extraction Prompt Strategy**:
   - Process chunks sequentially with entity deduplication
   - Use structured JSON output mode for reliable parsing
   - Entity types: `Person`, `Organization`, `Technology`, `Concept`, `Document`, `Location`
   - Relationship types: `MENTIONS`, `AUTHORED_BY`, `PART_OF`, `USES`, `RELATED_TO`

3. **Graph Construction**:
   ```cypher
   MERGE (e:Entity {id: $id, tenant_id: $tenant_id})
   SET e.name = $name, e.type = $type, e.properties = $props

   MERGE (source)-[r:$rel_type]->(target)
   SET r.confidence = $confidence, r.source_chunk = $chunk_id
   ```

4. **Dual Storage**:
   - Chunks with embeddings → pgvector
   - Entities and relationships → Neo4j
   - Cross-reference via `chunk_id` and `entity_id`

#### Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/agents/indexer.py` | Agentic Indexer agent |
| `backend/src/agentic_rag_backend/indexing/entity_extractor.py` | Entity extraction logic |
| `backend/src/agentic_rag_backend/indexing/graph_builder.py` | Neo4j graph construction |
| `backend/src/agentic_rag_backend/indexing/chunker.py` | Semantic chunking |
| `backend/src/agentic_rag_backend/indexing/embeddings.py` | Embedding generation |
| `backend/src/agentic_rag_backend/db/neo4j.py` | Neo4j client |
| `backend/src/agentic_rag_backend/db/postgres.py` | PostgreSQL/pgvector client |

#### Acceptance Criteria

- [ ] Chunks processed by Agentic Indexer
- [ ] Named entities identified with types
- [ ] Relationships extracted between entities
- [ ] Neo4j nodes created with appropriate labels
- [ ] Neo4j edges created with relationship types
- [ ] pgvector stores chunk embeddings
- [ ] Extraction decisions logged in trajectory

---

### 3.4 Story 4.4: Knowledge Graph Visualization

**As a** data engineer,
**I want** to visualize the current state of the knowledge graph,
**So that** I can identify gaps, orphan nodes, and data quality issues.

#### Technical Approach

1. **API Endpoints**:
   - `GET /api/v1/knowledge/graph` - Fetch graph data for visualization
   - `GET /api/v1/knowledge/stats` - Graph statistics (node/edge counts)
   - `GET /api/v1/knowledge/orphans` - List orphan nodes

2. **Graph Query**:
   ```cypher
   MATCH (n)
   WHERE n.tenant_id = $tenant_id
   OPTIONAL MATCH (n)-[r]-(m)
   RETURN n, r, m
   LIMIT $limit
   ```

3. **Frontend Component** (React Flow):
   - Force-directed graph layout
   - Node colors by entity type
   - Edge labels showing relationship types
   - Zoom, pan, node selection
   - Filter by entity type or date range
   - Highlight orphan nodes in warning color (Amber-400)

4. **Data Format** (API Response):
   ```json
   {
     "nodes": [
       {"id": "uuid", "label": "Entity Name", "type": "Technology", "properties": {...}}
     ],
     "edges": [
       {"source": "uuid1", "target": "uuid2", "type": "USES", "properties": {...}}
     ],
     "stats": {
       "nodeCount": 1500,
       "edgeCount": 3200,
       "orphanCount": 12
     }
   }
   ```

#### Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/api/routes/knowledge.py` | Knowledge graph API |
| `frontend/src/components/graphs/KnowledgeGraph.tsx` | React Flow visualization |
| `frontend/src/components/graphs/EntityNode.tsx` | Custom node component |
| `frontend/src/components/graphs/RelationshipEdge.tsx` | Custom edge component |
| `frontend/src/hooks/use-knowledge-graph.ts` | Graph data fetching hook |
| `frontend/src/app/(features)/knowledge/page.tsx` | Knowledge page |

#### Acceptance Criteria

- [ ] React Flow renders graph data
- [ ] Nodes display labels and types with colors
- [ ] Edges show relationship types
- [ ] Users can zoom, pan, and select nodes
- [ ] Orphan nodes highlighted in Amber-400
- [ ] Filter by entity type available
- [ ] Stats endpoint returns node/edge counts

---

## 4. Database Schema

### 4.1 PostgreSQL / pgvector Tables

```sql
-- Documents table (source documents)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    source_type VARCHAR(20) NOT NULL, -- 'url' | 'pdf' | 'text'
    source_url TEXT,
    filename TEXT,
    content_hash VARCHAR(64) NOT NULL, -- SHA-256 for deduplication
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- pending | processing | completed | failed
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (tenant_id, content_hash)
);

CREATE INDEX idx_documents_tenant_id ON documents(tenant_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);

-- Chunks table (document chunks with embeddings)
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    embedding vector(1536), -- OpenAI ada-002 dimension
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chunks_tenant_id ON chunks(tenant_id);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Ingestion jobs table
CREATE TABLE ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    document_id UUID REFERENCES documents(id),
    job_type VARCHAR(20) NOT NULL, -- 'crawl' | 'parse' | 'index'
    status VARCHAR(20) NOT NULL DEFAULT 'queued', -- queued | running | completed | failed
    progress JSONB, -- { "pages_crawled": 10, "total_pages": 50 }
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ingestion_jobs_tenant_id ON ingestion_jobs(tenant_id);
CREATE INDEX idx_ingestion_jobs_status ON ingestion_jobs(status);
```

### 4.2 Neo4j Node Labels and Relationships

```cypher
-- Node Labels and Properties

// Entity node (base for all entity types)
(:Entity {
    id: String,           // UUID
    tenant_id: String,    // UUID for multi-tenancy
    name: String,
    type: String,         // Person | Organization | Technology | Concept | Location
    description: String,
    properties: Map,
    source_chunks: [String], // Array of chunk IDs
    created_at: DateTime,
    updated_at: DateTime
})

// Document node (represents ingested documents)
(:Document {
    id: String,
    tenant_id: String,
    title: String,
    source_url: String,
    source_type: String,
    content_hash: String,
    created_at: DateTime
})

// Chunk node (for graph-based chunk navigation)
(:Chunk {
    id: String,
    tenant_id: String,
    document_id: String,
    chunk_index: Integer,
    preview: String,      // First 200 chars
    created_at: DateTime
})

-- Relationship Types

// Entity relationships
(:Entity)-[:MENTIONS {confidence: Float, chunk_id: String}]->(:Entity)
(:Entity)-[:AUTHORED_BY {confidence: Float}]->(:Entity)
(:Entity)-[:PART_OF {confidence: Float}]->(:Entity)
(:Entity)-[:USES {confidence: Float}]->(:Entity)
(:Entity)-[:RELATED_TO {confidence: Float, description: String}]->(:Entity)

// Document-Entity relationships
(:Document)-[:CONTAINS]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:Entity)
(:Chunk)-[:NEXT]->(:Chunk)  // Sequential chunk ordering

-- Indexes for Performance
CREATE INDEX entity_id FOR (e:Entity) ON (e.id);
CREATE INDEX entity_tenant ON (e:Entity) ON (e.tenant_id);
CREATE INDEX entity_type FOR (e:Entity) ON (e.type);
CREATE INDEX document_id FOR (d:Document) ON (d.id);
CREATE INDEX document_tenant FOR (d:Document) ON (d.tenant_id);
CREATE INDEX chunk_id FOR (c:Chunk) ON (c.id);
CREATE INDEX chunk_tenant FOR (c:Chunk) ON (c.tenant_id);
```

---

## 5. API Endpoints

### 5.1 Ingestion API

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| POST | `/api/v1/ingest/url` | Start URL crawl | `{ url, tenant_id, max_depth?, options? }` | `{ job_id, status }` |
| POST | `/api/v1/ingest/document` | Upload document | Multipart: file + metadata | `{ job_id, status }` |
| GET | `/api/v1/ingest/jobs/{job_id}` | Get job status | - | `{ job_id, status, progress, error? }` |
| GET | `/api/v1/ingest/jobs` | List jobs | `?tenant_id&status&limit&offset` | `{ jobs: [...], total }` |
| DELETE | `/api/v1/ingest/jobs/{job_id}` | Cancel job | - | `{ success }` |

### 5.2 Knowledge API

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/api/v1/knowledge/graph` | Get graph data | `?tenant_id&limit&entity_type&date_from` | `{ nodes, edges, stats }` |
| GET | `/api/v1/knowledge/stats` | Get graph stats | `?tenant_id` | `{ nodeCount, edgeCount, ... }` |
| GET | `/api/v1/knowledge/orphans` | List orphan nodes | `?tenant_id&limit` | `{ orphans: [...] }` |
| GET | `/api/v1/knowledge/entities/{id}` | Get entity details | - | `{ entity, relationships }` |
| DELETE | `/api/v1/knowledge/documents/{id}` | Delete document & related | - | `{ success, deleted_count }` |

### 5.3 Response Formats

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
  "detail": "The 'url' field must be a valid URL",
  "instance": "/api/v1/ingest/url"
}
```

---

## 6. File Structure

### 6.1 New Backend Files

```
backend/src/agentic_rag_backend/
├── agents/
│   └── indexer.py                    # NEW: Agentic Indexer agent
├── api/
│   ├── routes/
│   │   ├── __init__.py               # UPDATE: Add new routers
│   │   ├── ingest.py                 # NEW: Ingestion endpoints
│   │   └── knowledge.py              # NEW: Knowledge graph endpoints
│   └── middleware/
│       └── tenant.py                 # NEW: Multi-tenancy middleware
├── db/
│   ├── __init__.py                   # NEW
│   ├── postgres.py                   # NEW: PostgreSQL/pgvector client
│   ├── neo4j.py                      # NEW: Neo4j client
│   └── redis.py                      # NEW: Redis client
├── indexing/
│   ├── __init__.py                   # NEW
│   ├── crawler.py                    # NEW: Crawl4AI wrapper
│   ├── parser.py                     # NEW: Docling parser
│   ├── chunker.py                    # NEW: Semantic chunking
│   ├── embeddings.py                 # NEW: Embedding generation
│   ├── entity_extractor.py           # NEW: Entity extraction logic
│   ├── graph_builder.py              # NEW: Neo4j graph construction
│   ├── pipeline.py                   # NEW: Pipeline orchestration
│   └── workers/
│       ├── __init__.py               # NEW
│       ├── crawl_worker.py           # NEW: Async crawl worker
│       ├── parse_worker.py           # NEW: Async parse worker
│       └── index_worker.py           # NEW: Async indexing worker
├── models/
│   ├── __init__.py                   # NEW
│   ├── documents.py                  # NEW: Document schemas
│   ├── ingest.py                     # NEW: Ingestion schemas
│   ├── graphs.py                     # NEW: Graph schemas
│   └── events.py                     # NEW: Event schemas
└── core/
    ├── errors.py                     # NEW: Error types
    └── logging.py                    # NEW: Structured logging
```

### 6.2 New Frontend Files

```
frontend/src/
├── app/
│   └── (features)/
│       └── knowledge/
│           ├── page.tsx              # NEW: Knowledge graph page
│           └── [id]/
│               └── page.tsx          # NEW: Entity detail page
├── components/
│   └── graphs/
│       ├── KnowledgeGraph.tsx        # NEW: React Flow graph
│       ├── EntityNode.tsx            # NEW: Custom node component
│       └── RelationshipEdge.tsx      # NEW: Custom edge component
├── hooks/
│   └── use-knowledge-graph.ts        # NEW: Graph data hook
├── lib/
│   └── api.ts                        # UPDATE: Add knowledge API calls
└── types/
    └── graphs.ts                     # NEW: Graph types
```

---

## 7. Dependencies

### 7.1 Dependencies on Existing Code (Epic 1)

| Dependency | Location | Purpose |
|------------|----------|---------|
| FastAPI app | `main.py` | Add new routers |
| Settings | `config.py` | Database connection strings |
| Docker Compose | `docker-compose.yml` | Add Docling service (optional) |

### 7.2 External Dependencies (Backend)

```toml
# pyproject.toml additions
dependencies = [
    # Existing
    "agno==2.3.21",
    "fastapi==0.111.0",
    "uvicorn[standard]==0.30.0",
    "python-dotenv==1.0.1",

    # Epic 4 - New
    "crawl4ai>=0.3.0",           # Web crawling
    "docling==2.66.0",           # PDF parsing
    "neo4j>=5.0.0",              # Neo4j driver
    "asyncpg>=0.29.0",           # Async PostgreSQL
    "pgvector>=0.2.0",           # pgvector support
    "redis>=5.0.0",              # Redis client
    "openai>=1.0.0",             # Embeddings & entity extraction
    "tiktoken>=0.5.0",           # Token counting for chunking
    "pydantic>=2.0.0",           # Data validation
    "structlog>=24.0.0",         # Structured logging
    "tenacity>=8.0.0",           # Retry logic
]
```

### 7.3 External Dependencies (Frontend)

```json
// package.json additions
{
  "dependencies": {
    "reactflow": "^11.11.0",
    "@tanstack/react-query": "^5.0.0"
  }
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/indexing/test_crawler.py` | Crawl4AI wrapper functions |
| `tests/indexing/test_parser.py` | Docling parsing logic |
| `tests/indexing/test_chunker.py` | Semantic chunking |
| `tests/indexing/test_entity_extractor.py` | Entity extraction (mocked LLM) |
| `tests/indexing/test_graph_builder.py` | Neo4j operations (mocked) |
| `tests/api/test_ingest.py` | Ingestion API endpoints |
| `tests/api/test_knowledge.py` | Knowledge API endpoints |

### 8.2 Integration Tests

| Test File | Coverage |
|-----------|----------|
| `tests/integration/test_ingestion_pipeline.py` | End-to-end ingestion flow |
| `tests/integration/test_neo4j_integration.py` | Real Neo4j operations |
| `tests/integration/test_pgvector_integration.py` | Real pgvector operations |

### 8.3 Test Fixtures

```python
# tests/conftest.py additions

@pytest.fixture
def sample_pdf():
    """Provide a sample PDF for testing."""
    return Path(__file__).parent / "fixtures" / "sample.pdf"

@pytest.fixture
def sample_html():
    """Provide sample HTML content."""
    return "<html><body><h1>Test</h1><p>Content</p></body></html>"

@pytest.fixture
def mock_openai():
    """Mock OpenAI API for entity extraction."""
    with patch("openai.OpenAI") as mock:
        mock.return_value.chat.completions.create.return_value = ...
        yield mock
```

### 8.4 Frontend Tests

| Test File | Coverage |
|-----------|----------|
| `tests/components/knowledge-graph.test.tsx` | Graph component rendering |
| `tests/e2e/knowledge.spec.ts` | End-to-end knowledge page tests |

---

## 9. Acceptance Criteria Summary

### Story 4.1: URL Documentation Crawling
- [ ] `POST /api/v1/ingest/url` accepts valid URL and returns job_id
- [ ] Crawl4AI crawls documentation site respecting robots.txt
- [ ] Rate limiting enforced (default 1 req/sec)
- [ ] All linked pages extracted up to max_depth
- [ ] Job status endpoint shows progress
- [ ] Content queued for parsing pipeline

### Story 4.2: PDF Document Parsing
- [ ] `POST /api/v1/ingest/document` accepts PDF upload
- [ ] Docling extracts text preserving structure
- [ ] Tables converted to structured data
- [ ] Headers, sections, footnotes identified
- [ ] 50-page document processed in < 5 minutes
- [ ] Output normalized to UnifiedDocument

### Story 4.3: Agentic Entity Extraction
- [ ] Agentic Indexer processes document chunks
- [ ] Named entities extracted with types (Person, Organization, etc.)
- [ ] Relationships identified between entities
- [ ] Neo4j nodes created with labels
- [ ] Neo4j edges created with relationship types
- [ ] pgvector stores chunk embeddings
- [ ] Extraction logged in trajectory

### Story 4.4: Knowledge Graph Visualization
- [ ] React Flow renders graph from API data
- [ ] Nodes colored by entity type
- [ ] Edges labeled with relationship types
- [ ] Zoom, pan, node selection functional
- [ ] Orphan nodes highlighted in Amber-400
- [ ] Entity type filter available
- [ ] Stats endpoint returns counts

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Crawl4AI blocked by sites | High | Implement fallback to simple HTTP + BeautifulSoup |
| Docling parsing errors | Medium | Add error handling with partial extraction |
| LLM rate limits | Medium | Implement exponential backoff via tenacity |
| Neo4j performance at scale | Medium | Add proper indexing, use APOC for bulk operations |
| Entity deduplication issues | Medium | Use fuzzy matching + embeddings for entity resolution |
| Long ingestion times | High | Async workers with progress tracking |

---

## 11. Implementation Order

1. **Story 4.2 (PDF Parsing)** - Start here as it's self-contained and foundational
2. **Story 4.1 (URL Crawling)** - Builds on parsing infrastructure
3. **Story 4.3 (Entity Extraction)** - Requires both parsing pipelines
4. **Story 4.4 (Visualization)** - Frontend work after graph is populated

---

## 12. Configuration Updates

### 12.1 Environment Variables

Add to `.env.example`:

```bash
# Epic 4 - Ingestion Pipeline
CRAWL4AI_RATE_LIMIT=1.0           # Requests per second
CHUNK_SIZE=512                     # Tokens per chunk
CHUNK_OVERLAP=64                   # Token overlap between chunks
ENTITY_EXTRACTION_MODEL=gpt-4o     # Model for entity extraction
EMBEDDING_MODEL=text-embedding-ada-002
```

### 12.2 Docker Compose Update (Optional)

```yaml
# docker-compose.yml addition (optional for local Docling)
services:
  docling:
    image: ds4sd/docling:latest
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 13. Appendix

### 13.1 Entity Extraction Prompt Template

```
You are an expert at extracting structured information from text.

Given the following text chunk, extract:
1. Named entities (people, organizations, technologies, concepts, locations)
2. Relationships between entities

Output format (JSON):
{
  "entities": [
    {"name": "...", "type": "Person|Organization|Technology|Concept|Location", "description": "..."}
  ],
  "relationships": [
    {"source": "entity_name", "target": "entity_name", "type": "MENTIONS|AUTHORED_BY|USES|PART_OF|RELATED_TO", "confidence": 0.0-1.0}
  ]
}

Text chunk:
{chunk_content}
```

### 13.2 React Flow Node Styling

```typescript
const entityColors: Record<string, string> = {
  Person: '#3B82F6',        // Blue
  Organization: '#10B981',   // Emerald
  Technology: '#6366F1',     // Indigo
  Concept: '#8B5CF6',        // Violet
  Location: '#F59E0B',       // Amber
  orphan: '#F97316',         // Orange (warning)
};
```

---

**Document Version:** 1.0
**Last Updated:** 2025-12-28
**Author:** Development Team
