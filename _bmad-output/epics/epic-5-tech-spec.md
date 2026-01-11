# Epic 5 Tech Spec: Graphiti Temporal Knowledge Graph Integration

**Date:** 2025-12-29
**Status:** Complete
**Epic Owner:** Development Team

---

## 1. Overview

Epic 5 integrates Graphiti, Zep's temporal knowledge graph framework, to replace our custom entity extraction and graph building pipeline. This upgrade provides bi-temporal tracking, automatic contradiction resolution, and agent-optimized memory capabilities.

### 1.1 Epic Goals

- Replace custom entity extraction with Graphiti's episode-based ingestion
- Enable temporal queries (point-in-time historical lookups)
- Simplify codebase by removing custom graph management logic
- Improve retrieval quality with Graphiti's hybrid search (semantic + BM25 + graph traversal)

### 1.2 Motivation

Our Epic 4 implementation works but has limitations:
- **Static knowledge**: No temporal awareness - facts can't change over time
- **Manual entity extraction**: Custom LLM prompts for entity extraction
- **No contradiction handling**: Conflicting information not resolved
- **Complex codebase**: 25k+ lines of custom indexing logic

Graphiti provides:
- **Bi-temporal tracking**: Know when facts were true AND when ingested
- **Automatic extraction**: Type-safe entity extraction via Pydantic models
- **Edge invalidation**: Temporal resolution of contradictions
- **Production-proven**: 94.8% on Deep Memory Retrieval benchmark

### 1.3 Non-Functional Requirements

| NFR ID | Target | Impact |
|--------|--------|--------|
| NFR2 | < 5 min ingestion for 50-page doc | Graphiti parallel processing |
| NFR3 | < 2s query latency | Graphiti <100ms search + LLM |
| NFR5 | 1M+ nodes/edges | Enterprise scalability built-in |

### 1.4 Non-Goals

- Frontend visualization changes (reuse Epic 4.4)
- Query orchestrator changes beyond retrieval integration
- Full database migration (parallel running initially)

---

## 2. Architecture Decisions

### 2.1 High-Level Architecture Change

```
BEFORE (Epic 4):
┌─────────────────────────────────────────────────────────────────────────┐
│ Document → Chunker → EntityExtractor → GraphBuilder → Neo4j            │
│     ↓         ↓            ↓               ↓            ↓              │
│  Parser   tiktoken    OpenAI calls    Custom logic   Manual schema     │
└─────────────────────────────────────────────────────────────────────────┘

AFTER (Epic 5 with Graphiti):
┌─────────────────────────────────────────────────────────────────────────┐
│ Document → Parser → Graphiti.add_episode() → Neo4j (Graphiti-managed)  │
│     ↓         ↓              ↓                        ↓                │
│  Crawler  Docling    Automatic extraction      Temporal edges          │
│                      + type classification     + bi-temporal model     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack Changes

| Component | Before (Epic 4) | After (Epic 5) |
|-----------|-----------------|----------------|
| Entity Extraction | Custom OpenAI prompts | Graphiti SDK |
| Graph Building | Custom Neo4j queries | Graphiti episode ingestion |
| Embeddings | Custom pgvector integration | Graphiti built-in (BGE-m3) |
| Relationship Management | Manual deduplication | Automatic temporal edges |
| Search | Custom hybrid logic | Graphiti hybrid retrieval |

### 2.3 Key Dependencies

```toml
# backend/pyproject.toml additions
dependencies = [
  "graphiti-core>=0.5.0",  # Core Graphiti SDK
]
```

### 2.4 Custom Entity Types

```python
from graphiti_core.models import EntityModel
from pydantic import Field

class TechnicalConcept(EntityModel):
    """Technical concept from documentation."""
    domain: str = Field(description="Technical domain (frontend, backend, database, devops)")
    complexity: str = Field(description="Complexity level: basic, intermediate, advanced")

class CodePattern(EntityModel):
    """A code pattern, best practice, or anti-pattern."""
    language: str = Field(description="Programming language")
    pattern_type: str = Field(description="Type: design-pattern, anti-pattern, idiom, convention")

class APIEndpoint(EntityModel):
    """REST or GraphQL API endpoint."""
    method: str = Field(description="HTTP method (GET, POST, PUT, DELETE)")
    path: str = Field(description="Endpoint path pattern")

class ConfigurationOption(EntityModel):
    """Configuration setting or environment variable."""
    config_type: str = Field(description="Type: env-var, config-file, cli-flag")
    default_value: str = Field(description="Default value if any")
```

### 2.5 Edge Type Mapping

```python
edge_type_map = {
    ("TechnicalConcept", "TechnicalConcept"): ["RELATES_TO", "DEPENDS_ON", "EXTENDS"],
    ("TechnicalConcept", "CodePattern"): ["IMPLEMENTS", "DEMONSTRATES", "VIOLATES"],
    ("CodePattern", "CodePattern"): ["SIMILAR_TO", "ALTERNATIVE_TO", "REFACTORS_TO"],
    ("APIEndpoint", "TechnicalConcept"): ["USES", "REQUIRES", "RETURNS"],
    ("ConfigurationOption", "TechnicalConcept"): ["CONFIGURES", "ENABLES", "DISABLES"],
}
```

---

## 3. Module Changes

### 3.1 Modules to DELETE (replaced by Graphiti)

| Module | Lines | Replacement |
|--------|-------|-------------|
| `indexing/entity_extractor.py` | 352 | Graphiti entity extraction |
| `indexing/graph_builder.py` | 295 | Graphiti episode ingestion |
| `indexing/embeddings.py` | 228 | Graphiti built-in embeddings |
| `agents/indexer.py` (partial) | ~200 | Simplified Graphiti wrapper |

**Total reduction: ~1,075 lines of custom code**

### 3.2 Modules to MODIFY

| Module | Changes |
|--------|---------|
| `db/neo4j.py` | Simplify to Graphiti client wrapper |
| `agents/indexer.py` | Refactor for episode-based ingestion |
| `indexing/workers/index_worker.py` | Use Graphiti for graph operations |
| `main.py` | Initialize Graphiti client in lifespan |

### 3.3 Modules to KEEP (unchanged)

| Module | Reason |
|--------|--------|
| `indexing/crawler.py` | URL ingestion still needed |
| `indexing/parser.py` | PDF parsing still needed |
| `indexing/chunker.py` | May be simplified but kept for document sectioning |
| `api/routes/knowledge.py` | Graph visualization API unchanged |

### 3.4 New Modules

| Module | Purpose |
|--------|---------|
| `db/graphiti.py` | Graphiti client wrapper with custom entity types |
| `models/entity_types.py` | Pydantic entity type definitions |

---

## 4. Migration Strategy

### 4.1 Phase 1: Parallel Installation (Story 5.1)

- Install Graphiti alongside existing stack
- Configure custom entity types
- Initialize Graphiti client in application lifespan
- **No existing functionality affected**

### 4.2 Phase 2: Episode Ingestion Pipeline (Story 5.2)

- Create new ingestion path using Graphiti
- Documents ingested via `graphiti.add_episode()`
- **Old ingestion path remains for rollback**

### 4.3 Phase 3: Hybrid Retrieval Integration (Story 5.3)

- Integrate Graphiti search with query orchestrator
- Compare retrieval quality vs old approach
- **Feature flag for A/B testing**

### 4.4 Phase 4: Temporal Query Capabilities (Story 5.4)

- Add point-in-time query support to API
- Enable "what changed" queries
- **New feature, no migration needed**

### 4.5 Phase 5: Legacy Code Removal (Story 5.5)

- Migrate existing knowledge graph to Graphiti format
- Remove deprecated modules
- **Final cutover**

### 4.6 Phase 6: Test Suite Adaptation (Story 5.6)

- Update unit tests for Graphiti integration
- Add integration tests for temporal queries
- Ensure coverage parity

---

## 5. API Changes

### 5.1 New Endpoints

```
POST /api/v1/knowledge/temporal-query
  - Query knowledge graph at specific point in time
  - Request: { tenant_id, query, as_of_date? }
  - Response: { data: [...], temporal_context: {...} }

GET /api/v1/knowledge/changes
  - Get knowledge changes over time period
  - Query params: tenant_id, start_date, end_date, entity_type?
  - Response: { changes: [...], summary: {...} }
```

### 5.2 Modified Endpoints

```
POST /api/v1/ingest/url (unchanged interface)
  - Internal: Uses Graphiti episode ingestion instead of custom pipeline

POST /api/v1/ingest/document (unchanged interface)
  - Internal: Uses Graphiti episode ingestion instead of custom pipeline

GET /api/v1/knowledge/graph (unchanged interface)
  - Internal: Queries Graphiti-managed graph
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

- Custom entity type validation
- Graphiti client wrapper functions
- Episode creation from documents

### 6.2 Integration Tests

- Full ingestion pipeline with Graphiti
- Temporal query accuracy
- Hybrid search quality comparison

### 6.3 Migration Tests

- Existing graph data migrated correctly
- No data loss during cutover
- Rollback capability verified

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Graphiti API changes | HIGH | Pin version, monitor releases |
| Performance regression | MEDIUM | Benchmark before/after, parallel run |
| Entity type mismatch | MEDIUM | Extensive mapping tests |
| Neo4j schema conflicts | LOW | Separate database/namespace |

---

## 8. Success Criteria

1. All existing tests pass with Graphiti backend
2. Retrieval quality >= Epic 4 baseline
3. Temporal queries functional (<2s latency)
4. Code reduction >= 1,000 lines
5. No data loss in migration

---

## 9. References

- [Graphiti GitHub Repository](https://github.com/getzep/graphiti)
- [Zep: Temporal Knowledge Graph Architecture (arXiv)](https://arxiv.org/abs/2501.13956)
- [Custom Entity Types Documentation](https://deepwiki.com/getzep/graphiti/10.3-custom-entity-types)
- [Neo4j Blog: Graphiti Knowledge Graph Memory](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
