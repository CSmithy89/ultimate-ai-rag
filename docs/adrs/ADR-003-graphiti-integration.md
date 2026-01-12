# ADR-003: Graphiti Integration for Temporal Knowledge

**Status:** Accepted
**Date:** 2025-12-29
**Deciders:** Architecture Team
**Technical Story:** Epic 5 - Graphiti Temporal Knowledge Graph Integration

## Context

Epic 4 implemented a custom knowledge graph pipeline with:
- Custom entity extraction using OpenAI prompts (~352 lines)
- Custom graph builder for Neo4j (~295 lines)
- Custom embeddings management (~228 lines)
- Custom indexer agent integration (~200 lines)

While functional, this approach had significant limitations:

1. **No Temporal Awareness**: No tracking of when facts were true or when they changed
2. **Manual Deduplication**: Entity and relationship deduplication required custom logic
3. **Embedding Management**: Custom code for embedding generation and storage
4. **No Contradiction Handling**: No automatic invalidation of outdated facts
5. **Maintenance Burden**: ~1,075 lines of custom code to maintain

### Graphiti Overview

Graphiti is Zep's temporal knowledge graph framework that provides:
- **Bi-temporal tracking**: Fact validity time AND ingestion time
- **Automatic entity resolution**: Deduplication via embedding similarity
- **Episode-based ingestion**: Optimized for agent conversation workflows
- **Built-in embeddings**: BGE-m3 model with automatic chunking
- **Point-in-time queries**: Query knowledge at any historical moment

## Decision

We adopt **Graphiti** as the temporal knowledge graph layer, replacing our custom entity extraction and graph building pipeline.

### Integration Architecture

```
BEFORE (Epic 4):
Document --> Chunker --> EntityExtractor --> GraphBuilder --> Neo4j (custom schema)
                |              |                  |
                v              v                  v
          Custom code    OpenAI prompts    Custom Neo4j

AFTER (Epic 5):
Document --> Parser --> Graphiti.add_episode() --> Neo4j (Graphiti-managed temporal schema)
                              |
                              v
                     Automatic entity extraction
                     Automatic deduplication
                     Temporal edge management
                     Built-in embeddings (BGE-m3)
```

### Key Integration Points

1. **Episode-Based Ingestion**: Documents are ingested as "episodes" that track source, timestamp, and content
2. **Custom Entity Types**: We define domain-specific entity types via Pydantic models
3. **Hybrid Retrieval**: Graphiti's temporal search combined with pgvector semantic search
4. **Temporal Queries**: New API endpoints for point-in-time knowledge queries

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

## Consequences

### Positive

- **~1,075 Lines Removed**: Elimination of custom extraction/building code
- **Bi-Temporal Tracking**: Know when facts were true AND when they were ingested
- **Point-in-Time Queries**: Query knowledge graph at specific historical dates
- **Automatic Contradiction Resolution**: Temporal edge invalidation when facts change
- **Agent Memory Optimization**: Episode-based ingestion aligns with agent workflows
- **Reduced LLM Costs**: Graphiti's efficient extraction reduces token usage
- **Better Entity Resolution**: Embedding-based deduplication is more accurate

### Negative

- **Dependency on External Library**: Graphiti becomes a critical dependency
- **Schema Migration**: Existing graph data needs migration to Graphiti schema
- **Learning Curve**: Team needs to learn Graphiti's API and concepts
- **Less Control**: Black-box entity extraction vs. custom prompts

### Neutral

- **Neo4j Still Required**: Graphiti uses Neo4j as its backend
- **Embedding Model Change**: Switches from our chosen model to BGE-m3
- **API Changes**: New temporal query endpoints vs. existing graph queries

## Alternatives Considered

### 1. Keep Custom Implementation

**Rejected because:**
- High maintenance burden (~1,075 lines)
- No temporal tracking capability
- Manual deduplication is error-prone
- Missing contradiction resolution
- Would need to build temporal features from scratch

### 2. Microsoft GraphRAG

**Rejected because:**
- Heavy pre-computation (community summaries at index time)
- Not designed for real-time ingestion
- No bi-temporal tracking
- LangChain dependency conflicts with Agno

### 3. LightRAG

**Rejected because:**
- Newer, less mature ecosystem
- Limited temporal capabilities
- Smaller community support
- Would still require temporal feature development

### 4. Cognee

**Rejected because:**
- Different architectural approach (task-based)
- Less focus on temporal knowledge
- Would require significant adaptation
- Smaller user community

### 5. Build Custom Temporal Layer on Top of Existing Code

**Rejected because:**
- Significant development effort
- Would increase maintenance burden
- Graphiti already solves these problems
- Time better spent on other features

## Implementation Notes

### Migration Strategy

1. **Phase 1**: Parallel installation (Graphiti alongside existing)
2. **Phase 2**: Feature flag routing (new documents --> Graphiti)
3. **Phase 3**: Migration of existing knowledge graph data
4. **Phase 4**: Legacy code removal
5. **Phase 5**: Test suite adaptation

### New Dependencies

```toml
# backend/pyproject.toml
dependencies = [
  "graphiti-core>=0.5.0",  # Temporal knowledge graph SDK
]
```

### New Modules

| Module | Purpose |
|--------|---------|
| `db/graphiti.py` | Graphiti client wrapper with custom entity types |
| `models/entity_types.py` | Pydantic entity type definitions |
| `indexing/graphiti_ingestion.py` | Episode-based document ingestion |
| `retrieval/graphiti_retrieval.py` | Temporal-aware hybrid search |

### Deleted Modules

| Module | Lines Removed |
|--------|---------------|
| `indexing/entity_extractor.py` | -352 |
| `indexing/graph_builder.py` | -295 |
| `indexing/embeddings.py` | -228 |
| `agents/indexer.py` (simplified) | -200 |
| **Total** | **~1,075** |

### New API Endpoints

```
POST /api/v1/knowledge/temporal-query
  - Query knowledge graph at specific point in time

GET /api/v1/knowledge/changes
  - Get knowledge changes over time period

GET /api/v1/knowledge/entity/{id}/history
  - Get all temporal versions of an entity
```

### Configuration Options

| Config | Purpose | Default |
|--------|---------|---------|
| `GRAPHITI_ENABLED` | Enable Graphiti integration | `true` |
| `GRAPHITI_LLM_MODEL` | Model for entity extraction | `gpt-4o-mini` |
| `GRAPHITI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `GRAPHITI_NEO4J_URI` | Neo4j connection | `bolt://localhost:7687` |

## References

- [Architecture Addendum: Graphiti Integration](/home/chris/projects/work/Agentic Rag and Graphrag with copilot/_bmad-output/architecture.md#architecture-addendum-graphiti-integration-epic-5)
- [Epic 5 Tech Spec](/home/chris/projects/work/Agentic Rag and Graphrag with copilot/_bmad-output/epics/epic-5-tech-spec.md)
- [Graphiti GitHub Repository](https://github.com/getzep/graphiti)
- [Zep: Temporal Knowledge Graph Architecture (arXiv)](https://arxiv.org/abs/2501.13956)
- [Graphiti Documentation](https://docs.zep.ai/graphiti)
