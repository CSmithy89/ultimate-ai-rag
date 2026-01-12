# ADR-001: Hybrid Vector + Graph RAG Architecture

**Status:** Accepted
**Date:** 2025-12-28
**Deciders:** Architecture Team
**Technical Story:** FR11-FR14 (Hybrid Retrieval Requirements)

## Context

The Agentic RAG platform requires a retrieval system capable of answering complex queries that span factual information, relationships between entities, and temporal knowledge. Traditional vector-only RAG systems excel at semantic similarity but struggle with:

1. **Multi-hop reasoning** - Questions requiring traversal of entity relationships
2. **Temporal queries** - Understanding when facts were true
3. **Structured knowledge** - Explicit entity-relationship modeling
4. **Explainability** - Providing transparent retrieval reasoning

The project's PRD specifies 25 functional requirements across 5 domains, with FR11-FR14 specifically addressing hybrid retrieval capabilities:
- FR11: Semantic similarity via vector search
- FR12: Relationship traversal via graph queries
- FR13: Combined answer synthesis
- FR14: Graph-based explainability

## Decision

We adopt a **hybrid vector + graph RAG architecture** that combines:

1. **pgvector (PostgreSQL 16+)** for dense vector embeddings and semantic similarity search
2. **Neo4j 5.x Community** for knowledge graph storage, entity relationships, and graph traversal
3. **A synthesis layer** that combines results from both retrieval methods

### Architecture Overview

```
Query --> Orchestrator Agent
              |
    +---------+---------+
    |                   |
    v                   v
pgvector            Neo4j
(semantic)          (relationships)
    |                   |
    +---------+---------+
              |
              v
        Synthesis Layer
              |
              v
         Response + Explainability
```

### Key Design Decisions

1. **Separate Databases** - Vector and graph stores are kept separate rather than using Neo4j's built-in vector capabilities for:
   - Independent scaling of vector vs. graph workloads
   - Leverage PostgreSQL's mature ecosystem (backups, replication, tooling)
   - Better cost efficiency (pgvector is free, Neo4j vector requires enterprise)

2. **Agent-Driven Routing** - The orchestrator agent dynamically selects retrieval methods based on query characteristics:
   - Semantic queries --> Vector search
   - Relationship queries --> Graph traversal
   - Complex queries --> Both + synthesis

3. **Dual Reranking Pipeline**
   - Stage 1: Graphiti episode-based retrieval (temporal + semantic)
   - Stage 2: Cross-encoder reranking (configurable: Cohere, FlashRank, ColBERT)

4. **Explainability via Graph Paths** - Every answer includes the graph traversal path that contributed to it, enabling users to verify reasoning.

## Consequences

### Positive

- **Superior multi-hop reasoning**: Graph traversal handles questions like "Who are the collaborators of people mentioned in document X?"
- **Temporal awareness**: Bi-temporal tracking (fact validity time + ingestion time) via Graphiti
- **Transparent explainability**: Graph paths provide clear audit trails
- **Scalable to 1M+ nodes**: Neo4j handles graphs exceeding available memory
- **Competitive feature parity**: Matches capabilities of MS GraphRAG, LightRAG, and Cognee

### Negative

- **Increased operational complexity**: Two database systems to maintain
- **Higher latency**: Parallel retrieval + synthesis adds ~100-200ms
- **Learning curve**: Teams need expertise in both vector and graph paradigms
- **Storage duplication**: Some entity data appears in both stores

### Neutral

- **Cost**: pgvector is free; Neo4j Community is free but limited (no clustering)
- **Query complexity**: Requires careful query decomposition by the orchestrator agent

## Alternatives Considered

### 1. Vector-Only RAG (pgvector or Qdrant)

**Rejected because:**
- Cannot perform relationship traversal
- No explicit entity modeling
- Poor multi-hop reasoning
- No temporal fact tracking

### 2. Neo4j with Built-in Vector Search

**Rejected because:**
- Vector capabilities require Neo4j Enterprise (expensive)
- Less mature vector implementation compared to pgvector
- Couples vector scaling to graph scaling
- Limited embedding model support

### 3. Microsoft GraphRAG (LangChain)

**Rejected because:**
- Heavy pre-computation indexing (community summaries)
- Not designed for real-time ingestion
- LangChain dependency conflicts with Agno framework
- Less flexible for custom retrieval patterns

### 4. LightRAG or Cognee

**Rejected because:**
- Less mature ecosystems
- Limited temporal knowledge support
- Smaller community and documentation
- Would require significant customization for our use case

### 5. Single Store (Qdrant or Weaviate)

**Rejected because:**
- Hybrid vector+graph in single store is immature
- Less explicit relationship modeling
- Would sacrifice either vector or graph capabilities

## Implementation Notes

### Database Configuration

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg16
    # pgvector for semantic embeddings

  neo4j:
    image: neo4j:5-community
    environment:
      - NEO4J_PLUGINS=["apoc"]
    # Entity relationships + Graphiti temporal schema
```

### Multi-Tenancy Isolation

All queries MUST include `tenant_id` filtering:
- PostgreSQL: Schema-level isolation or row-level security
- Neo4j: Property-based filtering on all nodes/relationships

### Configuration Options

| Config | Purpose | Default |
|--------|---------|---------|
| `VECTOR_SEARCH_ENABLED` | Enable pgvector retrieval | `true` |
| `GRAPH_SEARCH_ENABLED` | Enable Neo4j retrieval | `true` |
| `RERANKER_ENABLED` | Cross-encoder reranking | `false` |
| `RERANKER_PROVIDER` | Reranker backend | `flashrank` |
| `GRADER_ENABLED` | CRAG quality grading | `false` |

## References

- [Architecture Document](/home/chris/projects/work/Agentic Rag and Graphrag with copilot/_bmad-output/architecture.md)
- [Epic 3: Hybrid Knowledge Retrieval](/home/chris/projects/work/Agentic Rag and Graphrag with copilot/_bmad-output/project-planning-artifacts/epics.md)
- [Epic 12: Advanced Retrieval](/home/chris/projects/work/Agentic Rag and Graphrag with copilot/_bmad-output/epics/epic-12-tech-spec.md)
- [Advanced Retrieval Configuration Guide](/home/chris/projects/work/Agentic Rag and Graphrag with copilot/docs/guides/advanced-retrieval-configuration.md)
