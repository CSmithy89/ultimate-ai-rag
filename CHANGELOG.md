# Changelog

## [Epic 5] - Graphiti Temporal Knowledge Graph Integration - 2025-12-29

### Added
- Graphiti integration with `graphiti-core>=0.5.0` for temporal knowledge graphs
- Custom entity types: `TechnicalConcept`, `CodePattern`, `APIEndpoint`, `ConfigurationOption`
- Edge type mappings for semantic relationship classification
- `GraphitiClient` wrapper with connection lifecycle management
- Episode-based document ingestion via `ingest_document_as_episode()`
- Hybrid retrieval with `graphiti_search()` and `search_with_backend_routing()`
- Temporal query capabilities:
  - Point-in-time search with `temporal_search()`
  - Knowledge change tracking with `get_knowledge_changes()`
- API endpoints for temporal queries (`/knowledge/temporal/search`, `/knowledge/temporal/changes`)
- Feature flags: `INGESTION_BACKEND` and `RETRIEVAL_BACKEND` (values: "graphiti" or "legacy")
- Integration test suite for end-to-end Graphiti workflow
- pytest-cov for coverage reporting

### Changed
- Default ingestion and retrieval backends now use Graphiti
- Legacy indexing modules (`embeddings`, `entity_extractor`, `graph_builder`) marked as deprecated

### Deprecated
- `EmbeddingGenerator` - use `ingest_document_as_episode()` instead
- `EntityExtractor` - use `ingest_document_as_episode()` instead
- `GraphBuilder` - Graphiti handles graph construction automatically

## [Epic 4] - Knowledge Ingestion Pipeline - 2025-12-29

### Added
- URL documentation crawling with Crawl4AI integration
- PDF document parsing with Docling
- Agentic entity extraction with LLM-powered NER
- Knowledge graph visualization endpoints
- Document chunking with semantic boundaries
- Multi-tenant document storage

### Changed
- None

### Fixed
- None

## [Epic 3] - Hybrid Knowledge Retrieval - 2025-12-29

### Added
- Vector semantic search with pgvector embeddings
- Graph relationship traversal via Neo4j Cypher queries
- Hybrid answer synthesis combining vector and graph results
- Graph-based explainability with source attribution
- Retrieval strategy routing based on query analysis

### Changed
- None

### Fixed
- None

## [Epic 2] - Agentic Query & Reasoning - 2025-12-28

### Added
- Orchestrator agent API with `POST /query`
- Multi-step execution planning with visible plan and thought list
- Retrieval strategy selection (vector/graph/hybrid)
- Persistent trajectory logging with Postgres storage
- Tenant-aware trajectory logging with indexes
- Query response envelope with request metadata
- Alembic scaffolding for database migrations
- Basic backend tests for routing and query responses

### Changed
- None

### Fixed
- None

## [Epic 1] - Foundation & Developer Quick Start - 2025-12-28

### Added
- Backend scaffold with Agno agent-api + FastAPI
- Frontend scaffold with Next.js App Router + CopilotKit dependencies
- Docker Compose dev stack with Postgres/pgvector, Neo4j, and Redis
- Environment configuration via `.env` validation

### Changed
- None

### Fixed
- None
