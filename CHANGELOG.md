# Changelog

## [Epic 4] - Knowledge Ingestion Pipeline - 2025-12-29

### Added
- URL documentation crawling with Crawl4AI
- PDF parsing via Docling with structured extraction
- Agentic entity extraction and graph construction pipeline
- Knowledge graph visualization UI

### Changed
- None

### Fixed
- None

## [Epic 3] - Hybrid Knowledge Retrieval - 2025-12-29

### Added
- Vector semantic search over pgvector embeddings
- Neo4j relationship traversal with tenant-scoped queries
- Hybrid answer synthesis combining vector + graph evidence
- Query response evidence for graph explainability (nodes, edges, paths)

### Changed
- Orchestrator now builds prompts with retrieval evidence
- Query responses include optional evidence payloads

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
