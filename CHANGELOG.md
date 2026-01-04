# Changelog

## [Epic 15] - Codebase Intelligence - 2026-01-04

### Added
- AST-based hallucination detection with symbol, path, import, and API validation
- Codebase RAG indexing pipeline with pgvector embeddings and Neo4j relationships
- Codebase API endpoints for validation, indexing, search, and cache management (`/api/v1/codebase/...`)
- CODEBASE_* configuration flags for detection, indexing, and chunking behavior
- Dependency-aware import validation via repo manifests (pyproject/requirements/package.json)

### Changed
- None

### Fixed
- None

## [Epic 11] - Code Cleanup & Migration - 2025-12-31

### Added
- Graphiti migration script (`backend/scripts/migrate_to_graphiti.py`) with optional backups and validation checks
- Redis-backed HITL validation checkpoints with query endpoints
- Workspace persistence for save/share/bookmark with load and share retrieval endpoints
- Neo4j connection pool configuration via environment settings
- A2A session persistence to Redis with on-demand recovery
- Parser-based HTML-to-Markdown conversion for crawler content
- Embedding token usage tracking for cost monitoring

### Changed
- Graphiti ingestion and retrieval now run without legacy feature flags
- Legacy ingestion modules removed in favor of Graphiti-only pipeline

### Fixed
- Datetime deprecation cleanup across backend utilities

## [Epic 10] - Testing Infrastructure - 2025-12-31

### Added
- Integration test framework with real Postgres, Neo4j, and Redis fixtures gated by `INTEGRATION_TESTS=1`
- Hybrid retrieval integration coverage (vector + graph) with tenant isolation checks
- PDF fixture set plus ingestion pipeline integration tests for URL ingest, parsing, entity extraction, and deduplication
- Optional Graphiti end-to-end integration test gated by `GRAPHITI_E2E=1`
- Protocol integration tests for MCP tool registry, A2A flows, and AG-UI streaming
- Benchmark suite for ingestion throughput and query latency with JSONL reporting
- Testing documentation for integration tests, benchmarks, and skipped test inventory
- CI updates for backend integration and benchmark jobs

### Changed
- None

### Fixed
- None

## [Epic 9] - Process & Quality Foundation - 2025-12-31

### Added
- Retrospective action item tracking registry and template with overdue check script
- Story template standards coverage section and authoring guide
- Pre-review PR checklist with protocol compliance prompts
- Protocol compliance checklist for API routes
- Story status validation script with CI enforcement
- Dev notes, test outcomes, and challenges requirements for story completion

### Changed
- Backend lint command now uses a repo-local UV cache for consistent runs

### Fixed
- None

## [Epic 8] - Operations & Observability - 2025-12-31

### Added
- Ops cost monitoring with `llm_usage_events` tracking and alert thresholds
- Intelligent model routing with configurable thresholds and model mappings
- Trajectory debugging endpoints and UI timeline viewer
- AES-256-GCM encrypted trajectory storage with controlled decryption

### Changed
- Orchestrator now selects models per request based on routing complexity

### Fixed
- None

## [Epic 7] - Protocol Integration & Extensibility - 2025-12-30

### Added
- MCP tool discovery and invocation endpoints (`/api/v1/mcp/tools`, `/api/v1/mcp/call`)
- A2A collaboration APIs for session creation and messaging (`/api/v1/a2a`)
- Universal AG-UI endpoint for protocol-agnostic clients (`/api/v1/ag-ui`)
- Python SDK client for MCP and A2A integrations
- MCP tool registry with `knowledge.query` and `knowledge.graph_stats` tools

### Changed
- None

### Fixed
- None

## [Epic 6] - Interactive Copilot Experience - 2025-12-30

### Added
- CopilotKit React integration with `@copilotkit/react-core` and `@copilotkit/react-ui`
- `CopilotProvider` wrapper component for application-wide CopilotKit context
- Next.js API route `/api/copilotkit` for CopilotKit runtime
- `ChatSidebar` component with polished chat interface using shadcn/ui styling
- `ThoughtTraceStepper` component showing agent progress with expandable details
- Generative UI components:
  - `SourceCard` for citation display with confidence indicators
  - `AnswerPanel` for formatted markdown responses with source references
  - `GraphPreview` for entity relationship visualization using React Flow
- `GenerativeUIRenderer` for dynamic component rendering via AG-UI protocol
- Human-in-the-Loop source validation:
  - `SourceValidationDialog` modal for reviewing/approving sources
  - `SourceValidationPanel` inline panel alternative
  - `SourceValidationCard` for individual source approval UI
  - `useSourceValidation` hook with auto-approve/reject thresholds
- Frontend actions system:
  - `ActionButtons` component (Save, Export, Share, Bookmark, Follow-up)
  - `ActionPanel` slide-out panel showing action history
  - `useCopilotActions` hook with CopilotKit action registration
  - Export dropdown supporting Markdown, PDF, JSON formats
  - Toast notification system with `useToast` hook
- Backend workspace API endpoints:
  - `POST /api/v1/workspace/save` - Save content to workspace
  - `POST /api/v1/workspace/export` - Export as markdown/PDF/JSON
  - `POST /api/v1/workspace/share` - Generate shareable link
  - `POST /api/v1/workspace/bookmark` - Bookmark content
  - `GET /api/v1/workspace/bookmarks` - List bookmarks
- TypeScript types and Zod schemas for all copilot data structures
- Comprehensive test suite: 315+ frontend tests, 21 backend workspace tests

### Changed
- `frontend/app/layout.tsx` now includes `CopilotProvider` and `Toaster`
- Design system follows "Professional Forge" direction (Indigo-600, Slate, Emerald-500)

### Fixed
- None

## [Epic 5] - Graphiti Temporal Knowledge Graph Integration - 2025-12-29

### Added
- Graphiti integration with `graphiti-core>=0.5.0` for temporal knowledge graphs
- Custom entity types: `TechnicalConcept`, `CodePattern`, `APIEndpoint`, `ConfigurationOption`
- Edge type mappings for semantic relationship classification
- `GraphitiClient` wrapper with connection lifecycle management
- Episode-based document ingestion via `ingest_document_as_episode()`
- Hybrid retrieval with `graphiti_search()`
- Temporal query capabilities:
  - Point-in-time search with `temporal_search()`
  - Knowledge change tracking with `get_knowledge_changes()`
- API endpoints for temporal queries (`/knowledge/temporal/search`, `/knowledge/temporal/changes`)
- Feature flags for backend selection (removed in Epic 11)
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
