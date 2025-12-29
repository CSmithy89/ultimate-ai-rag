# Story 5.1: Graphiti Installation and Custom Entity Types

Status: done

## Story

As a developer,
I want to install Graphiti and configure custom entity types,
so that I can leverage temporal knowledge graph capabilities for our RAG system.

## Acceptance Criteria

1. Given the backend project, when `uv sync` is run, then graphiti-core is installed with all dependencies.
2. Given custom entity types are defined, when Pydantic models are validated, then TechnicalConcept, CodePattern, APIEndpoint, and ConfigurationOption types are recognized.
3. Given the application starts, when the lifespan manager runs, then a Graphiti client is initialized and connected to Neo4j.
4. Given the Graphiti client is connected, when `graphiti.build_indices()` is called, then required Neo4j indexes are created.
5. Given edge type mappings are configured, when entities are extracted, then appropriate relationship types are applied based on entity pair types.

## Tasks / Subtasks

- [x] Add graphiti-core dependency to pyproject.toml (AC: 1)
  - [x] Add `graphiti-core>=0.5.0` to dependencies
  - [x] Run `uv lock` to update lock file
  - [x] Verify installation with `uv sync`

- [x] Create custom entity type definitions (AC: 2)
  - [x] Add `backend/src/agentic_rag_backend/models/entity_types.py`
  - [x] Define TechnicalConcept EntityModel with domain and complexity fields
  - [x] Define CodePattern EntityModel with language and pattern_type fields
  - [x] Define APIEndpoint EntityModel with method and path fields
  - [x] Define ConfigurationOption EntityModel with config_type and default_value fields
  - [x] Add unit tests for entity type validation

- [x] Create Graphiti client wrapper (AC: 3, 4)
  - [x] Add `backend/src/agentic_rag_backend/db/graphiti.py`
  - [x] Implement GraphitiClient class with Neo4j connection
  - [x] Add connect() and disconnect() async methods
  - [x] Add build_indices() method for index creation
  - [x] Configure entity types and edge type mappings

- [x] Integrate Graphiti into application lifespan (AC: 3)
  - [x] Update `backend/src/agentic_rag_backend/main.py` lifespan
  - [x] Initialize GraphitiClient in startup
  - [x] Store in app.state.graphiti
  - [x] Disconnect in shutdown

- [x] Configure edge type mappings (AC: 5)
  - [x] Define edge_type_map for entity pair relationships
  - [x] Add mapping for TechnicalConcept ↔ TechnicalConcept
  - [x] Add mapping for TechnicalConcept ↔ CodePattern
  - [x] Add mapping for CodePattern ↔ CodePattern
  - [x] Add mapping for APIEndpoint ↔ TechnicalConcept
  - [x] Add mapping for ConfigurationOption ↔ TechnicalConcept

- [x] Write unit tests (AC: 1-5)
  - [x] Add `backend/tests/db/test_graphiti.py`
  - [x] Test entity type definitions
  - [x] Test client initialization (mocked)
  - [x] Test edge type mapping configuration

## Technical Notes

### Graphiti Installation

```bash
# Using uv (our package manager)
uv add graphiti-core
```

### Entity Type Definition Pattern

```python
from graphiti_core.nodes import EntityNode
from pydantic import Field

class TechnicalConcept(EntityNode):
    """Technical concept from documentation."""
    domain: str = Field(description="Technical domain")
    complexity: str = Field(description="Complexity level")
```

### Client Initialization

```python
from graphiti_core import Graphiti

graphiti = Graphiti(
    uri=settings.neo4j_uri,
    user=settings.neo4j_user,
    password=settings.neo4j_password,
)
await graphiti.build_indices()
```

## Definition of Done

- [x] graphiti-core installed and importable
- [x] All 4 custom entity types defined and tested
- [x] GraphitiClient wrapper created with connect/disconnect
- [x] Application lifespan initializes Graphiti client
- [x] Edge type mappings configured
- [x] All unit tests passing (23 tests)
- [x] Code reviewed and merged

## Implementation Summary

### Files Created
- `backend/src/agentic_rag_backend/models/entity_types.py` - Custom entity types (TechnicalConcept, CodePattern, APIEndpoint, ConfigurationOption) with edge type mappings
- `backend/src/agentic_rag_backend/db/graphiti.py` - GraphitiClient wrapper with connection lifecycle management
- `backend/tests/db/test_graphiti.py` - 23 unit tests covering entity types, edge mappings, and client operations

### Files Modified
- `backend/pyproject.toml` - Added graphiti-core>=0.5.0 dependency
- `backend/src/agentic_rag_backend/db/__init__.py` - Exported GraphitiClient
- `backend/src/agentic_rag_backend/config.py` - Added graphiti_embedding_model and graphiti_llm_model settings
- `backend/src/agentic_rag_backend/main.py` - Integrated Graphiti into lifespan with SKIP_GRAPHITI env var support

### Environment Variables Added
- `GRAPHITI_EMBEDDING_MODEL` - Embedding model for Graphiti (default: text-embedding-3-small)
- `GRAPHITI_LLM_MODEL` - LLM model for entity extraction (default: gpt-4o-mini)
- `SKIP_GRAPHITI` - Set to "1" to skip Graphiti initialization (for testing)

### Test Results
- 23 Graphiti-specific tests: All passing
- Full test suite: 202 passed, 1 skipped (pre-existing DB-dependent test)
