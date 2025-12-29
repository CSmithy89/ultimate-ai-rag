# Story 5.1: Graphiti Installation and Custom Entity Types

Status: ready-for-dev

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

- [ ] Add graphiti-core dependency to pyproject.toml (AC: 1)
  - [ ] Add `graphiti-core>=0.5.0` to dependencies
  - [ ] Run `uv lock` to update lock file
  - [ ] Verify installation with `uv sync`

- [ ] Create custom entity type definitions (AC: 2)
  - [ ] Add `backend/src/agentic_rag_backend/models/entity_types.py`
  - [ ] Define TechnicalConcept EntityModel with domain and complexity fields
  - [ ] Define CodePattern EntityModel with language and pattern_type fields
  - [ ] Define APIEndpoint EntityModel with method and path fields
  - [ ] Define ConfigurationOption EntityModel with config_type and default_value fields
  - [ ] Add unit tests for entity type validation

- [ ] Create Graphiti client wrapper (AC: 3, 4)
  - [ ] Add `backend/src/agentic_rag_backend/db/graphiti.py`
  - [ ] Implement GraphitiClient class with Neo4j connection
  - [ ] Add connect() and disconnect() async methods
  - [ ] Add build_indices() method for index creation
  - [ ] Configure entity types and edge type mappings

- [ ] Integrate Graphiti into application lifespan (AC: 3)
  - [ ] Update `backend/src/agentic_rag_backend/main.py` lifespan
  - [ ] Initialize GraphitiClient in startup
  - [ ] Store in app.state.graphiti
  - [ ] Disconnect in shutdown

- [ ] Configure edge type mappings (AC: 5)
  - [ ] Define edge_type_map for entity pair relationships
  - [ ] Add mapping for TechnicalConcept ↔ TechnicalConcept
  - [ ] Add mapping for TechnicalConcept ↔ CodePattern
  - [ ] Add mapping for CodePattern ↔ CodePattern
  - [ ] Add mapping for APIEndpoint ↔ TechnicalConcept
  - [ ] Add mapping for ConfigurationOption ↔ TechnicalConcept

- [ ] Write unit tests (AC: 1-5)
  - [ ] Add `backend/tests/db/test_graphiti.py`
  - [ ] Test entity type definitions
  - [ ] Test client initialization (mocked)
  - [ ] Test edge type mapping configuration

## Technical Notes

### Graphiti Installation

```bash
# Using uv (our package manager)
uv add graphiti-core
```

### Entity Type Definition Pattern

```python
from graphiti_core.models import EntityModel
from pydantic import Field

class TechnicalConcept(EntityModel):
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

- [ ] graphiti-core installed and importable
- [ ] All 4 custom entity types defined and tested
- [ ] GraphitiClient wrapper created with connect/disconnect
- [ ] Application lifespan initializes Graphiti client
- [ ] Edge type mappings configured
- [ ] All unit tests passing
- [ ] Code reviewed and merged
