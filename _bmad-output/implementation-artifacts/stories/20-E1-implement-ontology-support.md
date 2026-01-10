# Story 20-E1: Implement Ontology Support

Status: done

## Story

As a developer building domain-specific AI applications,
I want OWL ontology support for domain-specific entity types and relationships,
so that entities extracted during ingestion can be typed according to domain ontologies.

## Context

This story is part of Epic 20: Advanced Retrieval Intelligence, specifically Group E: Advanced Features. It implements ontology support similar to Cognee's approach, enabling:

- **Domain-Specific Typing**: Entities can be classified using ontology classes
- **Relationship Semantics**: Properties define valid relationships between entity types
- **Enterprise Knowledge**: Load existing enterprise ontologies for consistent entity typing
- **Graphiti Integration**: Register ontology types for use during graph ingestion

**Competitive Positioning**: This feature directly competes with Cognee's ontology support.

**Dependencies**:
- Epic 5 (Graphiti) - Graph storage for ontology-typed entities
- owlready2 library - OWL ontology parsing

## Acceptance Criteria

1. Given an OWL ontology file, when loaded via OntologyLoader, then classes and properties are extracted.
2. Given a loaded ontology, when `get_entity_type()` is called with an entity name, then matching ontology class is returned.
3. Given ONTOLOGY_SUPPORT_ENABLED=false (default), when the system starts, then ontology features are not loaded.
4. Given ONTOLOGY_PATH is set and `load_ontology()` is called without a path, then the default path is used.
5. All ontology operations enforce tenant isolation via `tenant_id` filtering.
6. Ontology loading completes in <5 seconds for typical domain ontologies (<1000 classes).

## Technical Approach

### Module Structure

```
backend/src/agentic_rag_backend/
+-- ontology/                           # NEW: Ontology support module
|   +-- __init__.py
|   +-- loader.py                       # OntologyLoader class
|   +-- models.py                       # OntologyClass, OntologyProperty dataclasses
|   +-- adapter.py                      # OntologyAdapter for feature flag
```

### Core Components

1. **OntologyClass Dataclass** - Class from an ontology:
   - uri, name, description
   - parent_uris (inheritance hierarchy)
   - properties (list of property names)

2. **OntologyProperty Dataclass** - Property from an ontology:
   - uri, name, domain, range
   - description

3. **OntologyLoader Class** - Load and manage OWL ontologies:
   - `load_ontology()` - Parse OWL file and extract classes/properties
   - `get_entity_type()` - Find matching ontology class for an entity
   - `list_classes()` - List all loaded classes
   - `list_properties()` - List all loaded properties

4. **OntologyAdapter Class** - Feature flag wrapper

### Configuration

```bash
ONTOLOGY_SUPPORT_ENABLED=true|false          # Default: false
ONTOLOGY_PATH=/path/to/domain.owl            # Default ontology file
ONTOLOGY_AUTO_TYPE=true|false                # Auto-type entities during ingestion
```

## Tasks / Subtasks

- [x] Create ontology module structure (`backend/src/agentic_rag_backend/ontology/`)
- [x] Implement OntologyClass and OntologyProperty models (`models.py`)
- [x] Implement OntologyLoader class (`loader.py`)
  - [x] `load_ontology()` - Load and parse OWL file
  - [x] `get_entity_type()` - Match entity to ontology class
  - [x] `list_classes()` - List loaded classes
  - [x] `list_properties()` - List loaded properties
- [x] Implement OntologyAdapter with feature flag (`adapter.py`)
- [x] Add configuration variables to settings
  - [x] ONTOLOGY_SUPPORT_ENABLED
  - [x] ONTOLOGY_PATH
  - [x] ONTOLOGY_AUTO_TYPE
- [x] Create `ontology/__init__.py` with exports
- [x] Write unit tests for OntologyClass and OntologyProperty
- [x] Write unit tests for OntologyLoader
- [x] Write unit tests for OntologyAdapter
- [x] Test with sample OWL ontology
- [x] Add file size validation (MAX_ONTOLOGY_FILE_SIZE_BYTES = 50MB)
- [x] Add performance test for AC6 (<5 seconds)

## Testing Requirements

### Unit Tests
- OntologyClass model validation
- OntologyProperty model validation
- Ontology loading from valid OWL file
- Entity type matching logic
- Invalid/malformed ontology handling
- Feature flag behavior (enabled/disabled)
- Tenant isolation in loaded ontologies
- File size validation

### Integration Tests
- End-to-end ontology loading and entity typing
- Multiple ontologies for different tenants
- Performance test (<5 seconds for 100+ classes)

## Definition of Done

- [x] All acceptance criteria pass
- [x] All tasks completed
- [x] Unit test coverage >= 80% for ontology module
- [x] Feature flag (ONTOLOGY_SUPPORT_ENABLED) works correctly
- [x] Configuration documented
- [x] Code review approved
- [x] No regressions in existing tests

## Dev Notes

- Reference tech spec: `_bmad-output/epics/epic-20-tech-spec.md` (Story 20-E1 section)
- Use owlready2 library for OWL parsing
- owlready2 is optional dependency - handle ImportError gracefully
- Consider caching loaded ontologies per tenant
- Entity matching uses simple name substring matching (can be enhanced with NLP later)
- This story focuses on ontology loading; integration with Graphiti entity extraction is future work

### References

- `_bmad-output/epics/epic-20-tech-spec.md` (Group E: Advanced Features)
- owlready2 documentation: https://owlready2.readthedocs.io/

---

## Dev Agent Record

### File List

| File | Action | Description |
|------|--------|-------------|
| `backend/src/agentic_rag_backend/ontology/__init__.py` | NEW | Module exports and documentation |
| `backend/src/agentic_rag_backend/ontology/models.py` | NEW | OntologyClass, OntologyProperty, LoadedOntology, OntologyLoadResult dataclasses |
| `backend/src/agentic_rag_backend/ontology/loader.py` | NEW | OntologyLoader class with owlready2 integration |
| `backend/src/agentic_rag_backend/ontology/adapter.py` | NEW | OntologyAdapter with feature flag support |
| `backend/src/agentic_rag_backend/config.py` | MODIFIED | Added ONTOLOGY_SUPPORT_ENABLED, ONTOLOGY_PATH, ONTOLOGY_AUTO_TYPE settings |
| `backend/tests/ontology/__init__.py` | NEW | Test module init |
| `backend/tests/ontology/test_ontology.py` | NEW | 55 unit tests for ontology module |

### Change Log

| Date | Change | Details |
|------|--------|---------|
| 2026-01-06 | Initial implementation | Created ontology module with OntologyLoader, OntologyAdapter, and models |
| 2026-01-06 | Code review fixes | Added file size validation, improved exception handling, added performance test |
