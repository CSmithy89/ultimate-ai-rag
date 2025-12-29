# Story 5.5: Legacy Code Removal and Migration

Status: backlog

## Story

As a developer,
I want to migrate existing knowledge graph data to Graphiti format and remove legacy code,
so that the codebase is simplified and all knowledge benefits from temporal tracking.

## Acceptance Criteria

1. Given existing entities in Neo4j, when the migration script runs, then all entities are recreated as Graphiti-managed nodes with temporal metadata.
2. Given existing relationships in Neo4j, when they are migrated, then they become temporal edges with creation timestamps.
3. Given the migration completes, when entity counts are compared, then 100% of entities are migrated with no data loss.
4. Given the Graphiti backend is validated, when the feature flag is removed, then all ingestion uses Graphiti exclusively.
5. Given legacy modules are identified, when they are removed, then at least 1,000 lines of code are deleted.

## Tasks / Subtasks

- [ ] Create migration script for existing data (AC: 1, 2, 3)
  - [ ] Add `backend/scripts/migrate_to_graphiti.py`
  - [ ] Read existing entities from Neo4j
  - [ ] Map to custom entity types (TechnicalConcept, etc.)
  - [ ] Ingest as Graphiti episodes with historical timestamps
  - [ ] Migrate relationships as temporal edges
  - [ ] Add progress tracking and logging

- [ ] Implement migration validation (AC: 3)
  - [ ] Count entities before and after migration
  - [ ] Verify relationship preservation
  - [ ] Spot-check entity properties
  - [ ] Generate migration report

- [ ] Remove feature flag and legacy routing (AC: 4)
  - [ ] Remove `INGESTION_BACKEND` setting
  - [ ] Update index_worker to use Graphiti only
  - [ ] Remove legacy pipeline routing code
  - [ ] Update configuration documentation

- [ ] Delete deprecated modules (AC: 5)
  - [ ] Delete `backend/src/agentic_rag_backend/indexing/entity_extractor.py`
  - [ ] Delete `backend/src/agentic_rag_backend/indexing/graph_builder.py`
  - [ ] Delete `backend/src/agentic_rag_backend/indexing/embeddings.py`
  - [ ] Remove unused imports and dependencies
  - [ ] Update __init__.py exports

- [ ] Simplify remaining modules (AC: 5)
  - [ ] Refactor `db/neo4j.py` to thin wrapper
  - [ ] Simplify `agents/indexer.py` to Graphiti delegation
  - [ ] Remove dead code from index_worker.py
  - [ ] Clean up unused test fixtures

- [ ] Update documentation (AC: 4, 5)
  - [ ] Update architecture.md with new stack
  - [ ] Update API documentation
  - [ ] Remove references to deleted modules
  - [ ] Add Graphiti configuration guide

- [ ] Write migration tests (AC: 1-5)
  - [ ] Add `backend/tests/scripts/test_migration.py`
  - [ ] Test entity migration accuracy
  - [ ] Test relationship preservation
  - [ ] Test rollback capability

## Technical Notes

### Migration Script Pattern

```python
async def migrate_to_graphiti():
    # 1. Export existing entities
    existing_entities = await neo4j.get_all_entities(tenant_id)

    # 2. Map to entity types
    typed_entities = classify_entities(existing_entities)

    # 3. Create migration episodes
    for entity in typed_entities:
        await graphiti.add_episode(
            name=entity.name,
            episode_body=entity.description,
            reference_time=entity.created_at,  # Preserve original time
            entity_types=[type(entity)],
        )

    # 4. Validate migration
    new_count = await graphiti.count_entities()
    assert new_count >= len(existing_entities)
```

### Modules to Delete

| Module | Lines | Reason |
|--------|-------|--------|
| `indexing/entity_extractor.py` | 352 | Replaced by Graphiti |
| `indexing/graph_builder.py` | 295 | Replaced by Graphiti |
| `indexing/embeddings.py` | 228 | Replaced by Graphiti |
| Partial `agents/indexer.py` | ~200 | Simplified |
| **Total** | **~1,075** | |

## Definition of Done

- [ ] All existing entities migrated to Graphiti format
- [ ] All relationships converted to temporal edges
- [ ] Migration validation confirms 100% data preservation
- [ ] Feature flag removed, Graphiti is exclusive backend
- [ ] >= 1,000 lines of legacy code deleted
- [ ] Documentation updated
- [ ] All tests passing
- [ ] Code reviewed and merged
