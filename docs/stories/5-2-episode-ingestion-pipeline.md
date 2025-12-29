# Story 5.2: Episode Ingestion Pipeline

Status: backlog

## Story

As a data engineer,
I want documents to be ingested using Graphiti's episode-based approach,
so that entities and relationships are automatically extracted with temporal tracking.

## Acceptance Criteria

1. Given a parsed document, when `graphiti.add_episode()` is called, then the document content is ingested as an episode with automatic entity extraction.
2. Given custom entity types are configured, when entities are extracted, then they are classified into the appropriate types (TechnicalConcept, CodePattern, etc.).
3. Given an entity already exists, when a new episode mentions it, then the entity is deduplicated and linked with temporal edge tracking.
4. Given the ingestion completes, when the knowledge graph is queried, then new entities and relationships are visible.
5. Given the old ingestion pipeline exists, when a feature flag is set, then documents can be routed to either Graphiti or legacy pipeline.

## Tasks / Subtasks

- [ ] Create episode ingestion service (AC: 1, 2)
  - [ ] Add `backend/src/agentic_rag_backend/indexing/graphiti_ingestion.py`
  - [ ] Implement `ingest_document_as_episode()` function
  - [ ] Parse document sections into episode format
  - [ ] Call `graphiti.add_episode()` with entity types
  - [ ] Handle async ingestion with progress tracking

- [ ] Update index worker for Graphiti (AC: 1, 3, 4)
  - [ ] Modify `backend/src/agentic_rag_backend/indexing/workers/index_worker.py`
  - [ ] Add Graphiti ingestion path alongside legacy
  - [ ] Implement entity deduplication via Graphiti
  - [ ] Track temporal edges for updated entities

- [ ] Add feature flag for pipeline selection (AC: 5)
  - [ ] Add `INGESTION_BACKEND` setting (graphiti | legacy)
  - [ ] Update config.py with feature flag
  - [ ] Route ingestion based on flag value
  - [ ] Support gradual rollout

- [ ] Update IndexerAgent for Graphiti (AC: 1, 2, 3)
  - [ ] Refactor `backend/src/agentic_rag_backend/agents/indexer.py`
  - [ ] Replace custom entity extraction with Graphiti
  - [ ] Simplify graph building logic
  - [ ] Maintain trajectory logging

- [ ] Write integration tests (AC: 1-5)
  - [ ] Add `backend/tests/indexing/test_graphiti_ingestion.py`
  - [ ] Test episode creation from document
  - [ ] Test entity type classification
  - [ ] Test deduplication behavior
  - [ ] Test feature flag routing

## Technical Notes

### Episode Ingestion Pattern

```python
async def ingest_document_as_episode(
    graphiti: Graphiti,
    document: UnifiedDocument,
    tenant_id: str,
) -> EpisodeResult:
    episode = await graphiti.add_episode(
        name=document.title,
        episode_body=document.content,
        source_description=f"Document: {document.source_url}",
        reference_time=datetime.now(timezone.utc),
        entity_types=[TechnicalConcept, CodePattern, APIEndpoint, ConfigurationOption],
        group_id=tenant_id,
    )
    return episode
```

### Feature Flag Configuration

```python
class Settings:
    ingestion_backend: Literal["graphiti", "legacy"] = "graphiti"
```

## Definition of Done

- [ ] Documents ingestible via Graphiti episodes
- [ ] Custom entity types used for classification
- [ ] Entity deduplication working with temporal tracking
- [ ] Feature flag enables pipeline selection
- [ ] Legacy pipeline still functional (fallback)
- [ ] All integration tests passing
- [ ] Code reviewed and merged
