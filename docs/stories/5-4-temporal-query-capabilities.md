# Story 5.4: Temporal Query Capabilities

Status: backlog

## Story

As a user,
I want to query the knowledge graph at specific points in time,
so that I can understand how knowledge has evolved and what was true at particular dates.

## Acceptance Criteria

1. Given a temporal query with `as_of_date`, when the knowledge graph is searched, then only facts valid at that date are returned.
2. Given a date range, when the user requests changes, then all entity/relationship modifications within that period are returned.
3. Given an entity has been updated multiple times, when its history is requested, then all temporal versions are available.
4. Given a contradiction was resolved, when the resolution is queried, then both the old and new facts are accessible with timestamps.
5. Given the temporal API is called, when no date is specified, then the current (latest) state is returned.

## Tasks / Subtasks

- [ ] Create temporal query API endpoints (AC: 1, 2, 5)
  - [ ] Add `POST /api/v1/knowledge/temporal-query` endpoint
  - [ ] Add `GET /api/v1/knowledge/changes` endpoint
  - [ ] Implement request/response models for temporal queries
  - [ ] Handle default (current) state when no date provided

- [ ] Implement point-in-time queries (AC: 1, 5)
  - [ ] Add `query_at_point_in_time()` function in graphiti_retrieval.py
  - [ ] Use Graphiti's bi-temporal model for filtering
  - [ ] Return only facts valid at specified datetime
  - [ ] Include temporal metadata in response

- [ ] Implement change tracking queries (AC: 2, 4)
  - [ ] Add `get_changes_in_range()` function
  - [ ] Query edge invalidation timestamps
  - [ ] Return created, updated, and invalidated entities
  - [ ] Include contradiction resolution details

- [ ] Implement entity history queries (AC: 3, 4)
  - [ ] Add `GET /api/v1/knowledge/entity/{id}/history` endpoint
  - [ ] Retrieve all temporal versions of an entity
  - [ ] Include relationship changes over time
  - [ ] Show contradiction resolutions

- [ ] Update knowledge route handlers (AC: 1-5)
  - [ ] Modify `backend/src/agentic_rag_backend/api/routes/knowledge.py`
  - [ ] Add temporal query handlers
  - [ ] Integrate with Graphiti temporal APIs
  - [ ] Add rate limiting for temporal queries

- [ ] Write tests for temporal capabilities (AC: 1-5)
  - [ ] Add `backend/tests/api/test_temporal_queries.py`
  - [ ] Test point-in-time filtering
  - [ ] Test change range queries
  - [ ] Test entity history retrieval
  - [ ] Test contradiction resolution visibility

## Technical Notes

### Temporal Query Request Model

```python
class TemporalQueryRequest(BaseModel):
    tenant_id: UUID
    query: str
    as_of_date: datetime | None = None  # None = current state

class TemporalQueryResponse(BaseModel):
    data: list[SearchResult]
    temporal_context: TemporalContext

class TemporalContext(BaseModel):
    query_time: datetime
    as_of_date: datetime
    facts_valid_at: datetime
```

### Change Query Request

```python
class ChangeQueryRequest(BaseModel):
    tenant_id: UUID
    start_date: datetime
    end_date: datetime
    entity_type: str | None = None

class ChangeQueryResponse(BaseModel):
    changes: list[EntityChange]
    summary: ChangeSummary
```

### Graphiti Bi-Temporal Model

Graphiti tracks two timestamps:
- `valid_at`: When the fact was true in the real world
- `created_at`: When the fact was ingested into the system

This enables queries like "What did we know about X on date Y?"

## Definition of Done

- [ ] Point-in-time queries return temporally filtered results
- [ ] Change queries return entity modifications in date range
- [ ] Entity history shows all temporal versions
- [ ] Contradiction resolutions are accessible
- [ ] Default queries return current state
- [ ] All tests passing
- [ ] Code reviewed and merged
