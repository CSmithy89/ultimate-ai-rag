# Story 5.6: Test Suite Adaptation

Status: backlog

## Story

As a developer,
I want the test suite updated for Graphiti integration,
so that all functionality remains well-tested and regressions are caught.

## Acceptance Criteria

1. Given legacy entity extraction tests exist, when they are updated, then they test Graphiti entity type classification instead.
2. Given legacy graph builder tests exist, when they are updated, then they test Graphiti episode ingestion instead.
3. Given new temporal query features exist, when integration tests run, then point-in-time and change queries are tested.
4. Given the test suite runs, when coverage is measured, then it is >= 80% for Graphiti-related modules.
5. Given all tests pass, when CI runs, then the pipeline succeeds in under 5 minutes.

## Standards Coverage

- [ ] Multi-tenancy / tenant isolation: N/A - test suite changes
- [ ] Rate limiting / abuse protection: N/A - test suite changes
- [ ] Input validation / schema enforcement: N/A - test suite changes
- [ ] Tests (unit/integration): Planned - expand Graphiti-related test coverage
- [ ] Error handling + logging: Planned - verify error cases in tests
- [ ] Documentation updates: Planned - update README if coverage reporting changes

## Tasks / Subtasks

- [ ] Update entity extraction tests (AC: 1)
  - [ ] Refactor `backend/tests/indexing/test_entity_extractor.py`
  - [ ] Test custom entity type definitions
  - [ ] Test entity classification accuracy
  - [ ] Mock Graphiti SDK responses

- [ ] Update graph builder tests (AC: 2)
  - [ ] Refactor `backend/tests/indexing/test_graph_builder.py`
  - [ ] Convert to Graphiti episode ingestion tests
  - [ ] Test edge type mapping
  - [ ] Test deduplication behavior

- [ ] Add Graphiti client tests (AC: 1, 2)
  - [ ] Add `backend/tests/db/test_graphiti.py`
  - [ ] Test client initialization
  - [ ] Test connection management
  - [ ] Test error handling

- [ ] Add temporal query tests (AC: 3)
  - [ ] Add `backend/tests/api/test_temporal_queries.py`
  - [ ] Test point-in-time query filtering
  - [ ] Test change query date ranges
  - [ ] Test entity history retrieval

- [ ] Add hybrid retrieval tests (AC: 1, 3)
  - [ ] Add `backend/tests/retrieval/test_graphiti_retrieval.py`
  - [ ] Test search configuration
  - [ ] Test result ranking
  - [ ] Test latency requirements

- [ ] Update integration tests (AC: 3, 4)
  - [ ] Update end-to-end ingestion tests
  - [ ] Update retrieval integration tests
  - [ ] Add migration validation tests
  - [ ] Ensure test isolation

- [ ] Clean up deprecated tests (AC: 1, 2)
  - [ ] Remove tests for deleted modules
  - [ ] Remove unused fixtures
  - [ ] Update conftest.py for Graphiti mocks

- [ ] Verify coverage and CI (AC: 4, 5)
  - [ ] Run coverage report
  - [ ] Ensure >= 80% coverage
  - [ ] Verify CI pipeline timing
  - [ ] Add coverage badge to README

## Technical Notes

### Mocking Graphiti SDK

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_graphiti():
    graphiti = MagicMock()
    graphiti.add_episode = AsyncMock(return_value=mock_episode_result())
    graphiti.search = AsyncMock(return_value=mock_search_results())
    graphiti.build_indices = AsyncMock()
    return graphiti
```

### Test Structure Changes

```
backend/tests/
├── db/
│   ├── test_neo4j.py (simplified)
│   └── test_graphiti.py (NEW)
├── indexing/
│   ├── test_entity_extractor.py (REFACTORED → entity types)
│   ├── test_graph_builder.py (REFACTORED → episode ingestion)
│   └── test_graphiti_ingestion.py (NEW)
├── retrieval/
│   └── test_graphiti_retrieval.py (NEW)
└── api/
    └── test_temporal_queries.py (NEW)
```

### Coverage Requirements

| Module | Target |
|--------|--------|
| db/graphiti.py | 90% |
| models/entity_types.py | 100% |
| indexing/graphiti_ingestion.py | 85% |
| retrieval/graphiti_retrieval.py | 85% |
| api/routes/knowledge.py (temporal) | 80% |

## Definition of Done

- [ ] All legacy tests updated for Graphiti
- [ ] New tests for temporal queries added
- [ ] New tests for hybrid retrieval added
- [ ] Test coverage >= 80% for Graphiti modules
- [ ] CI pipeline passes in < 5 minutes
- [ ] No test flakiness
- [ ] All tests passing
- [ ] Code reviewed and merged
