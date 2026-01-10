# Story 20-E2: Implement Self-Improving Feedback Loop

Status: done

## Story

As a developer building AI-powered applications,
I want a feedback mechanism that uses user corrections and preferences,
so that retrieval quality improves over time based on real usage patterns.

## Context

This story is part of Epic 20: Advanced Retrieval Intelligence, specifically Group E: Advanced Features. It implements a self-improving feedback system similar to enterprise RAG systems, enabling:

- **User Corrections**: Learn from user-provided corrections to improve accuracy
- **Preference Learning**: Track user preferences between result options
- **Query Boost**: Apply learned boost factors to similar future queries
- **Feedback Decay**: Weight recent feedback more heavily

**Competitive Positioning**: Enterprise RAG systems commonly have feedback loops. This feature enables continuous improvement based on real usage.

**Dependencies**:
- PostgreSQL with pgvector for storing feedback and query embeddings
- Embedding provider for query similarity search

## Acceptance Criteria

1. Given user feedback, when recorded, then it is stored and aggregated.
2. Given a user correction, when learned from, then it influences future results.
3. Given similar queries, when retrieved, then they benefit from past feedback (boost factors).
4. Given FEEDBACK_LOOP_ENABLED=false (default), when the system starts, then feedback features are not active.
5. All feedback operations enforce tenant isolation via `tenant_id` filtering.
6. Feedback older than FEEDBACK_DECAY_DAYS contributes less to boost calculations.

## Technical Approach

### Module Structure

```
backend/src/agentic_rag_backend/
+-- feedback/                           # NEW: Feedback loop module
|   +-- __init__.py
|   +-- models.py                       # FeedbackType, UserFeedback, FeedbackStats
|   +-- loop.py                         # FeedbackLoop class
|   +-- adapter.py                      # FeedbackLoopAdapter for feature flag
```

### Core Components

1. **FeedbackType Enum** - Types of feedback:
   - RELEVANCE, ACCURACY, COMPLETENESS, PREFERENCE

2. **UserFeedback Dataclass** - Individual feedback record:
   - query_id, result_id, feedback_type, score, correction, tenant_id, user_id

3. **FeedbackLoop Class** - Main feedback system:
   - `record_feedback()` - Store and aggregate feedback
   - `get_query_boost()` - Calculate boost factors from similar query feedback
   - `_learn_from_correction()` - Learn from user corrections
   - `_find_similar_queries()` - Find similar past queries

4. **FeedbackLoopAdapter Class** - Feature flag wrapper

### Configuration

```bash
FEEDBACK_LOOP_ENABLED=true|false         # Default: false
FEEDBACK_MIN_SAMPLES=10                  # Min feedback before using
FEEDBACK_DECAY_DAYS=90                   # Feedback relevance decay
FEEDBACK_BOOST_MAX=1.5                   # Max boost factor
```

## Tasks / Subtasks

- [x] Create feedback module structure (`backend/src/agentic_rag_backend/feedback/`)
- [x] Implement FeedbackType enum and UserFeedback model (`models.py`)
- [x] Implement FeedbackLoop class (`loop.py`)
  - [x] `record_feedback()` - Store and aggregate feedback
  - [x] `get_query_boost()` - Calculate boost factors
  - [x] `_learn_from_correction()` - Learn from corrections
  - [x] `_find_similar_queries()` - Query similarity search
- [x] Implement FeedbackLoopAdapter with feature flag (`adapter.py`)
- [x] Add configuration variables to settings
  - [x] FEEDBACK_LOOP_ENABLED
  - [x] FEEDBACK_MIN_SAMPLES
  - [x] FEEDBACK_DECAY_DAYS
  - [x] FEEDBACK_BOOST_MAX
  - [x] FEEDBACK_BOOST_MIN
- [x] Create `feedback/__init__.py` with exports
- [x] Write unit tests for FeedbackType and UserFeedback
- [x] Write unit tests for FeedbackLoop
- [x] Write unit tests for FeedbackLoopAdapter

## Testing Requirements

### Unit Tests
- FeedbackType enum values
- UserFeedback model validation
- Feedback recording and aggregation
- Boost factor calculation
- Feedback decay logic
- Feature flag behavior
- Tenant isolation

### Integration Tests
- End-to-end feedback recording and boost retrieval
- Correction learning flow

## Definition of Done

- [x] All acceptance criteria pass
- [x] All tasks completed
- [x] Unit test coverage >= 80% for feedback module
- [x] Feature flag (FEEDBACK_LOOP_ENABLED) works correctly
- [x] Configuration documented
- [x] Code review approved
- [x] No regressions in existing tests

## Dev Notes

- Reference tech spec: `_bmad-output/epics/epic-20-tech-spec.md` (Story 20-E2 section)
- Feedback is stored in PostgreSQL (no new tables needed - use existing patterns)
- For now, in-memory storage for aggregations (can be moved to Redis later)
- Boost factor range: 0.5 to 1.5 (centered at 1.0)
- Query similarity uses embeddings from existing embedding provider

### References

- `_bmad-output/epics/epic-20-tech-spec.md` (Group E: Advanced Features)

---

## Dev Agent Record

### File List

| File | Action | Description |
|------|--------|-------------|
| `backend/src/agentic_rag_backend/feedback/__init__.py` | NEW | Module exports and documentation |
| `backend/src/agentic_rag_backend/feedback/models.py` | NEW | FeedbackType, UserFeedback, FeedbackStats, QueryBoost, FeedbackRecordResult dataclasses |
| `backend/src/agentic_rag_backend/feedback/loop.py` | NEW | FeedbackLoop class with embedding-based similarity search |
| `backend/src/agentic_rag_backend/feedback/adapter.py` | NEW | FeedbackLoopAdapter with feature flag support |
| `backend/src/agentic_rag_backend/config.py` | MODIFIED | Added FEEDBACK_LOOP_ENABLED, FEEDBACK_MIN_SAMPLES, FEEDBACK_DECAY_DAYS, FEEDBACK_BOOST_MAX, FEEDBACK_BOOST_MIN settings |
| `backend/tests/feedback/__init__.py` | NEW | Test module init |
| `backend/tests/feedback/test_feedback.py` | NEW | 58 unit tests for feedback module |

### Change Log

| Date | Change | Details |
|------|--------|---------|
| 2026-01-06 | Initial implementation | Created feedback module with FeedbackLoop, FeedbackLoopAdapter, and models |
| 2026-01-06 | Code review fixes | Fixed copy-paste error in docstring, added FeedbackType import, aligned config validation with model constraints |
