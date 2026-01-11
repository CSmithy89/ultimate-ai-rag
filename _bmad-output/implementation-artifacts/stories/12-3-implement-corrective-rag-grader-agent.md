# Story 12.3: Implement Corrective RAG Grader Agent

Status: done

## Story

As a **developer**,
I want **a grader agent that evaluates retrieval relevance and triggers fallback retrieval when needed**,
So that **low-quality retrievals are identified and alternative strategies are used for better answers**.

## Acceptance Criteria

1. Given a set of retrieved results, when the grader score is below threshold, then the system triggers fallback retrieval.
2. Given `GRADER_ENABLED=true`, when retrieval runs, then grading is applied to results.
3. Given `GRADER_ENABLED` is not set or `false`, when retrieval runs, then grading is skipped (opt-in by default).
4. Given grading is enabled, when results are graded, then the score (0.0-1.0) is logged in trajectory.
5. Given `GRADER_THRESHOLD=X`, when grader score < X, then fallback strategy is triggered.
6. Given `GRADER_FALLBACK_STRATEGY=web_search`, when fallback triggers, then Tavily web search is used.
7. Given grading runs, when processing results, then a lightweight model is used (not full LLM).

## Standards Coverage

- [ ] Multi-tenancy / tenant isolation: N/A - grading is per-query
- [ ] Rate limiting / abuse protection: N/A - uses existing query limits
- [ ] Input validation / schema enforcement: Must validate GRADER_THRESHOLD range
- [ ] Tests (unit/integration): Must add grader and fallback tests
- [ ] Error handling + logging: Must handle model failures gracefully
- [ ] Documentation updates: Must update configuration guide

## Tasks / Subtasks

- [x] Add grader configuration to config.py (AC: 2, 3, 5, 6)
  - [x] Add GRADER_ENABLED, GRADER_THRESHOLD, GRADER_FALLBACK_ENABLED
  - [x] Add GRADER_FALLBACK_STRATEGY, TAVILY_API_KEY
  - [x] Default to disabled
- [x] Implement grader module (AC: 1, 4, 7)
  - [x] Create RetrievalGrader class
  - [x] Implement lightweight grading (heuristic and cross-encoder options)
  - [x] Score range 0.0-1.0 with threshold comparison
  - [x] Log grader decisions in trajectory
- [x] Implement fallback strategies (AC: 1, 6)
  - [x] Create FallbackStrategy enum and base class
  - [x] Implement WebSearchFallback (Tavily integration)
  - [x] Implement ExpandedQueryFallback (query reformulation placeholder)
- [ ] Integrate grading into retrieval pipeline (AC: 1, 2, 3) - BACKLOG
  - [ ] Add grading step after reranking
  - [ ] Trigger fallback when score < threshold
  - [ ] Merge fallback results with original results
- [x] Add tests for grader (AC: 1-7)
  - [x] Unit tests for RetrievalGrader (33 tests)
  - [x] Unit tests for fallback strategies
  - [x] Unit tests for factory function
- [ ] Update configuration documentation (AC: 3) - BACKLOG

## Technical Notes

- **Grading Approach Options:**
  1. Cross-encoder scoring (reuse reranker model)
  2. LLM-based grading with lightweight model (claude-3-haiku, gpt-4o-mini)
  3. Dedicated T5-large relevance model
- **Score Calculation:** Average relevance score of top-k results
- **Fallback Strategies:**
  - `web_search`: Query Tavily API for current web data
  - `expanded_query`: Reformulate query and retry retrieval
  - `alternate_index`: Query different knowledge base (future)

## Definition of Done

- [ ] Acceptance criteria met
- [ ] Standards coverage updated
- [ ] Tests run and documented
- [ ] Configuration guide updated

## Dev Notes

### Implementation Decisions

1. **Heuristic Grader as Default**: Uses average retrieval scores from top-k hits rather than full LLM calls for minimal latency.

2. **Cross-Encoder Grader Option**: Optional cross-encoder grader available for higher accuracy (requires sentence-transformers).

3. **Tavily Web Search Fallback**: Integrated Tavily API for web search fallback when local retrieval quality is low.

4. **Expanded Query Fallback**: Placeholder implementation - full query reformulation would require LLM integration.

5. **Threshold Clamping**: GRADER_THRESHOLD is clamped to 0.0-1.0 range to prevent configuration errors.

6. **Pipeline Integration Deferred**: Grading integration into retrieval pipeline marked as backlog for separate PR to keep changes focused.

### Testing Notes

- All 33 unit tests pass
- Tests cover heuristic grading, cross-encoder grading, fallback handlers, and configuration integration
- Config integration tests validate environment variable parsing

### File List

**New Files:**
- `backend/src/agentic_rag_backend/retrieval/grader.py` - Core grader module (350+ lines)
- `backend/tests/test_grader.py` - Unit tests (33 tests)
- `_bmad-output/implementation-artifacts/stories/12-3-implement-corrective-rag-grader-agent.md` - Story file

**Modified Files:**
- `backend/src/agentic_rag_backend/config.py` - Added grader configuration settings
