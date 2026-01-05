# Story 20-B3: Implement Global/Local Query Routing

Status: drafted

## Story

As a developer building AI-powered applications,
I want a query classifier that routes queries to either global (community-level) or local (entity-level) retrieval,
so that abstract questions get answered using community summaries while specific questions get precise entity-level answers.

## Context

This story is part of Epic 20: Advanced Retrieval Intelligence, specifically Group B: Graph Intelligence. It implements the global/local query routing pattern from Microsoft GraphRAG, allowing the system to intelligently choose the appropriate retrieval strategy based on query characteristics.

**Competitive Positioning**: This feature directly implements Microsoft GraphRAG's core innovation - the distinction between "global" queries (that need corpus-wide understanding) and "local" queries (that need specific entity details). This completes our Group B Graph Intelligence feature set.

**Why This Matters**:
- **MS GraphRAG Parity:** Microsoft's GraphRAG distinguishes global vs local queries as its key innovation
- **Answer Quality:** Abstract questions benefit from community-level summaries; specific questions need precise entities
- **Efficiency:** Routing reduces unnecessary computation by selecting the right retrieval path
- **Flexibility:** Hybrid mode allows weighted combination when query intent is ambiguous

**Dependencies**:
- Story 20-B1 (Community Detection) - COMPLETED - provides community infrastructure for global queries
- Story 20-B2 (LazyRAG Pattern) - COMPLETED - provides query-time summarization for local queries
- Epic 5 (Graphiti) - Temporal graph storage for entities
- Neo4j - Graph database for entity/community traversal

**Enables**:
- Story 20-C2 (Dual-Level Retrieval) - Can use routing to combine entity and theme retrieval
- Enhanced retrieval orchestrator - Automatic strategy selection based on query type

## Acceptance Criteria

1. Given an abstract query like "What are the main themes?", when routed, then GLOBAL strategy is selected with confidence >= 0.7.
2. Given a specific query like "What is function X?", when routed, then LOCAL strategy is selected with confidence >= 0.7.
3. Given an ambiguous query, when routed, then HYBRID strategy is selected with weighted global/local combination.
4. Given QUERY_ROUTING_ENABLED=false (default), when the system starts, then query routing features are not loaded.
5. Rule-based classification completes in <10ms for pattern matching.
6. LLM classification (when QUERY_ROUTING_USE_LLM=true) completes in <500ms.
7. Routing adds <50ms total latency for rule-based classification.
8. All routing operations enforce tenant isolation via `tenant_id` filtering where applicable.
9. Routing decisions include confidence scores and reasoning for transparency.
10. Given a query with confidence below QUERY_ROUTING_CONFIDENCE_THRESHOLD, when routed, then HYBRID mode is used as fallback.

## Technical Approach

### Module Structure

```
backend/src/agentic_rag_backend/
+-- retrieval/                           # Existing retrieval module
|   +-- query_router.py                  # NEW: QueryRouter class
|   +-- models.py                        # ADD: QueryType enum, RoutingDecision dataclass
```

### Core Components

1. **QueryType Enum** - Query classification: GLOBAL, LOCAL, HYBRID
2. **RoutingDecision Dataclass** - Decision container with query_type, confidence, reasoning, global_weight, local_weight
3. **QueryRouter Class** - Main routing logic with rule-based and LLM-based classification

### Algorithm Flow

```
Query → Rule-Based Classification → (Confidence Check) → LLM Classification (optional) → Routing Decision
         (regex patterns)              (>= threshold?)       (for uncertain queries)
```

**Step 1: Rule-Based Classification**
- Match query against GLOBAL_PATTERNS (summary, themes, overview, etc.)
- Match query against LOCAL_PATTERNS (specific entities, what is X, who is Y)
- Calculate ratio and determine confidence

**Step 2: Confidence Check**
- If confidence >= QUERY_ROUTING_CONFIDENCE_THRESHOLD (default 0.7), return decision
- If confidence < threshold and QUERY_ROUTING_USE_LLM=true, proceed to LLM classification

**Step 3: LLM Classification (Optional)**
- Use configured LLM model to classify ambiguous queries
- Prompt asks LLM to determine if query is global, local, or hybrid
- Parse response and extract confidence

**Step 4: Fallback to Hybrid**
- If still uncertain, default to HYBRID with 50/50 weighting
- Allows retrieval to use both community and entity-level information

### Pattern Definitions

**Global Query Patterns** (indicate need for corpus-wide understanding):
- "what are the main/primary/key themes"
- "summarize", "summary", "overview"
- "all types/kinds/categories"
- "how many total/overall"
- "general understanding"

**Local Query Patterns** (indicate need for specific entity details):
- "what is [entity]"
- "who is/was [person]"
- "where is/does"
- "when did/was"
- "specific", "particular", "exact"
- "this/that [entity]"

### Integration with Retrieval

```python
# Usage in retrieval orchestrator
router = QueryRouter(llm_client=llm, use_llm_classification=True)
decision = await router.route_query(query)

if decision.query_type == QueryType.GLOBAL:
    # Use community-level retrieval (from 20-B1)
    results = await community_detector.search_communities(query, tenant_id)
elif decision.query_type == QueryType.LOCAL:
    # Use entity-level retrieval (from 20-B2 LazyRAG)
    results = await lazy_rag.retrieve_and_summarize(query, tenant_id)
else:  # HYBRID
    # Combine both with weights
    global_results = await community_detector.search_communities(query, tenant_id)
    local_results = await lazy_rag.retrieve_and_summarize(query, tenant_id)
    results = weighted_merge(global_results, local_results, decision.global_weight, decision.local_weight)
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query-router/route` | POST | Route a query and get routing decision |
| `/api/v1/query-router/patterns` | GET | Get current pattern definitions (for debugging) |
| `/api/v1/query-router/status` | GET | Get router configuration status |

### Configuration

```bash
# Epic 20 - Query Routing
QUERY_ROUTING_ENABLED=true|false             # Default: false
QUERY_ROUTING_USE_LLM=true|false             # Use LLM for uncertain queries (default: false)
QUERY_ROUTING_LLM_MODEL=gpt-4o-mini          # Classification model
QUERY_ROUTING_CONFIDENCE_THRESHOLD=0.7       # Below this, use hybrid or LLM
```

### LLM Classification Prompt

```
You are a query classifier. Determine if the following query requires:
- GLOBAL: High-level, abstract understanding across the entire knowledge base (themes, summaries, trends)
- LOCAL: Specific information about particular entities, facts, or details
- HYBRID: Both high-level context and specific details

Query: {query}

Respond with exactly one of: GLOBAL, LOCAL, or HYBRID
Then provide a confidence score from 0.0 to 1.0
Then provide a brief reasoning.

Format:
TYPE: [GLOBAL|LOCAL|HYBRID]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
```

## Tasks / Subtasks

- [ ] Add QueryType enum to retrieval models (`retrieval/models.py`)
- [ ] Add RoutingDecision dataclass to retrieval models (`retrieval/models.py`)
- [ ] Implement QueryRouter class (`retrieval/query_router.py`)
  - [ ] `__init__()` with llm_client and configuration
  - [ ] `route_query()` main entry point
  - [ ] `_rule_based_classification()` regex pattern matching
  - [ ] `_llm_classification()` LLM-based classification for uncertain queries
  - [ ] `_parse_llm_response()` extract type, confidence, reasoning from LLM output
  - [ ] Define GLOBAL_PATTERNS regex list
  - [ ] Define LOCAL_PATTERNS regex list
- [ ] Add configuration variables to settings (`core/config.py`)
  - [ ] QUERY_ROUTING_ENABLED (default: false)
  - [ ] QUERY_ROUTING_USE_LLM (default: false)
  - [ ] QUERY_ROUTING_LLM_MODEL (default: gpt-4o-mini)
  - [ ] QUERY_ROUTING_CONFIDENCE_THRESHOLD (default: 0.7)
- [ ] Add feature flag check (QUERY_ROUTING_ENABLED)
- [ ] Implement API routes (`api/routes/query_router.py`)
  - [ ] POST /api/v1/query-router/route
  - [ ] GET /api/v1/query-router/patterns
  - [ ] GET /api/v1/query-router/status
- [ ] Register routes in main.py (conditional on feature flag)
- [ ] Write unit tests for QueryRouter
  - [ ] Test GLOBAL pattern matching
  - [ ] Test LOCAL pattern matching
  - [ ] Test HYBRID fallback
  - [ ] Test confidence threshold behavior
  - [ ] Test LLM classification with mocked LLM
  - [ ] Test LLM response parsing
- [ ] Write API endpoint tests
- [ ] Update .env.example with query routing configuration variables
- [ ] Add performance logging for routing latency tracking

## Testing Requirements

### Unit Tests
- QueryType enum validation
- RoutingDecision dataclass serialization/deserialization
- Rule-based classification with global patterns
- Rule-based classification with local patterns
- Rule-based classification with mixed patterns (hybrid result)
- Confidence threshold behavior
- LLM classification with mocked LLM client
- LLM response parsing (valid and malformed responses)
- Pattern matching edge cases (empty query, very long query)
- Fallback to hybrid when LLM unavailable

### Integration Tests
- End-to-end routing with LLM classification
- Integration with CommunityDetector (20-B1) for global queries
- Integration with LazyRAGRetriever (20-B2) for local queries
- Weighted hybrid retrieval combining both strategies
- Performance: Rule-based routing <50ms

### Performance Tests
- Rule-based classification latency <10ms
- LLM classification latency <500ms
- Pattern matching performance with many patterns
- Memory usage for compiled regex patterns

### Security Tests
- Input validation for query parameters
- Regex pattern safety (no ReDoS vulnerabilities)
- LLM prompt injection prevention

## Definition of Done

- [ ] All acceptance criteria pass
- [ ] All tasks completed
- [ ] Unit test coverage >= 80% for query_router module
- [ ] API endpoints documented in OpenAPI spec
- [ ] Configuration documented in .env.example
- [ ] Feature flag (QUERY_ROUTING_ENABLED) works correctly
- [ ] Code review approved
- [ ] No regressions in existing retrieval tests
- [ ] Performance target met: <50ms for rule-based routing
- [ ] Integration with 20-B1 (Community Detection) verified
- [ ] Integration with 20-B2 (LazyRAG) verified

## Dev Notes

- Reference tech spec: `_bmad-output/epics/epic-20-tech-spec.md` (Story 20-B3 section)
- Use existing LLM client patterns from other modules
- Follow existing API patterns in `backend/src/agentic_rag_backend/api/routes/`
- LLM provider should use configured provider (QUERY_ROUTING_LLM_MODEL env var)
- Consider using `gpt-4o-mini` for cost-effective classification
- This story completes Group B: Graph Intelligence feature set
- The query router should be integrated into the main retrieval orchestrator

### Key Design Decisions

1. **Why rule-based first?**
   - Pattern matching is extremely fast (<10ms)
   - Many queries have clear intent that patterns can capture
   - Reduces LLM costs by only using LLM for uncertain queries
   - LLM is optional enhancement, not required

2. **Why configurable confidence threshold?**
   - Different use cases have different tolerance for uncertainty
   - High-stakes applications may prefer more LLM classification
   - Cost-sensitive applications may prefer more hybrid fallback

3. **Why weighted hybrid mode?**
   - Ambiguous queries may genuinely need both global and local context
   - Weights allow proportional combination based on pattern match ratios
   - Better answer quality than forcing a binary choice

4. **Why separate from retrieval?**
   - Routing logic is reusable across different retrieval strategies
   - Allows A/B testing of routing decisions
   - Enables logging and analysis of routing patterns

### Performance Considerations

- Compile regex patterns once at initialization, not per query
- Cache LLM classification results for identical queries (optional optimization)
- Rule-based classification should be the fast path for most queries
- LLM classification adds latency but improves accuracy for edge cases

### MS GraphRAG Comparison

| MS GraphRAG | Our Implementation |
|-------------|-------------------|
| Global: Uses community summaries | Uses CommunityDetector (20-B1) |
| Local: Uses entity retrieval | Uses LazyRAGRetriever (20-B2) |
| Static routing | Dynamic with confidence scores |
| Pre-computed | Query-time routing decision |

### References

- `_bmad-output/epics/epic-20-tech-spec.md` (Group B: Graph Intelligence)
- `backend/src/agentic_rag_backend/graph/community.py` (CommunityDetector from 20-B1)
- `backend/src/agentic_rag_backend/retrieval/lazy_rag.py` (LazyRAGRetriever from 20-B2)
- `backend/src/agentic_rag_backend/api/routes/` (API patterns)
- [MS GraphRAG Paper](https://arxiv.org/abs/2404.16130) - Global vs Local Queries
- [MS GraphRAG GitHub](https://github.com/microsoft/graphrag) - For comparison
