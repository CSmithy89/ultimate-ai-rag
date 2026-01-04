# Epic 19 Tech Spec: Quality Foundation & Tech Debt Resolution

**Date:** 2026-01-04
**Updated:** 2026-01-05 (Split from original Epic 19 - competitive features moved to Epic 20)
**Status:** Backlog
**Epic Owner:** Product and Engineering
**Origin:** Epic 12, 13, 14 Retrospective Carry-Forward Items + Quality Gates

---

## Overview

Epic 19 focuses on quality foundation and technical debt resolution. This epic establishes the testing infrastructure, observability, and code quality improvements identified in previous epic retrospectives.

### Strategic Context

This epic consolidates carry-forward items from:
- **Epic 12 Retro (2026-01-04):** Retrieval pipeline testing, grader configurability
- **Epic 13 Retro (2026-01-04):** Crawler improvements from gemini-code-assist and coderabbitai reviews
- **Epic 14 Retro (2026-01-05):** Tenant isolation tests, endpoint compliance

### Split Decision (2026-01-05)

Original Epic 19 was split into two epics:
- **Epic 19:** Quality Foundation & Tech Debt (this document) - 26 stories
- **Epic 20:** Advanced Retrieval Intelligence (competitive features) - 18 stories

This split ensures:
1. Quality gates are established before adding new features
2. Tech debt is resolved on a solid foundation
3. Observability is in place to measure improvement

### Goals

- Complete all code review carry-forward items from Epics 12, 13, 14
- Establish retrieval quality benchmarks and monitoring
- Ensure multi-tenancy security with automated tests
- Improve crawler robustness and configurability
- Create foundation for measuring Epic 20 feature improvements

---

## Story Groups

### Group C (Partial): Quality Observability

*Foundational observability for measuring retrieval quality - required before Epic 20 features*

#### Story 19-C4: Implement Retrieval Quality Benchmarks

**Objective:** Create evaluation framework for retrieval quality.

**Scope:**
- Evaluation dataset with labeled query-document pairs
- MRR@K (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Precision@K and Recall@K
- A/B comparison framework

**Acceptance Criteria:**
- Benchmark CLI command exists: `uv run benchmark-retrieval`
- Metrics are computed and reported in structured format
- Baseline scores are established and documented
- CI can run benchmarks on PRs (optional gate)
- Results are stored for historical comparison

**Technical Notes:**
- Store benchmark results in `tests/benchmarks/results/`
- Use BEIR or custom dataset format
- Support multiple retrieval configurations in single run

---

#### Story 19-C5: Implement Prometheus Metrics for Retrieval

**Objective:** Export retrieval quality metrics for production monitoring.

**Metrics to Export:**
```
# Counters
retrieval_requests_total{strategy="vector|graph|hybrid"}
retrieval_fallback_triggered_total{reason="low_score|empty_results|timeout"}
grader_evaluations_total{result="pass|fail|fallback"}

# Histograms
retrieval_latency_seconds{strategy,phase="embed|search|rerank|grade"}
reranking_improvement_ratio  # Post-rerank score / pre-rerank score
grader_score{model}

# Gauges
retrieval_precision{strategy,k}
retrieval_recall{strategy,k}
active_retrieval_operations
```

**Configuration:**
```bash
PROMETHEUS_ENABLED=true|false  # Default: false
PROMETHEUS_PORT=9090  # Default
PROMETHEUS_PATH=/metrics  # Default
```

**Acceptance Criteria:**
- Prometheus metrics endpoint exists at `/metrics`
- All retrieval operations emit metrics
- Grafana dashboard JSON templates provided in `docs/observability/`
- Alert rules defined for quality degradation (>20% drop)
- Metrics include tenant_id label for multi-tenant analysis

---

### Group F: Epic 12 Code Review Carry-Forward

*Origin: Epic 12 Code Review (2026-01-04)*

#### Story 19-F1: Add Full Retrieval Pipeline Integration Test

**Priority:** HIGH
**Origin:** Epic 12 Code Review (2026-01-04)
**Updated:** Epic 14 Retro - Include A2A API endpoint integration tests

**Objective:** Test complete pipeline: embed → search → rerank → grade → fallback.

**Test Coverage:**
```python
# tests/integration/test_retrieval_pipeline.py
class TestRetrievalPipeline:
    async def test_full_pipeline_vector_only(self):
        """Vector search → rerank → grade → response"""

    async def test_full_pipeline_hybrid(self):
        """Vector + Graph → merge → rerank → grade → response"""

    async def test_fallback_on_low_score(self):
        """Low grader score triggers Tavily fallback"""

    async def test_fallback_on_empty_results(self):
        """No results triggers fallback"""

    async def test_a2a_query_endpoint(self):
        """A2A API endpoint returns valid response"""

    async def test_mcp_tool_invocation(self):
        """MCP knowledge.query tool works end-to-end"""
```

**Acceptance Criteria:**
- Integration test covers full retrieval flow with all features enabled
- Test runs with realistic data (not mocked embeddings)
- Edge cases covered: empty results, low scores, timeouts, fallback trigger
- Test is included in CI pipeline with 60-second timeout
- A2A and MCP endpoints are tested for protocol compliance

---

#### Story 19-F2: Add Multi-Tenancy Enforcement Tests

**Priority:** HIGH
**Origin:** Epic 12 Code Review (2026-01-04)

**Objective:** Ensure tenant_id isolation in all retrieval paths.

**Test Coverage:**
```python
# tests/security/test_tenant_isolation.py
class TestTenantIsolation:
    async def test_vector_search_tenant_filter(self):
        """Vector search only returns tenant's documents"""

    async def test_graph_traversal_tenant_filter(self):
        """Graph queries respect tenant boundaries"""

    async def test_cross_tenant_access_denied(self):
        """Query with tenant_a cannot access tenant_b data"""

    async def test_reranker_preserves_tenant(self):
        """Reranked results maintain tenant isolation"""

    async def test_grader_preserves_tenant(self):
        """Grader cannot leak cross-tenant data"""

    async def test_a2a_session_tenant_isolation(self):
        """A2A sessions are isolated per tenant"""
```

**Acceptance Criteria:**
- Tests assert tenant_id is passed to ALL database queries
- Cross-tenant data leakage test explicitly fails when isolation broken
- All retrieval methods (vector, graph, hybrid) have tenant tests
- Security audit checklist passes
- Tests run in CI on every PR

---

#### Story 19-F3: Make CrossEncoderGrader Model Selectable

**Priority:** MEDIUM
**Origin:** Epic 12 Code Review (2026-01-04)

**Objective:** Allow configuration of grader model without code changes.

**Configuration:**
```bash
GRADER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  # Default (fast, good accuracy)
# Alternatives:
# GRADER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2  # Higher accuracy
# GRADER_MODEL=BAAI/bge-reranker-base  # BGE reranker
# GRADER_MODEL=BAAI/bge-reranker-large  # BGE large (best accuracy)
```

**Acceptance Criteria:**
- Grader model is configurable via `GRADER_MODEL` environment variable
- At least 3 models are tested and documented
- Model loading is lazy (on first grader use, not startup)
- Documentation in `docs/guides/advanced-retrieval-configuration.md` lists available models with accuracy/speed tradeoffs
- Fallback to default if configured model unavailable

---

#### Story 19-F4: Make Heuristic Content Length Weight Configurable

**Priority:** MEDIUM
**Origin:** Epic 12 Code Review (2026-01-04)

**Objective:** Allow tuning of length-based scoring heuristic.

**Configuration:**
```bash
GRADER_HEURISTIC_LENGTH_WEIGHT=0.5  # 0-1, how much length influences score (default: 0.5)
GRADER_HEURISTIC_MIN_LENGTH=50  # Minimum content length for full score (default: 50)
GRADER_HEURISTIC_MAX_LENGTH=2000  # Length at which bonus maxes out (default: 2000)
```

**Current Heuristic Logic:**
```python
# Longer content gets slight bonus (more context = better)
length_factor = min(len(content) / max_length, 1.0)
heuristic_score = base_score * (1 - weight) + length_factor * weight
```

**Acceptance Criteria:**
- Length weight is configurable via environment variable
- Documentation explains the heuristic rationale
- Tests cover weight values: 0 (disabled), 0.5 (default), 1.0 (max)
- Logging shows heuristic contribution to final score

---

#### Story 19-F5: Add Contextual Retrieval Cost Logging

**Priority:** LOW
**Origin:** Epic 12 Code Review (2026-01-04)

**Objective:** Track LLM costs for contextual retrieval enrichment.

**Logging Output:**
```
contextual_retrieval:
  chunks_enriched: 15
  model: claude-3-haiku
  input_tokens: 12500
  output_tokens: 3200
  estimated_cost_usd: 0.0048
  cache_hits: 8
  cache_misses: 7
```

**Acceptance Criteria:**
- Token usage is logged for each contextual enrichment call
- Cost estimates computed using model pricing (configurable)
- Aggregated cost metrics available via Prometheus (if enabled)
- Dashboard shows contextual retrieval costs over time
- Cache hit rate is tracked for prompt caching efficiency

---

### Group G: Epic 12 Retro Future Enhancements

*Origin: Epic 12 Retrospective Nice-to-Do Items*

#### Story 19-G1: Add Reranking Result Caching

**Origin:** Epic 12 Retro Nice-to-Do #1

**Objective:** Cache reranked results to reduce latency on repeated queries.

**Configuration:**
```bash
RERANKER_CACHE_ENABLED=true|false  # Default: false
RERANKER_CACHE_TTL_SECONDS=300  # Default: 5 minutes
RERANKER_CACHE_MAX_SIZE=1000  # Max cached queries
```

**Cache Key Strategy:**
```python
cache_key = hash(
    query_text +
    sorted(document_ids) +
    reranker_model +
    tenant_id
)
```

**Acceptance Criteria:**
- Reranked results cached by query hash + document set
- Cache TTL is configurable (default 5 minutes)
- Cache hit rate is logged and exposed as metric
- Repeated identical queries return faster (measured)
- Cache respects tenant isolation

---

#### Story 19-G2: Make Context Generation Prompt Configurable

**Origin:** Epic 12 Retro Nice-to-Do #2

**Objective:** Allow customization of contextual retrieval prompt.

**Configuration:**
```bash
CONTEXTUAL_RETRIEVAL_PROMPT_PATH=prompts/contextual_retrieval.txt
```

**Default Prompt Template:**
```
Given the following document:
{document}

And this specific chunk:
{chunk}

Generate a brief context (1-2 sentences) that situates this chunk within the document.
Focus on: what section this is from, key entities mentioned, and relationship to document theme.

Context:
```

**Acceptance Criteria:**
- Contextual retrieval prompt is loaded from configurable file path
- Template supports `{document}` and `{chunk}` placeholders
- Domain-specific prompts can be used (e.g., legal, medical, technical)
- Documentation provides prompt engineering examples
- Invalid template gracefully falls back to default

---

#### Story 19-G3: Add Cross-Encoder Model Preloading

**Origin:** Epic 12 Retro Nice-to-Do #3

**Objective:** Reduce first-query latency by preloading model at startup.

**Configuration:**
```bash
GRADER_PRELOAD_MODEL=true|false  # Default: false
RERANKER_PRELOAD_MODEL=true|false  # Default: false
```

**Acceptance Criteria:**
- When enabled, model loads during application startup
- First query latency reduced by ~2-5 seconds (measured)
- Memory usage impact documented (typical: +500MB-1GB)
- Startup time impact measured and logged
- Health check waits for model load completion

---

#### Story 19-G4: Support Custom Normalization Strategies

**Origin:** Epic 12 Retro Nice-to-Do #4

**Objective:** Allow pluggable scoring normalization algorithms.

**Built-in Strategies:**
```python
class NormalizationStrategy(Enum):
    MIN_MAX = "min_max"  # (score - min) / (max - min)
    Z_SCORE = "z_score"  # (score - mean) / std
    SOFTMAX = "softmax"  # exp(score) / sum(exp(scores))
    PERCENTILE = "percentile"  # Rank-based normalization
```

**Configuration:**
```bash
GRADER_NORMALIZATION_STRATEGY=min_max  # Default
```

**Acceptance Criteria:**
- At least 4 normalization strategies implemented
- Strategy is configurable via environment variable
- Custom strategies can be registered programmatically
- Documentation explains each strategy's use case
- A/B comparison shows impact on ranking quality

---

### Group I: Epic 13 Code Review Carry-Forward

*Origin: gemini-code-assist and coderabbitai reviews (2026-01-04)*
*Focus: Crawler robustness and configurability improvements*

#### Story 19-I1: Externalize Crawl Profile Domain Matching

**Priority:** MEDIUM
**Origin:** Epic 13 Code Review - coderabbitai

**Objective:** Move hardcoded domain profiles to external configuration.

**Current State:**
```python
# Hardcoded in crawler.py
_EXACT_DOMAIN_PROFILES = {
    "docs.python.org": "documentation",
    "github.com": "code_repository",
    ...
}
```

**Target State:**
```yaml
# config/crawl-profiles.yaml
domain_profiles:
  exact_match:
    "docs.python.org": "documentation"
    "github.com": "code_repository"
  pattern_match:
    "*.readthedocs.io": "documentation"
    "*.github.io": "documentation"
```

**Acceptance Criteria:**
- Domain-to-profile mapping moved to `config/crawl-profiles.yaml`
- YAML file is loaded at startup, validated
- Hot-reload supported (or restart required, documented)
- Fallback to embedded defaults if file missing
- Documentation explains how to add custom mappings

---

#### Story 19-I2: Add Dynamic User-Agent Rotation

**Priority:** MEDIUM
**Origin:** Epic 13 Code Review - gemini-code-assist

**Objective:** Replace hardcoded Chrome 131 user-agent with rotation.

**Current State:**
```python
# Hardcoded
USER_AGENT = "Mozilla/5.0 ... Chrome/131.0.0.0 ..."
```

**Target State:**
```bash
CRAWLER_USER_AGENT_STRATEGY=rotate|static|random  # Default: rotate
CRAWLER_USER_AGENT_LIST_PATH=config/user-agents.txt  # Optional custom list
```

**Options:**
1. `rotate`: Cycle through realistic browser user-agents
2. `static`: Use single configured user-agent
3. `random`: Random selection per request

**Acceptance Criteria:**
- User-agent is configurable, not hardcoded
- At least 10 realistic user-agents in rotation
- User-agent logged with each request for debugging
- Custom user-agent list can be provided
- fake-useragent library integration optional

---

#### Story 19-I3: Add Crawler Legacy Alias Deprecation Warnings

**Priority:** MEDIUM
**Origin:** Epic 13 Code Review - coderabbitai

**Objective:** Warn users about deprecated function aliases.

**Deprecated Aliases:**
```python
# Old names (deprecated)
extract_links()  # Use: get_links()
extract_title()  # Use: get_title()
```

**Implementation:**
```python
import warnings

def extract_links(*args, **kwargs):
    warnings.warn(
        "extract_links() is deprecated, use get_links() instead. "
        "Will be removed in v2.0.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_links(*args, **kwargs)
```

**Acceptance Criteria:**
- Deprecated aliases emit `DeprecationWarning`
- Warning includes migration path and removal timeline
- Documentation lists all deprecated functions
- Deprecation warnings logged at WARNING level
- Tests verify warnings are emitted

---

#### Story 19-I4: Add Profile Magic Number Documentation

**Priority:** LOW
**Origin:** Epic 13 Code Review - coderabbitai

**Objective:** Document rationale for crawl profile default values.

**Magic Numbers to Document:**
```python
rate_limit=5.0  # Why 5.0? Based on what?
max_concurrent=10  # Why 10?
js_wait=2.0  # Why 2 seconds?
max_depth=3  # Why depth 3?
```

**Acceptance Criteria:**
- Each magic number has inline comment explaining rationale
- `docs/guides/crawl-configuration.md` explains tuning guidelines
- Performance benchmarks justify default values
- Common override scenarios documented

---

#### Story 19-I5: Optimize BeautifulSoup Async Parsing

**Priority:** LOW
**Origin:** Epic 13 Code Review - gemini-code-assist

**Objective:** Consider async HTML parsing for large documents.

**Current State:**
```python
# Synchronous parsing blocks event loop
soup = BeautifulSoup(html, 'lxml')
```

**Target Options:**
1. Use `run_in_executor()` to offload parsing
2. Use `selectolax` for faster parsing
3. Stream parsing for very large documents

**Acceptance Criteria:**
- Documents >1MB use async-compatible parsing
- Parsing time reduced by >30% for large documents
- Memory usage improved for streaming scenarios
- Backward compatible with existing code

---

#### Story 19-I6: Add Visited Set Bloom Filter

**Priority:** LOW
**Origin:** Epic 13 Code Review - gemini-code-assist

**Objective:** Use bloom filter for large crawls to reduce memory.

**Current State:**
```python
visited = set()  # O(n) memory for n URLs
```

**Target State:**
```python
if len(urls_to_crawl) > BLOOM_FILTER_THRESHOLD:
    visited = BloomFilter(capacity=100000, error_rate=0.001)
else:
    visited = set()
```

**Configuration:**
```bash
CRAWLER_BLOOM_FILTER_THRESHOLD=10000  # Use bloom filter above this
CRAWLER_BLOOM_FILTER_ERROR_RATE=0.001  # 0.1% false positive rate
```

**Acceptance Criteria:**
- Crawls >10k pages use bloom filter instead of set()
- Memory usage reduced by >80% for large crawls
- False positive rate configurable and documented
- Fallback to set() for small crawls

---

#### Story 19-I7: Implement Config Validation Fail-Fast

**Priority:** LOW
**Origin:** Epic 13 Code Review - coderabbitai

**Objective:** Fail fast on invalid crawl configuration in production.

**Current State:**
```python
if invalid_profile:
    logger.warning("Invalid profile, using default")  # Continues silently
```

**Target State:**
```python
if invalid_profile:
    if settings.ENVIRONMENT == "production":
        raise ValueError(f"Invalid crawl profile: {profile}")
    else:
        logger.warning("Invalid profile, using default")
```

**Configuration:**
```bash
CRAWLER_STRICT_VALIDATION=true|false  # Default: true in prod, false in dev
```

**Acceptance Criteria:**
- Invalid configurations raise `ValueError` in production
- Development mode allows fallback with warning
- Clear error messages indicate what's invalid
- Validation runs at startup, not first use

---

#### Story 19-I8: Implement Crawl-Many Error Recovery

**Priority:** LOW
**Origin:** Epic 13 Code Review - gemini-code-assist

**Objective:** Return partial results on crawl failure.

**Current State:**
```python
# Single failure can abort entire batch
results = await crawl_many(urls)  # All or nothing
```

**Target State:**
```python
results = await crawl_many(urls, on_error="continue")
# Returns: CrawlManyResult(
#     successful=[...],
#     failed=[(url, error), ...],
#     partial=True
# )
```

**Acceptance Criteria:**
- Batch crawl returns partial results on individual failures
- Each failure includes URL and error context
- Error logs include URL for debugging
- Configurable behavior: "fail_fast" vs "continue"
- Success/failure counts in result metadata

---

#### Story 19-I9: Implement Rate Limiting Enforcement

**Priority:** LOW
**Origin:** Epic 13 Code Review - coderabbitai

**Objective:** Add actual rate limiting, not just advisory js_wait.

**Current State:**
```python
js_wait=2.0  # Advisory delay, not enforced
```

**Target State:**
```python
rate_limiter = AsyncLimiter(
    max_rate=profile.rate_limit,  # requests per second
    time_period=1.0
)
async with rate_limiter:
    response = await fetch(url)
```

**Acceptance Criteria:**
- Rate limiting is enforced at domain level
- `aiolimiter` or similar library used
- Rate limit config per profile is respected
- Metrics track rate limit waits
- Burst allowance configurable

---

#### Story 19-I10: Add Fallback Provider Key Validation

**Priority:** LOW
**Origin:** Epic 13 Code Review - coderabbitai

**Objective:** Validate Apify/BrightData keys when fallback is enabled.

**Current State:**
```python
CRAWL_FALLBACK_ENABLED=true
# But APIFY_API_KEY might be missing/invalid
```

**Target State:**
```python
# At startup
if settings.CRAWL_FALLBACK_ENABLED:
    if not settings.APIFY_API_KEY:
        raise ConfigurationError(
            "CRAWL_FALLBACK_ENABLED=true but APIFY_API_KEY not set"
        )
    # Optionally validate key with API call
```

**Acceptance Criteria:**
- Missing API keys detected at startup
- Optional API validation (check key is valid)
- Clear error message with configuration fix
- Validation can be skipped in development

---

#### Story 19-I11: Document Large Crawl Memory Implications

**Priority:** LOW
**Origin:** Epic 13 Code Review - gemini-code-assist

**Objective:** Document memory usage for large crawls.

**Documentation Content:**
```markdown
## Memory Usage Guidelines

| Crawl Size | Estimated Memory | Recommendation |
|------------|------------------|----------------|
| < 100 pages | < 500 MB | Default settings |
| 100-1000 pages | 500 MB - 2 GB | Increase container memory |
| 1000-10000 pages | 2-8 GB | Use bloom filter, streaming |
| > 10000 pages | 8+ GB | Consider distributed crawling |

### Streaming Mode
For very large crawls, enable streaming to process results incrementally:
```bash
CRAWLER_STREAMING_MODE=true
CRAWLER_BATCH_SIZE=100
```

**Acceptance Criteria:**
- Memory guidelines added to `docs/guides/crawl-configuration.md`
- Streaming mode documented for large crawls
- Docker memory recommendations provided
- Memory profiling results included

---

#### Story 19-I12: Add Profile Auto-Detection Examples

**Priority:** LOW
**Origin:** Epic 13 Code Review - coderabbitai

**Objective:** Add usage examples to profile detection docstring.

**Current State:**
```python
def get_profile_for_url(url: str) -> CrawlProfile:
    """Get appropriate crawl profile for URL."""
```

**Target State:**
```python
def get_profile_for_url(url: str) -> CrawlProfile:
    """Get appropriate crawl profile for URL.

    Profile is selected based on domain matching:
    1. Exact domain match (e.g., "docs.python.org" → documentation)
    2. Pattern match (e.g., "*.readthedocs.io" → documentation)
    3. Content-type detection (e.g., SPA detection → js_heavy)
    4. Default profile

    Examples:
        >>> get_profile_for_url("https://docs.python.org/3/library/")
        CrawlProfile(name="documentation", js_rendering=False, ...)

        >>> get_profile_for_url("https://github.com/owner/repo")
        CrawlProfile(name="code_repository", rate_limit=2.0, ...)

        >>> get_profile_for_url("https://unknown-site.com")
        CrawlProfile(name="default", ...)

    Args:
        url: The URL to get a profile for

    Returns:
        CrawlProfile configured for the URL's domain/type
    """
```

**Acceptance Criteria:**
- Docstring includes at least 3 usage examples
- Examples are tested with doctest
- Profile selection logic is explained
- Edge cases documented (unknown domains, malformed URLs)

---

### Group J: Epic 14 Retro Carry-Forward

*Origin: Epic 14 Retrospective (2026-01-05)*
*Focus: Protocol compliance and security testing*

#### Story 19-J1: Add Tenant Isolation Automated Tests

**Priority:** HIGH
**Origin:** Epic 14 Retrospective (2026-01-05)

**Objective:** Automated security test suite that attempts cross-tenant access.

**Test Coverage:**
```python
# tests/security/test_tenant_isolation_attacks.py
class TestTenantIsolationAttacks:
    async def test_query_injection_cross_tenant(self):
        """Attempt SQL/Cypher injection to access other tenant"""

    async def test_session_hijacking(self):
        """Attempt to use tenant_a session for tenant_b data"""

    async def test_a2a_cross_tenant_delegation(self):
        """Verify A2A cannot delegate across tenants"""

    async def test_mcp_tool_tenant_bypass(self):
        """Verify MCP tools respect tenant boundaries"""

    async def test_trajectory_cross_tenant_access(self):
        """Verify trajectory logs are tenant-isolated"""

    async def test_ingestion_cross_tenant_access(self):
        """Verify ingested documents respect tenant"""
```

**Acceptance Criteria:**
- Automated test suite runs as part of security CI job
- Tests attempt realistic attack patterns
- Any cross-tenant access fails the test
- Test results include security audit report
- Runs on every PR to security-sensitive code

---

#### Story 19-J2: Add Endpoint Spec Compliance Tests

**Priority:** MEDIUM
**Origin:** Epic 14 Retrospective (2026-01-05)

**Objective:** Validate endpoint paths and capabilities match story specs.

**Test Coverage:**
```python
# tests/compliance/test_endpoint_spec.py
class TestEndpointCompliance:
    def test_a2a_endpoints_match_spec(self):
        """Verify A2A endpoints match Epic 14 story spec"""
        expected = [
            ("POST", "/a2a/sessions", "Create session"),
            ("GET", "/a2a/sessions/{id}", "Get session"),
            ("POST", "/a2a/sessions/{id}/messages", "Add message"),
        ]
        # Verify all exist with correct methods

    def test_mcp_tools_match_spec(self):
        """Verify MCP tools match Epic 14 story spec"""
        expected_tools = [
            "knowledge.query",
            "knowledge.graph_stats",
            "vector_search",
            "ingest_url",
            "ingest_pdf",
            "ingest_youtube",
        ]
        # Verify all tools registered

    def test_error_responses_rfc7807(self):
        """Verify all errors use RFC 7807 format"""
```

**Acceptance Criteria:**
- Tests validate endpoint paths against story specifications
- Tests validate request/response schemas
- RFC 7807 error format compliance verified
- OpenAPI spec matches implementation
- Runs on every PR

---

#### Story 19-J3: Implement Dedicated PDF Parsing Tool

**Priority:** LOW
**Origin:** Epic 14 Retrospective (2026-01-05)

**Objective:** Dedicated PDF parsing with Docling instead of generic text flow.

**Current State:**
- PDF parsing uses generic text extraction
- Limited structure preservation
- No special MCP tool for PDF

**Target State:**
- Dedicated `ingest_pdf` MCP tool
- Docling-based parsing with layout awareness
- Structure preservation (headings, tables, lists)
- Page-level chunking option

**Acceptance Criteria:**
- `ingest_pdf` MCP tool uses Docling library
- Tables extracted with structure preserved
- Headings create chunk boundaries
- Page numbers tracked in metadata
- PDF-specific configuration options

---

## Technical Notes

### Dependencies

- Epic 12: Reranking, grading infrastructure (completed)
- Epic 13: Crawl4AI integration (completed)
- Epic 14: A2A/MCP protocols (completed)

### Testing Infrastructure

All stories in this epic contribute to testing coverage:
- Integration tests: 19-F1, 19-F2, 19-J1, 19-J2
- Security tests: 19-F2, 19-J1
- Compliance tests: 19-J2
- Benchmarks: 19-C4

### Implementation Order

**Phase 1 (Critical - Security/Quality Gates):**
1. 19-F1: Full retrieval pipeline integration test
2. 19-F2: Multi-tenancy enforcement tests
3. 19-J1: Tenant isolation automated tests
4. 19-C4: Retrieval quality benchmarks

**Phase 2 (Observability):**
5. 19-C5: Prometheus metrics
6. 19-F5: Contextual retrieval cost logging
7. 19-J2: Endpoint spec compliance tests

**Phase 3 (Configurability):**
8. 19-F3: CrossEncoder model selectable
9. 19-F4: Heuristic weight configurable
10. 19-G1-G4: Caching and tuning options

**Phase 4 (Crawler Improvements):**
11. 19-I1-I12: Crawler robustness (can be parallelized)

**Phase 5 (Polish):**
12. 19-J3: Dedicated PDF parsing

---

## Risks

- Large number of stories may extend timeline
- Crawler improvements (Group I) are low priority but numerous
- Security tests may uncover issues requiring fixes

**Mitigation:**
- Prioritize Groups F, J first (security/quality gates)
- Group I stories can be done incrementally
- Security issues get immediate hotfix priority

---

## Success Metrics

- 100% code coverage on retrieval pipeline
- Zero cross-tenant data leakage in security tests
- Retrieval benchmark baseline established
- All endpoints compliant with story specifications
- Crawler handles 10k+ page crawls reliably

---

## References

- `_bmad-output/implementation-artifacts/epic-12-retro-2026-01-04.md`
- `_bmad-output/implementation-artifacts/epic-13-retro-2026-01-04.md`
- `_bmad-output/implementation-artifacts/epic-14-retro-2026-01-05.md`
- `docs/guides/advanced-retrieval-configuration.md`
- `docs/guides/crawl-configuration.md`
