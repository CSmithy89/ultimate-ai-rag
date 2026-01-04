# Epic 13 Tech Spec: Enterprise Ingestion

**Date:** 2025-12-31
**Updated:** 2026-01-04 (Crawl4AI Migration Analysis)
**Status:** Backlog
**Epic Owner:** Product and Engineering

---

## Overview

Epic 13 hardens ingestion with enterprise-grade crawling providers and faster, more reliable capture of large sources. It adds robust external ingestion adapters, prioritizes YouTube transcripts, and **migrates from custom httpx crawler to actual Crawl4AI library**.

### Key Decision (2026-01-04)

**CRITICAL: Current crawler.py does NOT use Crawl4AI library.**

The installed `crawl4ai>=0.3.0` dependency is unused. Current implementation is a custom httpx-based crawler with significant limitations:

| Gap | Current | Target (Crawl4AI) |
|-----|---------|-------------------|
| JS Rendering | None (httpx) | Playwright browser |
| Concurrency | Sequential | `arun_many()` + MemoryAdaptiveDispatcher |
| Caching | None | CacheMode (BYPASS/READ/WRITE) |
| Proxy | None | Built-in proxy support |
| Throughput | ~1 page/sec | 10-50 pages/sec |

### Goals

- **Migrate to actual Crawl4AI library** for browser-based crawling.
- Add Apify and BrightData fallback ingestion for blocked sites.
- Optimize YouTube ingestion through transcript-first processing.
- Achieve 10x throughput improvement on standard benchmarks.

### Scope

**In scope**
- Crawl4AI library migration with AsyncWebCrawler.
- Parallel crawling with MemoryAdaptiveDispatcher.
- Ingestion adapter layer supporting Apify and BrightData.
- YouTube transcript ingestion path.
- Caching and proxy configuration.

**Out of scope**
- Full multimodal processing (moved to Epic 15 as codebase intelligence).

---

## Stories

### Story 13-1: Integrate Apify and BrightData Fallback

**Objective:** Add enterprise crawler adapters to avoid blocked ingestion.

**Acceptance Criteria**
- Given a crawl request, when Crawl4AI fails or is blocked, then the system can route the job to Apify or BrightData.
- Credentials and provider settings are configured via environment variables.
- Ingestion logs include provider selection and fallback reason.

**Configuration:**
```bash
INGESTION_FALLBACK_ENABLED=true|false  # Default: false
INGESTION_FALLBACK_PROVIDERS=apify,brightdata  # Fallback order
APIFY_API_KEY=xxx
BRIGHTDATA_API_KEY=xxx
```

### Story 13-2: Implement YouTube Transcript Ingestion

**Objective:** Ingest YouTube content via transcript-first workflow.

**Acceptance Criteria**
- Given a YouTube URL, when ingestion is triggered, then the system fetches transcripts using youtube-transcript-api.
- If a transcript is unavailable, the system records a clear error and optional fallback path.
- Transcript ingestion produces standard chunks with source metadata.
- **Ingestion completes in under 30 seconds for typical videos.**

**Configuration:**
```bash
YOUTUBE_TRANSCRIPT_LANGUAGES=en,es,fr  # Preferred languages
YOUTUBE_FALLBACK_TO_AUTO=true  # Use auto-generated if manual unavailable
```

### Story 13-3: Migrate to Crawl4AI Library

**Objective:** Replace custom httpx crawler with actual Crawl4AI library for browser-based crawling with JavaScript support and parallel execution.

**Why This Matters:**
The current `crawler.py` uses httpx which:
- Cannot render JavaScript (fails on SPAs, React sites)
- Crawls sequentially (1 page at a time)
- Has no caching (re-crawls everything)
- Has no proxy support (gets blocked easily)

**Migration Steps:**

1. **Replace CrawlerService with AsyncWebCrawler**
```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher

browser_config = BrowserConfig(
    headless=True,
    viewport_width=1280,
    viewport_height=720,
    # JavaScript is enabled by default in Playwright
)

run_config = CrawlerRunConfig(
    cache_mode=CacheMode.WRITE_ONLY,  # Cache for re-crawls
    word_count_threshold=10,
    excluded_tags=["script", "style", "nav", "footer"],
    exclude_external_links=True,
)

dispatcher = MemoryAdaptiveDispatcher(
    max_session_permit=10,           # Max concurrent sessions
    memory_threshold_percent=80.0,   # Auto-throttle at 80% memory
    check_interval=0.5,              # Check memory every 0.5 seconds
)

async with AsyncWebCrawler(config=browser_config) as crawler:
    results = await crawler.arun_many(urls, config=run_config, dispatcher=dispatcher)
```

2. **Update CrawledPage model** to include Crawl4AI result fields.

3. **Add caching layer** using CacheMode for incremental crawls.

4. **Add proxy configuration** for enterprise deployments.

**Acceptance Criteria**
- Given a crawl request, when using Crawl4AI AsyncWebCrawler, then JavaScript-rendered content is captured.
- Given multiple URLs, when `arun_many()` is called, then crawling happens in parallel with memory-adaptive concurrency.
- Crawl4AI respects robots.txt via built-in compliance.
- **Caching is enabled** - unchanged pages are not re-fetched.
- **Proxy support is configurable** for enterprise blocked sites.
- Retries and backoff use Crawl4AI's built-in mechanisms.
- **Throughput benchmark: 10x improvement** on standard doc site (e.g., 50 pages in <30 seconds).
- Legacy `crawler.py` httpx implementation is deprecated/removed.

**Configuration:**
```bash
# Crawl4AI Settings
CRAWL4AI_HEADLESS=true|false  # Default: true
CRAWL4AI_MAX_CONCURRENCY=10  # Max parallel sessions
CRAWL4AI_MEMORY_THRESHOLD=0.8  # Memory % before throttling
CRAWL4AI_CACHE_MODE=write_only|read_write|bypass  # Default: write_only
CRAWL4AI_TIMEOUT=30  # Seconds per page
CRAWL4AI_JS_ENABLED=true|false  # Default: true

# Proxy (optional)
CRAWL4AI_PROXY_URL=http://user:pass@proxy:8080
```

### Story 13-4: Optimize Crawl Configuration Profiles

**Objective:** Provide pre-configured crawl profiles for common use cases.

**Profiles:**

| Profile | Use Case | Settings |
|---------|----------|----------|
| `fast` | Documentation sites | headless, no screenshots, high concurrency |
| `thorough` | Complex SPAs | JS wait, screenshots, lower concurrency |
| `stealth` | Bot-protected sites | Proxy, random delays, fingerprint evasion |

**Acceptance Criteria**
- Given a profile name, when crawl is triggered, then appropriate settings are applied.
- Profiles are selectable via `CRAWL4AI_PROFILE` or API parameter.
- Custom profiles can be defined in configuration.

**Configuration:**
```bash
CRAWL4AI_PROFILE=fast|thorough|stealth  # Default: fast
```

---

## Technical Notes

### Migration Path

1. **Phase 1:** Add Crawl4AI wrapper alongside existing crawler.
2. **Phase 2:** Route new crawls through Crawl4AI.
3. **Phase 3:** Deprecate and remove legacy httpx crawler.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION ROUTER                          │
├─────────────────────────────────────────────────────────────┤
│  URL Type Detection                                          │
│      │                                                       │
│      ├── YouTube URL ──────► YouTube Transcript API          │
│      │                                                       │
│      ├── Standard URL ─────► Crawl4AI AsyncWebCrawler        │
│      │                           │                           │
│      │                           ├── Success ───► Chunks     │
│      │                           │                           │
│      │                           └── Blocked ───► Fallback   │
│      │                                               │       │
│      │                               ┌───────────────┤       │
│      │                               ▼               ▼       │
│      │                            Apify         BrightData   │
│      │                                                       │
│      └── PDF ──────────────► Docling                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Crawl4AI Key Features to Leverage

- **MemoryAdaptiveDispatcher:** Auto-throttles based on system memory.
- **CacheMode:** Avoid re-crawling unchanged content.
- **Session Management:** Persistent browser contexts for authenticated sites.
- **LLM-Optimized Markdown:** Better than markdownify for AI consumption.
- **Streaming:** Process results as they complete with `stream=True`.

### Files to Modify

| File | Change |
|------|--------|
| `indexing/crawler.py` | Replace with Crawl4AI wrapper |
| `indexing/workers/crawl_worker.py` | Update to use new crawler |
| `config.py` | Add Crawl4AI configuration |
| `models/ingest.py` | Update CrawlOptions model |
| `api/routes/ingest.py` | Add profile selection |

## Epic 12 Carry-Forward Items

The following items from Epic 12 code review should be addressed during Epic 13:

### Integration Tests (High Priority)

1. **Full Retrieval Pipeline Integration Test**
   - Location: `backend/tests/test_orchestrator_integration.py`
   - Test retrieval with reranking + grading + fallback all enabled
   - Verify: vector search → rerank → grade → fallback (if triggered)
   - Include assertions on trajectory logging, hit ordering, fallback execution

2. **Multi-Tenancy Enforcement Tests**
   - Add explicit `tenant_id` assertions in integration tests
   - Verify all database queries include tenant filtering
   - Prevent data leakage between tenants

### Configuration Enhancements (Medium Priority)

3. **Grader Model Selection**
   - Current: Hardcoded to `HeuristicGrader`
   - Target: Make `CrossEncoderGrader` selectable via `GRADER_MODEL=heuristic|cross_encoder`
   - Location: `backend/src/agentic_rag_backend/retrieval/grader.py:488-490`

4. **Heuristic Grader Content Length Weight**
   - Current: Hardcoded normalization factor (`avg_length / 1000`)
   - Target: Make configurable via `GRADER_CONTENT_LENGTH_WEIGHT` env var
   - Location: `backend/src/agentic_rag_backend/retrieval/grader.py:142-143`
   - Alternative: Document as known limitation in config guide

### Observability (Low Priority)

5. **Contextual Retrieval Cost Logging**
   - Log token usage and cost estimates for contextual enrichment
   - Help users understand spending on LLM calls per chunk

---

## Dependencies

- Crawl4AI library (already installed: `crawl4ai>=0.3.0`)
- Playwright browser binaries (installed by Crawl4AI)
- Existing ingestion pipeline (Epic 4).
- Storage and indexing pipeline for chunk processing.

## Risks

- **Browser resource usage:** Playwright uses more memory than httpx.
  - *Mitigation:* MemoryAdaptiveDispatcher throttles automatically.
- **Third-party provider API limits or pricing changes.**
  - *Mitigation:* Fallback chain with multiple providers.
- **Provider-specific HTML extraction inconsistencies.**
  - *Mitigation:* Standardize output format in adapter layer.

## Success Metrics

- **Throughput:** 10x improvement (target: 50 pages in <30 seconds).
- **JavaScript sites:** Successfully crawl React/Vue/Angular SPAs.
- **Fallback success rate:** >95% on previously blocked targets.
- **YouTube ingestion:** Completes in under 30 seconds for typical transcript.
- **Cache hit rate:** >50% on re-crawls of same site.

## References

- [Crawl4AI Documentation](https://github.com/unclecode/crawl4ai)
- [Crawl4AI AsyncWebCrawler API](https://github.com/unclecode/crawl4ai/blob/main/docs/md_v2/api/async-webcrawler.md)
- `docs/roadmap-decisions-2026-01-03.md` - Decision rationale
- `_bmad-output/prd.md`
- `_bmad-output/architecture.md`
- `_bmad-output/project-planning-artifacts/epics.md`
- `docs/recommendations_2025.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
