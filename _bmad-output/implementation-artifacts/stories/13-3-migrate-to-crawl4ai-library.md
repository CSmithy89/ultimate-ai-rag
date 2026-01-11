# Story 13.3: Migrate to Crawl4AI Library

Status: done

## Story Definition

**As a** developer,
**I want** to migrate from custom httpx crawler to actual Crawl4AI library,
**So that** JavaScript-rendered content is captured and crawling is parallelized.

## Why This Matters

The current `crawler.py` used httpx which cannot render JavaScript, crawls sequentially, has no caching, and has no proxy support. Crawl4AI provides all these capabilities out of the box.

## Acceptance Criteria

- [x] JavaScript-rendered content is captured (SPAs, React sites work)
- [x] Multiple URLs crawl in parallel via `arun_many()` with MemoryAdaptiveDispatcher
- [x] Caching is enabled (unchanged pages not re-fetched)
- [x] Proxy support is configurable for blocked sites
- [x] Throughput improves 10x (50 pages in <30 seconds capability)
- [x] Legacy httpx implementation is deprecated/removed

## Implementation Summary

### Files Modified

1. **`backend/src/agentic_rag_backend/indexing/crawler.py`**
   - Completely rewritten to use Crawl4AI AsyncWebCrawler
   - Added `CrawlerService` class with async context manager pattern
   - Implemented `crawl_page()` for single URL crawling
   - Implemented `crawl_many()` for parallel URL crawling with MemoryAdaptiveDispatcher
   - Added JavaScript rendering support via `delay_before_return_html`
   - Added proxy support via BrowserConfig
   - Added caching support via CacheMode enum
   - Removed legacy httpx-based implementation
   - Added legacy compatibility aliases for `extract_links` and `extract_title`

2. **`backend/src/agentic_rag_backend/config.py`**
   - Added new Crawl4AI configuration settings:
     - `crawl4ai_headless: bool` (default: True)
     - `crawl4ai_max_concurrent: int` (default: 10)
     - `crawl4ai_cache_enabled: bool` (default: True)
     - `crawl4ai_proxy_url: Optional[str]` (default: None)
     - `crawl4ai_js_wait_seconds: float` (default: 2.0)
     - `crawl4ai_page_timeout_ms: int` (default: 60000)

3. **`backend/src/agentic_rag_backend/indexing/__init__.py`**
   - Updated exports to include new functions and constants
   - Added `CRAWL4AI_AVAILABLE` flag for runtime feature detection
   - Added `DEFAULT_MAX_CONCURRENT` and `DEFAULT_JS_WAIT_SECONDS` constants

4. **`backend/tests/test_crawler_crawl4ai.py`** (new file)
   - Comprehensive test suite for new Crawl4AI implementation
   - 39 tests covering all functionality

5. **`backend/tests/indexing/test_crawler.py`**
   - Updated to work with new Crawl4AI implementation
   - Removed tests for deprecated components (RobotsTxtChecker, html_to_markdown)
   - 31 tests updated and passing

### Key Features

1. **JavaScript Rendering**
   - Uses Playwright-based browser automation
   - Configurable wait time for JavaScript to execute
   - Captures dynamically loaded content (SPAs, React, Vue, etc.)

2. **Parallel Crawling**
   - Uses `arun_many()` with MemoryAdaptiveDispatcher
   - Automatically adapts concurrency based on system resources
   - Configurable maximum concurrent sessions (default: 10)

3. **Caching**
   - Enabled by default via CacheMode.ENABLED
   - Avoids re-fetching unchanged pages
   - Can be disabled for fresh crawls

4. **Proxy Support**
   - Configurable via `crawl4ai_proxy_url` setting
   - Supports authenticated proxies
   - Useful for accessing blocked sites

### Configuration Environment Variables

```bash
# Crawl4AI Configuration
CRAWL4AI_HEADLESS=true          # Run browser in headless mode
CRAWL4AI_MAX_CONCURRENT=10      # Maximum concurrent crawl sessions
CRAWL4AI_CACHE_ENABLED=true     # Enable page caching
CRAWL4AI_PROXY_URL=             # Optional proxy URL (http://user:pass@host:port)
CRAWL4AI_JS_WAIT_SECONDS=2.0    # Seconds to wait for JavaScript rendering
CRAWL4AI_PAGE_TIMEOUT_MS=60000  # Page load timeout in milliseconds
```

### Usage Example

```python
from agentic_rag_backend.indexing import CrawlerService, crawl_url

# Using the convenience function
async for page in crawl_url("https://example.com", max_depth=2):
    print(f"Crawled: {page.url} - {page.title}")

# Using CrawlerService directly for more control
async with CrawlerService(
    headless=True,
    max_concurrent=10,
    cache_enabled=True,
    js_wait_seconds=2.0,
) as crawler:
    # Single page
    page = await crawler.crawl_page("https://example.com")

    # Multiple pages in parallel
    urls = ["https://example.com/page1", "https://example.com/page2"]
    async for page in crawler.crawl_many(urls, stream=True):
        print(f"Crawled: {page.url}")
```

## Test Results

```
70 tests passed (tests/indexing/test_crawler.py + tests/test_crawler_crawl4ai.py)
568 total backend tests passing
```

## Developer Notes

- The CrawlerService must be used as an async context manager
- The `crawl_url` convenience function handles context manager lifecycle
- Crawl4AI requires browser installation on first run (`crawl4ai-install`)
- The `CRAWL4AI_AVAILABLE` flag can be used to check if Crawl4AI is installed

## Completion Date

2026-01-04
