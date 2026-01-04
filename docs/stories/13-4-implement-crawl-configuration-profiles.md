# Story 13.4: Implement Crawl Configuration Profiles

## Status: Done

## Story Definition

**As a** developer,
**I want** pre-defined crawl configuration profiles,
**So that** I can easily configure the crawler for different scenarios (fast docs, SPAs, bot-protected sites).

## Why This Matters

Different websites require different crawling strategies:
- Static documentation sites can be crawled quickly with high concurrency
- SPAs need JavaScript rendering and wait conditions
- Bot-protected sites need stealth mode with slower rates and anti-detection measures

Pre-defined profiles simplify configuration and enable auto-detection based on URL patterns.

## Acceptance Criteria

- [x] Three profiles available: fast, thorough, stealth
- [x] Profiles are selectable via CRAWL4AI_PROFILE env var
- [x] Profile settings are logged for debugging
- [x] Auto-detection suggests appropriate profile based on URL
- [x] All tests pass

## Implementation Summary

### Files Created

1. **`backend/src/agentic_rag_backend/indexing/crawl_profiles.py`** (new file)
   - `CrawlProfileName` enum: FAST, THOROUGH, STEALTH
   - `CrawlProfile` frozen dataclass with fields:
     - name, description, headless, stealth, max_concurrent
     - rate_limit, wait_for, wait_timeout, proxy_config, cache_enabled
   - `CRAWL_PROFILES` dict with three pre-defined profiles
   - `get_crawl_profile(name)` function for retrieving profiles by name
   - `get_profile_for_url(url)` function for auto-detection
   - `apply_proxy_override(profile, proxy_url)` function for proxy overrides

### Files Modified

1. **`backend/src/agentic_rag_backend/config.py`**
   - Added `crawl4ai_profile: str` (default: "fast")
   - Added `crawl4ai_stealth_proxy: Optional[str]` (default: None)
   - Added validation for profile names

2. **`backend/src/agentic_rag_backend/indexing/crawler.py`**
   - Updated `CrawlerService.__init__` to accept profile parameter
   - Added stealth, wait_for, wait_timeout parameters
   - Updated `_create_browser_config` for stealth mode
   - Updated `_create_crawler_config` for wait_for conditions
   - Enhanced logging to include profile information
   - Updated `crawl_url` convenience function with profile support

3. **`backend/src/agentic_rag_backend/indexing/__init__.py`**
   - Added exports for crawl_profiles module

4. **`backend/tests/test_crawl_profiles.py`** (new file)
   - Comprehensive test suite for crawl profiles

### Profile Configurations

| Profile   | Description                     | Headless | Stealth | Concurrent | Rate Limit | Cache |
|-----------|--------------------------------|----------|---------|------------|------------|-------|
| fast      | High-speed for static docs     | True     | False   | 10         | 5.0/s      | True  |
| thorough  | SPAs with dynamic content      | True     | False   | 5          | 2.0/s      | True  |
| stealth   | Bot-protected sites            | False    | True    | 3          | 0.5/s      | False |

### Configuration Environment Variables

```bash
# Crawl Profile Configuration
CRAWL4AI_PROFILE=fast          # Profile to use: fast, thorough, or stealth
CRAWL4AI_STEALTH_PROXY=        # Optional proxy for stealth mode (http://user:pass@host:port)
```

### Auto-Detection Heuristics

The `get_profile_for_url()` function uses domain heuristics:

- **fast** profile for:
  - docs.*, documentation.*, readthedocs.*, gitbook.*, docusaurus.*
  - *.github.io, wiki.*

- **thorough** profile for:
  - app.*, dashboard.*, console.*, portal.*

- **stealth** profile for:
  - linkedin.com, facebook.com, twitter.com, x.com
  - cloudflare.com, indeed.com, glassdoor.com
  - amazon.com, google.com

### Usage Examples

```python
from agentic_rag_backend.indexing import (
    CrawlerService,
    crawl_url,
    get_crawl_profile,
    get_profile_for_url,
    CRAWL_PROFILES,
)

# Using a named profile
profile = get_crawl_profile("thorough")
async with CrawlerService(profile=profile) as crawler:
    page = await crawler.crawl_page("https://app.example.com")

# Using crawl_url with profile name
async for page in crawl_url("https://docs.python.org", profile_name="fast"):
    print(f"Crawled: {page.url}")

# Auto-detect profile based on URL
async for page in crawl_url("https://linkedin.com/jobs", auto_detect_profile=True):
    print(f"Crawled with stealth: {page.url}")

# Get suggested profile for planning
suggested = get_profile_for_url("https://docs.example.com")
print(f"Suggested profile: {suggested}")  # "fast"
```

## Test Results

```
All crawl profile tests passing
All existing crawler tests compatible
```

## Developer Notes

- Profiles are frozen dataclasses to ensure immutability
- Stealth mode uses a realistic Chrome user-agent
- Auto-detection defaults to "thorough" for unknown domains (safe middle ground)
- The stealth profile disables caching to ensure fresh content
- Profile settings can be overridden at the CrawlerService level if needed

## Completion Date

2026-01-04
