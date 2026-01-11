# Epic 13 Tech Spec: Enterprise Ingestion

**Version:** 1.0
**Created:** 2026-01-04
**Status:** Complete

---

## Overview

Epic 13 focuses on enterprise-grade ingestion capabilities: migrating from the custom httpx-based crawler to the actual Crawl4AI library for JavaScript rendering and parallel crawling, adding fallback crawl providers (Apify/BrightData) for blocked sites, implementing transcript-first YouTube ingestion, and creating pre-configured crawl profiles for common use cases.

### Key Decision (2026-01-04)

**Current `crawler.py` does NOT use the installed Crawl4AI library.** The existing implementation is a custom httpx-based crawler that:
- Cannot render JavaScript (misses SPA content, React sites, dynamically loaded content)
- Crawls URLs sequentially (slow for large documentation sites)
- Has no caching (re-fetches unchanged pages)
- Has no proxy support (cannot bypass blocked sites)
- Has no stealth mode (easily blocked by anti-bot systems)

The Crawl4AI library (already in `pyproject.toml` as `crawl4ai>=0.3.0`) provides all these capabilities through `AsyncWebCrawler`. **This epic prioritizes migration to use the actual library.**

### Business Value

- **10x throughput improvement**: Parallel crawling with `arun_many()` vs sequential httpx
- **JavaScript rendering**: Capture content from SPAs, React apps, and dynamically loaded pages
- **Resilient ingestion**: Fallback providers when primary crawling fails
- **Faster video ingestion**: Transcript-first approach vs full video processing
- **Simplified operations**: Pre-configured profiles for common crawl scenarios

---

## Current State Analysis

### Existing Implementation (`crawler.py`)

**Location:** `backend/src/agentic_rag_backend/indexing/crawler.py`

**Current Architecture:**
```python
# Uses httpx for HTTP requests (NOT Crawl4AI)
async def _fetch_url(self, url: str) -> tuple[str, int]:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, ...)
        return response.text, response.status_code
```

**Current Capabilities:**
| Feature | Status |
|---------|--------|
| Basic HTML fetch | Yes |
| robots.txt compliance | Yes |
| Rate limiting | Yes (per-request delay) |
| Markdown conversion | Yes (BeautifulSoup + markdownify) |
| Link extraction | Yes |
| Sequential crawling | Yes |
| JavaScript rendering | **NO** |
| Parallel crawling | **NO** |
| Caching | **NO** |
| Proxy support | **NO** |
| Stealth mode | **NO** |

**Current Limitations:**
1. **No JavaScript rendering** - Cannot capture content from SPAs, React sites
2. **Sequential crawling** - Each URL fetched one at a time with rate limit delays
3. **No caching** - Re-fetches identical pages on every crawl
4. **No proxy support** - Cannot bypass blocked sites
5. **Easily detected** - No stealth features, blocked by anti-bot systems

### Existing Worker Architecture

**Location:** `backend/src/agentic_rag_backend/indexing/workers/crawl_worker.py`

The `CrawlWorker` class consumes from Redis Streams and processes crawl jobs:
```python
class CrawlWorker:
    def __init__(self, ...):
        self.crawler = CrawlerService()  # Uses custom httpx crawler
```

This worker will need to be updated to use the new Crawl4AI-based crawler.

### Existing Configuration

**Location:** `backend/src/agentic_rag_backend/config.py`

Current crawl-related settings:
```python
crawl4ai_rate_limit: float  # CRAWL4AI_RATE_LIMIT env var
```

Additional settings needed for Epic 13.

---

## Target Architecture

### Post-Epic 13 Ingestion Pipeline

```
                         +-------------------+
                         |   Ingestion API   |
                         +-------------------+
                                  |
                    +-------------+-------------+
                    |             |             |
              +-----v-----+ +-----v-----+ +-----v-----+
              |   URL     | |  YouTube  | |   PDF     |
              |  Crawl    | |   Ingest  | |  Upload   |
              +-----------+ +-----------+ +-----------+
                    |             |             |
                    v             v             v
              +-----+-----+ +-----+-----+ +-----------+
              | Crawl4AI  | | Transcript| |  Docling  |
              | Wrapper   | |    API    | |  Parser   |
              +-----------+ +-----------+ +-----------+
                    |             |             |
                    |       +-----+-----+       |
                    |       |   Chunk   |       |
                    +------>|  + Index  |<------+
                            +-----------+
                                  |
                            +-----v-----+
                            | Graphiti  |
                            |  Episode  |
                            +-----------+
```

### New Module Structure

```
backend/src/agentic_rag_backend/indexing/
├── __init__.py           # Exports new modules
├── crawler.py            # DEPRECATED - to be removed/replaced
├── crawl4ai_crawler.py   # NEW: Crawl4AI AsyncWebCrawler wrapper
├── crawl_profiles.py     # NEW: Pre-configured crawl profiles
├── fallback_providers.py # NEW: Apify/BrightData adapters
├── youtube_ingestion.py  # NEW: YouTube transcript ingestion
├── chunker.py            # Existing
├── graphiti_ingestion.py # Existing
├── parser.py             # Existing
└── workers/
    ├── crawl_worker.py   # Updated to use new crawler
    └── ...
```

---

## Story Technical Details

### Story 13-1: Integrate Apify/BrightData Fallback

**Goal:** Provide fallback crawling when Crawl4AI is blocked by anti-bot systems.

**Technical Approach:**

1. **Adapter Pattern for Providers:**
```python
# backend/src/agentic_rag_backend/indexing/fallback_providers.py

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel

class CrawlResult(BaseModel):
    url: str
    content: str
    status_code: int
    error: Optional[str] = None

class CrawlProvider(ABC):
    """Abstract base class for crawl providers."""

    @abstractmethod
    async def crawl(self, url: str, options: dict) -> CrawlResult:
        """Crawl a single URL and return content."""
        pass

    @abstractmethod
    async def crawl_many(self, urls: list[str], options: dict) -> list[CrawlResult]:
        """Crawl multiple URLs in parallel."""
        pass

class ApifyProvider(CrawlProvider):
    """Apify web scraping API provider."""

    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.apify.com/v2"

    async def crawl(self, url: str, options: dict) -> CrawlResult:
        # Use Apify's Web Scraper actor
        # https://apify.com/apify/web-scraper
        ...

class BrightDataProvider(CrawlProvider):
    """BrightData (Luminati) web scraping provider."""

    def __init__(self, username: str, password: str, zone: str = "scraping_browser"):
        self.proxy_url = f"http://{username}:{password}@brd.superproxy.io:33335"
        self.zone = zone

    async def crawl(self, url: str, options: dict) -> CrawlResult:
        # Use BrightData's Scraping Browser
        ...
```

2. **Fallback Chain Logic:**
```python
class FallbackCrawler:
    """Crawler with automatic fallback to paid providers."""

    def __init__(
        self,
        primary: CrawlProvider,
        fallbacks: list[CrawlProvider],
        max_retries: int = 2,
    ):
        self.primary = primary
        self.fallbacks = fallbacks
        self.max_retries = max_retries

    async def crawl_with_fallback(
        self,
        url: str,
        options: dict
    ) -> tuple[CrawlResult, str]:
        """Returns (result, provider_used)."""

        # Try primary first
        try:
            result = await self.primary.crawl(url, options)
            if result.status_code == 200:
                return result, "crawl4ai"
        except Exception as e:
            logger.warning("primary_crawl_failed", url=url, error=str(e))

        # Try fallbacks in order
        for fallback in self.fallbacks:
            try:
                result = await fallback.crawl(url, options)
                if result.status_code == 200:
                    provider_name = fallback.__class__.__name__.lower()
                    return result, provider_name
            except Exception as e:
                logger.warning("fallback_crawl_failed",
                    provider=fallback.__class__.__name__,
                    url=url, error=str(e))

        raise CrawlError(f"All providers failed for {url}")
```

3. **Configuration:**
```python
# New settings in config.py
APIFY_API_TOKEN: Optional[str]           # APIFY_API_TOKEN env var
BRIGHTDATA_USERNAME: Optional[str]        # BRIGHTDATA_USERNAME
BRIGHTDATA_PASSWORD: Optional[str]        # BRIGHTDATA_PASSWORD
BRIGHTDATA_ZONE: str = "scraping_browser" # BRIGHTDATA_ZONE
CRAWL_FALLBACK_ENABLED: bool = True       # CRAWL_FALLBACK_ENABLED
CRAWL_FALLBACK_PROVIDERS: list[str]       # ["apify", "brightdata"]
```

**Acceptance Criteria:**
- When primary Crawl4AI fails with blocked/403/captcha, automatically try fallback
- Fallback provider selection is logged in trajectory
- Credentials are securely configured via environment variables
- Fallback usage is tracked for cost monitoring

---

### Story 13-2: Implement YouTube Transcript API Ingestion

**Goal:** Fast transcript-first YouTube video processing without full video download.

**Technical Approach:**

1. **YouTube Transcript Service:**
```python
# backend/src/agentic_rag_backend/indexing/youtube_ingestion.py

from dataclasses import dataclass
from typing import Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

@dataclass
class TranscriptSegment:
    text: str
    start: float  # seconds
    duration: float  # seconds

@dataclass
class YouTubeTranscriptResult:
    video_id: str
    title: Optional[str]
    language: str
    is_generated: bool
    segments: list[TranscriptSegment]
    full_text: str
    duration_seconds: float

class YouTubeIngestionService:
    """Transcript-first YouTube video ingestion."""

    def __init__(self, preferred_languages: list[str] = None):
        self.ytt_api = YouTubeTranscriptApi()
        self.preferred_languages = preferred_languages or ["en", "en-US", "en-GB"]

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats."""
        # Handles: youtube.com/watch?v=X, youtu.be/X, youtube.com/embed/X
        import re
        patterns = [
            r'(?:v=|/embed/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError(f"Could not extract video ID from: {url}")

    async def fetch_transcript(
        self,
        video_id: str,
        languages: list[str] = None,
    ) -> YouTubeTranscriptResult:
        """Fetch transcript for a YouTube video."""
        languages = languages or self.preferred_languages

        try:
            # Fetch transcript (runs in thread pool for async)
            transcript = await asyncio.to_thread(
                self.ytt_api.fetch,
                video_id,
                languages=languages,
            )

            # Parse segments
            segments = [
                TranscriptSegment(
                    text=seg["text"],
                    start=seg["start"],
                    duration=seg["duration"],
                )
                for seg in transcript
            ]

            # Build full text
            full_text = " ".join(seg.text for seg in segments)
            total_duration = max(s.start + s.duration for s in segments) if segments else 0

            return YouTubeTranscriptResult(
                video_id=video_id,
                title=None,  # Could fetch via YouTube Data API
                language=transcript.language,
                is_generated=transcript.is_generated,
                segments=segments,
                full_text=full_text,
                duration_seconds=total_duration,
            )

        except TranscriptsDisabled:
            raise IngestionError(
                document_id=video_id,
                reason="Subtitles are disabled for this video",
            )
        except NoTranscriptFound:
            raise IngestionError(
                document_id=video_id,
                reason=f"No transcript found in languages: {languages}",
            )
        except VideoUnavailable:
            raise IngestionError(
                document_id=video_id,
                reason="Video is unavailable",
            )

    async def ingest_video(
        self,
        url: str,
        tenant_id: str,
    ) -> UnifiedDocument:
        """Ingest a YouTube video as a document."""
        video_id = self.extract_video_id(url)
        result = await self.fetch_transcript(video_id)

        return UnifiedDocument(
            id=uuid4(),
            tenant_id=UUID(tenant_id),
            content=result.full_text,
            source_type=SourceType.URL,
            source_url=f"https://youtube.com/watch?v={video_id}",
            metadata=DocumentMetadata(
                title=result.title or f"YouTube Video {video_id}",
                extra={
                    "video_id": video_id,
                    "language": result.language,
                    "is_generated": result.is_generated,
                    "duration_seconds": result.duration_seconds,
                    "segment_count": len(result.segments),
                },
            ),
        )
```

2. **Chunking with Timestamps:**
```python
def chunk_transcript_with_timestamps(
    result: YouTubeTranscriptResult,
    chunk_duration_seconds: float = 120.0,  # 2 minute chunks
) -> list[ChunkData]:
    """Chunk transcript by time windows, preserving timestamps."""
    chunks = []
    current_chunk_text = []
    current_chunk_start = 0.0

    for segment in result.segments:
        current_chunk_text.append(segment.text)

        # Check if we've exceeded chunk duration
        if segment.start + segment.duration - current_chunk_start >= chunk_duration_seconds:
            chunk_text = " ".join(current_chunk_text)
            chunks.append(ChunkData(
                content=chunk_text,
                metadata={
                    "start_time": current_chunk_start,
                    "end_time": segment.start + segment.duration,
                    "video_id": result.video_id,
                },
            ))
            current_chunk_text = []
            current_chunk_start = segment.start + segment.duration

    # Don't forget the last chunk
    if current_chunk_text:
        chunks.append(ChunkData(
            content=" ".join(current_chunk_text),
            metadata={
                "start_time": current_chunk_start,
                "end_time": result.duration_seconds,
                "video_id": result.video_id,
            },
        ))

    return chunks
```

3. **Dependencies:**
```toml
# Add to pyproject.toml
"youtube-transcript-api>=0.6.0",
```

**Acceptance Criteria:**
- YouTube URLs are detected and routed to transcript ingestion
- Transcripts are fetched in under 30 seconds for typical videos
- Missing transcripts produce clear error messages
- Transcript chunks include timestamp metadata for deep linking
- Supports language fallback (try en, then any available)

---

### Story 13-3: Migrate to Crawl4AI Library

**Goal:** Replace custom httpx crawler with Crawl4AI's `AsyncWebCrawler` for JavaScript rendering and parallel crawling.

**Technical Approach:**

1. **New Crawl4AI Wrapper:**
```python
# backend/src/agentic_rag_backend/indexing/crawl4ai_crawler.py

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
    CacheMode,
)
from typing import AsyncGenerator, Optional
import structlog

logger = structlog.get_logger(__name__)

class Crawl4AICrawler:
    """Production crawler using Crawl4AI's AsyncWebCrawler."""

    def __init__(
        self,
        headless: bool = True,
        enable_stealth: bool = False,
        proxy_config: Optional[dict] = None,
        max_concurrent: int = 10,
        memory_threshold_percent: float = 70.0,
    ):
        self.browser_config = BrowserConfig(
            headless=headless,
            enable_stealth=enable_stealth,
            proxy_config=proxy_config,
            verbose=False,
        )

        self.dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=memory_threshold_percent,
            max_session_permit=max_concurrent,
            check_interval=1.0,
        )

    async def crawl_single(
        self,
        url: str,
        wait_for: Optional[str] = None,
        js_code: Optional[str] = None,
        cache_mode: CacheMode = CacheMode.ENABLED,
    ) -> CrawledPage:
        """Crawl a single URL with JavaScript rendering."""
        run_config = CrawlerRunConfig(
            cache_mode=cache_mode,
            wait_for=wait_for,
            js_code=[js_code] if js_code else None,
        )

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)

            if not result.success:
                logger.error("crawl_failed", url=url, error=result.error_message)
                return None

            return CrawledPage(
                url=url,
                title=result.metadata.get("title"),
                content=result.markdown.raw_markdown,
                content_hash=compute_content_hash(result.markdown.raw_markdown),
                crawl_timestamp=datetime.now(timezone.utc),
                links=result.links.get("internal", []),
            )

    async def crawl_many(
        self,
        urls: list[str],
        cache_mode: CacheMode = CacheMode.ENABLED,
        stream: bool = True,
    ) -> AsyncGenerator[CrawledPage, None]:
        """Crawl multiple URLs in parallel with memory-adaptive concurrency."""
        run_config = CrawlerRunConfig(
            cache_mode=cache_mode,
            stream=stream,
        )

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            if stream:
                # Streaming mode - yield results as they complete
                async for result in await crawler.arun_many(
                    urls=urls,
                    config=run_config,
                    dispatcher=self.dispatcher,
                ):
                    if result.success:
                        yield CrawledPage(
                            url=result.url,
                            title=result.metadata.get("title"),
                            content=result.markdown.raw_markdown,
                            content_hash=compute_content_hash(result.markdown.raw_markdown),
                            crawl_timestamp=datetime.now(timezone.utc),
                            links=result.links.get("internal", []),
                        )
                    else:
                        logger.warning("crawl_failed",
                            url=result.url,
                            error=result.error_message)
            else:
                # Batch mode - wait for all results
                results = await crawler.arun_many(
                    urls=urls,
                    config=run_config,
                    dispatcher=self.dispatcher,
                )
                for result in results:
                    if result.success:
                        yield CrawledPage(
                            url=result.url,
                            title=result.metadata.get("title"),
                            content=result.markdown.raw_markdown,
                            content_hash=compute_content_hash(result.markdown.raw_markdown),
                            crawl_timestamp=datetime.now(timezone.utc),
                            links=result.links.get("internal", []),
                        )

    async def crawl_site(
        self,
        start_url: str,
        max_depth: int = 3,
        max_pages: int = 100,
        options: CrawlOptions = None,
    ) -> AsyncGenerator[CrawledPage, None]:
        """Crawl a site starting from a URL, following links up to max_depth."""
        options = options or CrawlOptions()
        visited: set[str] = set()
        to_visit: list[tuple[str, int]] = [(start_url, 0)]

        while to_visit and len(visited) < max_pages:
            # Batch URLs at current depth for parallel processing
            batch = []
            current_depth = to_visit[0][1] if to_visit else 0

            while to_visit and to_visit[0][1] == current_depth and len(batch) < 10:
                url, depth = to_visit.pop(0)
                if url not in visited:
                    batch.append(url)
                    visited.add(url)

            if not batch:
                continue

            # Crawl batch in parallel
            async for page in self.crawl_many(batch):
                yield page

                # Queue links for next depth
                if current_depth < max_depth - 1 and options.follow_links:
                    for link in page.links:
                        if link not in visited:
                            if self._should_include_url(link, options):
                                to_visit.append((link, current_depth + 1))

    def _should_include_url(self, url: str, options: CrawlOptions) -> bool:
        """Check if URL matches include/exclude patterns."""
        import re

        if options.include_patterns:
            if not any(re.search(p, url) for p in options.include_patterns):
                return False

        if options.exclude_patterns:
            if any(re.search(p, url) for p in options.exclude_patterns):
                return False

        return True
```

2. **Updated CrawlerService (Bridge Pattern):**
```python
# Update CrawlerService to use Crawl4AI while maintaining API compatibility

class CrawlerService:
    """Crawler service using Crawl4AI with fallback support.

    MIGRATION NOTE: This replaces the httpx-based implementation.
    The old implementation is preserved in _legacy_crawler.py for reference.
    """

    def __init__(
        self,
        profile: str = "fast",
        fallback_providers: list[CrawlProvider] = None,
    ):
        self.profile = get_crawl_profile(profile)
        self.crawler = Crawl4AICrawler(
            headless=self.profile.headless,
            enable_stealth=self.profile.stealth,
            proxy_config=self.profile.proxy_config,
            max_concurrent=self.profile.max_concurrent,
        )
        self.fallback_providers = fallback_providers or []

    async def crawl(
        self,
        start_url: str,
        max_depth: int = 3,
        options: Optional[CrawlOptions] = None,
    ) -> AsyncGenerator[CrawledPage, None]:
        """Crawl a website starting from the given URL."""
        async for page in self.crawler.crawl_site(
            start_url, max_depth=max_depth, options=options
        ):
            yield page
```

3. **Worker Update:**
```python
# Update CrawlWorker to use new crawler

class CrawlWorker:
    def __init__(self, ...):
        # Use profile from job data or default
        profile = os.getenv("CRAWL4AI_PROFILE", "fast")
        self.crawler = CrawlerService(profile=profile)
```

**Acceptance Criteria:**
- JavaScript-rendered content is captured (SPAs, React sites work)
- Multiple URLs crawl in parallel via `arun_many()` with `MemoryAdaptiveDispatcher`
- Caching is enabled (unchanged pages not re-fetched)
- Proxy support is configurable for blocked sites
- Throughput improves 10x (50 pages in <30 seconds vs ~5 minutes sequential)
- Legacy httpx implementation is deprecated (moved to `_legacy_crawler.py`)

---

### Story 13-4: Implement Crawl Configuration Profiles

**Goal:** Pre-configured crawl profiles for common use cases.

**Technical Approach:**

1. **Profile Definitions:**
```python
# backend/src/agentic_rag_backend/indexing/crawl_profiles.py

from dataclasses import dataclass
from typing import Optional
from enum import Enum

class CrawlProfileName(str, Enum):
    FAST = "fast"
    THOROUGH = "thorough"
    STEALTH = "stealth"

@dataclass(frozen=True)
class CrawlProfile:
    """Pre-configured crawl settings for common use cases."""

    name: str
    description: str

    # Browser settings
    headless: bool
    stealth: bool

    # Concurrency settings
    max_concurrent: int
    rate_limit: float  # requests per second

    # Wait settings
    wait_for: Optional[str]  # CSS selector to wait for
    wait_timeout: float  # seconds

    # Proxy settings
    proxy_config: Optional[dict]

    # Cache settings
    cache_enabled: bool

# Pre-defined profiles
CRAWL_PROFILES: dict[str, CrawlProfile] = {
    "fast": CrawlProfile(
        name="fast",
        description="High-speed crawling for static documentation sites",
        headless=True,
        stealth=False,
        max_concurrent=10,
        rate_limit=5.0,
        wait_for=None,
        wait_timeout=5.0,
        proxy_config=None,
        cache_enabled=True,
    ),

    "thorough": CrawlProfile(
        name="thorough",
        description="Comprehensive crawling for complex SPAs with dynamic content",
        headless=True,
        stealth=False,
        max_concurrent=5,
        rate_limit=2.0,
        wait_for="css:body",  # Wait for body to be present
        wait_timeout=15.0,
        proxy_config=None,
        cache_enabled=True,
    ),

    "stealth": CrawlProfile(
        name="stealth",
        description="Bot-protected sites with fingerprint evasion and proxy rotation",
        headless=False,  # Non-headless is harder to detect
        stealth=True,
        max_concurrent=3,
        rate_limit=0.5,
        wait_for=None,
        wait_timeout=30.0,
        proxy_config=None,  # Set via env vars
        cache_enabled=False,  # Fresh requests each time
    ),
}

def get_crawl_profile(name: str) -> CrawlProfile:
    """Get a crawl profile by name."""
    if name not in CRAWL_PROFILES:
        raise ValueError(
            f"Unknown crawl profile: {name}. "
            f"Available: {list(CRAWL_PROFILES.keys())}"
        )
    return CRAWL_PROFILES[name]

def get_profile_for_url(url: str) -> CrawlProfile:
    """Automatically select profile based on URL characteristics."""
    # Simple heuristics - could be enhanced with ML
    known_spa_domains = ["app.", "dashboard.", "console."]
    known_protected_domains = ["linkedin.com", "twitter.com", "x.com"]

    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if any(d in domain for d in known_protected_domains):
        return CRAWL_PROFILES["stealth"]

    if any(d in domain for d in known_spa_domains):
        return CRAWL_PROFILES["thorough"]

    return CRAWL_PROFILES["fast"]
```

2. **API Integration:**
```python
# Update CrawlRequest model to accept profile

class CrawlRequest(BaseModel):
    url: HttpUrl
    tenant_id: UUID
    max_depth: int = Field(default=3, ge=1, le=10)
    profile: Optional[CrawlProfileName] = Field(
        default=None,
        description="Crawl profile: fast, thorough, or stealth. Auto-detected if not set."
    )
    options: CrawlOptions = Field(default_factory=CrawlOptions)
```

3. **Configuration:**
```python
# New settings in config.py
CRAWL4AI_PROFILE: str = "fast"           # Default profile
CRAWL4AI_MAX_CONCURRENT: int = 10        # Override max concurrent
CRAWL4AI_STEALTH_PROXY: Optional[str]    # Proxy for stealth mode
```

**Acceptance Criteria:**
- Three profiles available: fast, thorough, stealth
- Profiles are selectable via `CRAWL4AI_PROFILE` env var or API parameter
- Profile settings are logged for debugging
- Auto-detection suggests appropriate profile based on URL

---

## Dependencies & Configuration

### New Python Dependencies

```toml
# Add to pyproject.toml dependencies
"youtube-transcript-api>=0.6.0",  # YouTube transcripts
# crawl4ai>=0.3.0 already exists
```

### New Environment Variables

```bash
# Story 13-1: Fallback Providers
CRAWL_FALLBACK_ENABLED=true
CRAWL_FALLBACK_PROVIDERS=["apify", "brightdata"]  # JSON array
APIFY_API_TOKEN=apify_api_xxxx
BRIGHTDATA_USERNAME=brd-customer-xxx
BRIGHTDATA_PASSWORD=xxxx
BRIGHTDATA_ZONE=scraping_browser

# Story 13-2: YouTube Ingestion
YOUTUBE_PREFERRED_LANGUAGES=["en", "en-US"]  # JSON array
YOUTUBE_CHUNK_DURATION_SECONDS=120

# Story 13-3 & 13-4: Crawl4AI Configuration
CRAWL4AI_PROFILE=fast  # fast, thorough, stealth
CRAWL4AI_MAX_CONCURRENT=10
CRAWL4AI_MEMORY_THRESHOLD_PERCENT=70
CRAWL4AI_STEALTH_PROXY=http://user:pass@proxy:port
CRAWL4AI_CACHE_ENABLED=true
```

### Config Settings Updates

```python
# Add to Settings dataclass in config.py

# Story 13-1
crawl_fallback_enabled: bool
crawl_fallback_providers: list[str]
apify_api_token: Optional[str]
brightdata_username: Optional[str]
brightdata_password: Optional[str]
brightdata_zone: str

# Story 13-2
youtube_preferred_languages: list[str]
youtube_chunk_duration_seconds: int

# Story 13-3 & 13-4
crawl4ai_profile: str
crawl4ai_max_concurrent: int
crawl4ai_memory_threshold_percent: float
crawl4ai_stealth_proxy: Optional[str]
crawl4ai_cache_enabled: bool
```

---

## Testing Strategy

### Unit Tests

```python
# backend/tests/test_crawl4ai_crawler.py

import pytest
from agentic_rag_backend.indexing.crawl4ai_crawler import Crawl4AICrawler

class TestCrawl4AICrawler:
    @pytest.mark.asyncio
    async def test_crawl_single_static_page(self):
        """Test crawling a simple static page."""
        crawler = Crawl4AICrawler()
        result = await crawler.crawl_single("https://example.com")
        assert result is not None
        assert "Example Domain" in result.title

    @pytest.mark.asyncio
    async def test_crawl_many_parallel(self):
        """Test parallel crawling performance."""
        crawler = Crawl4AICrawler(max_concurrent=5)
        urls = [f"https://example.com/page{i}" for i in range(10)]
        results = []
        async for page in crawler.crawl_many(urls):
            results.append(page)
        # Should complete in reasonable time (not 10x sequential)

# backend/tests/test_youtube_ingestion.py

class TestYouTubeIngestion:
    @pytest.mark.asyncio
    async def test_extract_video_id_watch_url(self):
        """Test video ID extraction from watch URL."""
        service = YouTubeIngestionService()
        video_id = service.extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        assert video_id == "dQw4w9WgXcQ"

    @pytest.mark.asyncio
    async def test_fetch_transcript_success(self):
        """Test fetching transcript for video with subtitles."""
        service = YouTubeIngestionService()
        # Use a known video with transcripts
        result = await service.fetch_transcript("dQw4w9WgXcQ")
        assert result.full_text
        assert len(result.segments) > 0
```

### Integration Tests

```python
# backend/tests/integration/test_crawl_pipeline.py

@pytest.mark.integration
class TestCrawlPipeline:
    @pytest.mark.asyncio
    async def test_crawl_with_fallback(self):
        """Test fallback to alternative provider."""
        # Mock primary failure, verify fallback is used
        ...

    @pytest.mark.asyncio
    async def test_crawl_profiles_applied(self):
        """Test that profile settings are correctly applied."""
        ...
```

### Performance Benchmarks

```python
# backend/tests/benchmarks/test_crawl_performance.py

@pytest.mark.benchmark
class TestCrawlPerformance:
    @pytest.mark.asyncio
    async def test_parallel_vs_sequential(self):
        """Verify 10x improvement in parallel crawling."""
        urls = [...]  # 50 test URLs

        # Parallel with Crawl4AI
        start = time.perf_counter()
        async for _ in crawler.crawl_many(urls):
            pass
        parallel_time = time.perf_counter() - start

        # Should complete 50 pages in under 30 seconds
        assert parallel_time < 30.0
```

---

## Migration Plan

### Phase 1: Add New Modules (Story 13-3)
1. Create `crawl4ai_crawler.py` with new implementation
2. Create `crawl_profiles.py` with profile definitions
3. Keep existing `crawler.py` unchanged
4. Add feature flag `USE_CRAWL4AI=false` (default off)

### Phase 2: Parallel Testing
1. Enable `USE_CRAWL4AI=true` in staging
2. Run both crawlers in shadow mode
3. Compare results (content quality, speed, success rates)
4. Monitor memory usage with `MemoryAdaptiveDispatcher`

### Phase 3: Gradual Rollout
1. Enable for new crawl jobs only
2. Monitor error rates and fallback usage
3. Add Stories 13-1 (fallback) and 13-4 (profiles)
4. Complete Story 13-2 (YouTube) in parallel

### Phase 4: Deprecation
1. Move old `crawler.py` to `_legacy_crawler.py`
2. Update all imports to use new crawler
3. Remove feature flag, new crawler is default
4. Schedule removal of legacy code in Epic 14+

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Crawl4AI browser memory issues | High | Use MemoryAdaptiveDispatcher, configure thresholds |
| Fallback provider costs | Medium | Log usage, set budget alerts, rate limit |
| YouTube API rate limits | Medium | Implement backoff, cache transcripts |
| Breaking change in CrawlerService API | Medium | Use bridge pattern, deprecation warnings |
| Stealth mode detection bypass failure | Low | Multiple fallback providers, manual review |

---

## Out of Scope

- Full video download/transcription (Whisper) - covered by transcript API
- Image extraction from PDFs - existing Docling handles this
- Custom JavaScript execution per-site - future enhancement
- Distributed crawling across multiple nodes - future scaling epic

---

## References

- [Crawl4AI Documentation](https://github.com/unclecode/crawl4ai)
- [youtube-transcript-api Documentation](https://github.com/jdepoix/youtube-transcript-api)
- [Apify Web Scraper](https://apify.com/apify/web-scraper)
- [BrightData Scraping Browser](https://brightdata.com/products/scraping-browser)
- [Architecture Document](/_bmad-output/architecture.md)
- [Epic 4 Tech Spec](./epic-4-tech-spec.md) - Original ingestion pipeline
