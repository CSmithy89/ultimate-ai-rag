"""Crawl4AI-powered web crawler for autonomous documentation site crawling.

This module provides a high-performance web crawler using the Crawl4AI library,
which enables:
- JavaScript-rendered content capture (SPAs, React sites work)
- Parallel URL crawling via arun_many() with MemoryAdaptiveDispatcher
- Intelligent caching (unchanged pages not re-fetched)
- Proxy support for blocked sites
- 10x throughput improvement (50 pages in <30 seconds)

Story 13.3: Migrated from custom httpx crawler to Crawl4AI library.
"""

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional
from urllib.parse import urljoin, urlparse

import structlog

from agentic_rag_backend.core.errors import InvalidUrlError
from agentic_rag_backend.models.documents import CrawledPage
from agentic_rag_backend.models.ingest import CrawlOptions

# Crawl4AI imports
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher, SemaphoreDispatcher

    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    AsyncWebCrawler = None  # type: ignore
    BrowserConfig = None  # type: ignore
    CrawlerRunConfig = None  # type: ignore
    CacheMode = None  # type: ignore
    MemoryAdaptiveDispatcher = None  # type: ignore
    SemaphoreDispatcher = None  # type: ignore


logger = structlog.get_logger(__name__)

# Default configuration
DEFAULT_RATE_LIMIT = 1.0  # requests per second
DEFAULT_MAX_DEPTH = 3
DEFAULT_TIMEOUT = 60000  # milliseconds (Crawl4AI uses ms)
DEFAULT_USER_AGENT = (
    "AgenticRAG-Crawler/1.0 (+https://github.com/example/agentic-rag)"
)
DEFAULT_MAX_CONCURRENT = 10  # Maximum concurrent crawl sessions
DEFAULT_JS_WAIT_SECONDS = 2.0  # Wait for JavaScript to render
HTML_MARKDOWN_THREAD_THRESHOLD = 100_000


def compute_content_hash(content: str) -> str:
    """
    Compute SHA-256 hash of content for deduplication.

    Args:
        content: Content string to hash

    Returns:
        Hexadecimal hash string (64 characters)
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def is_valid_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL string to validate

    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def normalize_url(url: str, base_url: str) -> Optional[str]:
    """
    Normalize a URL relative to a base URL.

    Args:
        url: URL to normalize (may be relative)
        base_url: Base URL for resolving relative URLs

    Returns:
        Normalized absolute URL, or None if invalid
    """
    try:
        # Handle relative URLs
        if not url.startswith(("http://", "https://")):
            url = urljoin(base_url, url)

        parsed = urlparse(url)

        # Only allow http/https
        if parsed.scheme not in ("http", "https"):
            return None

        # Remove fragments
        url = url.split("#")[0]

        # Remove trailing slashes for consistency
        url = url.rstrip("/")

        return url
    except Exception:
        return None


def is_same_domain(url1: str, url2: str) -> bool:
    """
    Check if two URLs are on the same domain.

    Args:
        url1: First URL
        url2: Second URL

    Returns:
        True if URLs share the same domain
    """
    try:
        parsed1 = urlparse(url1)
        parsed2 = urlparse(url2)
        return parsed1.netloc == parsed2.netloc
    except Exception:
        return False


def extract_links_from_html(html: str, base_url: str) -> list[str]:
    """
    Extract links from HTML content.

    Args:
        html: HTML content
        base_url: Base URL for resolving relative links

    Returns:
        List of normalized absolute URLs
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if not href or not isinstance(href, str):
            continue
        normalized = normalize_url(href, base_url)
        if normalized and is_same_domain(normalized, base_url):
            links.append(normalized)

    return list(set(links))  # Remove duplicates


def extract_links_from_markdown(markdown: str, base_url: str) -> list[str]:
    """
    Extract links from markdown content.

    Args:
        markdown: Markdown content
        base_url: Base URL for resolving relative links

    Returns:
        List of normalized absolute URLs
    """
    # Match markdown links: [text](url) and bare URLs
    link_pattern = r'\[([^\]]*)\]\(([^)]+)\)'
    links = []

    for match in re.finditer(link_pattern, markdown):
        url = match.group(2)
        normalized = normalize_url(url, base_url)
        if normalized and is_same_domain(normalized, base_url):
            links.append(normalized)

    return list(set(links))


def extract_title_from_markdown(markdown: str) -> Optional[str]:
    """
    Extract title from markdown content.

    Args:
        markdown: Markdown content

    Returns:
        Title string (first H1), or None if not found
    """
    # Look for first H1 heading
    h1_pattern = r'^#\s+(.+)$'
    match = re.search(h1_pattern, markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def extract_title_from_html(html: str) -> Optional[str]:
    """
    Extract title from HTML content.

    Args:
        html: HTML content

    Returns:
        Title string, or None if not found
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None


class CrawlerService:
    """
    Crawl4AI-powered web crawler service.

    Provides high-performance crawling with:
    - JavaScript rendering for SPAs and dynamic content
    - Parallel crawling via MemoryAdaptiveDispatcher
    - Intelligent caching to avoid re-fetching unchanged pages
    - Proxy support for accessing blocked sites
    - Automatic markdown conversion

    Story 13.3: Migrated from httpx to Crawl4AI for 10x throughput improvement.
    """

    def __init__(
        self,
        headless: bool = True,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        cache_enabled: bool = True,
        proxy_url: Optional[str] = None,
        js_wait_seconds: float = DEFAULT_JS_WAIT_SECONDS,
        page_timeout_ms: int = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize Crawl4AI crawler service.

        Args:
            headless: Run browser in headless mode (default: True)
            max_concurrent: Maximum concurrent crawl sessions (default: 10)
            cache_enabled: Enable caching for unchanged pages (default: True)
            proxy_url: Optional proxy URL (format: "http://user:pass@host:port")
            js_wait_seconds: Seconds to wait for JavaScript rendering (default: 2.0)
            page_timeout_ms: Page load timeout in milliseconds (default: 60000)
        """
        if not CRAWL4AI_AVAILABLE:
            raise ImportError(
                "Crawl4AI is not installed. Install with: pip install crawl4ai"
            )

        self.headless = headless
        self.max_concurrent = max_concurrent
        self.cache_enabled = cache_enabled
        self.proxy_url = proxy_url
        self.js_wait_seconds = js_wait_seconds
        self.page_timeout_ms = page_timeout_ms

        self._crawler: Optional[AsyncWebCrawler] = None
        self._browser_config: Optional[BrowserConfig] = None

    def _create_browser_config(self) -> "BrowserConfig":
        """Create BrowserConfig for Crawl4AI."""
        config_kwargs = {
            "headless": self.headless,
            "verbose": False,
        }

        # Add proxy configuration if provided
        if self.proxy_url:
            config_kwargs["proxy_config"] = {"server": self.proxy_url}

        return BrowserConfig(**config_kwargs)

    def _create_crawler_config(
        self,
        wait_for_js: bool = True,
    ) -> "CrawlerRunConfig":
        """
        Create CrawlerRunConfig for a crawl operation.

        Args:
            wait_for_js: Whether to wait for JavaScript to render

        Returns:
            CrawlerRunConfig instance
        """
        # Determine cache mode
        if self.cache_enabled:
            cache_mode = CacheMode.ENABLED
        else:
            cache_mode = CacheMode.BYPASS

        config_kwargs = {
            "cache_mode": cache_mode,
            "page_timeout": self.page_timeout_ms,
            "verbose": False,
        }

        # Add JavaScript wait if needed
        if wait_for_js and self.js_wait_seconds > 0:
            # Wait for document to be fully loaded
            config_kwargs["delay_before_return_html"] = self.js_wait_seconds

        return CrawlerRunConfig(**config_kwargs)

    async def __aenter__(self) -> "CrawlerService":
        """Async context manager entry - initialize browser."""
        self._browser_config = self._create_browser_config()
        self._crawler = AsyncWebCrawler(config=self._browser_config)
        await self._crawler.__aenter__()
        logger.info(
            "crawler_initialized",
            headless=self.headless,
            max_concurrent=self.max_concurrent,
            cache_enabled=self.cache_enabled,
            proxy_configured=self.proxy_url is not None,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup browser."""
        if self._crawler:
            await self._crawler.__aexit__(exc_type, exc_val, exc_tb)
            self._crawler = None
        logger.info("crawler_shutdown")

    def _convert_result_to_crawled_page(
        self,
        result,
        depth: int = 0,
    ) -> Optional[CrawledPage]:
        """
        Convert Crawl4AI result to CrawledPage model.

        Args:
            result: Crawl4AI CrawlResult object
            depth: Crawl depth for this page

        Returns:
            CrawledPage if successful, None otherwise
        """
        if not result.success:
            logger.warning(
                "crawl_result_failed",
                url=result.url,
                error=result.error_message,
            )
            return None

        # Get markdown content (Crawl4AI provides this directly)
        # The result.markdown can be a MarkdownGenerationResult object or string
        markdown_content = ""
        if hasattr(result, 'markdown'):
            if hasattr(result.markdown, 'raw_markdown'):
                markdown_content = result.markdown.raw_markdown
            elif isinstance(result.markdown, str):
                markdown_content = result.markdown
            else:
                markdown_content = str(result.markdown) if result.markdown else ""

        if not markdown_content:
            # Fallback: if no markdown, we might have HTML
            if hasattr(result, 'html') and result.html:
                logger.debug("no_markdown_falling_back_to_html", url=result.url)
                # Return None for now - we expect Crawl4AI to provide markdown
                return None
            logger.warning("no_content_extracted", url=result.url)
            return None

        # Extract title from markdown or HTML
        title = extract_title_from_markdown(markdown_content)
        if not title and hasattr(result, 'html') and result.html:
            title = extract_title_from_html(result.html)

        # Extract links from HTML if available, otherwise from markdown
        links = []
        if hasattr(result, 'html') and result.html:
            links = extract_links_from_html(result.html, result.url)
        else:
            links = extract_links_from_markdown(markdown_content, result.url)

        # Compute content hash for deduplication
        content_hash = compute_content_hash(markdown_content)

        return CrawledPage(
            url=result.url,
            title=title,
            content=markdown_content,
            content_hash=content_hash,
            crawl_timestamp=datetime.now(timezone.utc),
            depth=depth,
            links=links,
        )

    async def crawl_page(self, url: str) -> Optional[CrawledPage]:
        """
        Crawl a single page using Crawl4AI.

        Args:
            url: URL to crawl

        Returns:
            CrawledPage if successful, None otherwise
        """
        if not is_valid_url(url):
            logger.warning("invalid_url", url=url)
            return None

        if not self._crawler:
            raise RuntimeError(
                "CrawlerService must be used as async context manager"
            )

        try:
            config = self._create_crawler_config(wait_for_js=True)
            result = await self._crawler.arun(url=url, config=config)
            return self._convert_result_to_crawled_page(result, depth=0)

        except Exception as e:
            logger.error("crawl_page_error", url=url, error=str(e))
            return None

    async def crawl_many(
        self,
        urls: list[str],
        stream: bool = False,
    ) -> AsyncGenerator[CrawledPage, None]:
        """
        Crawl multiple URLs in parallel using Crawl4AI's arun_many().

        Uses MemoryAdaptiveDispatcher for intelligent parallel crawling
        that adapts to available system resources.

        Args:
            urls: List of URLs to crawl
            stream: If True, yield results as they complete. If False, wait for all.

        Yields:
            CrawledPage objects for each successfully crawled page
        """
        if not self._crawler:
            raise RuntimeError(
                "CrawlerService must be used as async context manager"
            )

        # Filter valid URLs
        valid_urls = [url for url in urls if is_valid_url(url)]
        invalid_count = len(urls) - len(valid_urls)
        if invalid_count > 0:
            logger.warning("invalid_urls_skipped", count=invalid_count)

        if not valid_urls:
            return

        # Create dispatcher for parallel crawling
        dispatcher = MemoryAdaptiveDispatcher(
            max_session_permit=self.max_concurrent,
            memory_threshold_percent=85.0,
            check_interval=0.5,
        )

        # Create config
        config = self._create_crawler_config(wait_for_js=True)
        # Clone config with stream setting
        if hasattr(config, 'clone'):
            config = config.clone(stream=stream)
        else:
            # Fallback: create new config with stream
            config = CrawlerRunConfig(
                cache_mode=config.cache_mode,
                page_timeout=self.page_timeout_ms,
                delay_before_return_html=self.js_wait_seconds if self.js_wait_seconds > 0 else 0.1,
                verbose=False,
                stream=stream,
            )

        logger.info(
            "crawl_many_started",
            url_count=len(valid_urls),
            max_concurrent=self.max_concurrent,
            stream=stream,
        )

        try:
            if stream:
                # Streaming mode - yield results as they complete
                async for result in await self._crawler.arun_many(
                    urls=valid_urls,
                    config=config,
                    dispatcher=dispatcher,
                ):
                    page = self._convert_result_to_crawled_page(result, depth=0)
                    if page:
                        yield page
            else:
                # Batch mode - wait for all results
                results = await self._crawler.arun_many(
                    urls=valid_urls,
                    config=config,
                    dispatcher=dispatcher,
                )
                for result in results:
                    page = self._convert_result_to_crawled_page(result, depth=0)
                    if page:
                        yield page

        except Exception as e:
            logger.error("crawl_many_error", error=str(e))
            raise

        logger.info("crawl_many_completed", url_count=len(valid_urls))

    async def crawl(
        self,
        start_url: str,
        max_depth: int = DEFAULT_MAX_DEPTH,
        options: Optional[CrawlOptions] = None,
    ) -> AsyncGenerator[CrawledPage, None]:
        """
        Crawl a website starting from the given URL.

        Performs breadth-first crawling up to max_depth levels,
        using parallel crawling for discovered links.

        Args:
            start_url: Starting URL for the crawl
            max_depth: Maximum depth to crawl (1 = start page only)
            options: Optional crawl configuration

        Yields:
            CrawledPage objects for each successfully crawled page
        """
        if not is_valid_url(start_url):
            raise InvalidUrlError(start_url, "URL is not valid")

        if not self._crawler:
            raise RuntimeError(
                "CrawlerService must be used as async context manager"
            )

        options = options or CrawlOptions()

        visited: set[str] = set()
        current_level: list[str] = [start_url]

        logger.info(
            "crawl_started",
            start_url=start_url,
            max_depth=max_depth,
        )

        for depth in range(max_depth):
            # Filter out already visited URLs
            urls_to_crawl = []
            for url in current_level:
                normalized = normalize_url(url, start_url)
                if normalized and normalized not in visited:
                    visited.add(normalized)
                    urls_to_crawl.append(normalized)

            if not urls_to_crawl:
                break

            logger.info(
                "crawl_depth_started",
                depth=depth,
                url_count=len(urls_to_crawl),
            )

            # Crawl all URLs at this depth in parallel
            next_level: list[str] = []

            async for page in self.crawl_many(urls_to_crawl, stream=True):
                page.depth = depth
                yield page

                # Collect links for next depth if we haven't reached max depth
                if depth < max_depth - 1 and options.follow_links:
                    for link in page.links:
                        if link not in visited:
                            # Check include/exclude patterns
                            should_include = True

                            if options.include_patterns:
                                should_include = any(
                                    re.search(pattern, link)
                                    for pattern in options.include_patterns
                                )

                            if should_include and options.exclude_patterns:
                                should_include = not any(
                                    re.search(pattern, link)
                                    for pattern in options.exclude_patterns
                                )

                            if should_include:
                                next_level.append(link)

            current_level = list(set(next_level))  # Deduplicate

        logger.info(
            "crawl_completed",
            start_url=start_url,
            pages_crawled=len(visited),
        )


async def crawl_url(
    url: str,
    max_depth: int = DEFAULT_MAX_DEPTH,
    rate_limit: float = DEFAULT_RATE_LIMIT,
    options: Optional[CrawlOptions] = None,
    headless: bool = True,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    cache_enabled: bool = True,
    proxy_url: Optional[str] = None,
    js_wait_seconds: float = DEFAULT_JS_WAIT_SECONDS,
) -> AsyncGenerator[CrawledPage, None]:
    """
    Convenience function for crawling a URL using Crawl4AI.

    Args:
        url: URL to crawl
        max_depth: Maximum crawl depth
        rate_limit: Requests per second (legacy parameter, not used with Crawl4AI)
        options: Optional crawl configuration
        headless: Run browser in headless mode
        max_concurrent: Maximum concurrent sessions
        cache_enabled: Enable caching
        proxy_url: Optional proxy URL
        js_wait_seconds: Seconds to wait for JavaScript rendering

    Yields:
        CrawledPage objects for each crawled page
    """
    if options is None:
        options = CrawlOptions(rate_limit=rate_limit)

    async with CrawlerService(
        headless=headless,
        max_concurrent=max_concurrent,
        cache_enabled=cache_enabled,
        proxy_url=proxy_url,
        js_wait_seconds=js_wait_seconds,
    ) as crawler:
        async for page in crawler.crawl(url, max_depth=max_depth, options=options):
            yield page


# Legacy compatibility aliases
extract_links = extract_links_from_html
extract_title = extract_title_from_html
