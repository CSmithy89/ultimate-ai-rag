"""Crawl4AI-powered web crawler for autonomous documentation site crawling.

This module provides a high-performance web crawler using the Crawl4AI library,
which enables:
- JavaScript-rendered content capture (SPAs, React sites work)
- Parallel URL crawling via arun_many() with MemoryAdaptiveDispatcher
- Intelligent caching (unchanged pages not re-fetched)
- Proxy support for blocked sites
- 10x throughput improvement (50 pages in <30 seconds)
- Profile-based configuration (fast, thorough, stealth)

Story 13.3: Migrated from custom httpx crawler to Crawl4AI library.
Story 13.4: Added crawl configuration profiles for different scenarios.
"""

import asyncio
import hashlib
import ipaddress
import os
import re
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional
from urllib.parse import urljoin, urlparse

import structlog

from agentic_rag_backend.core.errors import InvalidUrlError
from agentic_rag_backend.models.documents import CrawledPage
from agentic_rag_backend.models.ingest import CrawlOptions
from agentic_rag_backend.indexing.crawl_profiles import (
    CrawlProfile,
    get_crawl_profile,
    get_profile_for_url,
    apply_proxy_override,
)

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
DEFAULT_MAX_PAGES = 1000  # Maximum pages to crawl in a single session


def sanitize_proxy_url(proxy_url: Optional[str]) -> Optional[str]:
    """
    Remove credentials from proxy URL for safe logging.

    Args:
        proxy_url: Proxy URL that may contain credentials

    Returns:
        Sanitized URL with credentials replaced by '***:***', or None
    """
    if not proxy_url:
        return None
    try:
        parsed = urlparse(proxy_url)
        if parsed.username or parsed.password:
            # Reconstruct URL without credentials
            host_port = parsed.hostname or ""
            if parsed.port:
                host_port = f"{host_port}:{parsed.port}"
            return f"{parsed.scheme}://***:***@{host_port}"
        return proxy_url
    except Exception:
        return "***"


def compute_content_hash(content: str) -> str:
    """
    Compute SHA-256 hash of content for deduplication.

    Args:
        content: Content string to hash

    Returns:
        Hexadecimal hash string (64 characters)
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def is_valid_url(url: str, allow_private: bool = False) -> bool:
    """
    Validate URL format and optionally reject private/internal IPs (SSRF protection).

    Args:
        url: URL string to validate
        allow_private: If False (default), reject private IP ranges for SSRF protection

    Returns:
        True if URL is valid and safe, False otherwise
    """
    try:
        result = urlparse(url)
        if result.scheme not in ("http", "https"):
            return False
        if not result.netloc:
            return False

        hostname = result.hostname
        if not hostname:
            return False

        # SSRF protection: reject private/internal IPs unless explicitly allowed
        if not allow_private:
            # Reject localhost variants
            if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
                logger.warning("ssrf_blocked_localhost", url=url, hostname=hostname)
                return False

            # Check if hostname is an IP address and reject private ranges
            try:
                ip = ipaddress.ip_address(hostname)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    logger.warning(
                        "ssrf_blocked_private_ip",
                        url=url,
                        hostname=hostname,
                        ip_type="private" if ip.is_private else "reserved",
                    )
                    return False
            except ValueError:
                # Not an IP address, likely a domain name - that's fine
                pass

        return True
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
    # Match markdown links: [text](url)
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
    - Profile-based configuration (fast, thorough, stealth)

    Story 13.3: Migrated from httpx to Crawl4AI for 10x throughput improvement.
    Story 13.4: Added crawl configuration profiles for different scenarios.
    """

    def __init__(
        self,
        headless: bool = True,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        cache_enabled: bool = True,
        proxy_url: Optional[str] = None,
        js_wait_seconds: float = DEFAULT_JS_WAIT_SECONDS,
        page_timeout_ms: int = DEFAULT_TIMEOUT,
        profile: Optional[CrawlProfile] = None,
        stealth: bool = False,
        wait_for: Optional[str] = None,
        wait_timeout: float = 10.0,
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
            profile: Optional CrawlProfile to apply (overrides individual settings)
            stealth: Enable stealth mode for anti-detection (default: False)
            wait_for: CSS selector or JS expression to wait for before capturing
            wait_timeout: Timeout in seconds for wait_for condition (default: 10.0)
        """
        if not CRAWL4AI_AVAILABLE:
            raise ImportError(
                "Crawl4AI is not installed. Install with: pip install crawl4ai"
            )

        # If a profile is provided, use its settings
        if profile is not None:
            self.headless = profile.headless
            self.max_concurrent = profile.max_concurrent
            self.cache_enabled = profile.cache_enabled
            self.proxy_url = profile.proxy_config
            self.stealth = profile.stealth
            self.wait_for = profile.wait_for
            self.wait_timeout = profile.wait_timeout
            # Use profile rate limit to calculate js_wait_seconds if not explicitly set
            # Guard against division by zero with max(0.1, rate_limit)
            safe_rate_limit = max(0.1, profile.rate_limit)
            self.js_wait_seconds = js_wait_seconds if js_wait_seconds != DEFAULT_JS_WAIT_SECONDS else max(1.0, 1.0 / safe_rate_limit)
            self.profile_name = profile.name
            logger.info(
                "crawler_profile_applied",
                profile=profile.name,
                description=profile.description,
                headless=self.headless,
                stealth=self.stealth,
                max_concurrent=self.max_concurrent,
                cache_enabled=self.cache_enabled,
                wait_for=self.wait_for,
                proxy_configured=sanitize_proxy_url(self.proxy_url),
            )
        else:
            self.headless = headless
            self.max_concurrent = max_concurrent
            self.cache_enabled = cache_enabled
            self.proxy_url = proxy_url
            self.js_wait_seconds = js_wait_seconds
            self.stealth = stealth
            self.wait_for = wait_for
            self.wait_timeout = wait_timeout
            self.profile_name = None

        self.page_timeout_ms = page_timeout_ms

        # Validate and auto-clamp timeout configuration
        # wait_timeout must be less than page_timeout to avoid guaranteed failures
        max_wait_timeout_s = (self.page_timeout_ms - 1000) / 1000  # Leave 1s buffer
        if self.wait_timeout > max_wait_timeout_s:
            original_wait_timeout = self.wait_timeout
            self.wait_timeout = max(1.0, max_wait_timeout_s)  # Clamp to safe value
            logger.warning(
                "wait_timeout_clamped",
                original_wait_timeout_s=original_wait_timeout,
                clamped_wait_timeout_s=self.wait_timeout,
                page_timeout_ms=self.page_timeout_ms,
                reason="wait_timeout exceeded page_timeout, auto-clamped",
            )

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

        # Add stealth mode configuration if enabled
        # Stealth mode helps bypass bot detection
        if self.stealth:
            # Use current Chrome user agent to appear more like a real browser
            # Chrome 130+ is recommended to avoid bot detection flags
            user_agent = os.getenv(
                "CRAWL4AI_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            )
            config_kwargs["user_agent"] = user_agent
            logger.debug("stealth_mode_enabled", headless=self.headless, user_agent=user_agent[:50])

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

        # Add wait_for condition if specified (from profile or explicit setting)
        if self.wait_for:
            config_kwargs["wait_for"] = self.wait_for
            # Pass wait_for_timeout in milliseconds to CrawlerRunConfig
            if self.wait_timeout > 0:
                config_kwargs["wait_for_timeout"] = int(self.wait_timeout * 1000)
                config_kwargs["wait_until"] = "domcontentloaded"
            logger.debug(
                "wait_for_condition_set",
                wait_for=self.wait_for,
                wait_timeout_ms=int(self.wait_timeout * 1000),
            )

        return CrawlerRunConfig(**config_kwargs)

    async def __aenter__(self) -> "CrawlerService":
        """Async context manager entry - initialize browser."""
        self._browser_config = self._create_browser_config()
        self._crawler = AsyncWebCrawler(config=self._browser_config)
        await self._crawler.__aenter__()
        logger.info(
            "crawler_initialized",
            profile=self.profile_name,
            headless=self.headless,
            stealth=self.stealth,
            max_concurrent=self.max_concurrent,
            cache_enabled=self.cache_enabled,
            proxy_configured=self.proxy_url is not None,
            wait_for=self.wait_for,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup browser with robust error handling."""
        if self._crawler:
            try:
                await self._crawler.__aexit__(exc_type, exc_val, exc_tb)
                logger.info(
                    "crawler_browser_closed",
                    exc_occurred=exc_type is not None,
                    profile=self.profile_name,
                )
            except Exception as e:
                logger.error(
                    "crawler_cleanup_error",
                    error=str(e),
                    profile=self.profile_name,
                )
            finally:
                self._crawler = None
        logger.info("crawler_shutdown", profile=self.profile_name)

    def _convert_result_to_crawled_page(
        self,
        result,
        depth: int = 0,
        tenant_id: Optional[str] = None,
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
            # Fallback: if no markdown, try to convert HTML
            if hasattr(result, 'html') and result.html:
                logger.debug("no_markdown_converting_from_html", url=result.url)
                try:
                    from markdownify import markdownify
                    markdown_content = markdownify(result.html, heading_style="ATX")
                    if not markdown_content or not markdown_content.strip():
                        logger.warning("html_to_markdown_empty", url=result.url)
                        return None
                except Exception as e:
                    logger.warning(
                        "html_to_markdown_failed",
                        url=result.url,
                        error=str(e),
                    )
                    return None
            else:
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
            tenant_id=tenant_id,
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
        max_pages: int = DEFAULT_MAX_PAGES,
        tenant_id: Optional[str] = None,
    ) -> AsyncGenerator[CrawledPage, None]:
        """
        Crawl multiple URLs in parallel using Crawl4AI's arun_many().

        Uses MemoryAdaptiveDispatcher for intelligent parallel crawling
        that adapts to available system resources.

        Args:
            urls: List of URLs to crawl
            stream: If True, yield results as they complete. If False, wait for all.
            max_pages: Maximum number of pages to crawl (memory safeguard)
            tenant_id: Optional tenant ID for multi-tenancy tracking

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

        # Enforce max_pages limit to prevent unbounded memory usage
        if len(valid_urls) > max_pages:
            logger.warning(
                "crawl_urls_truncated",
                requested=len(valid_urls),
                max_pages=max_pages,
            )
            valid_urls = valid_urls[:max_pages]

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
            # Fallback: create new config with stream, preserving all settings
            fallback_kwargs = {
                "cache_mode": config.cache_mode,
                "page_timeout": self.page_timeout_ms,
                "delay_before_return_html": self.js_wait_seconds if self.js_wait_seconds > 0 else 0.1,
                "verbose": False,
                "stream": stream,
            }
            # Preserve wait_for settings from original config
            if self.wait_for:
                fallback_kwargs["wait_for"] = self.wait_for
                if self.wait_timeout > 0:
                    fallback_kwargs["wait_for_timeout"] = int(self.wait_timeout * 1000)
                    fallback_kwargs["wait_until"] = "domcontentloaded"
            config = CrawlerRunConfig(**fallback_kwargs)

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
                    page = self._convert_result_to_crawled_page(result, depth=0, tenant_id=tenant_id)
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
                    page = self._convert_result_to_crawled_page(result, depth=0, tenant_id=tenant_id)
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
        max_pages: int = DEFAULT_MAX_PAGES,
        options: Optional[CrawlOptions] = None,
        tenant_id: Optional[str] = None,
    ) -> AsyncGenerator[CrawledPage, None]:
        """
        Crawl a website starting from the given URL.

        Performs breadth-first crawling up to max_depth levels,
        using parallel crawling for discovered links.

        Args:
            start_url: Starting URL for the crawl
            max_depth: Maximum depth to crawl (1 = start page only)
            max_pages: Maximum pages to crawl (prevents unbounded memory growth)
            options: Optional crawl configuration
            tenant_id: Optional tenant ID for multi-tenancy tracking

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
        pages_crawled = 0

        logger.info(
            "crawl_started",
            start_url=start_url,
            max_depth=max_depth,
            max_pages=max_pages,
            tenant_id=tenant_id,
        )

        for depth in range(max_depth):
            # Check max_pages limit before starting new depth
            if pages_crawled >= max_pages:
                logger.warning(
                    "crawl_max_pages_reached",
                    max_pages=max_pages,
                    pages_crawled=pages_crawled,
                    depth=depth,
                )
                break

            # Filter out already visited URLs
            urls_to_crawl = []
            for url in current_level:
                normalized = normalize_url(url, start_url)
                if normalized and normalized not in visited:
                    visited.add(normalized)
                    urls_to_crawl.append(normalized)

            if not urls_to_crawl:
                break

            # Limit URLs to remaining page budget
            remaining_budget = max_pages - pages_crawled
            if len(urls_to_crawl) > remaining_budget:
                urls_to_crawl = urls_to_crawl[:remaining_budget]
                logger.info(
                    "crawl_urls_limited_by_budget",
                    original_count=len(current_level),
                    limited_to=remaining_budget,
                )

            logger.info(
                "crawl_depth_started",
                depth=depth,
                url_count=len(urls_to_crawl),
            )

            # Crawl all URLs at this depth in parallel
            next_level: list[str] = []

            async for page in self.crawl_many(urls_to_crawl, stream=True, max_pages=remaining_budget, tenant_id=tenant_id):
                page.depth = depth
                pages_crawled += 1
                yield page

                # Check if we've hit the max_pages limit
                if pages_crawled >= max_pages:
                    logger.info("crawl_max_pages_limit_hit", max_pages=max_pages)
                    break

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

            # Use set for efficient deduplication
            current_level = list(set(next_level))

        logger.info(
            "crawl_completed",
            start_url=start_url,
            pages_crawled=pages_crawled,
            visited_urls=len(visited),
            tenant_id=tenant_id,
        )


async def crawl_url(
    url: str,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_pages: int = DEFAULT_MAX_PAGES,
    rate_limit: float = DEFAULT_RATE_LIMIT,
    options: Optional[CrawlOptions] = None,
    headless: bool = True,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    cache_enabled: bool = True,
    proxy_url: Optional[str] = None,
    js_wait_seconds: float = DEFAULT_JS_WAIT_SECONDS,
    profile: Optional[CrawlProfile] = None,
    profile_name: Optional[str] = None,
    auto_detect_profile: bool = False,
    tenant_id: Optional[str] = None,
) -> AsyncGenerator[CrawledPage, None]:
    """
    Convenience function for crawling a URL using Crawl4AI.

    Args:
        url: URL to crawl
        max_depth: Maximum crawl depth
        max_pages: Maximum pages to crawl (prevents unbounded memory growth)
        rate_limit: Requests per second (legacy parameter, not used with Crawl4AI)
        options: Optional crawl configuration
        headless: Run browser in headless mode
        max_concurrent: Maximum concurrent sessions
        cache_enabled: Enable caching
        proxy_url: Optional proxy URL
        js_wait_seconds: Seconds to wait for JavaScript rendering
        profile: Optional CrawlProfile instance to use
        profile_name: Optional profile name (fast, thorough, stealth)
        auto_detect_profile: If True, auto-detect profile based on URL
        tenant_id: Optional tenant ID for multi-tenancy tracking

    Yields:
        CrawledPage objects for each crawled page
    """
    if options is None:
        options = CrawlOptions(rate_limit=rate_limit)

    # Determine profile to use
    effective_profile = profile
    if effective_profile is None and profile_name:
        effective_profile = get_crawl_profile(profile_name)
    elif effective_profile is None and auto_detect_profile:
        detected_name = get_profile_for_url(url)
        effective_profile = get_crawl_profile(detected_name)
        logger.info(
            "crawl_profile_auto_detected",
            url=url,
            profile=detected_name,
        )

    async with CrawlerService(
        headless=headless,
        max_concurrent=max_concurrent,
        cache_enabled=cache_enabled,
        proxy_url=proxy_url,
        js_wait_seconds=js_wait_seconds,
        profile=effective_profile,
    ) as crawler:
        async for page in crawler.crawl(
            url,
            max_depth=max_depth,
            max_pages=max_pages,
            options=options,
            tenant_id=tenant_id,
        ):
            yield page


# Legacy compatibility aliases
extract_links = extract_links_from_html
extract_title = extract_title_from_html
