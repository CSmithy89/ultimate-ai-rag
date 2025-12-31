"""Crawl4AI wrapper for autonomous documentation site crawling."""

import asyncio
import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional
from urllib.parse import urljoin, urlparse

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agentic_rag_backend.core.errors import InvalidUrlError
from agentic_rag_backend.models.documents import CrawledPage
from agentic_rag_backend.models.ingest import CrawlOptions

logger = structlog.get_logger(__name__)

# Default configuration
DEFAULT_RATE_LIMIT = 1.0  # requests per second
DEFAULT_MAX_DEPTH = 3
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_USER_AGENT = (
    "AgenticRAG-Crawler/1.0 (+https://github.com/example/agentic-rag)"
)


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


def extract_links(html: str, base_url: str) -> list[str]:
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


def html_to_markdown(html: str, title: Optional[str] = None) -> str:
    """
    Convert HTML to markdown format.

    Uses BeautifulSoup to parse HTML and convert common elements into markdown.

    Args:
        html: HTML content
        title: Optional title to prepend

    Returns:
        Markdown-formatted content
    """
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(html, "html.parser")

    def _table_to_markdown(table: Any) -> str:
        rows: list[list[str]] = []
        for row in table.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in row.find_all(["th", "td"])
            ]
            if cells:
                rows.append(cells)

        if not rows:
            return ""

        col_count = max(len(row) for row in rows)
        normalized = [row + [""] * (col_count - len(row)) for row in rows]
        header = normalized[0]
        body = normalized[1:]

        def _format_row(values: list[str]) -> str:
            return "| " + " | ".join(values) + " |"

        lines = [
            _format_row(header),
            "| " + " | ".join(["---"] * col_count) + " |",
        ]
        for row in body:
            lines.append(_format_row(row))

        return "\n".join(lines)

    for element in soup(["script", "style"]):
        element.decompose()

    for table in soup.find_all("table"):
        table.replace_with(_table_to_markdown(table))

    content = markdownify(
        str(soup),
        heading_style="ATX",
        bullets="-",
    )

    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[ \t]+", " ", content)
    content = content.strip()

    if title:
        content = f"# {title}\n\n{content}"

    return content


def extract_title(html: str) -> Optional[str]:
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


class RobotsTxtChecker:
    """
    Simple robots.txt compliance checker.

    Caches robots.txt content to avoid repeated fetches.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[bool, list[str]]] = {}

    async def is_allowed(
        self,
        url: str,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> bool:
        """
        Check if URL is allowed by robots.txt.

        Args:
            url: URL to check
            user_agent: User agent string

        Returns:
            True if crawling is allowed
        """
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = f"{base}/robots.txt"

        if base not in self._cache:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(robots_url, timeout=10.0)
                    if response.status_code == 200:
                        disallowed = self._parse_robots_txt(response.text)
                        self._cache[base] = (True, disallowed)
                    else:
                        # No robots.txt or error - allow all
                        self._cache[base] = (True, [])
            except Exception:
                # Error fetching - allow all
                self._cache[base] = (True, [])

        _, disallowed_paths = self._cache[base]

        # Check if path is disallowed
        path = parsed.path or "/"
        for pattern in disallowed_paths:
            if path.startswith(pattern):
                return False

        return True

    def _parse_robots_txt(self, content: str) -> list[str]:
        """Parse robots.txt and return disallowed paths for all user agents."""
        disallowed = []
        current_applies = False

        for line in content.split("\n"):
            line = line.strip().lower()

            if line.startswith("user-agent:"):
                agent = line.split(":", 1)[1].strip()
                current_applies = agent == "*"

            elif line.startswith("disallow:") and current_applies:
                path = line.split(":", 1)[1].strip()
                if path:
                    disallowed.append(path)

        return disallowed


class CrawlerService:
    """
    Crawl4AI-style web crawler service.

    Provides autonomous crawling with:
    - robots.txt compliance
    - Rate limiting
    - Depth-limited link following
    - Content extraction and markdown conversion
    """

    def __init__(
        self,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
        respect_robots_txt: bool = True,
    ) -> None:
        """
        Initialize crawler service.

        Args:
            rate_limit: Maximum requests per second
            timeout: Request timeout in seconds
            user_agent: User agent string for requests
            respect_robots_txt: Whether to respect robots.txt
        """
        self.rate_limit = rate_limit
        self.delay = 1.0 / rate_limit
        self.timeout = timeout
        self.user_agent = user_agent
        self.respect_robots_txt = respect_robots_txt
        self._robots_checker = RobotsTxtChecker()
        self._last_request_time: float = 0

    async def _rate_limit_wait(self) -> None:
        """Wait to respect rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self._last_request_time = time.monotonic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    )
    async def _fetch_url(self, url: str) -> tuple[str, int]:
        """
        Fetch URL content with retry logic.

        Args:
            url: URL to fetch

        Returns:
            Tuple of (content, status_code)

        Raises:
            CrawlError: If fetch fails after retries
        """
        await self._rate_limit_wait()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
            return response.text, response.status_code

    async def crawl_page(self, url: str) -> Optional[CrawledPage]:
        """
        Crawl a single page.

        Args:
            url: URL to crawl

        Returns:
            CrawledPage if successful, None otherwise
        """
        if not is_valid_url(url):
            logger.warning("invalid_url", url=url)
            return None

        # Check robots.txt
        if self.respect_robots_txt:
            if not await self._robots_checker.is_allowed(url):
                logger.info("robots_txt_disallowed", url=url)
                return None

        try:
            content, status_code = await self._fetch_url(url)

            if status_code != 200:
                logger.warning("fetch_failed", url=url, status_code=status_code)
                return None

            # Extract title and convert to markdown
            title = extract_title(content)
            markdown = html_to_markdown(content, title)
            content_hash = compute_content_hash(markdown)
            links = extract_links(content, url)

            return CrawledPage(
                url=url,
                title=title,
                content=markdown,
                content_hash=content_hash,
                crawl_timestamp=datetime.now(timezone.utc),
                links=links,
            )

        except Exception as e:
            logger.error("crawl_error", url=url, error=str(e))
            return None

    async def crawl(
        self,
        start_url: str,
        max_depth: int = DEFAULT_MAX_DEPTH,
        options: Optional[CrawlOptions] = None,
    ) -> AsyncGenerator[CrawledPage, None]:
        """
        Crawl a website starting from the given URL.

        Performs breadth-first crawling up to max_depth levels,
        respecting rate limits and robots.txt.

        Args:
            start_url: Starting URL for the crawl
            max_depth: Maximum depth to crawl (1 = start page only)
            options: Optional crawl configuration

        Yields:
            CrawledPage objects for each successfully crawled page
        """
        if not is_valid_url(start_url):
            raise InvalidUrlError(start_url, "URL is not valid")

        options = options or CrawlOptions()

        # Apply options
        self.rate_limit = options.rate_limit
        self.delay = 1.0 / options.rate_limit
        self.respect_robots_txt = options.respect_robots_txt

        visited: set[str] = set()
        to_visit: list[tuple[str, int]] = [(start_url, 0)]  # (url, depth)

        logger.info(
            "crawl_started",
            start_url=start_url,
            max_depth=max_depth,
            rate_limit=options.rate_limit,
        )

        while to_visit:
            url, depth = to_visit.pop(0)

            # Skip if already visited
            normalized = normalize_url(url, start_url)
            if not normalized or normalized in visited:
                continue

            visited.add(normalized)

            # Crawl the page
            page = await self.crawl_page(normalized)
            if page is None:
                continue

            page.depth = depth
            yield page

            # Add links for next depth level
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
                            to_visit.append((link, depth + 1))

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
) -> AsyncGenerator[CrawledPage, None]:
    """
    Convenience function for crawling a URL.

    Args:
        url: URL to crawl
        max_depth: Maximum crawl depth
        rate_limit: Requests per second
        options: Optional crawl configuration

    Yields:
        CrawledPage objects for each crawled page
    """
    if options is None:
        options = CrawlOptions(rate_limit=rate_limit)

    crawler = CrawlerService(rate_limit=options.rate_limit)
    async for page in crawler.crawl(url, max_depth=max_depth, options=options):
        yield page
