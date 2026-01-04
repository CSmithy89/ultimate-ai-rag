"""Tests for Crawl4AI-powered crawler service.

Story 13.3: Migrate to Crawl4AI Library
Tests the new CrawlerService implementation using Crawl4AI for:
- JavaScript-rendered content capture
- Parallel URL crawling via arun_many()
- Caching behavior
- Proxy configuration
"""

from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, patch
import pytest

from agentic_rag_backend.indexing.crawler import (
    CrawlerService,
    crawl_url,
    compute_content_hash,
    is_valid_url,
    normalize_url,
    is_same_domain,
    extract_links_from_html,
    extract_links_from_markdown,
    extract_title_from_html,
    extract_title_from_markdown,
    CRAWL4AI_AVAILABLE,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_JS_WAIT_SECONDS,
)
from agentic_rag_backend.models.documents import CrawledPage
from agentic_rag_backend.models.ingest import CrawlOptions
from agentic_rag_backend.core.errors import InvalidUrlError


class TestHelperFunctions:
    """Tests for utility functions."""

    def test_compute_content_hash(self):
        """Test SHA-256 hash computation."""
        content = "Hello, World!"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)

        assert len(hash1) == 64  # SHA-256 produces 64 hex chars
        assert hash1 == hash2  # Same content produces same hash
        assert compute_content_hash("Different") != hash1  # Different content produces different hash

    def test_is_valid_url_valid(self):
        """Test URL validation with valid URLs."""
        assert is_valid_url("https://example.com") is True
        assert is_valid_url("http://example.com") is True
        assert is_valid_url("https://example.com/path/to/page") is True
        assert is_valid_url("https://example.com?query=param") is True

    def test_is_valid_url_invalid(self):
        """Test URL validation with invalid URLs."""
        assert is_valid_url("") is False
        assert is_valid_url("not-a-url") is False
        assert is_valid_url("ftp://example.com") is False
        assert is_valid_url("file:///local/path") is False
        assert is_valid_url("javascript:alert(1)") is False

    def test_normalize_url_absolute(self):
        """Test URL normalization with absolute URLs."""
        base = "https://example.com/page"

        # Already absolute
        assert normalize_url("https://example.com/other", base) == "https://example.com/other"

        # Remove fragment
        assert normalize_url("https://example.com/page#section", base) == "https://example.com/page"

        # Remove trailing slash
        assert normalize_url("https://example.com/page/", base) == "https://example.com/page"

    def test_normalize_url_relative(self):
        """Test URL normalization with relative URLs."""
        base = "https://example.com/docs/intro"

        assert normalize_url("/other", base) == "https://example.com/other"
        assert normalize_url("../guide", base) == "https://example.com/guide"
        assert normalize_url("./page", base) == "https://example.com/docs/page"

    def test_normalize_url_invalid(self):
        """Test URL normalization with invalid URLs."""
        base = "https://example.com"

        assert normalize_url("javascript:void(0)", base) is None
        assert normalize_url("ftp://other.com", base) is None

    def test_is_same_domain(self):
        """Test domain comparison."""
        assert is_same_domain("https://example.com/a", "https://example.com/b") is True
        assert is_same_domain("http://example.com", "https://example.com") is True
        assert is_same_domain("https://example.com", "https://other.com") is False
        assert is_same_domain("https://sub.example.com", "https://example.com") is False

    def test_extract_links_from_html(self):
        """Test HTML link extraction."""
        html = """
        <html>
            <body>
                <a href="/page1">Page 1</a>
                <a href="https://example.com/page2">Page 2</a>
                <a href="https://other.com/page3">External</a>
                <a href="#section">Anchor</a>
            </body>
        </html>
        """
        base_url = "https://example.com"
        links = extract_links_from_html(html, base_url)

        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links
        assert "https://other.com/page3" not in links  # Different domain
        # Anchors should be excluded (normalize removes fragments)

    def test_extract_links_from_markdown(self):
        """Test markdown link extraction."""
        markdown = """
        # Test Page

        Check out [Page 1](/page1) and [Page 2](https://example.com/page2).
        Also see [External](https://other.com/page3).
        """
        base_url = "https://example.com"
        links = extract_links_from_markdown(markdown, base_url)

        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links
        assert "https://other.com/page3" not in links  # Different domain

    def test_extract_title_from_html(self):
        """Test HTML title extraction."""
        html = "<html><head><title>My Page Title</title></head></html>"
        assert extract_title_from_html(html) == "My Page Title"

        html_no_title = "<html><body>Content</body></html>"
        assert extract_title_from_html(html_no_title) is None

    def test_extract_title_from_markdown(self):
        """Test markdown title extraction."""
        markdown = "# My Page Title\n\nSome content here."
        assert extract_title_from_markdown(markdown) == "My Page Title"

        markdown_no_h1 = "## Subtitle\n\nContent without H1."
        assert extract_title_from_markdown(markdown_no_h1) is None

        markdown_multiple_h1 = "# First\n\n# Second"
        assert extract_title_from_markdown(markdown_multiple_h1) == "First"


class MockCrawlResult:
    """Mock Crawl4AI CrawlResult for testing."""

    def __init__(
        self,
        url: str,
        success: bool = True,
        markdown: Optional[str] = None,
        html: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        self.url = url
        self.success = success
        self.error_message = error_message
        self.html = html

        # Mock markdown object with raw_markdown attribute
        if markdown:
            self.markdown = MagicMock()
            self.markdown.raw_markdown = markdown
        else:
            self.markdown = None


class MockAsyncWebCrawler:
    """Mock AsyncWebCrawler for testing."""

    def __init__(self, results: Optional[dict[str, MockCrawlResult]] = None):
        self.results = results or {}
        self._entered = False

    async def __aenter__(self):
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._entered = False

    async def arun(self, url: str, config: Any = None) -> MockCrawlResult:
        if url in self.results:
            return self.results[url]
        # Default successful result
        return MockCrawlResult(
            url=url,
            success=True,
            markdown="# Test Page\n\nContent for " + url,
            html="<html><head><title>Test</title></head><body><h1>Test Page</h1></body></html>",
        )

    async def arun_many(
        self,
        urls: list[str],
        config: Any = None,
        dispatcher: Any = None,
    ) -> list[MockCrawlResult]:
        results = []
        for url in urls:
            results.append(await self.arun(url, config))
        return results


@pytest.fixture
def mock_crawl4ai():
    """Fixture to mock Crawl4AI imports."""
    with patch.dict('sys.modules', {
        'crawl4ai': MagicMock(),
        'crawl4ai.async_dispatcher': MagicMock(),
    }):
        yield


class TestCrawlerService:
    """Tests for CrawlerService class."""

    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    def test_init_default_settings(self):
        """Test CrawlerService initialization with defaults."""
        crawler = CrawlerService()

        assert crawler.headless is True
        assert crawler.max_concurrent == DEFAULT_MAX_CONCURRENT
        assert crawler.cache_enabled is True
        assert crawler.proxy_url is None
        assert crawler.js_wait_seconds == DEFAULT_JS_WAIT_SECONDS

    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    def test_init_custom_settings(self):
        """Test CrawlerService initialization with custom settings."""
        crawler = CrawlerService(
            headless=False,
            max_concurrent=5,
            cache_enabled=False,
            proxy_url="http://proxy:8080",
            js_wait_seconds=3.0,
            page_timeout_ms=30000,
        )

        assert crawler.headless is False
        assert crawler.max_concurrent == 5
        assert crawler.cache_enabled is False
        assert crawler.proxy_url == "http://proxy:8080"
        assert crawler.js_wait_seconds == 3.0
        assert crawler.page_timeout_ms == 30000

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    async def test_context_manager_required(self):
        """Test that CrawlerService requires context manager usage."""
        crawler = CrawlerService()

        with pytest.raises(RuntimeError, match="context manager"):
            await crawler.crawl_page("https://example.com")

    @pytest.mark.asyncio
    async def test_crawl_page_invalid_url(self):
        """Test crawl_page with invalid URL."""
        with patch('agentic_rag_backend.indexing.crawler.CRAWL4AI_AVAILABLE', True):
            with patch('agentic_rag_backend.indexing.crawler.AsyncWebCrawler', MockAsyncWebCrawler):
                with patch('agentic_rag_backend.indexing.crawler.BrowserConfig', MagicMock):
                    with patch('agentic_rag_backend.indexing.crawler.CrawlerRunConfig', MagicMock):
                        with patch('agentic_rag_backend.indexing.crawler.CacheMode', MagicMock):
                            with patch('agentic_rag_backend.indexing.crawler.MemoryAdaptiveDispatcher', MagicMock):
                                crawler = CrawlerService.__new__(CrawlerService)
                                crawler.headless = True
                                crawler.max_concurrent = 10
                                crawler.cache_enabled = True
                                crawler.proxy_url = None
                                crawler.js_wait_seconds = 2.0
                                crawler.page_timeout_ms = 60000
                                crawler._crawler = MockAsyncWebCrawler()

                                result = await crawler.crawl_page("not-a-valid-url")
                                assert result is None

    @pytest.mark.asyncio
    async def test_convert_result_success(self):
        """Test converting successful Crawl4AI result to CrawledPage."""
        with patch('agentic_rag_backend.indexing.crawler.CRAWL4AI_AVAILABLE', True):
            crawler = CrawlerService.__new__(CrawlerService)
            crawler.headless = True
            crawler.max_concurrent = 10
            crawler.cache_enabled = True
            crawler.proxy_url = None
            crawler.js_wait_seconds = 2.0
            crawler.page_timeout_ms = 60000

            result = MockCrawlResult(
                url="https://example.com/page",
                success=True,
                markdown="# My Title\n\nContent here.",
                html="<html><head><title>My Title</title></head><body><a href='/other'>Link</a></body></html>",
            )

            page = crawler._convert_result_to_crawled_page(result, depth=1)

            assert page is not None
            assert page.url == "https://example.com/page"
            assert page.title == "My Title"
            assert "Content here" in page.content
            assert page.depth == 1
            assert len(page.content_hash) == 64
            assert isinstance(page.crawl_timestamp, datetime)

    @pytest.mark.asyncio
    async def test_convert_result_failure(self):
        """Test converting failed Crawl4AI result."""
        with patch('agentic_rag_backend.indexing.crawler.CRAWL4AI_AVAILABLE', True):
            crawler = CrawlerService.__new__(CrawlerService)
            crawler.headless = True

            result = MockCrawlResult(
                url="https://example.com/error",
                success=False,
                error_message="Connection timeout",
            )

            page = crawler._convert_result_to_crawled_page(result, depth=0)

            assert page is None

    @pytest.mark.asyncio
    async def test_convert_result_no_content(self):
        """Test converting result with no content."""
        with patch('agentic_rag_backend.indexing.crawler.CRAWL4AI_AVAILABLE', True):
            crawler = CrawlerService.__new__(CrawlerService)
            crawler.headless = True

            result = MockCrawlResult(
                url="https://example.com/empty",
                success=True,
                markdown="",
                html="",
            )

            page = crawler._convert_result_to_crawled_page(result, depth=0)

            assert page is None


class TestCrawlUrlFunction:
    """Tests for the crawl_url convenience function."""

    @pytest.mark.asyncio
    async def test_crawl_url_invalid(self):
        """Test crawl_url with invalid URL raises error."""
        with pytest.raises(InvalidUrlError):
            async for _ in crawl_url("not-a-url"):
                pass


class TestCrawlOptions:
    """Tests for CrawlOptions handling."""

    def test_crawl_options_defaults(self):
        """Test CrawlOptions default values."""
        options = CrawlOptions()

        assert options.follow_links is True
        assert options.respect_robots_txt is True
        assert options.rate_limit == 1.0
        assert options.include_patterns == []
        assert options.exclude_patterns == []

    def test_crawl_options_custom(self):
        """Test CrawlOptions with custom values."""
        options = CrawlOptions(
            follow_links=False,
            respect_robots_txt=False,
            rate_limit=2.0,
            include_patterns=[r".*docs.*"],
            exclude_patterns=[r".*api.*"],
        )

        assert options.follow_links is False
        assert options.rate_limit == 2.0
        assert len(options.include_patterns) == 1
        assert len(options.exclude_patterns) == 1


class TestModuleExports:
    """Tests for module-level exports and constants."""

    def test_default_constants(self):
        """Test default constant values."""
        assert DEFAULT_MAX_CONCURRENT == 10
        assert DEFAULT_JS_WAIT_SECONDS == 2.0

    def test_crawl4ai_available_flag(self):
        """Test CRAWL4AI_AVAILABLE flag exists."""
        # Should be either True or False depending on environment
        assert isinstance(CRAWL4AI_AVAILABLE, bool)


class TestCacheConfiguration:
    """Tests for cache configuration."""

    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    def test_cache_enabled(self):
        """Test cache enabled configuration."""
        crawler = CrawlerService(cache_enabled=True)
        assert crawler.cache_enabled is True

    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    def test_cache_disabled(self):
        """Test cache disabled configuration."""
        crawler = CrawlerService(cache_enabled=False)
        assert crawler.cache_enabled is False


class TestProxyConfiguration:
    """Tests for proxy configuration."""

    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    def test_proxy_url_none(self):
        """Test no proxy configuration."""
        crawler = CrawlerService(proxy_url=None)
        assert crawler.proxy_url is None

    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    def test_proxy_url_set(self):
        """Test proxy URL configuration."""
        proxy = "http://user:pass@proxy.example.com:8080"
        crawler = CrawlerService(proxy_url=proxy)
        assert crawler.proxy_url == proxy


class TestJavaScriptRendering:
    """Tests for JavaScript rendering configuration."""

    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    def test_js_wait_default(self):
        """Test default JS wait time."""
        crawler = CrawlerService()
        assert crawler.js_wait_seconds == DEFAULT_JS_WAIT_SECONDS

    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    def test_js_wait_custom(self):
        """Test custom JS wait time."""
        crawler = CrawlerService(js_wait_seconds=5.0)
        assert crawler.js_wait_seconds == 5.0


class TestParallelCrawling:
    """Tests for parallel crawling functionality."""

    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    def test_max_concurrent_default(self):
        """Test default max concurrent sessions."""
        crawler = CrawlerService()
        assert crawler.max_concurrent == DEFAULT_MAX_CONCURRENT

    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    def test_max_concurrent_custom(self):
        """Test custom max concurrent sessions."""
        crawler = CrawlerService(max_concurrent=20)
        assert crawler.max_concurrent == 20


class TestLinkExtraction:
    """Additional tests for link extraction edge cases."""

    def test_extract_links_html_empty(self):
        """Test link extraction from empty HTML."""
        links = extract_links_from_html("", "https://example.com")
        assert links == []

    def test_extract_links_html_no_links(self):
        """Test link extraction from HTML without links."""
        html = "<html><body><p>No links here</p></body></html>"
        links = extract_links_from_html(html, "https://example.com")
        assert links == []

    def test_extract_links_html_duplicate_removal(self):
        """Test that duplicate links are removed."""
        html = """
        <a href="/page">Link 1</a>
        <a href="/page">Link 2</a>
        <a href="/page">Link 3</a>
        """
        links = extract_links_from_html(html, "https://example.com")
        assert len(links) == 1
        assert "https://example.com/page" in links

    def test_extract_links_markdown_empty(self):
        """Test link extraction from empty markdown."""
        links = extract_links_from_markdown("", "https://example.com")
        assert links == []


class TestTitleExtraction:
    """Additional tests for title extraction edge cases."""

    def test_extract_title_html_whitespace(self):
        """Test title extraction handles whitespace."""
        html = "<html><head><title>  Spaced Title  </title></head></html>"
        assert extract_title_from_html(html) == "Spaced Title"

    def test_extract_title_markdown_with_formatting(self):
        """Test title extraction from markdown with formatting."""
        markdown = "# **Bold Title**\n\nContent"
        # The regex will include the formatting markers
        title = extract_title_from_markdown(markdown)
        assert "Bold Title" in title


class TestCrawledPageModel:
    """Tests for CrawledPage model creation."""

    def test_crawled_page_creation(self):
        """Test creating a CrawledPage instance."""
        page = CrawledPage(
            url="https://example.com",
            title="Test Page",
            content="# Test\n\nContent here.",
            content_hash="a" * 64,
            crawl_timestamp=datetime.now(timezone.utc),
            depth=0,
            links=["https://example.com/other"],
        )

        assert page.url == "https://example.com"
        assert page.title == "Test Page"
        assert page.depth == 0
        assert len(page.links) == 1

    def test_crawled_page_defaults(self):
        """Test CrawledPage default values."""
        page = CrawledPage(
            url="https://example.com",
            content="Content",
            content_hash="b" * 64,
        )

        assert page.title is None
        assert page.depth == 0
        assert page.links == []
