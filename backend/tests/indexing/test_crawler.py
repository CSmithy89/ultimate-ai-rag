"""Tests for the crawler service.

Story 13.3: Updated tests for Crawl4AI migration.
Tests helper functions and CrawlerService with Crawl4AI backend.
"""

from unittest.mock import MagicMock, patch

import pytest

from agentic_rag_backend.indexing.crawler import (
    CrawlerService,
    compute_content_hash,
    crawl_url,
    extract_links_from_html,
    extract_title_from_html,
    is_same_domain,
    is_valid_url,
    normalize_url,
    CRAWL4AI_AVAILABLE,
)
from agentic_rag_backend.models.ingest import CrawlOptions
from agentic_rag_backend.core.errors import InvalidUrlError


class TestUrlValidation:
    """Tests for URL validation functions."""

    def test_is_valid_url_http(self):
        """Test valid HTTP URL."""
        assert is_valid_url("http://example.com") is True

    def test_is_valid_url_https(self):
        """Test valid HTTPS URL."""
        assert is_valid_url("https://docs.example.com/path") is True

    def test_is_valid_url_invalid_scheme(self):
        """Test URL with invalid scheme."""
        assert is_valid_url("ftp://example.com") is False
        assert is_valid_url("file:///etc/passwd") is False

    def test_is_valid_url_no_scheme(self):
        """Test URL without scheme."""
        assert is_valid_url("example.com") is False

    def test_is_valid_url_empty(self):
        """Test empty string."""
        assert is_valid_url("") is False

    def test_is_valid_url_malformed(self):
        """Test malformed URLs."""
        assert is_valid_url("not a url at all") is False


class TestUrlNormalization:
    """Tests for URL normalization."""

    def test_normalize_absolute_url(self):
        """Test normalizing absolute URL."""
        result = normalize_url("https://example.com/page", "https://example.com")
        assert result == "https://example.com/page"

    def test_normalize_relative_url(self):
        """Test normalizing relative URL."""
        result = normalize_url("/docs/intro", "https://example.com")
        assert result == "https://example.com/docs/intro"

    def test_normalize_removes_fragment(self):
        """Test that fragments are removed."""
        result = normalize_url("https://example.com/page#section", "https://example.com")
        assert result == "https://example.com/page"

    def test_normalize_removes_trailing_slash(self):
        """Test that trailing slashes are removed."""
        result = normalize_url("https://example.com/page/", "https://example.com")
        assert result == "https://example.com/page"

    def test_normalize_invalid_scheme(self):
        """Test normalizing URL with invalid scheme."""
        result = normalize_url("javascript:void(0)", "https://example.com")
        assert result is None


class TestDomainComparison:
    """Tests for domain comparison."""

    def test_is_same_domain_true(self):
        """Test same domain detection."""
        assert is_same_domain(
            "https://example.com/page1",
            "https://example.com/page2",
        ) is True

    def test_is_same_domain_false(self):
        """Test different domain detection."""
        assert is_same_domain(
            "https://example.com/page",
            "https://other.com/page",
        ) is False

    def test_is_same_domain_different_subdomains(self):
        """Test subdomains are treated as different."""
        assert is_same_domain(
            "https://docs.example.com/page",
            "https://api.example.com/page",
        ) is False


class TestContentHash:
    """Tests for content hashing."""

    def test_compute_content_hash_returns_64_chars(self):
        """Test hash is 64 characters (SHA-256 hex)."""
        result = compute_content_hash("test content")
        assert len(result) == 64

    def test_compute_content_hash_consistent(self):
        """Test same content produces same hash."""
        hash1 = compute_content_hash("test content")
        hash2 = compute_content_hash("test content")
        assert hash1 == hash2

    def test_compute_content_hash_different_content(self):
        """Test different content produces different hash."""
        hash1 = compute_content_hash("content 1")
        hash2 = compute_content_hash("content 2")
        assert hash1 != hash2


class TestExtractTitle:
    """Tests for title extraction."""

    def test_extract_title_present(self):
        """Test extracting title when present."""
        html = "<html><head><title>Page Title</title></head><body></body></html>"
        result = extract_title_from_html(html)
        assert result == "Page Title"

    def test_extract_title_missing(self):
        """Test when title is missing."""
        html = "<html><head></head><body></body></html>"
        result = extract_title_from_html(html)
        assert result is None

    def test_extract_title_with_whitespace(self):
        """Test title with whitespace is trimmed."""
        html = "<title>  Spaced Title  </title>"
        result = extract_title_from_html(html)
        assert result == "Spaced Title"


class TestExtractLinks:
    """Tests for link extraction."""

    def test_extract_links_basic(self):
        """Test basic link extraction."""
        html = '<a href="/page1">Link 1</a><a href="/page2">Link 2</a>'
        result = extract_links_from_html(html, "https://example.com")
        assert "https://example.com/page1" in result
        assert "https://example.com/page2" in result

    def test_extract_links_same_domain_only(self):
        """Test only same-domain links are extracted."""
        html = '<a href="/local">Local</a><a href="https://other.com/page">External</a>'
        result = extract_links_from_html(html, "https://example.com")
        assert "https://example.com/local" in result
        assert "https://other.com/page" not in result

    def test_extract_links_removes_duplicates(self):
        """Test duplicate links are removed."""
        html = '<a href="/page">Link 1</a><a href="/page">Link 2</a>'
        result = extract_links_from_html(html, "https://example.com")
        assert result.count("https://example.com/page") == 1


class MockCrawlResult:
    """Mock Crawl4AI CrawlResult for testing."""

    def __init__(
        self,
        url: str,
        success: bool = True,
        markdown: str = "",
        html: str = "",
        error_message: str = None,
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


class TestCrawlerService:
    """Tests for the CrawlerService class."""

    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    def test_crawler_service_init(self):
        """Test CrawlerService initialization."""
        crawler = CrawlerService(
            headless=True,
            max_concurrent=5,
            cache_enabled=True,
        )
        assert crawler.headless is True
        assert crawler.max_concurrent == 5
        assert crawler.cache_enabled is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CRAWL4AI_AVAILABLE, reason="Crawl4AI not installed")
    async def test_crawler_requires_context_manager(self):
        """Test CrawlerService requires async context manager."""
        crawler = CrawlerService()
        with pytest.raises(RuntimeError, match="context manager"):
            await crawler.crawl_page("https://example.com")

    @pytest.mark.asyncio
    async def test_crawl_page_invalid_url(self):
        """Test crawl returns None for invalid URL."""
        # Create a mock crawler that bypasses context manager check
        with patch('agentic_rag_backend.indexing.crawler.CRAWL4AI_AVAILABLE', True):
            crawler = CrawlerService.__new__(CrawlerService)
            crawler.headless = True
            crawler.max_concurrent = 10
            crawler.cache_enabled = True
            crawler.proxy_url = None
            crawler.js_wait_seconds = 2.0
            crawler.page_timeout_ms = 60000
            crawler._crawler = MagicMock()  # Mock as initialized

            result = await crawler.crawl_page("not-a-url")
            assert result is None

    @pytest.mark.asyncio
    async def test_convert_result_to_crawled_page_success(self):
        """Test converting successful Crawl4AI result."""
        with patch('agentic_rag_backend.indexing.crawler.CRAWL4AI_AVAILABLE', True):
            crawler = CrawlerService.__new__(CrawlerService)
            crawler.headless = True

            result = MockCrawlResult(
                url="https://example.com/page",
                success=True,
                markdown="# Test Page\n\nContent here.",
                html="<html><head><title>Test</title></head><body><p>Content</p></body></html>",
            )

            page = crawler._convert_result_to_crawled_page(result, depth=0)

            assert page is not None
            assert page.url == "https://example.com/page"
            assert "Content here" in page.content
            assert len(page.content_hash) == 64

    @pytest.mark.asyncio
    async def test_convert_result_to_crawled_page_failure(self):
        """Test converting failed Crawl4AI result."""
        with patch('agentic_rag_backend.indexing.crawler.CRAWL4AI_AVAILABLE', True):
            crawler = CrawlerService.__new__(CrawlerService)
            crawler.headless = True

            result = MockCrawlResult(
                url="https://example.com/error",
                success=False,
                error_message="Connection failed",
            )

            page = crawler._convert_result_to_crawled_page(result, depth=0)
            assert page is None


class TestCrawlUrlFunction:
    """Tests for the convenience crawl_url function."""

    @pytest.mark.asyncio
    async def test_crawl_url_invalid_url(self):
        """Test crawl_url raises error for invalid URL."""
        with pytest.raises(InvalidUrlError):
            async for _ in crawl_url("not-a-url"):
                pass


class TestCrawlOptions:
    """Tests for CrawlOptions model."""

    def test_crawl_options_defaults(self):
        """Test CrawlOptions default values."""
        options = CrawlOptions()
        assert options.follow_links is True
        assert options.respect_robots_txt is True
        assert options.rate_limit == 1.0

    def test_crawl_options_custom(self):
        """Test CrawlOptions with custom values."""
        options = CrawlOptions(
            follow_links=False,
            rate_limit=2.0,
            include_patterns=[r".*docs.*"],
        )
        assert options.follow_links is False
        assert options.rate_limit == 2.0
        assert len(options.include_patterns) == 1
