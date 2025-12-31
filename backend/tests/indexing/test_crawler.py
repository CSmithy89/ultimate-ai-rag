"""Tests for the crawler service."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag_backend.indexing.crawler import (
    CrawlerService,
    RobotsTxtChecker,
    compute_content_hash,
    crawl_url,
    extract_links,
    extract_title,
    html_to_markdown,
    is_same_domain,
    is_valid_url,
    normalize_url,
)
from agentic_rag_backend.models.ingest import CrawlOptions


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


class TestHtmlToMarkdown:
    """Tests for HTML to Markdown conversion."""

    def test_convert_headers(self):
        """Test header conversion."""
        html = "<h1>Title</h1><h2>Subtitle</h2>"
        result = html_to_markdown(html)
        assert "# Title" in result
        assert "## Subtitle" in result

    def test_convert_paragraphs(self):
        """Test paragraph conversion."""
        html = "<p>First paragraph</p><p>Second paragraph</p>"
        result = html_to_markdown(html)
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_convert_links(self):
        """Test link conversion."""
        html = '<a href="https://example.com">Link text</a>'
        result = html_to_markdown(html)
        assert "[Link text](https://example.com)" in result

    def test_convert_bold(self):
        """Test bold text conversion."""
        html = "<strong>bold text</strong>"
        result = html_to_markdown(html)
        assert "**bold text**" in result

    def test_convert_italic(self):
        """Test italic text conversion."""
        html = "<em>italic text</em>"
        result = html_to_markdown(html)
        assert "*italic text*" in result

    def test_convert_code(self):
        """Test inline code conversion."""
        html = "<code>code</code>"
        result = html_to_markdown(html)
        assert "`code`" in result

    def test_removes_script_tags(self):
        """Test script tags are removed."""
        html = "<p>Content</p><script>alert('xss')</script>"
        result = html_to_markdown(html)
        assert "script" not in result
        assert "alert" not in result

    def test_removes_style_tags(self):
        """Test style tags are removed."""
        html = "<style>.foo { color: red; }</style><p>Content</p>"
        result = html_to_markdown(html)
        assert "style" not in result
        assert "color" not in result

    def test_with_title(self):
        """Test prepending title."""
        html = "<p>Content</p>"
        result = html_to_markdown(html, title="Page Title")
        assert result.startswith("# Page Title")

    def test_convert_tables(self):
        """Test table conversion preserves structure."""
        html = (
            "<table>"
            "<tr><th>Name</th><th>Age</th></tr>"
            "<tr><td>Ada</td><td>36</td></tr>"
            "</table>"
        )
        result = html_to_markdown(html)
        assert "| Name | Age |" in result
        assert "| --- | --- |" in result
        assert "| Ada | 36 |" in result

    def test_strips_unsafe_link_attributes(self):
        """Test javascript: links are stripped."""
        html = '<a href="javascript:alert(1)">Click</a>'
        result = html_to_markdown(html)
        assert "javascript:" not in result


class TestExtractTitle:
    """Tests for title extraction."""

    def test_extract_title_present(self):
        """Test extracting title when present."""
        html = "<html><head><title>Page Title</title></head><body></body></html>"
        result = extract_title(html)
        assert result == "Page Title"

    def test_extract_title_missing(self):
        """Test when title is missing."""
        html = "<html><head></head><body></body></html>"
        result = extract_title(html)
        assert result is None

    def test_extract_title_with_whitespace(self):
        """Test title with whitespace is trimmed."""
        html = "<title>  Spaced Title  </title>"
        result = extract_title(html)
        assert result == "Spaced Title"


class TestExtractLinks:
    """Tests for link extraction."""

    def test_extract_links_basic(self):
        """Test basic link extraction."""
        html = '<a href="/page1">Link 1</a><a href="/page2">Link 2</a>'
        result = extract_links(html, "https://example.com")
        assert "https://example.com/page1" in result
        assert "https://example.com/page2" in result

    def test_extract_links_same_domain_only(self):
        """Test only same-domain links are extracted."""
        html = '<a href="/local">Local</a><a href="https://other.com/page">External</a>'
        result = extract_links(html, "https://example.com")
        assert "https://example.com/local" in result
        assert "https://other.com/page" not in result

    def test_extract_links_removes_duplicates(self):
        """Test duplicate links are removed."""
        html = '<a href="/page">Link 1</a><a href="/page">Link 2</a>'
        result = extract_links(html, "https://example.com")
        assert result.count("https://example.com/page") == 1


class TestRobotsTxtChecker:
    """Tests for robots.txt compliance checker."""

    @pytest.mark.asyncio
    async def test_is_allowed_no_robots_txt(self):
        """Test URL is allowed when no robots.txt exists."""
        checker = RobotsTxtChecker()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            result = await checker.is_allowed("https://example.com/page")
            assert result is True

    @pytest.mark.asyncio
    async def test_is_allowed_with_robots_txt(self):
        """Test URL checking against robots.txt."""
        checker = RobotsTxtChecker()

        robots_content = """
User-agent: *
Disallow: /private/
Disallow: /admin/
"""

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = robots_content
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            # Allowed path
            result = await checker.is_allowed("https://example.com/public/page")
            assert result is True

            # Disallowed path
            result = await checker.is_allowed("https://example.com/private/secret")
            assert result is False


class TestCrawlerService:
    """Tests for the CrawlerService class."""

    @pytest.mark.asyncio
    async def test_crawl_page_success(self):
        """Test successful single page crawl."""
        crawler = CrawlerService(respect_robots_txt=False)

        with patch.object(crawler, "_fetch_url") as mock_fetch:
            mock_fetch.return_value = (
                "<html><head><title>Test</title></head><body><p>Content</p></body></html>",
                200,
            )

            result = await crawler.crawl_page("https://example.com/page")

            assert result is not None
            assert result.url == "https://example.com/page"
            assert result.title == "Test"
            assert "Content" in result.content
            assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_crawl_page_not_found(self):
        """Test crawl returns None for 404."""
        crawler = CrawlerService(respect_robots_txt=False)

        with patch.object(crawler, "_fetch_url") as mock_fetch:
            mock_fetch.return_value = ("Not Found", 404)

            result = await crawler.crawl_page("https://example.com/missing")
            assert result is None

    @pytest.mark.asyncio
    async def test_crawl_page_invalid_url(self):
        """Test crawl returns None for invalid URL."""
        crawler = CrawlerService()
        result = await crawler.crawl_page("not-a-url")
        assert result is None

    @pytest.mark.asyncio
    async def test_crawl_respects_depth_limit(self):
        """Test crawl respects max_depth parameter."""
        crawler = CrawlerService(respect_robots_txt=False)

        pages_crawled = []

        async def mock_crawl_page(url):
            from agentic_rag_backend.models.documents import CrawledPage

            if len(pages_crawled) < 5:
                pages_crawled.append(url)
                return CrawledPage(
                    url=url,
                    title="Test",
                    content="# Test",
                    content_hash="a" * 64,
                    crawl_timestamp=datetime.now(timezone.utc),
                    links=[f"https://example.com/page{len(pages_crawled)}"],
                )
            return None

        with patch.object(crawler, "crawl_page", side_effect=mock_crawl_page):
            results = []
            async for page in crawler.crawl(
                "https://example.com",
                max_depth=2,
                options=CrawlOptions(follow_links=True, respect_robots_txt=False),
            ):
                results.append(page)

            # With max_depth=2, should get start page + 1 level of links
            assert len(results) <= 2


@pytest.mark.asyncio
async def test_crawl_url_function():
    """Test the convenience crawl_url function."""
    with patch("agentic_rag_backend.indexing.crawler.CrawlerService") as MockCrawler:
        mock_instance = AsyncMock()

        async def mock_crawl(*args, **kwargs):
            from agentic_rag_backend.models.documents import CrawledPage

            yield CrawledPage(
                url="https://example.com",
                title="Test",
                content="# Test",
                content_hash="a" * 64,
                crawl_timestamp=datetime.now(timezone.utc),
                links=[],
            )

        mock_instance.crawl = mock_crawl
        MockCrawler.return_value = mock_instance

        results = []
        async for page in crawl_url("https://example.com"):
            results.append(page)

        assert len(results) == 1
        assert results[0].url == "https://example.com"
