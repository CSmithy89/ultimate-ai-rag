"""Tests for fallback crawl providers.

Story 13-1: Integrate Apify BrightData Fallback
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from agentic_rag_backend.indexing.fallback_providers import (
    CrawlResult,
    CrawlProvider,
    ApifyProvider,
    BrightDataProvider,
    FallbackCrawler,
    SimpleCrawlProvider,
    create_fallback_crawler,
    BLOCKED_STATUS_CODES,
    BLOCKED_CONTENT_PATTERNS,
)
from agentic_rag_backend.core.errors import CrawlError


class TestCrawlResult:
    """Tests for CrawlResult Pydantic model."""

    def test_create_crawl_result_minimal(self):
        """Test creating CrawlResult with required fields only."""
        result = CrawlResult(
            url="https://example.com",
            content="<html>Test</html>",
            status_code=200,
        )
        assert result.url == "https://example.com"
        assert result.content == "<html>Test</html>"
        assert result.status_code == 200
        assert result.error is None
        assert result.provider is None
        assert result.elapsed_ms is None

    def test_create_crawl_result_full(self):
        """Test creating CrawlResult with all fields."""
        result = CrawlResult(
            url="https://example.com",
            content="<html>Test</html>",
            status_code=200,
            error=None,
            provider="apify",
            elapsed_ms=1500,
        )
        assert result.provider == "apify"
        assert result.elapsed_ms == 1500

    def test_create_crawl_result_with_error(self):
        """Test creating CrawlResult with an error."""
        result = CrawlResult(
            url="https://example.com",
            content="",
            status_code=403,
            error="Access denied",
            provider="simple",
        )
        assert result.error == "Access denied"
        assert result.status_code == 403


class TestSimpleCrawlProvider:
    """Tests for SimpleCrawlProvider."""

    @pytest.fixture
    def provider(self):
        """Create a SimpleCrawlProvider instance."""
        return SimpleCrawlProvider(timeout_seconds=10.0)

    def test_provider_name(self, provider):
        """Test provider name property."""
        assert provider.name == "simple"

    @pytest.mark.asyncio
    async def test_crawl_success(self, provider):
        """Test successful crawl."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>Hello World</body></html>"
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await provider.crawl("https://example.com")

            assert result.status_code == 200
            assert result.content == "<html><body>Hello World</body></html>"
            assert result.provider == "simple"
            assert result.error is None

    @pytest.mark.asyncio
    async def test_crawl_blocked_status(self, provider):
        """Test crawl detects blocked status codes."""
        mock_response = MagicMock()
        mock_response.text = "Access Denied"
        mock_response.status_code = 403

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await provider.crawl("https://example.com")

            assert result.status_code == 403
            assert result.error == "Response indicates blocking"

    @pytest.mark.asyncio
    async def test_crawl_blocked_content(self, provider):
        """Test crawl detects blocked content patterns."""
        mock_response = MagicMock()
        mock_response.text = "<html>Please complete the captcha to continue</html>"
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await provider.crawl("https://example.com")

            assert result.error == "Response indicates blocking"

    @pytest.mark.asyncio
    async def test_crawl_timeout(self, provider):
        """Test crawl handles timeout errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await provider.crawl("https://example.com")

            assert result.status_code == 0
            assert result.error == "Request timed out"


class TestApifyProvider:
    """Tests for ApifyProvider."""

    @pytest.fixture
    def provider(self):
        """Create an ApifyProvider instance."""
        return ApifyProvider(api_token="test_token", timeout_seconds=10.0)

    def test_provider_name(self, provider):
        """Test provider name property."""
        assert provider.name == "apify"

    @pytest.mark.asyncio
    async def test_crawl_success(self, provider):
        """Test successful Apify crawl."""
        # Mock responses for actor run and dataset fetch
        run_response = MagicMock()
        run_response.status_code = 201
        run_response.json.return_value = {
            "data": {
                "id": "run123",
                "status": "SUCCEEDED",
                "defaultDatasetId": "dataset123",
            }
        }

        dataset_response = MagicMock()
        dataset_response.status_code = 200
        dataset_response.json.return_value = [
            {"url": "https://example.com", "html": "<html>Success</html>"}
        ]

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=run_response)
            mock_client_instance.get = AsyncMock(return_value=dataset_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await provider.crawl("https://example.com")

            assert result.status_code == 200
            assert result.content == "<html>Success</html>"
            assert result.provider == "apify"
            assert result.error is None

    @pytest.mark.asyncio
    async def test_crawl_api_error(self, provider):
        """Test Apify API error handling."""
        run_response = MagicMock()
        run_response.status_code = 401
        run_response.text = "Unauthorized"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=run_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await provider.crawl("https://example.com")

            assert result.status_code == 401
            assert "Apify API error" in result.error

    @pytest.mark.asyncio
    async def test_crawl_run_failed(self, provider):
        """Test handling of failed actor run."""
        run_response = MagicMock()
        run_response.status_code = 201
        run_response.json.return_value = {
            "data": {
                "id": "run123",
                "status": "FAILED",
                "defaultDatasetId": "dataset123",
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=run_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await provider.crawl("https://example.com")

            assert result.status_code == 500
            assert "Actor run failed" in result.error

    @pytest.mark.asyncio
    async def test_crawl_no_results(self, provider):
        """Test handling of empty dataset results."""
        run_response = MagicMock()
        run_response.status_code = 201
        run_response.json.return_value = {
            "data": {
                "id": "run123",
                "status": "SUCCEEDED",
                "defaultDatasetId": "dataset123",
            }
        }

        dataset_response = MagicMock()
        dataset_response.status_code = 200
        dataset_response.json.return_value = []

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=run_response)
            mock_client_instance.get = AsyncMock(return_value=dataset_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await provider.crawl("https://example.com")

            assert result.error == "No results returned from actor"


class TestBrightDataProvider:
    """Tests for BrightDataProvider."""

    @pytest.fixture
    def provider(self):
        """Create a BrightDataProvider instance."""
        return BrightDataProvider(
            username="brd-customer-test",
            password="test_password",
            zone="scraping_browser",
            timeout_seconds=10.0,
        )

    def test_provider_name(self, provider):
        """Test provider name property."""
        assert provider.name == "brightdata"

    def test_proxy_url_format(self, provider):
        """Test proxy URL is correctly formatted."""
        proxy_url = provider._proxy_url
        assert "brd-customer-test-zone-scraping_browser" in proxy_url
        assert "test_password" in proxy_url
        assert "brd.superproxy.io:33335" in proxy_url

    @pytest.mark.asyncio
    async def test_crawl_success(self, provider):
        """Test successful BrightData crawl."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>Proxied Content</body></html>"
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await provider.crawl("https://example.com")

            assert result.status_code == 200
            assert result.content == "<html><body>Proxied Content</body></html>"
            assert result.provider == "brightdata"
            assert result.error is None

    @pytest.mark.asyncio
    async def test_crawl_blocked(self, provider):
        """Test BrightData crawl detects blocking."""
        mock_response = MagicMock()
        mock_response.text = "Access Denied"
        mock_response.status_code = 403

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await provider.crawl("https://example.com")

            assert result.error == "Response indicates blocking"

    @pytest.mark.asyncio
    async def test_crawl_proxy_error(self, provider):
        """Test BrightData handles proxy errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(
                side_effect=httpx.ProxyError("Proxy connection failed")
            )
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await provider.crawl("https://example.com")

            assert result.status_code == 0
            assert "Proxy connection failed" in result.error


class TestFallbackCrawler:
    """Tests for FallbackCrawler chain logic."""

    @pytest.fixture
    def mock_primary(self):
        """Create a mock primary provider."""
        provider = MagicMock(spec=CrawlProvider)
        provider.name = "primary"
        return provider

    @pytest.fixture
    def mock_fallback1(self):
        """Create first mock fallback provider."""
        provider = MagicMock(spec=CrawlProvider)
        provider.name = "fallback1"
        return provider

    @pytest.fixture
    def mock_fallback2(self):
        """Create second mock fallback provider."""
        provider = MagicMock(spec=CrawlProvider)
        provider.name = "fallback2"
        return provider

    @pytest.fixture
    def crawler(self, mock_primary, mock_fallback1, mock_fallback2):
        """Create a FallbackCrawler instance."""
        return FallbackCrawler(
            primary=mock_primary,
            fallbacks=[mock_fallback1, mock_fallback2],
        )

    @pytest.mark.asyncio
    async def test_primary_success(self, crawler, mock_primary):
        """Test primary succeeds on first try."""
        mock_primary.crawl = AsyncMock(
            return_value=CrawlResult(
                url="https://example.com",
                content="<html>Success</html>",
                status_code=200,
                provider="primary",
            )
        )

        result, provider_name = await crawler.crawl_with_fallback("https://example.com")

        assert result.status_code == 200
        assert provider_name == "primary"
        mock_primary.crawl.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_primary_error(
        self, crawler, mock_primary, mock_fallback1
    ):
        """Test falls back when primary returns error."""
        mock_primary.crawl = AsyncMock(
            return_value=CrawlResult(
                url="https://example.com",
                content="",
                status_code=403,
                error="Blocked",
                provider="primary",
            )
        )
        mock_fallback1.crawl = AsyncMock(
            return_value=CrawlResult(
                url="https://example.com",
                content="<html>Fallback Success</html>",
                status_code=200,
                provider="fallback1",
            )
        )

        result, provider_name = await crawler.crawl_with_fallback("https://example.com")

        assert result.status_code == 200
        assert provider_name == "fallback1"
        mock_primary.crawl.assert_called_once()
        mock_fallback1.crawl.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_primary_exception(
        self, crawler, mock_primary, mock_fallback1
    ):
        """Test falls back when primary raises exception."""
        mock_primary.crawl = AsyncMock(side_effect=Exception("Connection failed"))
        mock_fallback1.crawl = AsyncMock(
            return_value=CrawlResult(
                url="https://example.com",
                content="<html>Fallback Success</html>",
                status_code=200,
                provider="fallback1",
            )
        )

        result, provider_name = await crawler.crawl_with_fallback("https://example.com")

        assert result.status_code == 200
        assert provider_name == "fallback1"

    @pytest.mark.asyncio
    async def test_second_fallback_on_first_failure(
        self, crawler, mock_primary, mock_fallback1, mock_fallback2
    ):
        """Test uses second fallback when first fallback fails."""
        mock_primary.crawl = AsyncMock(side_effect=Exception("Primary failed"))
        mock_fallback1.crawl = AsyncMock(
            return_value=CrawlResult(
                url="https://example.com",
                content="Blocked",
                status_code=403,
                error="Blocked",
                provider="fallback1",
            )
        )
        mock_fallback2.crawl = AsyncMock(
            return_value=CrawlResult(
                url="https://example.com",
                content="<html>Second Fallback Success</html>",
                status_code=200,
                provider="fallback2",
            )
        )

        result, provider_name = await crawler.crawl_with_fallback("https://example.com")

        assert result.status_code == 200
        assert provider_name == "fallback2"
        mock_fallback1.crawl.assert_called_once()
        mock_fallback2.crawl.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_providers_fail(
        self, crawler, mock_primary, mock_fallback1, mock_fallback2
    ):
        """Test raises CrawlError when all providers fail."""
        mock_primary.crawl = AsyncMock(side_effect=Exception("Primary failed"))
        mock_fallback1.crawl = AsyncMock(side_effect=Exception("Fallback1 failed"))
        mock_fallback2.crawl = AsyncMock(side_effect=Exception("Fallback2 failed"))

        with pytest.raises(CrawlError) as exc_info:
            await crawler.crawl_with_fallback("https://example.com")

        assert "All providers failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fallback_usage_tracking(
        self, crawler, mock_primary, mock_fallback1
    ):
        """Test fallback usage is tracked for cost monitoring."""
        mock_primary.crawl = AsyncMock(
            return_value=CrawlResult(
                url="https://example.com",
                content="",
                status_code=403,
                error="Blocked",
                provider="primary",
            )
        )
        mock_fallback1.crawl = AsyncMock(
            return_value=CrawlResult(
                url="https://example.com",
                content="<html>Success</html>",
                status_code=200,
                provider="fallback1",
            )
        )

        # Make multiple fallback calls
        await crawler.crawl_with_fallback("https://example.com")
        await crawler.crawl_with_fallback("https://example.com/page2")

        usage = crawler.get_fallback_usage()
        assert usage["fallback1"] == 2

    @pytest.mark.asyncio
    async def test_empty_content_triggers_fallback(
        self, crawler, mock_primary, mock_fallback1
    ):
        """Test empty content in response triggers fallback."""
        mock_primary.crawl = AsyncMock(
            return_value=CrawlResult(
                url="https://example.com",
                content="",  # Empty content
                status_code=200,
                provider="primary",
            )
        )
        mock_fallback1.crawl = AsyncMock(
            return_value=CrawlResult(
                url="https://example.com",
                content="<html>Real Content</html>",
                status_code=200,
                provider="fallback1",
            )
        )

        result, provider_name = await crawler.crawl_with_fallback("https://example.com")

        assert provider_name == "fallback1"
        assert result.content == "<html>Real Content</html>"


class TestBlockedResponseDetection:
    """Tests for blocked response detection patterns."""

    @pytest.fixture
    def provider(self):
        """Create a SimpleCrawlProvider for testing."""
        return SimpleCrawlProvider()

    @pytest.mark.parametrize("status_code", BLOCKED_STATUS_CODES)
    def test_blocked_status_codes(self, provider, status_code):
        """Test that blocked status codes are detected."""
        assert provider._is_blocked_response("any content", status_code) is True

    @pytest.mark.parametrize("pattern", BLOCKED_CONTENT_PATTERNS)
    def test_blocked_content_patterns(self, provider, pattern):
        """Test that blocked content patterns are detected."""
        content = f"<html>This page contains {pattern} verification</html>"
        assert provider._is_blocked_response(content, 200) is True

    def test_normal_content_not_blocked(self, provider):
        """Test that normal content is not detected as blocked."""
        content = "<html><body>Normal page content here</body></html>"
        assert provider._is_blocked_response(content, 200) is False


class TestCreateFallbackCrawler:
    """Tests for factory function create_fallback_crawler."""

    @pytest.fixture
    def mock_settings_disabled(self):
        """Create mock settings with fallback disabled."""
        settings = MagicMock()
        settings.crawl_fallback_enabled = False
        return settings

    @pytest.fixture
    def mock_settings_enabled(self):
        """Create mock settings with fallback enabled and credentials."""
        settings = MagicMock()
        settings.crawl_fallback_enabled = True
        settings.crawl_fallback_providers = ["apify", "brightdata"]
        settings.apify_api_token = "test_apify_token"
        settings.brightdata_username = "brd-customer-test"
        settings.brightdata_password = "test_password"
        settings.brightdata_zone = "scraping_browser"
        return settings

    @pytest.fixture
    def mock_settings_partial(self):
        """Create mock settings with only Apify configured."""
        settings = MagicMock()
        settings.crawl_fallback_enabled = True
        settings.crawl_fallback_providers = ["apify", "brightdata"]
        settings.apify_api_token = "test_apify_token"
        settings.brightdata_username = None
        settings.brightdata_password = None
        settings.brightdata_zone = "scraping_browser"
        return settings

    def test_disabled_returns_none(self, mock_settings_disabled):
        """Test returns None when fallback is disabled."""
        with patch(
            "agentic_rag_backend.indexing.fallback_providers.get_settings",
            return_value=mock_settings_disabled,
        ):
            result = create_fallback_crawler()
            assert result is None

    def test_enabled_creates_crawler(self, mock_settings_enabled):
        """Test creates FallbackCrawler when enabled with credentials."""
        with patch(
            "agentic_rag_backend.indexing.fallback_providers.get_settings",
            return_value=mock_settings_enabled,
        ):
            result = create_fallback_crawler()

            assert result is not None
            assert isinstance(result, FallbackCrawler)
            assert result.primary.name == "simple"
            assert len(result.fallbacks) == 2
            assert result.fallbacks[0].name == "apify"
            assert result.fallbacks[1].name == "brightdata"

    def test_partial_credentials_skips_provider(self, mock_settings_partial):
        """Test skips provider when credentials are missing."""
        with patch(
            "agentic_rag_backend.indexing.fallback_providers.get_settings",
            return_value=mock_settings_partial,
        ):
            result = create_fallback_crawler()

            assert result is not None
            assert len(result.fallbacks) == 1
            assert result.fallbacks[0].name == "apify"

    def test_custom_primary_provider(self, mock_settings_enabled):
        """Test using custom primary provider."""
        custom_primary = MagicMock(spec=CrawlProvider)
        custom_primary.name = "custom"

        with patch(
            "agentic_rag_backend.indexing.fallback_providers.get_settings",
            return_value=mock_settings_enabled,
        ):
            result = create_fallback_crawler(primary_provider=custom_primary)

            assert result.primary.name == "custom"


class TestCrawlProviderCrawlMany:
    """Tests for crawl_many method on providers."""

    @pytest.fixture
    def provider(self):
        """Create a SimpleCrawlProvider for testing."""
        return SimpleCrawlProvider(timeout_seconds=5.0)

    @pytest.mark.asyncio
    async def test_crawl_many_success(self, provider):
        """Test crawl_many returns results for multiple URLs."""
        urls = ["https://example.com/page1", "https://example.com/page2"]

        mock_responses = [
            MagicMock(text="<html>Page 1</html>", status_code=200),
            MagicMock(text="<html>Page 2</html>", status_code=200),
        ]

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(side_effect=mock_responses)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            results = await provider.crawl_many(urls)

            assert len(results) == 2
            assert results[0].content == "<html>Page 1</html>"
            assert results[1].content == "<html>Page 2</html>"

    @pytest.mark.asyncio
    async def test_crawl_many_partial_failure(self, provider):
        """Test crawl_many handles partial failures gracefully."""
        urls = ["https://example.com/good", "https://example.com/bad"]

        mock_response_good = MagicMock(text="<html>Good</html>", status_code=200)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(
                side_effect=[mock_response_good, httpx.TimeoutException("Timeout")]
            )
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            results = await provider.crawl_many(urls)

            assert len(results) == 2
            assert results[0].content == "<html>Good</html>"
            assert results[1].error == "Request timed out"
