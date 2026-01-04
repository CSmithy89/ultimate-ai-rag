"""Fallback crawl providers for anti-bot bypass.

This module implements a fallback chain pattern for web crawling when
the primary crawler fails due to anti-bot measures, rate limiting,
or blocking. Supports Apify and BrightData as fallback providers.

Story 13-1: Integrate Apify BrightData Fallback
"""

from abc import ABC, abstractmethod
from typing import Optional
import time

import httpx
import structlog
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

from agentic_rag_backend.config import get_settings
from agentic_rag_backend.core.errors import CrawlError

logger = structlog.get_logger(__name__)

# Error detection patterns for triggering fallback
BLOCKED_STATUS_CODES = {403, 429, 503}
BLOCKED_CONTENT_PATTERNS = [
    "captcha",
    "challenge",
    "verify you are human",
    "access denied",
    "blocked",
    "rate limit",
    "too many requests",
]

# Apify API constants
APIFY_BASE_URL = "https://api.apify.com/v2"
APIFY_DEFAULT_ACTOR = "apify/web-scraper"

# BrightData proxy constants
BRIGHTDATA_PROXY_HOST = "brd.superproxy.io"
BRIGHTDATA_PROXY_PORT = 33335

# Default timeouts
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_APIFY_WAIT_SECONDS = 120


class CrawlResult(BaseModel):
    """Unified result from any crawl provider."""

    url: str = Field(..., description="Crawled URL")
    content: str = Field(..., description="Page content (HTML or text)")
    status_code: int = Field(..., description="HTTP status code")
    error: Optional[str] = Field(default=None, description="Error message if any")
    provider: Optional[str] = Field(default=None, description="Provider that succeeded")
    elapsed_ms: Optional[int] = Field(default=None, description="Time taken in milliseconds")


class CrawlProvider(ABC):
    """Abstract base class for crawl providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and tracking."""
        pass

    @abstractmethod
    async def crawl(self, url: str, options: Optional[dict] = None) -> CrawlResult:
        """
        Crawl a single URL and return content.

        Args:
            url: The URL to crawl
            options: Optional provider-specific options

        Returns:
            CrawlResult with the crawled content
        """
        pass

    async def crawl_many(
        self, urls: list[str], options: Optional[dict] = None
    ) -> list[CrawlResult]:
        """
        Crawl multiple URLs. Default implementation is sequential.

        Args:
            urls: List of URLs to crawl
            options: Optional provider-specific options

        Returns:
            List of CrawlResult objects
        """
        results = []
        for url in urls:
            try:
                result = await self.crawl(url, options)
                results.append(result)
            except Exception as e:
                logger.warning(
                    "crawl_many_item_failed",
                    provider=self.name,
                    url=url,
                    error=str(e),
                )
                results.append(
                    CrawlResult(
                        url=url,
                        content="",
                        status_code=0,
                        error=str(e),
                        provider=self.name,
                    )
                )
        return results

    def _is_blocked_response(self, content: str, status_code: int) -> bool:
        """
        Check if response indicates blocking.

        Args:
            content: Response content
            status_code: HTTP status code

        Returns:
            True if response indicates blocking
        """
        if status_code in BLOCKED_STATUS_CODES:
            return True

        content_lower = content.lower()
        for pattern in BLOCKED_CONTENT_PATTERNS:
            if pattern in content_lower:
                return True

        return False


class ApifyProvider(CrawlProvider):
    """
    Apify Web Scraper provider.

    Uses Apify's actor API to crawl URLs with JavaScript rendering
    and automatic anti-bot bypass.

    API Documentation: https://docs.apify.com/api/v2
    """

    def __init__(
        self,
        api_token: str,
        actor_id: str = APIFY_DEFAULT_ACTOR,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        wait_seconds: int = DEFAULT_APIFY_WAIT_SECONDS,
    ) -> None:
        """
        Initialize Apify provider.

        Args:
            api_token: Apify API token
            actor_id: Actor to use for crawling
            timeout_seconds: HTTP timeout for API calls
            wait_seconds: Max seconds to wait for actor completion
        """
        self._api_token = api_token
        self._actor_id = actor_id
        self._timeout = timeout_seconds
        self._wait_seconds = wait_seconds

    @property
    def name(self) -> str:
        """Provider name."""
        return "apify"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    )
    async def crawl(self, url: str, options: Optional[dict] = None) -> CrawlResult:
        """
        Crawl a URL using Apify Web Scraper.

        Args:
            url: The URL to crawl
            options: Optional crawl options (e.g., waitForSelector)

        Returns:
            CrawlResult with crawled content
        """
        start_time = time.monotonic()
        options = options or {}

        logger.info("apify_crawl_start", url=url, actor_id=self._actor_id)

        # Prepare actor input
        actor_input = {
            "startUrls": [{"url": url}],
            "maxRequestsPerCrawl": 1,
            "maxConcurrency": 1,
            "pageFunction": """async function pageFunction(context) {
                const { page, request } = context;
                const title = await page.title();
                const content = await page.content();
                return {
                    url: request.url,
                    title,
                    html: content,
                };
            }""",
            **options,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # Start actor run with synchronous wait
            run_url = (
                f"{APIFY_BASE_URL}/acts/{self._actor_id}/runs"
                f"?token={self._api_token}&waitForFinish={self._wait_seconds}"
            )

            response = await client.post(run_url, json=actor_input)

            if response.status_code != 201:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                error_msg = f"Apify API error: {response.status_code}"
                logger.warning(
                    "apify_crawl_api_error",
                    url=url,
                    status_code=response.status_code,
                    response_text=response.text[:500],
                )
                return CrawlResult(
                    url=url,
                    content="",
                    status_code=response.status_code,
                    error=error_msg,
                    provider=self.name,
                    elapsed_ms=elapsed_ms,
                )

            run_data = response.json().get("data", {})
            run_status = run_data.get("status")
            dataset_id = run_data.get("defaultDatasetId")

            if run_status != "SUCCEEDED":
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                error_msg = f"Actor run failed with status: {run_status}"
                logger.warning(
                    "apify_crawl_run_failed",
                    url=url,
                    run_status=run_status,
                    run_id=run_data.get("id"),
                )
                return CrawlResult(
                    url=url,
                    content="",
                    status_code=500,
                    error=error_msg,
                    provider=self.name,
                    elapsed_ms=elapsed_ms,
                )

            # Fetch results from dataset
            dataset_url = f"{APIFY_BASE_URL}/datasets/{dataset_id}/items?token={self._api_token}"
            dataset_response = await client.get(dataset_url)

            if dataset_response.status_code != 200:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                error_msg = f"Failed to fetch dataset: {dataset_response.status_code}"
                logger.warning(
                    "apify_crawl_dataset_error",
                    url=url,
                    dataset_id=dataset_id,
                    status_code=dataset_response.status_code,
                )
                return CrawlResult(
                    url=url,
                    content="",
                    status_code=dataset_response.status_code,
                    error=error_msg,
                    provider=self.name,
                    elapsed_ms=elapsed_ms,
                )

            items = dataset_response.json()
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            if not items:
                logger.warning("apify_crawl_no_results", url=url, dataset_id=dataset_id)
                return CrawlResult(
                    url=url,
                    content="",
                    status_code=200,
                    error="No results returned from actor",
                    provider=self.name,
                    elapsed_ms=elapsed_ms,
                )

            # Extract content from first result
            item = items[0]
            content = item.get("html", "") or item.get("text", "") or ""

            logger.info(
                "apify_crawl_success",
                url=url,
                content_length=len(content),
                elapsed_ms=elapsed_ms,
            )

            return CrawlResult(
                url=url,
                content=content,
                status_code=200,
                provider=self.name,
                elapsed_ms=elapsed_ms,
            )


class BrightDataProvider(CrawlProvider):
    """
    BrightData Scraping Browser provider.

    Uses BrightData's proxy infrastructure with JavaScript rendering
    and automatic anti-bot bypass.

    Documentation: https://docs.brightdata.com/scraping-browser
    """

    def __init__(
        self,
        username: str,
        password: str,
        zone: str = "scraping_browser",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Initialize BrightData provider.

        Args:
            username: BrightData username (format: brd-customer-xxx)
            password: BrightData password
            zone: BrightData zone for proxy selection
            timeout_seconds: HTTP timeout for requests
        """
        self._username = username
        self._password = password
        self._zone = zone
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        """Provider name."""
        return "brightdata"

    @property
    def _proxy_url(self) -> str:
        """Build proxy URL with zone."""
        # Format: http://username-zone:password@host:port
        user_with_zone = f"{self._username}-zone-{self._zone}"
        return f"http://{user_with_zone}:{self._password}@{BRIGHTDATA_PROXY_HOST}:{BRIGHTDATA_PROXY_PORT}"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    )
    async def crawl(self, url: str, options: Optional[dict] = None) -> CrawlResult:
        """
        Crawl a URL using BrightData Scraping Browser.

        Args:
            url: The URL to crawl
            options: Optional options (e.g., headers)

        Returns:
            CrawlResult with crawled content
        """
        start_time = time.monotonic()
        options = options or {}

        logger.info("brightdata_crawl_start", url=url, zone=self._zone)

        headers = options.get("headers", {})
        if "User-Agent" not in headers:
            headers["User-Agent"] = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

        try:
            # Use proxy for the request
            async with httpx.AsyncClient(
                proxy=self._proxy_url,
                timeout=self._timeout,
                verify=False,  # BrightData proxy may use self-signed certs
            ) as client:
                response = await client.get(url, headers=headers)

                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                content = response.text

                # Check for blocking
                if self._is_blocked_response(content, response.status_code):
                    logger.warning(
                        "brightdata_crawl_blocked",
                        url=url,
                        status_code=response.status_code,
                        elapsed_ms=elapsed_ms,
                    )
                    return CrawlResult(
                        url=url,
                        content=content,
                        status_code=response.status_code,
                        error="Response indicates blocking",
                        provider=self.name,
                        elapsed_ms=elapsed_ms,
                    )

                logger.info(
                    "brightdata_crawl_success",
                    url=url,
                    status_code=response.status_code,
                    content_length=len(content),
                    elapsed_ms=elapsed_ms,
                )

                return CrawlResult(
                    url=url,
                    content=content,
                    status_code=response.status_code,
                    provider=self.name,
                    elapsed_ms=elapsed_ms,
                )

        except httpx.ProxyError as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            error_msg = f"Proxy connection failed: {str(e)}"
            logger.error(
                "brightdata_crawl_proxy_error",
                url=url,
                error=str(e),
                elapsed_ms=elapsed_ms,
            )
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                error=error_msg,
                provider=self.name,
                elapsed_ms=elapsed_ms,
            )


class FallbackCrawler:
    """
    Crawler with automatic fallback to paid providers.

    Implements a chain-of-responsibility pattern where the primary
    crawler is tried first, falling back to alternative providers
    if the primary fails or is blocked.
    """

    def __init__(
        self,
        primary: CrawlProvider,
        fallbacks: list[CrawlProvider],
    ) -> None:
        """
        Initialize fallback crawler.

        Args:
            primary: Primary crawl provider to try first
            fallbacks: List of fallback providers in order of preference
        """
        self.primary = primary
        self.fallbacks = fallbacks
        self._fallback_usage_counts: dict[str, int] = {}

    async def crawl_with_fallback(
        self,
        url: str,
        options: Optional[dict] = None,
    ) -> tuple[CrawlResult, str]:
        """
        Attempt primary crawl, fall back to alternatives on failure.

        Args:
            url: URL to crawl
            options: Optional crawl options

        Returns:
            Tuple of (CrawlResult, provider_name)

        Raises:
            CrawlError: If all providers fail
        """
        providers_tried: list[str] = []
        last_error: Optional[str] = None

        # Try primary first
        try:
            logger.info("primary_crawl_attempt", url=url, provider=self.primary.name)
            result = await self.primary.crawl(url, options)

            if self._is_success(result):
                logger.info(
                    "primary_crawl_success",
                    url=url,
                    provider=self.primary.name,
                    elapsed_ms=result.elapsed_ms,
                )
                return result, self.primary.name

            # Primary returned but with error/blocking
            last_error = result.error or f"Status code: {result.status_code}"
            providers_tried.append(self.primary.name)
            logger.warning(
                "primary_crawl_failed",
                url=url,
                provider=self.primary.name,
                status_code=result.status_code,
                error=last_error,
            )

        except Exception as e:
            last_error = str(e)
            providers_tried.append(self.primary.name)
            logger.warning(
                "primary_crawl_exception",
                url=url,
                provider=self.primary.name,
                error=str(e),
            )

        # Try fallbacks in order
        for fallback in self.fallbacks:
            try:
                logger.info(
                    "fallback_triggered",
                    url=url,
                    provider=fallback.name,
                    reason=last_error,
                    providers_tried=providers_tried,
                )

                result = await fallback.crawl(url, options)

                if self._is_success(result):
                    # Track usage for cost monitoring
                    self._fallback_usage_counts[fallback.name] = (
                        self._fallback_usage_counts.get(fallback.name, 0) + 1
                    )

                    logger.info(
                        "fallback_crawl_succeeded",
                        url=url,
                        provider=fallback.name,
                        elapsed_ms=result.elapsed_ms,
                        total_usage=self._fallback_usage_counts[fallback.name],
                    )
                    return result, fallback.name

                # Fallback returned but with error/blocking
                last_error = result.error or f"Status code: {result.status_code}"
                providers_tried.append(fallback.name)
                logger.warning(
                    "fallback_crawl_failed",
                    url=url,
                    provider=fallback.name,
                    status_code=result.status_code,
                    error=last_error,
                )

            except Exception as e:
                last_error = str(e)
                providers_tried.append(fallback.name)
                logger.warning(
                    "fallback_crawl_exception",
                    url=url,
                    provider=fallback.name,
                    error=str(e),
                )

        # All providers failed
        logger.error(
            "all_providers_failed",
            url=url,
            providers_tried=providers_tried,
            last_error=last_error,
        )
        raise CrawlError(url, f"All providers failed: {', '.join(providers_tried)}")

    def _is_success(self, result: CrawlResult) -> bool:
        """
        Check if crawl result indicates success.

        Args:
            result: The crawl result to check

        Returns:
            True if result indicates successful crawl
        """
        return (
            result.status_code == 200
            and result.error is None
            and len(result.content.strip()) > 0
        )

    def get_fallback_usage(self) -> dict[str, int]:
        """
        Get fallback provider usage counts for cost monitoring.

        Returns:
            Dictionary mapping provider names to usage counts
        """
        return self._fallback_usage_counts.copy()


class SimpleCrawlProvider(CrawlProvider):
    """
    Simple httpx-based crawler for testing and as primary provider.

    This is a basic implementation that can be used as the primary
    provider before falling back to paid services.
    """

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        user_agent: Optional[str] = None,
    ) -> None:
        """
        Initialize simple crawler.

        Args:
            timeout_seconds: Request timeout
            user_agent: Custom user agent string
        """
        self._timeout = timeout_seconds
        self._user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "simple"

    async def crawl(self, url: str, options: Optional[dict] = None) -> CrawlResult:
        """
        Crawl a URL using simple httpx client.

        Args:
            url: The URL to crawl
            options: Optional options (e.g., headers)

        Returns:
            CrawlResult with crawled content
        """
        start_time = time.monotonic()
        options = options or {}

        headers = {"User-Agent": self._user_agent, **options.get("headers", {})}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)

                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                content = response.text

                # Check for blocking
                if self._is_blocked_response(content, response.status_code):
                    return CrawlResult(
                        url=url,
                        content=content,
                        status_code=response.status_code,
                        error="Response indicates blocking",
                        provider=self.name,
                        elapsed_ms=elapsed_ms,
                    )

                return CrawlResult(
                    url=url,
                    content=content,
                    status_code=response.status_code,
                    provider=self.name,
                    elapsed_ms=elapsed_ms,
                )

        except httpx.TimeoutException:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                error="Request timed out",
                provider=self.name,
                elapsed_ms=elapsed_ms,
            )

        except httpx.RequestError as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                error=str(e),
                provider=self.name,
                elapsed_ms=elapsed_ms,
            )


def create_fallback_crawler(
    primary_provider: Optional[CrawlProvider] = None,
) -> Optional[FallbackCrawler]:
    """
    Factory function to create FallbackCrawler from configuration.

    Creates provider instances based on environment configuration and
    returns a configured FallbackCrawler, or None if fallback is disabled.

    Args:
        primary_provider: Optional primary provider to use instead of SimpleCrawlProvider

    Returns:
        Configured FallbackCrawler instance, or None if disabled
    """
    settings = get_settings()

    if not settings.crawl_fallback_enabled:
        logger.info("fallback_crawler_disabled")
        return None

    # Build primary provider if not provided
    primary = primary_provider or SimpleCrawlProvider()

    # Build fallback providers based on configuration
    fallbacks: list[CrawlProvider] = []

    for provider_name in settings.crawl_fallback_providers:
        if provider_name == "apify":
            if settings.apify_api_token:
                fallbacks.append(
                    ApifyProvider(api_token=settings.apify_api_token)
                )
                logger.info("fallback_provider_configured", provider="apify")
            else:
                logger.warning(
                    "fallback_provider_skipped",
                    provider="apify",
                    reason="APIFY_API_TOKEN not set",
                )

        elif provider_name == "brightdata":
            if settings.brightdata_username and settings.brightdata_password:
                fallbacks.append(
                    BrightDataProvider(
                        username=settings.brightdata_username,
                        password=settings.brightdata_password,
                        zone=settings.brightdata_zone,
                    )
                )
                logger.info(
                    "fallback_provider_configured",
                    provider="brightdata",
                    zone=settings.brightdata_zone,
                )
            else:
                logger.warning(
                    "fallback_provider_skipped",
                    provider="brightdata",
                    reason="BRIGHTDATA_USERNAME or BRIGHTDATA_PASSWORD not set",
                )

        else:
            logger.warning(
                "fallback_provider_unknown",
                provider=provider_name,
            )

    if not fallbacks:
        logger.warning(
            "no_fallback_providers_configured",
            hint="Set APIFY_API_TOKEN or BRIGHTDATA credentials to enable fallbacks",
        )

    logger.info(
        "fallback_crawler_created",
        primary=primary.name,
        fallbacks=[f.name for f in fallbacks],
    )

    return FallbackCrawler(primary=primary, fallbacks=fallbacks)
