"""Atlassian Confluence sync connector.

Story 20-H3: Implement External Data Source Sync

This module provides a connector for syncing pages from Atlassian Confluence.
"""

import asyncio
import base64
import re
from datetime import datetime
from typing import AsyncIterator, Optional

import httpx
import structlog

from .base import BaseConnector
from .models import SyncConfig, SyncContent, SyncItem, SyncSourceType

logger = structlog.get_logger(__name__)


class ConfluenceConnector(BaseConnector):
    """Sync connector for Atlassian Confluence.

    Syncs pages from Confluence spaces, supporting incremental sync
    via last modified timestamps.

    Example:
        config = SyncConfig(
            source_type=SyncSourceType.CONFLUENCE,
            credentials={
                "url": "https://your-domain.atlassian.net/wiki",
                "email": "user@example.com",
                "api_token": "your-api-token",
            },
            settings={"spaces": ["SPACE1", "SPACE2"]},
        )
        connector = ConfluenceConnector(config)
        result = await connector.sync()
    """

    def __init__(self, config: SyncConfig) -> None:
        """Initialize Confluence connector.

        Args:
            config: Configuration with Confluence settings
        """
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def source_type(self) -> SyncSourceType:
        """Return Confluence source type."""
        return SyncSourceType.CONFLUENCE

    @property
    def base_url(self) -> str:
        """Return the Confluence base URL."""
        url = self._config.credentials.get("url", "")
        return url.rstrip("/")

    @property
    def spaces(self) -> list[str]:
        """Return the configured space keys."""
        spaces = self._config.settings.get("spaces", [])
        if isinstance(spaces, str):
            return [s.strip() for s in spaces.split(",") if s.strip()]
        return spaces

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            email = self._config.credentials.get("email", "")
            api_token = self._config.credentials.get("api_token", "")

            # Basic auth with email:api_token
            auth_string = f"{email}:{api_token}"
            auth_bytes = base64.b64encode(auth_string.encode()).decode()

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Basic {auth_bytes}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=self._config.http_timeout_seconds,
            )

        return self._client

    async def list_items(
        self,
        max_items: Optional[int] = None,
        incremental: bool = True,
    ) -> AsyncIterator[SyncItem]:
        """List pages in configured Confluence spaces.

        Args:
            max_items: Maximum number of items to list
            incremental: If True, only list items changed since last sync

        Yields:
            SyncItem for each page
        """
        client = await self._get_client()

        if not self.spaces:
            self._logger.warning("confluence_no_spaces_configured")
            return

        items_yielded = 0

        for space_key in self.spaces:
            start = 0
            limit = 25

            while True:
                try:
                    # Use Confluence REST API v2
                    response = await client.get(
                        f"/api/v2/spaces/{space_key}/pages",
                        params={
                            "start": start,
                            "limit": limit,
                            "body-format": "storage",
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    results = data.get("results", [])
                    if not results:
                        break

                    for page in results:
                        page_id = page.get("id", "")
                        title = page.get("title", "")
                        version = page.get("version", {})
                        last_modified_str = version.get("createdAt")

                        last_modified = None
                        if last_modified_str:
                            try:
                                last_modified = datetime.fromisoformat(
                                    last_modified_str.replace("Z", "+00:00")
                                )
                            except (ValueError, TypeError):
                                pass

                        item = SyncItem(
                            id=page_id,
                            source_type=SyncSourceType.CONFLUENCE,
                            name=title,
                            path=f"/spaces/{space_key}/pages/{page_id}",
                            content_type="text/html",
                            last_modified=last_modified,
                            etag=str(version.get("number", "")),
                            metadata={
                                "space_key": space_key,
                                "version": version.get("number"),
                                "status": page.get("status"),
                            },
                        )

                        # Check if item needs sync (incremental mode)
                        if incremental and not self._should_sync_item(item):
                            continue

                        yield item
                        items_yielded += 1

                        if max_items and items_yielded >= max_items:
                            return

                    # Check for more pages
                    links = data.get("_links", {})
                    if "next" not in links:
                        break

                    start += limit
                    await asyncio.sleep(0.1)  # Rate limiting

                except httpx.HTTPStatusError as e:
                    self._logger.error(
                        "confluence_list_failed",
                        space_key=space_key,
                        status_code=e.response.status_code,
                        error=str(e),
                    )
                    break
                except Exception as e:
                    self._logger.error(
                        "confluence_list_error",
                        space_key=space_key,
                        error=str(e),
                    )
                    break

    async def fetch_content(self, item: SyncItem) -> SyncContent:
        """Fetch page content from Confluence.

        Args:
            item: The page to fetch

        Returns:
            SyncContent with the page content
        """
        client = await self._get_client()

        try:
            # Fetch page with body content
            response = await client.get(
                f"/api/v2/pages/{item.id}",
                params={"body-format": "storage"},
            )
            response.raise_for_status()
            data = response.json()

            # Get body content
            body = data.get("body", {})
            storage = body.get("storage", {})
            html_content = storage.get("value", "")

            # Extract plain text from HTML
            text = self._extract_text(html_content)

            self._update_state_for_item(item)

            return SyncContent(
                item=item,
                content=html_content,
                text=text,
                metadata={
                    "title": data.get("title"),
                    "space_id": data.get("spaceId"),
                    "parent_id": data.get("parentId"),
                    "version": data.get("version", {}).get("number"),
                },
            )

        except httpx.HTTPStatusError as e:
            self._logger.error(
                "confluence_fetch_failed",
                page_id=item.id,
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            self._logger.error(
                "confluence_fetch_error",
                page_id=item.id,
                error=str(e),
            )
            raise

    def _extract_text(self, html_content: str) -> str:
        """Extract plain text from Confluence storage format.

        Args:
            html_content: HTML/XHTML content from Confluence

        Returns:
            Plain text content
        """
        if not html_content:
            return ""

        try:
            # Remove macro tags
            text = re.sub(r"<ac:[^>]+>", "", html_content)
            text = re.sub(r"</ac:[^>]+>", "", text)

            # Remove ri tags
            text = re.sub(r"<ri:[^>]+>", "", text)
            text = re.sub(r"</ri:[^>]+>", "", text)

            # Remove CDATA sections but keep content
            text = re.sub(r"<!\[CDATA\[", "", text)
            text = re.sub(r"\]\]>", "", text)

            # Remove HTML tags
            text = re.sub(r"<[^>]+>", " ", text)

            # Decode HTML entities
            import html
            text = html.unescape(text)

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()

            return text

        except Exception as e:
            self._logger.warning(
                "confluence_text_extraction_failed",
                error=str(e),
            )
            return ""

    async def validate_connection(self) -> bool:
        """Validate Confluence connection.

        Returns:
            True if connection is valid
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/v2/spaces", params={"limit": 1})
            response.raise_for_status()

            self._logger.info(
                "confluence_connection_validated",
                base_url=self.base_url,
            )
            return True

        except Exception as e:
            self._logger.warning(
                "confluence_connection_failed",
                base_url=self.base_url,
                error=str(e),
            )
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
