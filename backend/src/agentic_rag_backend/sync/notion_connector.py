"""Notion sync connector.

Story 20-H3: Implement External Data Source Sync

This module provides a connector for syncing pages from Notion workspaces.
"""

import asyncio
from datetime import datetime
from typing import Any, AsyncIterator, Optional

import httpx
import structlog

from .base import BaseConnector
from .models import SyncConfig, SyncContent, SyncItem, SyncSourceType

logger = structlog.get_logger(__name__)

NOTION_API_VERSION = "2022-06-28"
NOTION_API_BASE = "https://api.notion.com/v1"

# Maximum pagination iterations to prevent infinite loops
MAX_PAGINATION_PAGES = 1000


class NotionConnector(BaseConnector):
    """Sync connector for Notion workspaces.

    Syncs pages from Notion databases, supporting incremental sync
    via last_edited_time.

    Example:
        config = SyncConfig(
            source_type=SyncSourceType.NOTION,
            credentials={"api_key": "secret_xxx"},
            settings={"database_ids": ["db1", "db2"]},
        )
        connector = NotionConnector(config)
        result = await connector.sync()
    """

    def __init__(self, config: SyncConfig) -> None:
        """Initialize Notion connector.

        Args:
            config: Configuration with Notion settings
        """
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def source_type(self) -> SyncSourceType:
        """Return Notion source type."""
        return SyncSourceType.NOTION

    @property
    def database_ids(self) -> list[str]:
        """Return the configured database IDs."""
        db_ids = self._config.settings.get("database_ids", [])
        if isinstance(db_ids, str):
            return [d.strip() for d in db_ids.split(",") if d.strip()]
        return db_ids

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            api_key = self._config.credentials.get("api_key", "")

            self._client = httpx.AsyncClient(
                base_url=NOTION_API_BASE,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Notion-Version": NOTION_API_VERSION,
                    "Content-Type": "application/json",
                },
                timeout=self._config.http_timeout_seconds,
            )

        return self._client

    async def list_items(
        self,
        max_items: Optional[int] = None,
        incremental: bool = True,
    ) -> AsyncIterator[SyncItem]:
        """List pages in configured Notion databases.

        Args:
            max_items: Maximum number of items to list
            incremental: If True, only list items changed since last sync

        Yields:
            SyncItem for each page
        """
        client = await self._get_client()

        if not self.database_ids:
            # If no databases specified, search all accessible pages
            async for item in self._list_all_pages(client, max_items, incremental):
                yield item
            return

        items_yielded = 0

        for database_id in self.database_ids:
            start_cursor = None

            while True:
                try:
                    # Query database for pages
                    payload: dict[str, Any] = {"page_size": 100}
                    if start_cursor:
                        payload["start_cursor"] = start_cursor

                    response = await client.post(
                        f"/databases/{database_id}/query",
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()

                    results = data.get("results", [])
                    if not results:
                        break

                    for page in results:
                        item = self._page_to_item(page, database_id)
                        if item is None:
                            continue

                        # Check if item needs sync (incremental mode)
                        if incremental and not self._should_sync_item(item):
                            continue

                        yield item
                        items_yielded += 1

                        if max_items and items_yielded >= max_items:
                            return

                    # Check for more pages
                    if not data.get("has_more"):
                        break

                    start_cursor = data.get("next_cursor")
                    await asyncio.sleep(0.1)  # Rate limiting

                except httpx.HTTPStatusError as e:
                    self._logger.error(
                        "notion_list_failed",
                        database_id=database_id,
                        status_code=e.response.status_code,
                        error=str(e),
                    )
                    break
                except Exception as e:
                    self._logger.error(
                        "notion_list_error",
                        database_id=database_id,
                        error=str(e),
                    )
                    break

    async def _list_all_pages(
        self,
        client: httpx.AsyncClient,
        max_items: Optional[int],
        incremental: bool,
    ) -> AsyncIterator[SyncItem]:
        """List all accessible pages via search.

        Args:
            client: HTTP client
            max_items: Maximum items to list
            incremental: If True, only list changed items

        Yields:
            SyncItem for each page
        """
        items_yielded = 0
        start_cursor = None

        while True:
            try:
                payload: dict[str, Any] = {
                    "filter": {"property": "object", "value": "page"},
                    "page_size": 100,
                }
                if start_cursor:
                    payload["start_cursor"] = start_cursor

                response = await client.post("/search", json=payload)
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                for page in results:
                    item = self._page_to_item(page)
                    if item is None:
                        continue

                    if incremental and not self._should_sync_item(item):
                        continue

                    yield item
                    items_yielded += 1

                    if max_items and items_yielded >= max_items:
                        return

                if not data.get("has_more"):
                    break

                start_cursor = data.get("next_cursor")
                await asyncio.sleep(0.1)

            except Exception as e:
                self._logger.error("notion_search_error", error=str(e))
                break

    def _page_to_item(
        self,
        page: dict[str, Any],
        database_id: Optional[str] = None,
    ) -> Optional[SyncItem]:
        """Convert Notion page to SyncItem.

        Args:
            page: Notion page object
            database_id: Parent database ID if known

        Returns:
            SyncItem or None if conversion fails
        """
        try:
            page_id = page.get("id", "")
            title = self._extract_title(page)

            last_edited = page.get("last_edited_time")
            last_modified = None
            if last_edited:
                try:
                    last_modified = datetime.fromisoformat(
                        last_edited.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            return SyncItem(
                id=page_id,
                source_type=SyncSourceType.NOTION,
                name=title or f"Untitled ({page_id[:8]})",
                path=f"/pages/{page_id}",
                content_type="text/markdown",
                last_modified=last_modified,
                etag=last_edited or "",
                metadata={
                    "database_id": database_id,
                    "parent_type": page.get("parent", {}).get("type"),
                    "archived": page.get("archived", False),
                    "url": page.get("url"),
                },
            )
        except Exception as e:
            self._logger.warning(
                "notion_page_conversion_failed",
                error=str(e),
            )
            return None

    def _extract_title(self, page: dict[str, Any]) -> str:
        """Extract title from Notion page.

        Args:
            page: Notion page object

        Returns:
            Page title or empty string
        """
        properties = page.get("properties", {})

        # Look for title property
        for prop in properties.values():
            if prop.get("type") == "title":
                title_parts = prop.get("title", [])
                return "".join(
                    t.get("plain_text", "") for t in title_parts
                )

        return ""

    async def fetch_content(self, item: SyncItem) -> SyncContent:
        """Fetch page content from Notion.

        Args:
            item: The page to fetch

        Returns:
            SyncContent with the page content
        """
        client = await self._get_client()

        try:
            # Fetch page blocks
            blocks = await self._fetch_blocks(client, item.id)

            # Convert blocks to text
            text = self._blocks_to_text(blocks)

            self._update_state_for_item(item)

            return SyncContent(
                item=item,
                content=text,
                text=text,
                metadata={
                    "block_count": len(blocks),
                },
            )

        except httpx.HTTPStatusError as e:
            self._logger.error(
                "notion_fetch_failed",
                page_id=item.id,
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            self._logger.error(
                "notion_fetch_error",
                page_id=item.id,
                error=str(e),
            )
            raise

    async def _fetch_blocks(
        self,
        client: httpx.AsyncClient,
        block_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch all blocks for a page.

        Args:
            client: HTTP client
            block_id: Page or block ID

        Returns:
            List of block objects
        """
        blocks = []
        start_cursor = None
        page_count = 0

        while page_count < MAX_PAGINATION_PAGES:
            params: dict[str, Any] = {"page_size": 100}
            if start_cursor:
                params["start_cursor"] = start_cursor

            response = await client.get(
                f"/blocks/{block_id}/children",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            blocks.extend(results)
            page_count += 1

            if not data.get("has_more"):
                break

            start_cursor = data.get("next_cursor")
            await asyncio.sleep(0.05)  # Rate limiting

        if page_count >= MAX_PAGINATION_PAGES:
            self._logger.warning(
                "notion_pagination_limit_reached",
                block_id=block_id,
                max_pages=MAX_PAGINATION_PAGES,
                blocks_fetched=len(blocks),
            )

        return blocks

    def _blocks_to_text(self, blocks: list[dict[str, Any]], depth: int = 0) -> str:
        """Convert Notion blocks to plain text.

        Args:
            blocks: List of Notion block objects
            depth: Current indentation depth

        Returns:
            Plain text representation
        """
        lines = []
        indent = "  " * depth

        for block in blocks:
            block_type = block.get("type", "")
            block_data = block.get(block_type, {})

            # Extract text from rich text arrays
            rich_text = block_data.get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in rich_text)

            if block_type == "paragraph":
                if text:
                    lines.append(f"{indent}{text}")

            elif block_type in ("heading_1", "heading_2", "heading_3"):
                level = int(block_type[-1])
                prefix = "#" * level
                lines.append(f"{indent}{prefix} {text}")

            elif block_type == "bulleted_list_item":
                lines.append(f"{indent}- {text}")

            elif block_type == "numbered_list_item":
                lines.append(f"{indent}1. {text}")

            elif block_type == "to_do":
                checked = block_data.get("checked", False)
                checkbox = "[x]" if checked else "[ ]"
                lines.append(f"{indent}- {checkbox} {text}")

            elif block_type == "code":
                language = block_data.get("language", "")
                lines.append(f"{indent}```{language}")
                lines.append(f"{indent}{text}")
                lines.append(f"{indent}```")

            elif block_type == "quote":
                lines.append(f"{indent}> {text}")

            elif block_type == "divider":
                lines.append(f"{indent}---")

            elif block_type == "callout":
                emoji = block_data.get("icon", {}).get("emoji", "")
                lines.append(f"{indent}{emoji} {text}")

            elif block_type == "toggle":
                lines.append(f"{indent}▸ {text}")

            # Add any text content
            elif text:
                lines.append(f"{indent}{text}")

        return "\n".join(lines)

    async def validate_connection(self) -> bool:
        """Validate Notion connection.

        Returns:
            True if connection is valid
        """
        try:
            client = await self._get_client()
            response = await client.get("/users/me")
            response.raise_for_status()

            self._logger.info("notion_connection_validated")
            return True

        except Exception as e:
            self._logger.warning(
                "notion_connection_failed",
                error=str(e),
            )
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
