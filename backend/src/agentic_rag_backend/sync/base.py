"""Base connector protocol for external data source sync.

Story 20-H3: Implement External Data Source Sync

This module defines the BaseConnector protocol that all sync connectors
must implement.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import structlog

from .models import (
    SyncConfig,
    SyncContent,
    SyncItem,
    SyncResult,
    SyncSourceType,
    SyncState,
    SyncStatus,
)

logger = structlog.get_logger(__name__)


class BaseConnector(ABC):
    """Abstract base class for sync connectors.

    All sync connectors must implement this interface to be used
    with the SyncManager.

    Example:
        class MyConnector(BaseConnector):
            @property
            def source_type(self) -> SyncSourceType:
                return SyncSourceType.S3

            async def list_items(self) -> AsyncIterator[SyncItem]:
                ...

            async def fetch_content(self, item: SyncItem) -> SyncContent:
                ...
    """

    def __init__(self, config: SyncConfig) -> None:
        """Initialize the connector.

        Args:
            config: Configuration for this connector
        """
        self._config = config
        self._state: Optional[SyncState] = None
        self._logger = logger.bind(
            source_type=self.source_type.value,
            source_id=config.source_id,
        )

    @property
    @abstractmethod
    def source_type(self) -> SyncSourceType:
        """Return the type of sync source this connector handles."""
        ...

    @property
    def config(self) -> SyncConfig:
        """Return the connector configuration."""
        return self._config

    @property
    def state(self) -> Optional[SyncState]:
        """Return the current sync state."""
        return self._state

    @state.setter
    def state(self, value: SyncState) -> None:
        """Set the sync state."""
        self._state = value

    @abstractmethod
    async def list_items(
        self,
        max_items: Optional[int] = None,
        incremental: bool = True,
    ) -> AsyncIterator[SyncItem]:
        """List items available for sync.

        Args:
            max_items: Maximum number of items to list
            incremental: If True, only list items changed since last sync

        Yields:
            SyncItem objects representing items to sync
        """
        ...

    @abstractmethod
    async def fetch_content(self, item: SyncItem) -> SyncContent:
        """Fetch content for a specific item.

        Args:
            item: The item to fetch content for

        Returns:
            SyncContent with the fetched content
        """
        ...

    async def validate_connection(self) -> bool:
        """Validate that the connector can connect to the source.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Default implementation tries to list one item
            async for _ in self.list_items(max_items=1, incremental=False):
                return True
            return True  # Empty source is still valid
        except Exception as e:
            self._logger.warning(
                "connection_validation_failed",
                error=str(e),
            )
            return False

    async def sync(
        self,
        max_items: Optional[int] = None,
        incremental: bool = True,
    ) -> SyncResult:
        """Run a full sync operation.

        Args:
            max_items: Maximum number of items to sync
            incremental: If True, only sync items changed since last sync

        Returns:
            SyncResult with sync statistics
        """
        result = SyncResult(source_type=self.source_type)
        result.mark_started()

        max_items = max_items or self._config.max_items_per_sync

        try:
            self._logger.info(
                "sync_started",
                incremental=incremental,
                max_items=max_items,
            )

            items_processed = 0
            async for item in self.list_items(
                max_items=max_items,
                incremental=incremental,
            ):
                result.items_found += 1

                try:
                    content = await self.fetch_content(item)
                    # Content is available for downstream processing
                    # In a full implementation, this would be passed to indexing
                    result.items_synced += 1

                    self._logger.debug(
                        "item_synced",
                        item_id=item.id,
                        item_name=item.name,
                        content_size=len(content.text),
                    )
                except Exception as e:
                    result.add_error(item.id, str(e))
                    self._logger.warning(
                        "item_sync_failed",
                        item_id=item.id,
                        error=str(e),
                    )

                items_processed += 1
                if max_items and items_processed >= max_items:
                    break

            result.mark_completed()

            self._logger.info(
                "sync_completed",
                status=result.status.value,
                items_found=result.items_found,
                items_synced=result.items_synced,
                items_failed=result.items_failed,
                duration_seconds=result.duration_seconds,
            )

        except Exception as e:
            result.status = SyncStatus.FAILED
            result.error_message = str(e)
            result.mark_completed()

            self._logger.error(
                "sync_failed",
                error=str(e),
            )

        return result

    def _should_sync_item(self, item: SyncItem) -> bool:
        """Check if an item should be synced based on state.

        Args:
            item: The item to check

        Returns:
            True if item should be synced
        """
        if not self._state:
            return True

        if not item.etag:
            return True

        stored_etag = self._state.etags.get(item.id)
        return stored_etag != item.etag

    def _update_state_for_item(self, item: SyncItem) -> None:
        """Update state after syncing an item.

        Args:
            item: The item that was synced
        """
        if not self._state:
            self._state = SyncState(
                source_type=self.source_type,
                source_id=self._config.source_id,
            )

        if item.etag:
            self._state.etags[item.id] = item.etag
