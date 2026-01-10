"""Sync manager for orchestrating external data source connectors.

Story 20-H3: Implement External Data Source Sync

This module provides the SyncManager that orchestrates sync operations
across multiple connectors.
"""

import asyncio
from datetime import datetime
from typing import Any, Optional

import structlog

from .base import BaseConnector
from .confluence_connector import ConfluenceConnector
from .models import (
    SyncConfig,
    SyncResult,
    SyncSourceType,
    SyncState,
    SyncStatus,
)
from .notion_connector import NotionConnector
from .s3_connector import S3Connector

logger = structlog.get_logger(__name__)

# Default configuration values
DEFAULT_EXTERNAL_SYNC_ENABLED = False
DEFAULT_SYNC_INTERVAL_MINUTES = 60
DEFAULT_MAX_ITEMS_PER_SYNC = 1000

# Connector class mapping
CONNECTOR_CLASSES: dict[SyncSourceType, type[BaseConnector]] = {
    SyncSourceType.S3: S3Connector,
    SyncSourceType.CONFLUENCE: ConfluenceConnector,
    SyncSourceType.NOTION: NotionConnector,
}


class SyncManager:
    """Orchestrates sync operations across multiple connectors.

    Manages multiple sync connectors, handles scheduling, and provides
    a unified interface for sync operations.

    Example:
        manager = SyncManager(enabled=True)
        manager.register_connector(s3_config)
        manager.register_connector(confluence_config)

        # Run all syncs
        results = await manager.sync_all()

        # Run specific sync
        result = await manager.sync_source(SyncSourceType.S3)
    """

    def __init__(
        self,
        enabled: bool = DEFAULT_EXTERNAL_SYNC_ENABLED,
        max_concurrent: int = 3,
    ) -> None:
        """Initialize the sync manager.

        Args:
            enabled: Whether sync is enabled
            max_concurrent: Maximum concurrent sync operations
        """
        self._enabled = enabled
        self._max_concurrent = max_concurrent
        self._connectors: dict[str, BaseConnector] = {}
        self._states: dict[str, SyncState] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

        self._logger = logger.bind(component="SyncManager")

        if enabled:
            self._logger.info(
                "sync_manager_enabled",
                max_concurrent=max_concurrent,
            )
        else:
            self._logger.info("sync_manager_disabled")

    @property
    def enabled(self) -> bool:
        """Check if sync is enabled."""
        return self._enabled

    @property
    def connectors(self) -> dict[str, BaseConnector]:
        """Return registered connectors."""
        return self._connectors.copy()

    def register_connector(self, config: SyncConfig) -> str:
        """Register a new connector.

        Args:
            config: Configuration for the connector

        Returns:
            Connector ID
        """
        if not self._enabled:
            self._logger.warning("sync_manager_disabled_cannot_register")
            return ""

        connector_class = CONNECTOR_CLASSES.get(config.source_type)
        if not connector_class:
            self._logger.error(
                "sync_unknown_source_type",
                source_type=config.source_type.value,
            )
            return ""

        connector = connector_class(config)
        connector_id = config.source_id

        # Restore state if available
        if connector_id in self._states:
            connector.state = self._states[connector_id]

        self._connectors[connector_id] = connector

        self._logger.info(
            "sync_connector_registered",
            connector_id=connector_id,
            source_type=config.source_type.value,
        )

        return connector_id

    def unregister_connector(self, connector_id: str) -> bool:
        """Unregister a connector.

        Args:
            connector_id: ID of connector to remove

        Returns:
            True if connector was removed
        """
        if connector_id in self._connectors:
            # Save state before removing
            connector = self._connectors[connector_id]
            if connector.state:
                self._states[connector_id] = connector.state

            del self._connectors[connector_id]

            self._logger.info(
                "sync_connector_unregistered",
                connector_id=connector_id,
            )
            return True

        return False

    async def sync_all(
        self,
        incremental: bool = True,
        max_items: Optional[int] = None,
    ) -> dict[str, SyncResult]:
        """Run sync for all registered connectors.

        Args:
            incremental: If True, only sync changed items
            max_items: Maximum items per connector

        Returns:
            Dict mapping connector IDs to their results
        """
        if not self._enabled:
            self._logger.warning("sync_manager_disabled_cannot_sync")
            return {}

        if not self._connectors:
            self._logger.info("sync_no_connectors_registered")
            return {}

        self._logger.info(
            "sync_all_started",
            connector_count=len(self._connectors),
            incremental=incremental,
        )

        tasks = [
            self._sync_with_semaphore(connector_id, incremental, max_items)
            for connector_id in self._connectors
        ]

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for connector_id, result in zip(self._connectors.keys(), results_list):
            if isinstance(result, Exception):
                results[connector_id] = SyncResult(
                    source_type=self._connectors[connector_id].source_type,
                    status=SyncStatus.FAILED,
                    error_message=str(result),
                )
            else:
                results[connector_id] = result

        # Log summary
        completed = sum(1 for r in results.values() if r.status == SyncStatus.COMPLETED)
        failed = sum(1 for r in results.values() if r.status == SyncStatus.FAILED)
        partial = sum(1 for r in results.values() if r.status == SyncStatus.PARTIAL)

        self._logger.info(
            "sync_all_completed",
            completed=completed,
            partial=partial,
            failed=failed,
        )

        return results

    async def sync_source(
        self,
        connector_id: str,
        incremental: bool = True,
        max_items: Optional[int] = None,
    ) -> Optional[SyncResult]:
        """Run sync for a specific connector.

        Args:
            connector_id: ID of connector to sync
            incremental: If True, only sync changed items
            max_items: Maximum items to sync

        Returns:
            SyncResult or None if connector not found
        """
        if not self._enabled:
            self._logger.warning("sync_manager_disabled_cannot_sync")
            return None

        connector = self._connectors.get(connector_id)
        if not connector:
            self._logger.warning(
                "sync_connector_not_found",
                connector_id=connector_id,
            )
            return None

        return await self._sync_with_semaphore(connector_id, incremental, max_items)

    async def _sync_with_semaphore(
        self,
        connector_id: str,
        incremental: bool,
        max_items: Optional[int],
    ) -> SyncResult:
        """Run sync with semaphore for concurrency control.

        Args:
            connector_id: ID of connector to sync
            incremental: If True, only sync changed items
            max_items: Maximum items to sync

        Returns:
            SyncResult
        """
        async with self._semaphore:
            connector = self._connectors[connector_id]
            result = await connector.sync(
                max_items=max_items,
                incremental=incremental,
            )

            # Update state
            if connector.state:
                connector.state.last_sync = datetime.utcnow()
                self._states[connector_id] = connector.state

            return result

    async def validate_all(self) -> dict[str, bool]:
        """Validate connections for all connectors.

        Returns:
            Dict mapping connector IDs to validation status
        """
        if not self._enabled:
            return {}

        results = {}
        for connector_id, connector in self._connectors.items():
            try:
                results[connector_id] = await connector.validate_connection()
            except Exception as e:
                self._logger.warning(
                    "sync_validation_error",
                    connector_id=connector_id,
                    error=str(e),
                )
                results[connector_id] = False

        return results

    def get_status(self) -> dict[str, Any]:
        """Get status of all connectors.

        Returns:
            Status information dict
        """
        connectors_status = []
        for connector_id, connector in self._connectors.items():
            state = connector.state or self._states.get(connector_id)
            connectors_status.append({
                "id": connector_id,
                "source_type": connector.source_type.value,
                "enabled": connector.config.enabled,
                "last_sync": state.last_sync.isoformat() if state and state.last_sync else None,
                "items_tracked": len(state.etags) if state else 0,
            })

        return {
            "enabled": self._enabled,
            "connector_count": len(self._connectors),
            "connectors": connectors_status,
        }

    async def close_all(self) -> None:
        """Close all connectors."""
        for connector_id, connector in self._connectors.items():
            try:
                if hasattr(connector, "close"):
                    await connector.close()
            except Exception as e:
                self._logger.warning(
                    "sync_close_error",
                    connector_id=connector_id,
                    error=str(e),
                )


def create_sync_manager(
    enabled: bool = DEFAULT_EXTERNAL_SYNC_ENABLED,
    configs: Optional[list[SyncConfig]] = None,
    max_concurrent: int = 3,
) -> SyncManager:
    """Factory function to create a configured SyncManager.

    Args:
        enabled: Whether sync is enabled
        configs: List of connector configurations
        max_concurrent: Maximum concurrent syncs

    Returns:
        Configured SyncManager instance
    """
    manager = SyncManager(enabled=enabled, max_concurrent=max_concurrent)

    if enabled and configs:
        for config in configs:
            manager.register_connector(config)

    return manager
