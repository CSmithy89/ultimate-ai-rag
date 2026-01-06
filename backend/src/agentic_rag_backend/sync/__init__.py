"""External data source sync module.

Story 20-H3: Implement External Data Source Sync

This module provides connectors for syncing content from external sources:
- S3: AWS S3 bucket sync
- Confluence: Atlassian Confluence page sync
- Notion: Notion workspace sync

Example:
    from agentic_rag_backend.sync import (
        SyncManager,
        SyncConfig,
        SyncSourceType,
        create_sync_manager,
    )

    # Create manager
    manager = create_sync_manager(enabled=True)

    # Register S3 connector
    manager.register_connector(SyncConfig(
        source_type=SyncSourceType.S3,
        credentials={"aws_access_key_id": "xxx", "aws_secret_access_key": "xxx"},
        settings={"bucket": "my-docs", "prefix": "documents/"},
    ))

    # Run sync
    results = await manager.sync_all()
"""

from .base import BaseConnector
from .confluence_connector import ConfluenceConnector
from .manager import (
    DEFAULT_EXTERNAL_SYNC_ENABLED,
    DEFAULT_MAX_ITEMS_PER_SYNC,
    DEFAULT_SYNC_INTERVAL_MINUTES,
    SyncManager,
    create_sync_manager,
)
from .models import (
    SyncConfig,
    SyncContent,
    SyncItem,
    SyncResult,
    SyncSourceType,
    SyncState,
    SyncStatus,
)
from .notion_connector import NotionConnector
from .s3_connector import S3Connector

__all__ = [
    # Models
    "SyncSourceType",
    "SyncStatus",
    "SyncItem",
    "SyncContent",
    "SyncResult",
    "SyncState",
    "SyncConfig",
    # Base
    "BaseConnector",
    # Connectors
    "S3Connector",
    "ConfluenceConnector",
    "NotionConnector",
    # Manager
    "SyncManager",
    "create_sync_manager",
    # Defaults
    "DEFAULT_EXTERNAL_SYNC_ENABLED",
    "DEFAULT_SYNC_INTERVAL_MINUTES",
    "DEFAULT_MAX_ITEMS_PER_SYNC",
]
