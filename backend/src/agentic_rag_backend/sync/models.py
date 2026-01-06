"""Data models for external data source sync.

Story 20-H3: Implement External Data Source Sync

This module defines the core data models used by sync connectors
and the sync manager.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SyncSourceType(str, Enum):
    """Supported sync source types."""

    S3 = "s3"
    CONFLUENCE = "confluence"
    NOTION = "notion"
    GOOGLE_DRIVE = "google_drive"
    DISCORD = "discord"


class SyncStatus(str, Enum):
    """Status of a sync operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class SyncItem:
    """Represents a single item to sync.

    Attributes:
        id: Unique identifier for the item
        source_type: Type of source (s3, confluence, etc.)
        name: Human-readable name
        path: Path or location in source
        content_type: MIME type or content category
        size_bytes: Size in bytes (if known)
        last_modified: Last modification timestamp
        etag: Entity tag for change detection
        metadata: Additional source-specific metadata
    """

    id: str
    source_type: SyncSourceType
    name: str
    path: str
    content_type: str = "application/octet-stream"
    size_bytes: Optional[int] = None
    last_modified: Optional[datetime] = None
    etag: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncContent:
    """Content fetched from a sync source.

    Attributes:
        item: The SyncItem this content belongs to
        content: Raw content (bytes or text)
        text: Extracted text content
        metadata: Additional metadata extracted from content
    """

    item: SyncItem
    content: bytes | str
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncResult:
    """Result of a sync operation.

    Attributes:
        source_type: Type of source that was synced
        status: Status of the sync operation
        items_found: Number of items found
        items_synced: Number of items successfully synced
        items_failed: Number of items that failed
        started_at: When sync started
        completed_at: When sync completed
        duration_seconds: Total duration in seconds
        error_message: Error message if failed
        errors: List of individual item errors
    """

    source_type: SyncSourceType
    status: SyncStatus = SyncStatus.PENDING
    items_found: int = 0
    items_synced: int = 0
    items_failed: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    errors: list[dict[str, str]] = field(default_factory=list)

    def mark_started(self) -> None:
        """Mark the sync as started."""
        self.status = SyncStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()

    def mark_completed(self) -> None:
        """Mark the sync as completed."""
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()

        if self.items_failed == 0:
            self.status = SyncStatus.COMPLETED
        elif self.items_synced > 0:
            self.status = SyncStatus.PARTIAL
        else:
            self.status = SyncStatus.FAILED

    def add_error(self, item_id: str, error: str) -> None:
        """Add an error for a specific item."""
        self.items_failed += 1
        self.errors.append({"item_id": item_id, "error": error})


@dataclass
class SyncState:
    """Persistent state for a sync source.

    Attributes:
        source_type: Type of source
        source_id: Unique identifier for this source instance
        last_sync: When last successful sync completed
        last_cursor: Cursor/token for incremental sync
        etags: Mapping of item IDs to their ETags
        metadata: Additional state metadata
    """

    source_type: SyncSourceType
    source_id: str
    last_sync: Optional[datetime] = None
    last_cursor: Optional[str] = None
    etags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncConfig:
    """Configuration for a sync source.

    Attributes:
        source_type: Type of source
        enabled: Whether this source is enabled
        source_id: Unique identifier for this source instance
        credentials: Authentication credentials
        settings: Source-specific settings
        sync_interval_minutes: How often to sync (0 = manual only)
        max_items_per_sync: Maximum items to sync per run
    """

    source_type: SyncSourceType
    enabled: bool = True
    source_id: str = ""
    credentials: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    sync_interval_minutes: int = 60
    max_items_per_sync: int = 1000

    def __post_init__(self) -> None:
        """Generate source_id if not provided."""
        if not self.source_id:
            self.source_id = f"{self.source_type.value}-default"
