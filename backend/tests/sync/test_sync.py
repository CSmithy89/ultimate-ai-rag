"""Unit tests for external data source sync (Story 20-H3).

Tests cover:
- SyncModels (SyncItem, SyncResult, SyncState, SyncConfig)
- BaseConnector protocol
- S3Connector with mocked boto3
- ConfluenceConnector with mocked API
- NotionConnector with mocked API
- SyncManager orchestration
- Feature flag behavior
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag_backend.sync import (
    BaseConnector,
    ConfluenceConnector,
    DEFAULT_EXTERNAL_SYNC_ENABLED,
    NotionConnector,
    S3Connector,
    SyncConfig,
    SyncContent,
    SyncItem,
    SyncManager,
    SyncResult,
    SyncSourceType,
    SyncState,
    SyncStatus,
    create_sync_manager,
)


# ============================================================================
# SyncModels Tests
# ============================================================================


class TestSyncItem:
    """Tests for SyncItem dataclass."""

    def test_create_item(self):
        """Test creating a sync item."""
        item = SyncItem(
            id="test-123",
            source_type=SyncSourceType.S3,
            name="document.txt",
            path="docs/document.txt",
            content_type="text/plain",
            size_bytes=1024,
        )
        assert item.id == "test-123"
        assert item.source_type == SyncSourceType.S3
        assert item.name == "document.txt"
        assert item.size_bytes == 1024

    def test_item_with_etag(self):
        """Test item with ETag for change detection."""
        item = SyncItem(
            id="test-456",
            source_type=SyncSourceType.S3,
            name="file.json",
            path="data/file.json",
            etag="abc123",
            last_modified=datetime.utcnow(),
        )
        assert item.etag == "abc123"
        assert item.last_modified is not None


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_create_result(self):
        """Test creating a sync result."""
        result = SyncResult(source_type=SyncSourceType.S3)
        assert result.status == SyncStatus.PENDING
        assert result.items_found == 0
        assert result.items_synced == 0

    def test_mark_started(self):
        """Test marking sync as started."""
        result = SyncResult(source_type=SyncSourceType.CONFLUENCE)
        result.mark_started()
        assert result.status == SyncStatus.IN_PROGRESS
        assert result.started_at is not None

    def test_mark_completed_success(self):
        """Test marking sync as completed (success)."""
        result = SyncResult(source_type=SyncSourceType.NOTION)
        result.mark_started()
        result.items_synced = 5
        result.mark_completed()
        assert result.status == SyncStatus.COMPLETED
        assert result.completed_at is not None
        assert result.duration_seconds >= 0

    def test_mark_completed_partial(self):
        """Test marking sync as completed (partial)."""
        result = SyncResult(source_type=SyncSourceType.S3)
        result.mark_started()
        result.items_synced = 3
        result.items_failed = 2
        result.mark_completed()
        assert result.status == SyncStatus.PARTIAL

    def test_mark_completed_failed(self):
        """Test marking sync as completed (all failed)."""
        result = SyncResult(source_type=SyncSourceType.S3)
        result.mark_started()
        result.items_failed = 5
        result.mark_completed()
        assert result.status == SyncStatus.FAILED

    def test_add_error(self):
        """Test adding item errors."""
        result = SyncResult(source_type=SyncSourceType.S3)
        result.add_error("item-1", "Connection timeout")
        assert result.items_failed == 1
        assert len(result.errors) == 1
        assert result.errors[0]["item_id"] == "item-1"


class TestSyncState:
    """Tests for SyncState dataclass."""

    def test_create_state(self):
        """Test creating sync state."""
        state = SyncState(
            source_type=SyncSourceType.S3,
            source_id="s3-default",
        )
        assert state.source_type == SyncSourceType.S3
        assert state.last_sync is None
        assert len(state.etags) == 0

    def test_state_with_etags(self):
        """Test state with stored ETags."""
        state = SyncState(
            source_type=SyncSourceType.S3,
            source_id="s3-default",
            etags={"item-1": "etag-1", "item-2": "etag-2"},
        )
        assert len(state.etags) == 2


class TestSyncConfig:
    """Tests for SyncConfig dataclass."""

    def test_create_config(self):
        """Test creating sync config."""
        config = SyncConfig(
            source_type=SyncSourceType.S3,
            credentials={"aws_access_key_id": "xxx"},
            settings={"bucket": "my-bucket"},
        )
        assert config.source_type == SyncSourceType.S3
        assert config.enabled is True
        assert config.source_id == "s3-default"

    def test_config_custom_id(self):
        """Test config with custom source ID."""
        config = SyncConfig(
            source_type=SyncSourceType.CONFLUENCE,
            source_id="confluence-prod",
        )
        assert config.source_id == "confluence-prod"


# ============================================================================
# S3Connector Tests
# ============================================================================


class TestS3Connector:
    """Tests for S3Connector class."""

    @pytest.fixture
    def s3_config(self):
        """Create S3 config fixture."""
        return SyncConfig(
            source_type=SyncSourceType.S3,
            credentials={
                "aws_access_key_id": "test-key",
                "aws_secret_access_key": "test-secret",
            },
            settings={"bucket": "test-bucket", "prefix": "documents/"},
        )

    def test_initialization(self, s3_config):
        """Test S3 connector initialization."""
        connector = S3Connector(s3_config)
        assert connector.source_type == SyncSourceType.S3
        assert connector.bucket == "test-bucket"
        assert connector.prefix == "documents/"

    @pytest.mark.asyncio
    async def test_list_items_with_mock(self, s3_config):
        """Test listing S3 items with mocked boto3."""
        connector = S3Connector(s3_config)

        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "documents/file1.txt",
                        "Size": 1024,
                        "ETag": '"abc123"',
                        "LastModified": datetime.utcnow(),
                    },
                    {
                        "Key": "documents/file2.json",
                        "Size": 2048,
                        "ETag": '"def456"',
                        "LastModified": datetime.utcnow(),
                    },
                ]
            }
        ]

        connector._client = mock_client

        items = []
        async for item in connector.list_items(incremental=False):
            items.append(item)

        assert len(items) == 2
        assert items[0].name == "file1.txt"
        assert items[1].name == "file2.json"

    @pytest.mark.asyncio
    async def test_fetch_content_with_mock(self, s3_config):
        """Test fetching S3 content with mock."""
        connector = S3Connector(s3_config)

        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"Hello, World!"
        mock_client.get_object.return_value = {
            "Body": mock_body,
            "ContentLength": 13,
        }

        connector._client = mock_client

        item = SyncItem(
            id="documents/test.txt",
            source_type=SyncSourceType.S3,
            name="test.txt",
            path="documents/test.txt",
            content_type="text/plain",
        )

        content = await connector.fetch_content(item)
        assert content.text == "Hello, World!"

    @pytest.mark.asyncio
    async def test_validate_connection_with_mock(self, s3_config):
        """Test validating S3 connection."""
        connector = S3Connector(s3_config)

        mock_client = MagicMock()
        mock_client.head_bucket.return_value = {}
        connector._client = mock_client

        result = await connector.validate_connection()
        assert result is True


# ============================================================================
# ConfluenceConnector Tests
# ============================================================================


class TestConfluenceConnector:
    """Tests for ConfluenceConnector class."""

    @pytest.fixture
    def confluence_config(self):
        """Create Confluence config fixture."""
        return SyncConfig(
            source_type=SyncSourceType.CONFLUENCE,
            credentials={
                "url": "https://test.atlassian.net/wiki",
                "email": "test@example.com",
                "api_token": "test-token",
            },
            settings={"spaces": ["SPACE1", "SPACE2"]},
        )

    def test_initialization(self, confluence_config):
        """Test Confluence connector initialization."""
        connector = ConfluenceConnector(confluence_config)
        assert connector.source_type == SyncSourceType.CONFLUENCE
        assert connector.base_url == "https://test.atlassian.net/wiki"
        assert connector.spaces == ["SPACE1", "SPACE2"]

    @pytest.mark.asyncio
    async def test_list_items_with_mock(self, confluence_config):
        """Test listing Confluence pages with mocked API."""
        connector = ConfluenceConnector(confluence_config)

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "page-1",
                    "title": "Test Page 1",
                    "version": {"number": 1, "createdAt": "2024-01-01T00:00:00Z"},
                    "status": "current",
                },
                {
                    "id": "page-2",
                    "title": "Test Page 2",
                    "version": {"number": 2, "createdAt": "2024-01-02T00:00:00Z"},
                    "status": "current",
                },
            ],
            "_links": {},
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        connector._client = mock_client

        items = []
        async for item in connector.list_items(incremental=False):
            items.append(item)

        # 2 pages per space * 2 spaces = 4 items
        assert len(items) == 4
        assert items[0].name == "Test Page 1"

    @pytest.mark.asyncio
    async def test_fetch_content_with_mock(self, confluence_config):
        """Test fetching Confluence page content."""
        connector = ConfluenceConnector(confluence_config)

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "page-1",
            "title": "Test Page",
            "body": {
                "storage": {
                    "value": "<p>Hello World</p>",
                }
            },
            "version": {"number": 1},
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        connector._client = mock_client

        item = SyncItem(
            id="page-1",
            source_type=SyncSourceType.CONFLUENCE,
            name="Test Page",
            path="/spaces/SPACE1/pages/page-1",
        )

        content = await connector.fetch_content(item)
        assert "Hello World" in content.text

    def test_extract_text(self, confluence_config):
        """Test HTML text extraction."""
        connector = ConfluenceConnector(confluence_config)

        html = "<p>Hello</p><p>World</p>"
        text = connector._extract_text(html)
        assert "Hello" in text
        assert "World" in text


# ============================================================================
# NotionConnector Tests
# ============================================================================


class TestNotionConnector:
    """Tests for NotionConnector class."""

    @pytest.fixture
    def notion_config(self):
        """Create Notion config fixture."""
        return SyncConfig(
            source_type=SyncSourceType.NOTION,
            credentials={"api_key": "secret_test"},
            settings={"database_ids": ["db-1", "db-2"]},
        )

    def test_initialization(self, notion_config):
        """Test Notion connector initialization."""
        connector = NotionConnector(notion_config)
        assert connector.source_type == SyncSourceType.NOTION
        assert connector.database_ids == ["db-1", "db-2"]

    @pytest.mark.asyncio
    async def test_list_items_with_mock(self, notion_config):
        """Test listing Notion pages with mocked API."""
        connector = NotionConnector(notion_config)

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "page-1",
                    "last_edited_time": "2024-01-01T00:00:00.000Z",
                    "properties": {
                        "title": {
                            "type": "title",
                            "title": [{"plain_text": "Test Page 1"}],
                        }
                    },
                    "parent": {"type": "database_id"},
                    "archived": False,
                },
            ],
            "has_more": False,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        connector._client = mock_client

        items = []
        async for item in connector.list_items(incremental=False):
            items.append(item)

        # 1 page per database * 2 databases = 2 items
        assert len(items) == 2
        assert items[0].name == "Test Page 1"

    @pytest.mark.asyncio
    async def test_fetch_content_with_mock(self, notion_config):
        """Test fetching Notion page content."""
        connector = NotionConnector(notion_config)

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"plain_text": "Hello World"}],
                    },
                }
            ],
            "has_more": False,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        connector._client = mock_client

        item = SyncItem(
            id="page-1",
            source_type=SyncSourceType.NOTION,
            name="Test Page",
            path="/pages/page-1",
        )

        content = await connector.fetch_content(item)
        assert "Hello World" in content.text

    def test_blocks_to_text(self, notion_config):
        """Test block to text conversion."""
        connector = NotionConnector(notion_config)

        blocks = [
            {
                "type": "heading_1",
                "heading_1": {"rich_text": [{"plain_text": "Title"}]},
            },
            {
                "type": "paragraph",
                "paragraph": {"rich_text": [{"plain_text": "Content"}]},
            },
        ]

        text = connector._blocks_to_text(blocks)
        assert "# Title" in text
        assert "Content" in text


# ============================================================================
# SyncManager Tests
# ============================================================================


class TestSyncManager:
    """Tests for SyncManager class."""

    def test_manager_disabled_by_default(self):
        """Test manager is disabled by default."""
        manager = SyncManager()
        assert not manager.enabled

    def test_manager_enabled(self):
        """Test manager can be enabled."""
        manager = SyncManager(enabled=True)
        assert manager.enabled

    def test_register_connector(self):
        """Test registering a connector."""
        manager = SyncManager(enabled=True)
        config = SyncConfig(
            source_type=SyncSourceType.S3,
            source_id="s3-test",
            settings={"bucket": "test"},
        )

        connector_id = manager.register_connector(config)
        assert connector_id == "s3-test"
        assert "s3-test" in manager.connectors

    def test_register_when_disabled(self):
        """Test registering fails when disabled."""
        manager = SyncManager(enabled=False)
        config = SyncConfig(source_type=SyncSourceType.S3)

        connector_id = manager.register_connector(config)
        assert connector_id == ""
        assert len(manager.connectors) == 0

    def test_unregister_connector(self):
        """Test unregistering a connector."""
        manager = SyncManager(enabled=True)
        config = SyncConfig(
            source_type=SyncSourceType.S3,
            source_id="s3-test",
        )
        manager.register_connector(config)

        result = manager.unregister_connector("s3-test")
        assert result is True
        assert "s3-test" not in manager.connectors

    @pytest.mark.asyncio
    async def test_sync_all_empty(self):
        """Test sync all with no connectors."""
        manager = SyncManager(enabled=True)
        results = await manager.sync_all()
        assert results == {}

    @pytest.mark.asyncio
    async def test_sync_when_disabled(self):
        """Test sync fails when disabled."""
        manager = SyncManager(enabled=False)
        results = await manager.sync_all()
        assert results == {}

    def test_get_status(self):
        """Test getting manager status."""
        manager = SyncManager(enabled=True)
        status = manager.get_status()

        assert status["enabled"] is True
        assert status["connector_count"] == 0
        assert status["connectors"] == []


class TestCreateSyncManager:
    """Tests for create_sync_manager factory function."""

    def test_create_disabled(self):
        """Test creating disabled manager."""
        manager = create_sync_manager(enabled=False)
        assert not manager.enabled

    def test_create_with_configs(self):
        """Test creating manager with configs."""
        configs = [
            SyncConfig(source_type=SyncSourceType.S3, source_id="s3-1"),
            SyncConfig(source_type=SyncSourceType.NOTION, source_id="notion-1"),
        ]
        manager = create_sync_manager(enabled=True, configs=configs)

        assert manager.enabled
        assert len(manager.connectors) == 2


# ============================================================================
# Default Constants Tests
# ============================================================================


class TestDefaultConstants:
    """Tests for default configuration constants."""

    def test_default_sync_disabled(self):
        """Test external sync is disabled by default."""
        assert DEFAULT_EXTERNAL_SYNC_ENABLED is False


# ============================================================================
# SyncSourceType Tests
# ============================================================================


class TestSyncSourceType:
    """Tests for SyncSourceType enum."""

    def test_all_source_types(self):
        """Test all expected source types exist."""
        assert SyncSourceType.S3 == "s3"
        assert SyncSourceType.CONFLUENCE == "confluence"
        assert SyncSourceType.NOTION == "notion"
        assert SyncSourceType.GOOGLE_DRIVE == "google_drive"
        assert SyncSourceType.DISCORD == "discord"
