"""Tests for workspace API routes."""

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agentic_rag_backend.main import app
from agentic_rag_backend.config import load_settings
from agentic_rag_backend.api.routes.workspace import (
    BookmarkContentRequest,
    ExportContentRequest,
    MAX_CONTENT_SIZE,
    SaveContentRequest,
    ShareContentRequest,
    SourceInfo,
    get_postgres,
    limiter,
)
from agentic_rag_backend.db.postgres import PostgresClient


@pytest.fixture
def mock_workspace_postgres():
    """Provide a PostgresClient mock with in-memory workspace storage."""
    workspace_store: dict[UUID, dict[str, object]] = {}
    share_store: dict[UUID, dict[str, object]] = {}
    bookmark_store: dict[UUID, list[dict[str, object]]] = {}

    async def create_workspace_item(
        workspace_id: UUID,
        tenant_id: UUID,
        content_id: str,
        content: str,
        title: str | None,
        query: str | None,
        sources: list[dict[str, object]] | None,
        session_id: str | None,
        trajectory_id: str | None,
    ) -> datetime:
        created_at = datetime.now(timezone.utc)
        workspace_store[workspace_id] = {
            "id": workspace_id,
            "tenant_id": tenant_id,
            "content_id": content_id,
            "content": content,
            "title": title,
            "query": query,
            "sources": sources,
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "created_at": created_at,
        }
        return created_at

    async def get_workspace_item(tenant_id: UUID, workspace_id: UUID):
        row = workspace_store.get(workspace_id)
        if not row or row["tenant_id"] != tenant_id:
            return None
        return row

    async def create_workspace_share(
        share_id: UUID,
        tenant_id: UUID,
        content_id: str,
        content: str,
        title: str | None,
        query: str | None,
        sources: list[dict[str, object]] | None,
        token: str,
        expires_at: datetime | None,
    ) -> datetime:
        created_at = datetime.now(timezone.utc)
        share_store[share_id] = {
            "id": share_id,
            "tenant_id": tenant_id,
            "content_id": content_id,
            "content": content,
            "title": title,
            "query": query,
            "sources": sources,
            "token": token,
            "created_at": created_at,
            "expires_at": expires_at,
        }
        return created_at

    async def get_workspace_share(share_id: UUID):
        return share_store.get(share_id)

    async def create_workspace_bookmark(
        bookmark_id: UUID,
        tenant_id: UUID,
        content_id: str,
        content: str,
        title: str | None,
        query: str | None,
        session_id: str | None,
    ) -> datetime:
        created_at = datetime.now(timezone.utc)
        bookmark_store.setdefault(tenant_id, []).append({
            "id": bookmark_id,
            "content_id": content_id,
            "content": content,
            "title": title,
            "query": query,
            "session_id": session_id,
            "created_at": created_at,
        })
        return created_at

    async def list_workspace_bookmarks(tenant_id: UUID, limit: int, offset: int):
        rows = bookmark_store.get(tenant_id, [])
        return rows[offset:offset + limit], len(rows)

    client = MagicMock(spec=PostgresClient)
    client.create_workspace_item = AsyncMock(side_effect=create_workspace_item)
    client.get_workspace_item = AsyncMock(side_effect=get_workspace_item)
    client.create_workspace_share = AsyncMock(side_effect=create_workspace_share)
    client.get_workspace_share = AsyncMock(side_effect=get_workspace_share)
    client.create_workspace_bookmark = AsyncMock(side_effect=create_workspace_bookmark)
    client.list_workspace_bookmarks = AsyncMock(side_effect=list_workspace_bookmarks)
    return client


@pytest.fixture
def client(mock_workspace_postgres):
    """Create test client with mocked Postgres dependency."""
    app.state.settings = load_settings()
    limiter.enabled = False

    async def override_get_postgres():
        return mock_workspace_postgres

    app.dependency_overrides[get_postgres] = override_get_postgres

    with TestClient(app) as test_client:
        yield test_client

    limiter.enabled = True
    app.dependency_overrides.clear()


class TestSaveContentRequest:
    """Tests for SaveContentRequest model."""

    def test_valid_save_request(self):
        """Test valid save request."""
        request = SaveContentRequest(
            content_id="test-123",
            content="This is test content",
            title="Test Title",
            query="What is test?",
            tenant_id="11111111-1111-1111-1111-111111111111",
        )

        assert request.content_id == "test-123"
        assert request.content == "This is test content"
        assert request.title == "Test Title"
        assert request.tenant_id == "11111111-1111-1111-1111-111111111111"

    def test_save_request_requires_content_id(self):
        """Test that content_id is required."""
        with pytest.raises(ValidationError):
            SaveContentRequest(
                content="Test content",
                title="Test",
            )

    def test_save_request_requires_content(self):
        """Test that content is required."""
        with pytest.raises(ValidationError):
            SaveContentRequest(
                content_id="test-123",
                title="Test",
            )

    def test_save_request_optional_fields(self):
        """Test optional fields."""
        request = SaveContentRequest(
            content_id="test-123",
            content="Content only",
        )

        assert request.title is None
        assert request.query is None
        assert request.tenant_id is None


class TestExportContentRequest:
    """Tests for ExportContentRequest model."""

    def test_valid_export_request_markdown(self):
        """Test valid export request with markdown format."""
        request = ExportContentRequest(
            content_id="test-123",
            content="Export this content",
            format="markdown",
        )

        assert request.format == "markdown"

    def test_valid_export_request_json(self):
        """Test valid export request with JSON format."""
        request = ExportContentRequest(
            content_id="test-123",
            content="Export this content",
            format="json",
        )

        assert request.format == "json"

    def test_valid_export_request_pdf(self):
        """Test valid export request with PDF format."""
        request = ExportContentRequest(
            content_id="test-123",
            content="Export this content",
            format="pdf",
        )

        assert request.format == "pdf"

    def test_invalid_export_format(self):
        """Test invalid export format."""
        with pytest.raises(ValidationError):
            ExportContentRequest(
                content_id="test-123",
                content="Export this content",
                format="xml",
            )


class TestShareContentRequest:
    """Tests for ShareContentRequest model."""

    def test_valid_share_request(self):
        """Test valid share request."""
        request = ShareContentRequest(
            content_id="test-123",
            content="Share this content",
            title="Shared Content",
            tenant_id="11111111-1111-1111-1111-111111111111",
        )

        assert request.content_id == "test-123"
        assert request.title == "Shared Content"

    def test_share_request_with_sources(self):
        """Test share request with sources as SourceInfo objects."""
        request = ShareContentRequest(
            content_id="test-123",
            content="Content with sources",
            sources=[
                SourceInfo(id="source-1", title="Source 1"),
                SourceInfo(id="source-2", title="Source 2", url="https://example.com"),
            ],
        )

        assert len(request.sources) == 2
        assert request.sources[0].id == "source-1"


class TestBookmarkContentRequest:
    """Tests for BookmarkContentRequest model."""

    def test_valid_bookmark_request(self):
        """Test valid bookmark request."""
        request = BookmarkContentRequest(
            content_id="test-123",
            content="Bookmark this content",
            title="Bookmarked Item",
            tenant_id="11111111-1111-1111-1111-111111111111",
        )

        assert request.content_id == "test-123"
        assert request.title == "Bookmarked Item"


class TestSaveContentEndpoint:
    """Tests for save_content endpoint."""

    def test_save_content_success(self, client):
        """Test successful content save."""
        response = client.post(
            "/api/v1/workspace/save",
            json={
                "content_id": "content-123",
                "content": "AI response content to save",
                "title": "Saved Response",
                "query": "What was the question?",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["content_id"] == "content-123"
        assert data["data"]["workspace_id"] is not None
        assert data["data"]["saved_at"] is not None

    def test_save_content_with_no_title(self, client):
        """Test save content without title."""
        response = client.post(
            "/api/v1/workspace/save",
            json={
                "content_id": "content-456",
                "content": "Response without title",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["content_id"] == "content-456"

    def test_save_requires_tenant_id(self, client):
        """Test save fails without tenant_id."""
        response = client.post(
            "/api/v1/workspace/save",
            json={
                "content_id": "content-123",
                "content": "Content to save",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["title"] == "Tenant Required"


class TestLoadWorkspaceEndpoint:
    """Tests for load_workspace_content endpoint."""

    def test_load_workspace_success(self, client):
        """Test loading saved workspace content."""
        save_response = client.post(
            "/api/v1/workspace/save",
            json={
                "content_id": "content-123",
                "content": "Saved content",
                "title": "Saved Item",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )
        workspace_id = save_response.json()["data"]["workspace_id"]

        response = client.get(
            f"/api/v1/workspace/{workspace_id}",
            params={"tenant_id": "11111111-1111-1111-1111-111111111111"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["workspace_id"] == workspace_id
        assert data["data"]["content"] == "Saved content"

    def test_load_requires_tenant_id(self, client):
        """Test load fails without tenant_id."""
        response = client.get(
            "/api/v1/workspace/11111111-1111-1111-1111-111111111111",
        )

        assert response.status_code == 400


class TestExportContentEndpoint:
    """Tests for export_content endpoint."""

    def test_export_markdown_success(self, client):
        """Test successful markdown export returns Response."""
        response = client.post(
            "/api/v1/workspace/export",
            json={
                "content_id": "content-123",
                "content": "# Heading\n\nParagraph content",
                "title": "Markdown Doc",
                "format": "markdown",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/markdown; charset=utf-8"

    def test_export_json_success(self, client):
        """Test successful JSON export."""
        response = client.post(
            "/api/v1/workspace/export",
            json={
                "content_id": "content-123",
                "content": "Plain text content",
                "format": "json",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

    def test_export_pdf_returns_not_implemented(self, client):
        """Test PDF export returns 501 Not Implemented."""
        response = client.post(
            "/api/v1/workspace/export",
            json={
                "content_id": "content-123",
                "content": "Content for PDF",
                "format": "pdf",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        assert response.status_code == 501
        data = response.json()
        assert data["status"] == 501

    def test_export_requires_tenant_id(self, client):
        """Test export fails without tenant_id."""
        response = client.post(
            "/api/v1/workspace/export",
            json={
                "content_id": "content-123",
                "content": "Content to export",
                "format": "markdown",
            },
        )

        assert response.status_code == 400


class TestShareContentEndpoint:
    """Tests for share_content endpoint."""

    def test_share_content_success(self, client):
        """Test successful content share."""
        response = client.post(
            "/api/v1/workspace/share",
            json={
                "content_id": "content-123",
                "content": "Content to share",
                "title": "Shared Response",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["share_url"] is not None
        assert "share" in data["data"]["share_url"]
        assert data["data"]["expires_at"] is not None

    def test_share_content_unique_urls(self, client):
        """Test each share generates unique URL."""
        payload = {
            "content_id": "content-123",
            "content": "Same content",
            "tenant_id": "11111111-1111-1111-1111-111111111111",
        }

        result1 = client.post("/api/v1/workspace/share", json=payload)
        result2 = client.post("/api/v1/workspace/share", json=payload)

        assert result1.json()["data"]["share_url"] != result2.json()["data"]["share_url"]

    def test_share_content_includes_token(self, client):
        """Test share URL includes signed token."""
        response = client.post(
            "/api/v1/workspace/share",
            json={
                "content_id": "content-123",
                "content": "Content to share",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "token=" in data["data"]["share_url"]

    def test_share_content_retrieval(self, client):
        """Test retrieving shared content with token."""
        response = client.post(
            "/api/v1/workspace/share",
            json={
                "content_id": "content-123",
                "content": "Content to share",
                "title": "Shared Response",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        share_url = response.json()["data"]["share_url"]
        parsed_url = urlparse(share_url)
        share_id = parsed_url.path.rstrip("/").split("/")[-1]
        token = parse_qs(parsed_url.query)["token"][0]

        get_response = client.get(
            f"/api/v1/workspace/share/{share_id}",
            params={"token": token},
        )

        assert get_response.status_code == 200
        data = get_response.json()
        assert data["data"]["content_id"] == "content-123"
        assert data["data"]["content"] == "Content to share"

    def test_share_content_retrieval_rejects_invalid_token(self, client):
        """Test retrieving shared content rejects invalid token."""
        response = client.post(
            "/api/v1/workspace/share",
            json={
                "content_id": "content-123",
                "content": "Content to share",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        share_url = response.json()["data"]["share_url"]
        parsed_url = urlparse(share_url)
        share_id = parsed_url.path.rstrip("/").split("/")[-1]

        get_response = client.get(
            f"/api/v1/workspace/share/{share_id}",
            params={"token": "invalid-token"},
        )

        assert get_response.status_code == 403
        payload = get_response.json()
        assert payload["status"] == 403
        assert payload["title"] == "Forbidden"
        assert payload["type"].endswith("/invalid-token")

    def test_share_content_retrieval_rejects_tampered_token(self, client):
        """Test retrieving shared content rejects tampered token."""
        response = client.post(
            "/api/v1/workspace/share",
            json={
                "content_id": "content-123",
                "content": "Content to share",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        share_url = response.json()["data"]["share_url"]
        parsed_url = urlparse(share_url)
        share_id = parsed_url.path.rstrip("/").split("/")[-1]
        token = parse_qs(parsed_url.query)["token"][0]
        tampered = token[:-1] + ("0" if token[-1] != "0" else "1")

        get_response = client.get(
            f"/api/v1/workspace/share/{share_id}",
            params={"token": tampered},
        )

        assert get_response.status_code == 403
        payload = get_response.json()
        assert payload["status"] == 403
        assert payload["title"] == "Forbidden"
        assert payload["type"].endswith("/invalid-token")

    def test_share_requires_tenant_id(self, client):
        """Test share fails without tenant_id."""
        response = client.post(
            "/api/v1/workspace/share",
            json={
                "content_id": "content-123",
                "content": "Content to share",
            },
        )

        assert response.status_code == 400


class TestBookmarkContentEndpoint:
    """Tests for bookmark_content endpoint."""

    def test_bookmark_content_success(self, client):
        """Test successful content bookmark."""
        response = client.post(
            "/api/v1/workspace/bookmark",
            json={
                "content_id": "content-123",
                "content": "AI response to bookmark",
                "title": "Bookmarked Response",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["bookmark_id"] is not None
        assert data["data"]["bookmarked_at"] is not None

    def test_bookmark_requires_tenant_id(self, client):
        """Test bookmark fails without tenant_id."""
        response = client.post(
            "/api/v1/workspace/bookmark",
            json={
                "content_id": "content-123",
                "content": "Content to bookmark",
            },
        )

        assert response.status_code == 400


class TestGetBookmarksEndpoint:
    """Tests for get_bookmarks endpoint."""

    def test_get_bookmarks_returns_list(self, client):
        """Test get bookmarks returns list."""
        response = client.get(
            "/api/v1/workspace/bookmarks",
            params={"tenant_id": "11111111-1111-1111-1111-111111111111"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"] is not None
        assert isinstance(data["data"], list)

    def test_get_bookmarks_empty_for_new_tenant(self, client):
        """Test get bookmarks returns empty for new tenant."""
        response = client.get(
            "/api/v1/workspace/bookmarks",
            params={"tenant_id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []


class TestMultiTenantIsolation:
    """Tests for multi-tenant data isolation."""

    def test_bookmarks_isolated_between_tenants(self, client):
        """Test bookmarks are isolated between tenants."""
        # Create bookmark for 11111111-1111-1111-1111-111111111111
        client.post(
            "/api/v1/workspace/bookmark",
            json={
                "content_id": "content-123",
                "content": "Tenant 1 content",
                "title": "Tenant 1 Bookmark",
                "tenant_id": "11111111-1111-1111-1111-111111111111",
            },
        )

        # Create bookmark for 22222222-2222-2222-2222-222222222222
        client.post(
            "/api/v1/workspace/bookmark",
            json={
                "content_id": "content-456",
                "content": "Tenant 2 content",
                "title": "Tenant 2 Bookmark",
                "tenant_id": "22222222-2222-2222-2222-222222222222",
            },
        )

        # Get bookmarks for 11111111-1111-1111-1111-111111111111
        result1 = client.get(
            "/api/v1/workspace/bookmarks",
            params={"tenant_id": "11111111-1111-1111-1111-111111111111"},
        )
        data1 = result1.json()
        assert len(data1["data"]) == 1
        assert data1["data"][0]["content_id"] == "content-123"

        # Get bookmarks for 22222222-2222-2222-2222-222222222222
        result2 = client.get(
            "/api/v1/workspace/bookmarks",
            params={"tenant_id": "22222222-2222-2222-2222-222222222222"},
        )
        data2 = result2.json()
        assert len(data2["data"]) == 1
        assert data2["data"][0]["content_id"] == "content-456"

        # Get bookmarks for 33333333-3333-3333-3333-333333333333 (should be empty)
        result3 = client.get(
            "/api/v1/workspace/bookmarks",
            params={"tenant_id": "33333333-3333-3333-3333-333333333333"},
        )
        data3 = result3.json()
        assert len(data3["data"]) == 0


class TestContentSizeLimits:
    """Tests for content size validation."""

    def test_save_content_rejects_oversized_content(self):
        """Test save request rejects content over MAX_CONTENT_SIZE bytes."""
        oversized_content = "x" * (MAX_CONTENT_SIZE + 1)

        with pytest.raises(ValidationError):
            SaveContentRequest(
                content_id="content-123",
                content=oversized_content,
                tenant_id="11111111-1111-1111-1111-111111111111",
            )

    def test_export_content_rejects_oversized_content(self):
        """Test export request rejects content over MAX_CONTENT_SIZE bytes."""
        oversized_content = "x" * (MAX_CONTENT_SIZE + 1)

        with pytest.raises(ValidationError):
            ExportContentRequest(
                content_id="content-123",
                content=oversized_content,
                format="markdown",
                tenant_id="11111111-1111-1111-1111-111111111111",
            )

    def test_share_content_rejects_oversized_content(self):
        """Test share request rejects content over MAX_CONTENT_SIZE bytes."""
        oversized_content = "x" * (MAX_CONTENT_SIZE + 1)

        with pytest.raises(ValidationError):
            ShareContentRequest(
                content_id="content-123",
                content=oversized_content,
                tenant_id="11111111-1111-1111-1111-111111111111",
            )

    def test_bookmark_content_rejects_oversized_content(self):
        """Test bookmark request rejects content over MAX_CONTENT_SIZE bytes."""
        oversized_content = "x" * (MAX_CONTENT_SIZE + 1)

        with pytest.raises(ValidationError):
            BookmarkContentRequest(
                content_id="content-123",
                content=oversized_content,
                tenant_id="11111111-1111-1111-1111-111111111111",
            )

    def test_save_content_accepts_max_size_content(self):
        """Test save request accepts content at MAX_CONTENT_SIZE bytes."""
        max_content = "x" * MAX_CONTENT_SIZE

        # Should not raise
        request = SaveContentRequest(
            content_id="content-123",
            content=max_content,
            tenant_id="11111111-1111-1111-1111-111111111111",
        )
        assert len(request.content) == MAX_CONTENT_SIZE

    def test_multibyte_content_validates_bytes_not_chars(self):
        """Test content validation uses byte size, not character count.

        Multi-byte UTF-8 characters (like emoji) should count as more than 1.
        This prevents DoS attacks using emoji-heavy content.
        """
        # Each emoji is 4 bytes in UTF-8
        emoji = "\U0001F600"  # grinning face emoji
        assert len(emoji) == 1  # 1 character
        assert len(emoji.encode("utf-8")) == 4  # 4 bytes

        # Create content that is under char limit but over byte limit
        # MAX_CONTENT_SIZE / 4 emojis would be at the byte limit
        # Add 1 more to go over
        emoji_count = (MAX_CONTENT_SIZE // 4) + 1
        emoji_content = emoji * emoji_count

        # Character count is under limit but byte count is over
        assert len(emoji_content) < MAX_CONTENT_SIZE
        assert len(emoji_content.encode("utf-8")) > MAX_CONTENT_SIZE

        # Should reject based on byte size
        with pytest.raises(ValidationError):
            SaveContentRequest(
                content_id="content-123",
                content=emoji_content,
                tenant_id="11111111-1111-1111-1111-111111111111",
            )
