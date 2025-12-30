"""Tests for workspace API routes.

Story 6-5: Frontend Actions
Code Review Fixes: Security and validation improvements
"""

import pytest
from pydantic import ValidationError

from agentic_rag_backend.api.routes.workspace import (
    SaveContentRequest,
    SaveContentResponse,
    ExportContentRequest,
    ExportContentResponse,
    ShareContentRequest,
    ShareContentResponse,
    BookmarkContentRequest,
    BookmarkContentResponse,
    BookmarksListResponse,
    SourceInfo,
    save_content,
    export_content,
    share_content,
    bookmark_content,
    get_bookmarks,
    MAX_CONTENT_SIZE,
    _bookmarks_storage,
)
from agentic_rag_backend.core.errors import TenantRequiredError


@pytest.fixture(autouse=True)
def clear_bookmark_storage():
    """Clear bookmark storage before each test for isolation."""
    _bookmarks_storage.clear()
    yield
    _bookmarks_storage.clear()


class TestSaveContentRequest:
    """Tests for SaveContentRequest model."""

    def test_valid_save_request(self):
        """Test valid save request."""
        request = SaveContentRequest(
            content_id="test-123",
            content="This is test content",
            title="Test Title",
            query="What is test?",
            tenant_id="tenant-1",
        )
        
        assert request.content_id == "test-123"
        assert request.content == "This is test content"
        assert request.title == "Test Title"
        assert request.tenant_id == "tenant-1"

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
            tenant_id="tenant-1",
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
            tenant_id="tenant-1",
        )
        
        assert request.content_id == "test-123"
        assert request.title == "Bookmarked Item"


class TestSaveContentEndpoint:
    """Tests for save_content endpoint."""

    @pytest.mark.asyncio
    async def test_save_content_success(self):
        """Test successful content save."""
        request_body = SaveContentRequest(
            content_id="content-123",
            content="AI response content to save",
            title="Saved Response",
            query="What was the question?",
            tenant_id="tenant-1",
        )

        result = await save_content(request_body=request_body)

        assert result.data is not None
        assert result.data.content_id == "content-123"
        assert result.data.workspace_id is not None
        assert result.data.saved_at is not None

    @pytest.mark.asyncio
    async def test_save_content_with_no_title(self):
        """Test save content without title."""
        request_body = SaveContentRequest(
            content_id="content-456",
            content="Response without title",
            tenant_id="tenant-1",
        )

        result = await save_content(request_body=request_body)

        assert result.data is not None
        assert result.data.content_id == "content-456"

    @pytest.mark.asyncio
    async def test_save_requires_tenant_id(self):
        """Test save fails without tenant_id."""
        request_body = SaveContentRequest(
            content_id="content-123",
            content="Content to save",
        )

        with pytest.raises(TenantRequiredError):
            # Must pass x_tenant_id=None explicitly when calling directly
            # (not through FastAPI) since default is Header object not None
            await save_content(request_body=request_body, x_tenant_id=None)


class TestExportContentEndpoint:
    """Tests for export_content endpoint."""

    @pytest.mark.asyncio
    async def test_export_markdown_success(self):
        """Test successful markdown export returns Response."""
        request_body = ExportContentRequest(
            content_id="content-123",
            content="# Heading\n\nParagraph content",
            title="Markdown Doc",
            format="markdown",
            tenant_id="tenant-1",
        )

        result = await export_content(request_body=request_body)

        # Returns a Response object for download
        assert result.media_type == "text/markdown"
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_export_json_success(self):
        """Test successful JSON export."""
        request_body = ExportContentRequest(
            content_id="content-123",
            content="Plain text content",
            format="json",
            tenant_id="tenant-1",
        )

        result = await export_content(request_body=request_body)

        assert result.media_type == "application/json"
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_export_pdf_returns_not_implemented(self):
        """Test PDF export returns 501 Not Implemented."""
        request_body = ExportContentRequest(
            content_id="content-123",
            content="Content for PDF",
            format="pdf",
            tenant_id="tenant-1",
        )

        result = await export_content(request_body=request_body)

        assert result.status_code == 501

    @pytest.mark.asyncio
    async def test_export_requires_tenant_id(self):
        """Test export fails without tenant_id."""
        request_body = ExportContentRequest(
            content_id="content-123",
            content="Content to export",
            format="markdown",
        )

        with pytest.raises(TenantRequiredError):
            await export_content(request_body=request_body, x_tenant_id=None)


class TestShareContentEndpoint:
    """Tests for share_content endpoint."""

    @pytest.mark.asyncio
    async def test_share_content_success(self):
        """Test successful content share."""
        request_body = ShareContentRequest(
            content_id="content-123",
            content="Content to share",
            title="Shared Response",
            tenant_id="tenant-1",
        )

        result = await share_content(request_body=request_body)

        assert result.data is not None
        assert result.data.share_url is not None
        assert "share" in result.data.share_url
        assert result.data.expires_at is not None  # Check expiration is set

    @pytest.mark.asyncio
    async def test_share_content_unique_urls(self):
        """Test each share generates unique URL."""
        request_body = ShareContentRequest(
            content_id="content-123",
            content="Same content",
            tenant_id="tenant-1",
        )

        result1 = await share_content(request_body=request_body)
        result2 = await share_content(request_body=request_body)

        assert result1.data.share_url != result2.data.share_url

    @pytest.mark.asyncio
    async def test_share_content_includes_token(self):
        """Test share URL includes signed token."""
        request_body = ShareContentRequest(
            content_id="content-123",
            content="Content to share",
            tenant_id="tenant-1",
        )

        result = await share_content(request_body=request_body)

        assert "token=" in result.data.share_url

    @pytest.mark.asyncio
    async def test_share_requires_tenant_id(self):
        """Test share fails without tenant_id."""
        request_body = ShareContentRequest(
            content_id="content-123",
            content="Content to share",
        )

        with pytest.raises(TenantRequiredError):
            await share_content(request_body=request_body, x_tenant_id=None)


class TestBookmarkContentEndpoint:
    """Tests for bookmark_content endpoint."""

    @pytest.mark.asyncio
    async def test_bookmark_content_success(self):
        """Test successful content bookmark."""
        request_body = BookmarkContentRequest(
            content_id="content-123",
            content="AI response to bookmark",
            title="Bookmarked Response",
            tenant_id="tenant-1",
        )

        result = await bookmark_content(request_body=request_body)

        assert result.data is not None
        assert result.data.bookmark_id is not None
        assert result.data.bookmarked_at is not None

    @pytest.mark.asyncio
    async def test_bookmark_requires_tenant_id(self):
        """Test bookmark fails without tenant_id."""
        request_body = BookmarkContentRequest(
            content_id="content-123",
            content="Content to bookmark",
        )

        with pytest.raises(TenantRequiredError):
            await bookmark_content(request_body=request_body, x_tenant_id=None)


class TestGetBookmarksEndpoint:
    """Tests for get_bookmarks endpoint."""

    @pytest.mark.asyncio
    async def test_get_bookmarks_returns_list(self):
        """Test get bookmarks returns list."""
        # Call with explicit values (not Query objects)
        result = await get_bookmarks(tenant_id="tenant-1", limit=50, offset=0)
        
        assert result.data is not None
        assert isinstance(result.data, list)

    @pytest.mark.asyncio
    async def test_get_bookmarks_empty_for_new_tenant(self):
        """Test get bookmarks returns empty for new tenant."""
        result = await get_bookmarks(tenant_id="new-tenant-xyz", limit=50, offset=0)

        assert result.data == []


class TestMultiTenantIsolation:
    """Tests for multi-tenant data isolation."""

    @pytest.mark.asyncio
    async def test_bookmarks_isolated_between_tenants(self):
        """Test bookmarks are isolated between tenants."""
        # Create bookmark for tenant-1
        request1 = BookmarkContentRequest(
            content_id="content-123",
            content="Tenant 1 content",
            title="Tenant 1 Bookmark",
            tenant_id="tenant-1",
        )
        await bookmark_content(request_body=request1)

        # Create bookmark for tenant-2
        request2 = BookmarkContentRequest(
            content_id="content-456",
            content="Tenant 2 content",
            title="Tenant 2 Bookmark",
            tenant_id="tenant-2",
        )
        await bookmark_content(request_body=request2)

        # Get bookmarks for tenant-1
        result1 = await get_bookmarks(tenant_id="tenant-1", limit=50, offset=0)
        assert len(result1.data) == 1
        assert result1.data[0].content_id == "content-123"

        # Get bookmarks for tenant-2
        result2 = await get_bookmarks(tenant_id="tenant-2", limit=50, offset=0)
        assert len(result2.data) == 1
        assert result2.data[0].content_id == "content-456"

        # Get bookmarks for tenant-3 (should be empty)
        result3 = await get_bookmarks(tenant_id="tenant-3", limit=50, offset=0)
        assert len(result3.data) == 0


class TestContentSizeLimits:
    """Tests for content size validation."""

    def test_save_content_rejects_oversized_content(self):
        """Test save request rejects content over MAX_CONTENT_SIZE."""
        oversized_content = "x" * (MAX_CONTENT_SIZE + 1)

        with pytest.raises(ValidationError):
            SaveContentRequest(
                content_id="content-123",
                content=oversized_content,
                tenant_id="tenant-1",
            )

    def test_export_content_rejects_oversized_content(self):
        """Test export request rejects content over MAX_CONTENT_SIZE."""
        oversized_content = "x" * (MAX_CONTENT_SIZE + 1)

        with pytest.raises(ValidationError):
            ExportContentRequest(
                content_id="content-123",
                content=oversized_content,
                format="markdown",
                tenant_id="tenant-1",
            )

    def test_share_content_rejects_oversized_content(self):
        """Test share request rejects content over MAX_CONTENT_SIZE."""
        oversized_content = "x" * (MAX_CONTENT_SIZE + 1)

        with pytest.raises(ValidationError):
            ShareContentRequest(
                content_id="content-123",
                content=oversized_content,
                tenant_id="tenant-1",
            )

    def test_bookmark_content_rejects_oversized_content(self):
        """Test bookmark request rejects content over MAX_CONTENT_SIZE."""
        oversized_content = "x" * (MAX_CONTENT_SIZE + 1)

        with pytest.raises(ValidationError):
            BookmarkContentRequest(
                content_id="content-123",
                content=oversized_content,
                tenant_id="tenant-1",
            )

    def test_save_content_accepts_max_size_content(self):
        """Test save request accepts content at MAX_CONTENT_SIZE."""
        max_content = "x" * MAX_CONTENT_SIZE

        # Should not raise
        request = SaveContentRequest(
            content_id="content-123",
            content=max_content,
            tenant_id="tenant-1",
        )
        assert len(request.content) == MAX_CONTENT_SIZE
