"""Workspace API routes for frontend actions.

Story 6-5: Frontend Actions

Provides endpoints for:
- POST /workspace/save - Save content to workspace
- POST /workspace/export - Export as markdown/PDF/JSON
- POST /workspace/share - Generate shareable link
- POST /workspace/bookmark - Bookmark content
- GET /workspace/bookmarks - List bookmarks
"""

from datetime import datetime, timezone
import json
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Header, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field


router = APIRouter(prefix="/workspace", tags=["workspace"])


# ============================================
# Request/Response Models
# ============================================

class SourceInfo(BaseModel):
    """Source reference for saved content."""

    id: str
    title: str
    url: Optional[str] = None


class SaveContentRequest(BaseModel):
    """Request to save content to workspace."""

    content_id: str = Field(..., description="Unique ID of the content")
    content: str = Field(..., description="The content to save")
    title: Optional[str] = Field(None, description="Title for saved content")
    query: Optional[str] = Field(None, description="Original query")
    sources: Optional[list[SourceInfo]] = Field(None, description="Source references")
    session_id: Optional[str] = Field(None, description="Session ID")
    trajectory_id: Optional[str] = Field(None, description="Trajectory ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")


class SavedContentData(BaseModel):
    """Saved content response data."""

    content_id: str
    workspace_id: str
    saved_at: str


class SaveContentResponse(BaseModel):
    """Response from save endpoint."""

    data: SavedContentData
    meta: dict[str, Any] = Field(default_factory=dict)


class ExportContentRequest(BaseModel):
    """Request to export content."""

    content_id: str = Field(..., description="Unique ID of the content")
    content: str = Field(..., description="The content to export")
    title: Optional[str] = Field(None, description="Title for export")
    query: Optional[str] = Field(None, description="Original query")
    sources: Optional[list[SourceInfo]] = Field(None, description="Source references")
    format: Literal["markdown", "pdf", "json"] = Field(..., description="Export format")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")


class ExportContentResponse(BaseModel):
    """Response from export endpoint."""

    filename: str
    format: str


class ShareContentRequest(BaseModel):
    """Request to share content."""

    content_id: str = Field(..., description="Unique ID of the content")
    content: str = Field(..., description="The content to share")
    title: Optional[str] = Field(None, description="Title for shared content")
    query: Optional[str] = Field(None, description="Original query")
    sources: Optional[list[SourceInfo]] = Field(None, description="Source references")
    session_id: Optional[str] = Field(None, description="Session ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")


class SharedContentData(BaseModel):
    """Shared content response data."""

    share_id: str
    share_url: str
    expires_at: Optional[str] = None


class ShareContentResponse(BaseModel):
    """Response from share endpoint."""

    data: SharedContentData
    meta: dict[str, Any] = Field(default_factory=dict)


class BookmarkContentRequest(BaseModel):
    """Request to bookmark content."""

    content_id: str = Field(..., description="Unique ID of the content")
    content: str = Field(..., description="The content to bookmark")
    title: Optional[str] = Field(None, description="Title for bookmark")
    query: Optional[str] = Field(None, description="Original query")
    session_id: Optional[str] = Field(None, description="Session ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")


class BookmarkedContentData(BaseModel):
    """Bookmarked content response data."""

    bookmark_id: str
    bookmarked_at: str


class BookmarkContentResponse(BaseModel):
    """Response from bookmark endpoint."""

    data: BookmarkedContentData
    meta: dict[str, Any] = Field(default_factory=dict)


class BookmarkItem(BaseModel):
    """A single bookmark item."""

    bookmark_id: str
    content_id: str
    title: str
    query: Optional[str] = None
    bookmarked_at: str


class BookmarksListResponse(BaseModel):
    """Response from list bookmarks endpoint."""

    data: list[BookmarkItem]
    meta: dict[str, Any] = Field(default_factory=dict)


# ============================================
# In-memory storage (to be replaced with DB)
# ============================================

# Simple in-memory storage for demo purposes
# In production, this would use PostgreSQL
_workspace_storage: dict[str, dict[str, Any]] = {}
_shares_storage: dict[str, dict[str, Any]] = {}
_bookmarks_storage: dict[str, list[dict[str, Any]]] = {}


# ============================================
# Endpoints
# ============================================

@router.post("/save", response_model=SaveContentResponse)
async def save_content(
    request_body: SaveContentRequest,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> SaveContentResponse:
    """Save content to the user's workspace.

    Args:
        request_body: Content to save
        x_tenant_id: Tenant ID from header (optional)

    Returns:
        SaveContentResponse with workspace reference
    """
    tenant_id = request_body.tenant_id or x_tenant_id or "default"
    workspace_id = str(uuid4())
    saved_at = datetime.now(timezone.utc).isoformat()

    # Store in memory (replace with DB in production)
    if tenant_id not in _workspace_storage:
        _workspace_storage[tenant_id] = {}

    _workspace_storage[tenant_id][workspace_id] = {
        "content_id": request_body.content_id,
        "content": request_body.content,
        "title": request_body.title or f"Response - {saved_at[:10]}",
        "query": request_body.query,
        "sources": [s.model_dump() for s in request_body.sources] if request_body.sources else None,
        "session_id": request_body.session_id,
        "trajectory_id": request_body.trajectory_id,
        "saved_at": saved_at,
    }

    return SaveContentResponse(
        data=SavedContentData(
            content_id=request_body.content_id,
            workspace_id=workspace_id,
            saved_at=saved_at,
        ),
        meta={"request_id": str(uuid4()), "timestamp": saved_at},
    )


@router.post("/export")
async def export_content(
    request_body: ExportContentRequest,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Response:
    """Export content in the specified format.

    Args:
        request_body: Content and format to export
        x_tenant_id: Tenant ID from header (optional)

    Returns:
        Response with exported content as file download
    """
    export_format = request_body.format
    title = request_body.title or "AI Response"
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if export_format == "markdown":
        # Generate markdown content
        md_lines = [
            f"# {title}",
            "",
        ]

        if request_body.query:
            md_lines.extend([
                "## Query",
                "",
                request_body.query,
                "",
            ])

        md_lines.extend([
            "## Response",
            "",
            request_body.content,
            "",
        ])

        if request_body.sources:
            md_lines.extend([
                "## Sources",
                "",
            ])
            for source in request_body.sources:
                if source.url:
                    md_lines.append(f"- [{source.title}]({source.url})")
                else:
                    md_lines.append(f"- {source.title}")

        md_lines.extend([
            "",
            "---",
            f"*Exported on {date_str}*",
        ])

        content = "\n".join(md_lines)
        filename = f"response-{date_str}.md"

        return Response(
            content=content.encode("utf-8"),
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    elif export_format == "json":
        # Generate JSON export
        export_data = {
            "title": title,
            "content_id": request_body.content_id,
            "content": request_body.content,
            "query": request_body.query,
            "sources": [s.model_dump() for s in request_body.sources] if request_body.sources else None,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        content = json.dumps(export_data, indent=2)
        filename = f"response-{date_str}.json"

        return Response(
            content=content.encode("utf-8"),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    else:  # PDF
        # PDF generation would require additional libraries (e.g., weasyprint, reportlab)
        # For now, return a placeholder PDF response
        # In production, implement proper PDF generation
        placeholder_pdf = b"%PDF-1.4\n%Placeholder PDF - implement with proper library\n%%EOF"
        filename = f"response-{date_str}.pdf"

        return Response(
            content=placeholder_pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )


@router.post("/share", response_model=ShareContentResponse)
async def share_content(
    request_body: ShareContentRequest,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> ShareContentResponse:
    """Generate a shareable link for content.

    Args:
        request_body: Content to share
        x_tenant_id: Tenant ID from header (optional)

    Returns:
        ShareContentResponse with share URL
    """
    tenant_id = request_body.tenant_id or x_tenant_id or "default"
    share_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # Store share data (replace with DB in production)
    _shares_storage[share_id] = {
        "tenant_id": tenant_id,
        "content_id": request_body.content_id,
        "content": request_body.content,
        "title": request_body.title,
        "query": request_body.query,
        "sources": [s.model_dump() for s in request_body.sources] if request_body.sources else None,
        "created_at": created_at,
    }

    # Generate share URL (in production, use proper URL construction)
    base_url = "https://app.example.com"  # Would come from config
    share_url = f"{base_url}/share/{share_id}"

    return ShareContentResponse(
        data=SharedContentData(
            share_id=share_id,
            share_url=share_url,
        ),
        meta={"request_id": str(uuid4()), "timestamp": created_at},
    )


@router.post("/bookmark", response_model=BookmarkContentResponse)
async def bookmark_content(
    request_body: BookmarkContentRequest,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> BookmarkContentResponse:
    """Bookmark content for quick access later.

    Args:
        request_body: Content to bookmark
        x_tenant_id: Tenant ID from header (optional)

    Returns:
        BookmarkContentResponse with bookmark reference
    """
    tenant_id = request_body.tenant_id or x_tenant_id or "default"
    bookmark_id = str(uuid4())
    bookmarked_at = datetime.now(timezone.utc).isoformat()

    # Store bookmark (replace with DB in production)
    if tenant_id not in _bookmarks_storage:
        _bookmarks_storage[tenant_id] = []

    _bookmarks_storage[tenant_id].append({
        "bookmark_id": bookmark_id,
        "content_id": request_body.content_id,
        "content": request_body.content,
        "title": request_body.title or "Bookmarked Response",
        "query": request_body.query,
        "session_id": request_body.session_id,
        "bookmarked_at": bookmarked_at,
    })

    return BookmarkContentResponse(
        data=BookmarkedContentData(
            bookmark_id=bookmark_id,
            bookmarked_at=bookmarked_at,
        ),
        meta={"request_id": str(uuid4()), "timestamp": bookmarked_at},
    )


@router.get("/bookmarks", response_model=BookmarksListResponse)
async def get_bookmarks(
    tenant_id: str = Query(..., description="Tenant ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum bookmarks to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> BookmarksListResponse:
    """List bookmarks for a tenant.

    Args:
        tenant_id: Tenant ID to list bookmarks for
        limit: Maximum number of bookmarks to return
        offset: Pagination offset

    Returns:
        BookmarksListResponse with list of bookmarks
    """
    bookmarks = _bookmarks_storage.get(tenant_id, [])

    # Apply pagination
    paginated = bookmarks[offset : offset + limit]

    items = [
        BookmarkItem(
            bookmark_id=b["bookmark_id"],
            content_id=b["content_id"],
            title=b["title"],
            query=b.get("query"),
            bookmarked_at=b["bookmarked_at"],
        )
        for b in paginated
    ]

    return BookmarksListResponse(
        data=items,
        meta={
            "request_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": len(bookmarks),
            "limit": limit,
            "offset": offset,
        },
    )
