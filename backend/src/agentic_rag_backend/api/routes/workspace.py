"""Workspace API routes for frontend actions.

Story 6-5: Frontend Actions

Provides endpoints for:
- POST /workspace/save - Save content to workspace
- POST /workspace/export - Export as markdown/PDF/JSON
- POST /workspace/share - Generate shareable link
- POST /workspace/bookmark - Bookmark content
- GET /workspace/bookmarks - List bookmarks
"""

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import logging
import secrets
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Header, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from agentic_rag_backend.config import get_settings
from agentic_rag_backend.core.errors import TenantRequiredError


router = APIRouter(prefix="/workspace", tags=["workspace"])
logger = logging.getLogger(__name__)


# ============================================
# Constants
# ============================================

# Maximum content size to prevent DoS (100KB)
MAX_CONTENT_SIZE = 100_000

# Share link expiration (24 hours)
SHARE_LINK_TTL_HOURS = 24

# Secret for signing share tokens (in production, use from config)
_SHARE_SECRET = secrets.token_hex(32)


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
    content: str = Field(..., description="The content to save", max_length=MAX_CONTENT_SIZE)
    title: Optional[str] = Field(None, description="Title for saved content", max_length=500)
    query: Optional[str] = Field(None, description="Original query", max_length=10_000)
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
    content: str = Field(..., description="The content to export", max_length=MAX_CONTENT_SIZE)
    title: Optional[str] = Field(None, description="Title for export", max_length=500)
    query: Optional[str] = Field(None, description="Original query", max_length=10_000)
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
    content: str = Field(..., description="The content to share", max_length=MAX_CONTENT_SIZE)
    title: Optional[str] = Field(None, description="Title for shared content", max_length=500)
    query: Optional[str] = Field(None, description="Original query", max_length=10_000)
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
    content: str = Field(..., description="The content to bookmark", max_length=MAX_CONTENT_SIZE)
    title: Optional[str] = Field(None, description="Title for bookmark", max_length=500)
    query: Optional[str] = Field(None, description="Original query", max_length=10_000)
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
# Helper Functions
# ============================================

def _get_tenant_id(body_tenant_id: Optional[str], header_tenant_id: Optional[str]) -> str:
    """Extract and validate tenant ID from request body or header.

    Args:
        body_tenant_id: Tenant ID from request body
        header_tenant_id: Tenant ID from X-Tenant-ID header

    Returns:
        Valid tenant ID

    Raises:
        TenantRequiredError: If no tenant ID is provided
    """
    tenant_id = body_tenant_id or header_tenant_id
    if not tenant_id:
        raise TenantRequiredError()
    return tenant_id


def _generate_share_token(share_id: str, expires_at: datetime) -> str:
    """Generate a signed token for share URL validation.

    Args:
        share_id: The share ID
        expires_at: Expiration timestamp

    Returns:
        HMAC signature for the share link
    """
    message = f"{share_id}:{expires_at.isoformat()}"
    return hmac.new(
        _SHARE_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()[:16]


def _verify_share_token(share_id: str, expires_at: datetime, token: str) -> bool:
    """Verify a share token is valid.

    Args:
        share_id: The share ID
        expires_at: Expiration timestamp
        token: The token to verify

    Returns:
        True if token is valid, False otherwise
    """
    expected = _generate_share_token(share_id, expires_at)
    return hmac.compare_digest(expected, token)


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
        x_tenant_id: Tenant ID from header (required if not in body)

    Returns:
        SaveContentResponse with workspace reference

    Raises:
        TenantRequiredError: If no tenant ID is provided
    """
    tenant_id = _get_tenant_id(request_body.tenant_id, x_tenant_id)
    workspace_id = str(uuid4())
    saved_at = datetime.now(timezone.utc).isoformat()

    logger.info("Saving content to workspace", extra={
        "tenant_id": tenant_id,
        "content_id": request_body.content_id,
        "workspace_id": workspace_id,
    })

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
        x_tenant_id: Tenant ID from header (required if not in body)

    Returns:
        Response with exported content as file download

    Raises:
        TenantRequiredError: If no tenant ID is provided
    """
    tenant_id = _get_tenant_id(request_body.tenant_id, x_tenant_id)
    export_format = request_body.format
    title = request_body.title or "AI Response"
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info("Exporting content", extra={
        "tenant_id": tenant_id,
        "content_id": request_body.content_id,
        "format": export_format,
    })

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
        # PDF generation requires additional libraries (weasyprint, reportlab)
        # Return 501 Not Implemented until proper PDF support is added
        logger.warning("PDF export not implemented", extra={
            "tenant_id": tenant_id,
            "content_id": request_body.content_id,
        })
        return JSONResponse(
            status_code=501,
            content={
                "type": "https://api.example.com/errors/not-implemented",
                "title": "Not Implemented",
                "status": 501,
                "detail": "PDF export is not yet implemented. Please use markdown or json format.",
                "instance": "/api/v1/workspace/export",
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
        x_tenant_id: Tenant ID from header (required if not in body)

    Returns:
        ShareContentResponse with share URL

    Raises:
        TenantRequiredError: If no tenant ID is provided
    """
    tenant_id = _get_tenant_id(request_body.tenant_id, x_tenant_id)
    share_id = str(uuid4())
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(hours=SHARE_LINK_TTL_HOURS)

    # Generate signed token for URL validation
    token = _generate_share_token(share_id, expires_at)

    logger.info("Creating share link", extra={
        "tenant_id": tenant_id,
        "content_id": request_body.content_id,
        "share_id": share_id,
        "expires_at": expires_at.isoformat(),
    })

    # Store share data (replace with DB in production)
    _shares_storage[share_id] = {
        "tenant_id": tenant_id,
        "content_id": request_body.content_id,
        "content": request_body.content,
        "title": request_body.title,
        "query": request_body.query,
        "sources": [s.model_dump() for s in request_body.sources] if request_body.sources else None,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "token": token,
    }

    # Generate share URL with signed token
    settings = get_settings()
    base_url = getattr(settings, 'frontend_url', 'https://app.example.com')
    share_url = f"{base_url}/share/{share_id}?token={token}"

    return ShareContentResponse(
        data=SharedContentData(
            share_id=share_id,
            share_url=share_url,
            expires_at=expires_at.isoformat(),
        ),
        meta={"request_id": str(uuid4()), "timestamp": created_at.isoformat()},
    )


@router.post("/bookmark", response_model=BookmarkContentResponse)
async def bookmark_content(
    request_body: BookmarkContentRequest,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> BookmarkContentResponse:
    """Bookmark content for quick access later.

    Args:
        request_body: Content to bookmark
        x_tenant_id: Tenant ID from header (required if not in body)

    Returns:
        BookmarkContentResponse with bookmark reference

    Raises:
        TenantRequiredError: If no tenant ID is provided
    """
    tenant_id = _get_tenant_id(request_body.tenant_id, x_tenant_id)
    bookmark_id = str(uuid4())
    bookmarked_at = datetime.now(timezone.utc).isoformat()

    logger.info("Creating bookmark", extra={
        "tenant_id": tenant_id,
        "content_id": request_body.content_id,
        "bookmark_id": bookmark_id,
    })

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
        tenant_id: Tenant ID to list bookmarks for (required)
        limit: Maximum number of bookmarks to return
        offset: Pagination offset

    Returns:
        BookmarksListResponse with list of bookmarks
    """
    logger.info("Listing bookmarks", extra={
        "tenant_id": tenant_id,
        "limit": limit,
        "offset": offset,
    })

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
