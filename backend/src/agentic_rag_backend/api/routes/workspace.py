"""Workspace API routes for frontend actions.

Story 6-5: Frontend Actions

Provides endpoints for:
- POST /workspace/save - Save content to workspace
- GET /workspace/{workspace_id} - Load saved content
- POST /workspace/export - Export as markdown/PDF/JSON
- POST /workspace/share - Generate shareable link
- GET /workspace/share/{share_id} - Retrieve shared content
- POST /workspace/bookmark - Bookmark content
- GET /workspace/bookmarks - List bookmarks
"""

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import logging
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from agentic_rag_backend.config import get_settings
from agentic_rag_backend.core.errors import TenantRequiredError
from agentic_rag_backend.db.postgres import PostgresClient


# Rate limiter instance - key function extracts client IP
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/workspace", tags=["workspace"])
logger = logging.getLogger(__name__)


# ============================================
# Constants
# ============================================

# Maximum content size to prevent DoS (100KB bytes, not characters)
MAX_CONTENT_SIZE = 100_000

# Share link expiration (24 hours)
SHARE_LINK_TTL_HOURS = 24


def _get_share_secret() -> str:
    """Get share secret from config (cached via lru_cache in get_settings)."""
    return get_settings().share_secret


async def get_postgres(request: Request) -> PostgresClient:
    """Get PostgreSQL client from app.state."""
    return request.app.state.postgres


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
    title: Optional[str] = Field(None, description="Title for saved content", max_length=500)
    query: Optional[str] = Field(None, description="Original query", max_length=10_000)
    sources: Optional[list[SourceInfo]] = Field(None, description="Source references")
    session_id: Optional[str] = Field(None, description="Session ID")
    trajectory_id: Optional[str] = Field(None, description="Trajectory ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")

    @field_validator("content")
    @classmethod
    def validate_content_size(cls, v: str) -> str:
        """Validate content byte size (UTF-8)."""
        return _validate_byte_size(v, MAX_CONTENT_SIZE)


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
    title: Optional[str] = Field(None, description="Title for export", max_length=500)
    query: Optional[str] = Field(None, description="Original query", max_length=10_000)
    sources: Optional[list[SourceInfo]] = Field(None, description="Source references")
    format: Literal["markdown", "pdf", "json"] = Field(..., description="Export format")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")

    @field_validator("content")
    @classmethod
    def validate_content_size(cls, v: str) -> str:
        """Validate content byte size (UTF-8)."""
        return _validate_byte_size(v, MAX_CONTENT_SIZE)


class ExportContentResponse(BaseModel):
    """Response from export endpoint."""

    filename: str
    format: str


class ShareContentRequest(BaseModel):
    """Request to share content."""

    content_id: str = Field(..., description="Unique ID of the content")
    content: str = Field(..., description="The content to share")
    title: Optional[str] = Field(None, description="Title for shared content", max_length=500)
    query: Optional[str] = Field(None, description="Original query", max_length=10_000)
    sources: Optional[list[SourceInfo]] = Field(None, description="Source references")
    session_id: Optional[str] = Field(None, description="Session ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")

    @field_validator("content")
    @classmethod
    def validate_content_size(cls, v: str) -> str:
        """Validate content byte size (UTF-8)."""
        return _validate_byte_size(v, MAX_CONTENT_SIZE)


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
    title: Optional[str] = Field(None, description="Title for bookmark", max_length=500)
    query: Optional[str] = Field(None, description="Original query", max_length=10_000)
    session_id: Optional[str] = Field(None, description="Session ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")

    @field_validator("content")
    @classmethod
    def validate_content_size(cls, v: str) -> str:
        """Validate content byte size (UTF-8)."""
        return _validate_byte_size(v, MAX_CONTENT_SIZE)


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


class WorkspaceContentData(BaseModel):
    """Workspace content response data."""

    workspace_id: str
    content_id: str
    content: str
    title: Optional[str] = None
    query: Optional[str] = None
    sources: Optional[list[SourceInfo]] = None
    session_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    saved_at: str


class WorkspaceContentResponse(BaseModel):
    """Response for loading saved workspace content."""

    data: WorkspaceContentData
    meta: dict[str, Any] = Field(default_factory=dict)


class SharedContentDetails(BaseModel):
    """Shared content details response data."""

    share_id: str
    content_id: str
    content: str
    title: Optional[str] = None
    query: Optional[str] = None
    sources: Optional[list[SourceInfo]] = None
    created_at: str
    expires_at: Optional[str] = None


class SharedContentRetrieveResponse(BaseModel):
    """Response for retrieving shared content."""

    data: SharedContentDetails
    meta: dict[str, Any] = Field(default_factory=dict)


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


def _parse_tenant_uuid(tenant_id: str) -> UUID:
    """Parse tenant UUID or raise a 400 error."""
    try:
        return UUID(tenant_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="tenant_id must be a valid UUID",
        ) from exc


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
        _get_share_secret().encode(),
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


def _validate_byte_size(value: str, max_bytes: int) -> str:
    """Validate string byte size (UTF-8).

    Args:
        value: String to validate
        max_bytes: Maximum allowed byte size

    Returns:
        The original string if valid

    Raises:
        ValueError: If byte size exceeds maximum
    """
    byte_size = len(value.encode('utf-8'))
    if byte_size > max_bytes:
        raise ValueError(
            f'Content size ({byte_size} bytes) exceeds maximum of {max_bytes} bytes'
        )
    return value


# ============================================
# Endpoints
# ============================================

@router.post("/save", response_model=SaveContentResponse)
@limiter.limit("30/minute")
async def save_content(
    request: Request,
    request_body: SaveContentRequest,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    postgres: PostgresClient = Depends(get_postgres),
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
    tenant_uuid = _parse_tenant_uuid(tenant_id)
    workspace_uuid = uuid4()
    saved_at = datetime.now(timezone.utc)
    title = request_body.title or f"Response - {saved_at.isoformat()[:10]}"

    logger.info("Saving content to workspace", extra={
        "tenant_id": tenant_id,
        "content_id": request_body.content_id,
        "workspace_id": str(workspace_uuid),
    })

    try:
        created_at = await postgres.create_workspace_item(
            workspace_id=workspace_uuid,
            tenant_id=tenant_uuid,
            content_id=request_body.content_id,
            content=request_body.content,
            title=title,
            query=request_body.query,
            sources=[s.model_dump() for s in request_body.sources] if request_body.sources else None,
            session_id=request_body.session_id,
            trajectory_id=request_body.trajectory_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    return SaveContentResponse(
        data=SavedContentData(
            content_id=request_body.content_id,
            workspace_id=str(workspace_uuid),
            saved_at=created_at.isoformat(),
        ),
        meta={"request_id": str(uuid4()), "timestamp": created_at.isoformat()},
    )


@router.post("/export")
@limiter.limit("20/minute")
async def export_content(
    request: Request,
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
@limiter.limit("10/minute")
async def share_content(
    request: Request,
    request_body: ShareContentRequest,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    postgres: PostgresClient = Depends(get_postgres),
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
    tenant_uuid = _parse_tenant_uuid(tenant_id)
    share_uuid = uuid4()
    share_id = str(share_uuid)
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

    try:
        stored_at = await postgres.create_workspace_share(
            share_id=share_uuid,
            tenant_id=tenant_uuid,
            content_id=request_body.content_id,
            content=request_body.content,
            title=request_body.title,
            query=request_body.query,
            sources=[s.model_dump() for s in request_body.sources] if request_body.sources else None,
            token=token,
            expires_at=expires_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

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
        meta={"request_id": str(uuid4()), "timestamp": stored_at.isoformat()},
    )


@router.get("/share/{share_id}", response_model=SharedContentRetrieveResponse)
@limiter.limit("30/minute")
async def get_shared_content(
    request: Request,
    share_id: UUID,
    token: str = Query(..., description="Share token"),
    postgres: PostgresClient = Depends(get_postgres),
) -> SharedContentRetrieveResponse:
    """Retrieve shared content with token validation."""
    def _problem(status: int, title: str, detail: str, type_suffix: str) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content={
                "type": f"https://api.example.com/errors/{type_suffix}",
                "title": title,
                "status": status,
                "detail": detail,
                "instance": request.url.path,
            },
        )

    row = await postgres.get_workspace_share(share_id)
    if not row:
        return _problem(404, "Not Found", "Shared content not found", "not-found")

    expires_at = row.get("expires_at")
    if expires_at and expires_at < datetime.now(timezone.utc):
        return _problem(410, "Gone", "Share link has expired", "share-expired")

    stored_token = row.get("token")
    if stored_token:
        if not hmac.compare_digest(stored_token, token):
            return _problem(403, "Forbidden", "Invalid share token", "invalid-token")
    elif expires_at and not _verify_share_token(str(share_id), expires_at, token):
        return _problem(403, "Forbidden", "Invalid share token", "invalid-token")

    created_at = row["created_at"].isoformat() if row.get("created_at") else None
    expires_at_str = expires_at.isoformat() if expires_at else None

    return SharedContentRetrieveResponse(
        data=SharedContentDetails(
            share_id=str(row["id"]),
            content_id=row["content_id"],
            content=row["content"],
            title=row.get("title"),
            query=row.get("query"),
            sources=row.get("sources"),
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            expires_at=expires_at_str,
        ),
        meta={"request_id": str(uuid4()), "timestamp": datetime.now(timezone.utc).isoformat()},
    )


@router.post("/bookmark", response_model=BookmarkContentResponse)
@limiter.limit("30/minute")
async def bookmark_content(
    request: Request,
    request_body: BookmarkContentRequest,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    postgres: PostgresClient = Depends(get_postgres),
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
    tenant_uuid = _parse_tenant_uuid(tenant_id)
    bookmark_uuid = uuid4()

    logger.info("Creating bookmark", extra={
        "tenant_id": tenant_id,
        "content_id": request_body.content_id,
        "bookmark_id": str(bookmark_uuid),
    })

    try:
        stored_at = await postgres.create_workspace_bookmark(
            bookmark_id=bookmark_uuid,
            tenant_id=tenant_uuid,
            content_id=request_body.content_id,
            content=request_body.content,
            title=request_body.title or "Bookmarked Response",
            query=request_body.query,
            session_id=request_body.session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    return BookmarkContentResponse(
        data=BookmarkedContentData(
            bookmark_id=str(bookmark_uuid),
            bookmarked_at=stored_at.isoformat(),
        ),
        meta={"request_id": str(uuid4()), "timestamp": stored_at.isoformat()},
    )


@router.get("/bookmarks", response_model=BookmarksListResponse)
@limiter.limit("60/minute")
async def get_bookmarks(
    request: Request,
    tenant_id: Optional[str] = Query(None, description="Tenant ID"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum bookmarks to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    postgres: PostgresClient = Depends(get_postgres),
) -> BookmarksListResponse:
    """List bookmarks for a tenant.

    Args:
        tenant_id: Tenant ID to list bookmarks for (required)
        limit: Maximum number of bookmarks to return
        offset: Pagination offset

    Returns:
        BookmarksListResponse with list of bookmarks
    """
    tenant_value = _get_tenant_id(tenant_id, x_tenant_id)
    tenant_uuid = _parse_tenant_uuid(tenant_value)

    logger.info("Listing bookmarks", extra={
        "tenant_id": tenant_value,
        "limit": limit,
        "offset": offset,
    })

    bookmarks, total = await postgres.list_workspace_bookmarks(
        tenant_id=tenant_uuid,
        limit=limit,
        offset=offset,
    )

    items = [
        BookmarkItem(
            bookmark_id=str(b["id"]),
            content_id=b["content_id"],
            title=b.get("title") or "Bookmarked Response",
            query=b.get("query"),
            bookmarked_at=b["created_at"].isoformat(),
        )
        for b in bookmarks
    ]

    return BookmarksListResponse(
        data=items,
        meta={
            "request_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": total,
            "limit": limit,
            "offset": offset,
        },
    )


@router.get("/{workspace_id}", response_model=WorkspaceContentResponse)
@limiter.limit("60/minute")
async def load_workspace_content(
    request: Request,
    workspace_id: UUID,
    tenant_id: Optional[str] = Query(None, description="Tenant ID"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    postgres: PostgresClient = Depends(get_postgres),
) -> WorkspaceContentResponse:
    """Load saved workspace content by workspace ID."""
    tenant_value = _get_tenant_id(tenant_id, x_tenant_id)
    tenant_uuid = _parse_tenant_uuid(tenant_value)

    row = await postgres.get_workspace_item(
        tenant_id=tenant_uuid,
        workspace_id=workspace_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Workspace item not found")

    saved_at = row["created_at"].isoformat() if row.get("created_at") else None

    return WorkspaceContentResponse(
        data=WorkspaceContentData(
            workspace_id=str(row["id"]),
            content_id=row["content_id"],
            content=row["content"],
            title=row.get("title"),
            query=row.get("query"),
            sources=row.get("sources"),
            session_id=row.get("session_id"),
            trajectory_id=row.get("trajectory_id"),
            saved_at=saved_at or datetime.now(timezone.utc).isoformat(),
        ),
        meta={"request_id": str(uuid4()), "timestamp": datetime.now(timezone.utc).isoformat()},
    )
