"""Pydantic models for documents."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Type of document source."""

    URL = "url"
    PDF = "pdf"
    TEXT = "text"


class DocumentStatus(str, Enum):
    """Processing status of a document."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Metadata extracted from a document."""

    title: Optional[str] = Field(default=None, description="Document title")
    author: Optional[str] = Field(default=None, description="Document author")
    description: Optional[str] = Field(
        default=None, description="Document description"
    )
    language: Optional[str] = Field(default=None, description="Document language")
    crawl_timestamp: Optional[datetime] = Field(
        default=None, description="When the document was crawled"
    )
    source_url: Optional[str] = Field(
        default=None, description="Original source URL"
    )
    page_count: Optional[int] = Field(
        default=None, ge=0, description="Number of pages (for PDFs)"
    )
    word_count: Optional[int] = Field(
        default=None, ge=0, description="Approximate word count"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class UnifiedDocument(BaseModel):
    """
    Unified document model for normalized document representation.

    Both Crawl4AI output and Docling output are normalized to this
    common model before processing.
    """

    id: UUID = Field(..., description="Unique document identifier")
    tenant_id: UUID = Field(..., description="Tenant identifier for multi-tenancy")
    source_type: SourceType = Field(..., description="Type of document source")
    source_url: Optional[str] = Field(
        default=None, description="Source URL for web documents"
    )
    filename: Optional[str] = Field(
        default=None, description="Filename for uploaded documents"
    )
    content: str = Field(..., description="Document content (markdown format)")
    content_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of content for deduplication",
    )
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING, description="Processing status"
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata, description="Document metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "tenant_id": "123e4567-e89b-12d3-a456-426614174001",
            "source_type": "url",
            "source_url": "https://docs.example.com/intro",
            "content": "# Introduction\n\nThis is the document content.",
            "content_hash": "a" * 64,
            "status": "pending",
            "metadata": {"title": "Introduction"},
        }
    ]}}


class CrawledPage(BaseModel):
    """Represents a single crawled page before normalization."""

    url: str = Field(..., description="Page URL")
    title: Optional[str] = Field(default=None, description="Page title")
    content: str = Field(..., description="Page content in markdown format")
    content_hash: str = Field(..., description="SHA-256 hash of content")
    crawl_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the page was crawled"
    )
    depth: int = Field(default=0, ge=0, description="Crawl depth from start URL")
    links: list[str] = Field(
        default_factory=list, description="Links discovered on this page"
    )
