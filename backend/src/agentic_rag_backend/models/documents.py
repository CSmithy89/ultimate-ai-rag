"""Pydantic models for documents."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional
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
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
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
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the page was crawled",
    )
    depth: int = Field(default=0, ge=0, description="Crawl depth from start URL")
    links: list[str] = Field(
        default_factory=list, description="Links discovered on this page"
    )


# Story 4.2 - PDF Document Parsing Models


class DocumentSection(BaseModel):
    """Represents a document section with hierarchy."""

    heading: Optional[str] = Field(default=None, description="Section heading text")
    level: int = Field(
        default=1, ge=1, le=6, description="Heading level (1-6 for h1-h6)"
    )
    content: str = Field(..., description="Section content text")
    page_number: int = Field(default=1, ge=1, description="Page number where section appears")

    model_config = {"json_schema_extra": {"examples": [
        {
            "heading": "Introduction",
            "level": 1,
            "content": "This section introduces the topic...",
            "page_number": 1,
        }
    ]}}


class TableContent(BaseModel):
    """Extracted table in markdown format."""

    caption: Optional[str] = Field(default=None, description="Table caption if available")
    markdown: str = Field(..., description="Table content in markdown format")
    row_count: int = Field(default=0, ge=0, description="Number of rows in the table")
    column_count: int = Field(default=0, ge=0, description="Number of columns in the table")
    page_number: int = Field(default=1, ge=1, description="Page number where table appears")

    model_config = {"json_schema_extra": {"examples": [
        {
            "caption": "Sales Data Q4 2024",
            "markdown": "| Product | Sales |\n|---------|-------|\n| A | 100 |\n| B | 200 |",
            "row_count": 3,
            "column_count": 2,
            "page_number": 5,
        }
    ]}}


class FootnoteContent(BaseModel):
    """Extracted footnote content."""

    reference: str = Field(..., description="Footnote reference marker (e.g., '1', '*')")
    content: str = Field(..., description="Footnote text content")
    page_number: int = Field(default=1, ge=1, description="Page number where footnote appears")

    model_config = {"json_schema_extra": {"examples": [
        {
            "reference": "1",
            "content": "This is a reference to the original source.",
            "page_number": 3,
        }
    ]}}


class ParsedDocument(BaseModel):
    """
    Extended document model for parsed PDF content.

    Contains structured content extracted by Docling including sections,
    tables, and footnotes with their semantic relationships preserved.
    """

    id: UUID = Field(..., description="Unique document identifier")
    tenant_id: UUID = Field(..., description="Tenant identifier for multi-tenancy")
    source_type: Literal["pdf"] = Field(default="pdf", description="Source type (always 'pdf')")
    filename: str = Field(..., description="Original filename")
    content_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of file content for deduplication",
    )
    file_size: int = Field(..., ge=0, description="File size in bytes")
    page_count: int = Field(..., ge=1, description="Number of pages in the document")
    sections: list[DocumentSection] = Field(
        default_factory=list, description="Extracted document sections"
    )
    tables: list[TableContent] = Field(
        default_factory=list, description="Extracted tables in markdown format"
    )
    footnotes: list[FootnoteContent] = Field(
        default_factory=list, description="Extracted footnotes"
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata, description="Document metadata"
    )
    processing_time_ms: int = Field(
        default=0, ge=0, description="Processing time in milliseconds (NFR2 tracking)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "tenant_id": "123e4567-e89b-12d3-a456-426614174001",
            "source_type": "pdf",
            "filename": "document.pdf",
            "content_hash": "a" * 64,
            "file_size": 1024567,
            "page_count": 25,
            "sections": [{"heading": "Introduction", "level": 1, "content": "...", "page_number": 1}],
            "tables": [],
            "footnotes": [],
            "metadata": {"title": "Technical Document"},
            "processing_time_ms": 45000,
        }
    ]}}

    def to_unified_document(self) -> UnifiedDocument:
        """
        Convert ParsedDocument to UnifiedDocument format.

        Concatenates all sections and tables into a single markdown string
        for downstream processing (chunking, embedding, etc.).

        Returns:
            UnifiedDocument with combined content
        """
        # Build markdown content from sections and tables
        content_parts = []

        for section in self.sections:
            if section.heading:
                content_parts.append(f"{'#' * section.level} {section.heading}\n")
            content_parts.append(section.content + "\n")

        for table in self.tables:
            if table.caption:
                content_parts.append(f"\n**{table.caption}**\n")
            content_parts.append(table.markdown + "\n")

        if self.footnotes:
            content_parts.append("\n---\n**Footnotes:**\n")
            for fn in self.footnotes:
                content_parts.append(f"[{fn.reference}] {fn.content}\n")

        content = "\n".join(content_parts)

        return UnifiedDocument(
            id=self.id,
            tenant_id=self.tenant_id,
            source_type=SourceType.PDF,
            filename=self.filename,
            content=content,
            content_hash=self.content_hash,
            status=DocumentStatus.COMPLETED,
            metadata=DocumentMetadata(
                title=self.metadata.title,
                author=self.metadata.author,
                page_count=self.page_count,
                extra={
                    "file_size": self.file_size,
                    "tables_count": len(self.tables),
                    "sections_count": len(self.sections),
                    "processing_time_ms": self.processing_time_ms,
                },
            ),
            created_at=self.created_at,
        )
