"""Pydantic models for ingestion requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class JobType(str, Enum):
    """Type of ingestion job."""

    CRAWL = "crawl"
    PARSE = "parse"
    INDEX = "index"


class JobStatusEnum(str, Enum):
    """Status of an ingestion job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CrawlOptions(BaseModel):
    """Optional crawl configuration."""

    follow_links: bool = Field(
        default=True, description="Whether to follow links on the page"
    )
    respect_robots_txt: bool = Field(
        default=True, description="Whether to respect robots.txt directives"
    )
    rate_limit: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Requests per second"
    )
    include_patterns: list[str] = Field(
        default_factory=list,
        description="URL patterns to include (regex)",
    )
    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="URL patterns to exclude (regex)",
    )


class CrawlRequest(BaseModel):
    """Request model for URL crawl initiation."""

    url: HttpUrl = Field(..., description="URL to crawl")
    tenant_id: UUID = Field(..., description="Tenant identifier for multi-tenancy")
    max_depth: int = Field(
        default=3, ge=1, le=10, description="Maximum crawl depth for linked pages"
    )
    options: CrawlOptions = Field(
        default_factory=CrawlOptions, description="Additional crawl options"
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "url": "https://docs.example.com",
            "tenant_id": "123e4567-e89b-12d3-a456-426614174000",
            "max_depth": 3,
            "options": {"rate_limit": 1.0},
        }
    ]}}


class CrawlResponse(BaseModel):
    """Response model for crawl job creation."""

    job_id: UUID = Field(..., description="Unique identifier for the crawl job")
    status: JobStatusEnum = Field(
        default=JobStatusEnum.QUEUED, description="Initial job status"
    )


class JobProgress(BaseModel):
    """Progress metrics for a crawl job."""

    pages_crawled: int = Field(default=0, ge=0, description="Number of pages crawled")
    pages_discovered: int = Field(
        default=0, ge=0, description="Total pages discovered"
    )
    pages_failed: int = Field(default=0, ge=0, description="Number of failed pages")
    current_url: Optional[str] = Field(
        default=None, description="URL currently being crawled"
    )


class JobStatus(BaseModel):
    """Job status with progress metrics."""

    job_id: UUID = Field(..., description="Unique job identifier")
    tenant_id: UUID = Field(..., description="Tenant identifier")
    job_type: JobType = Field(..., description="Type of job")
    status: JobStatusEnum = Field(..., description="Current job status")
    progress: Optional[JobProgress] = Field(
        default=None, description="Progress metrics if available"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if job failed"
    )
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(
        default=None, description="Job start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="Job completion timestamp"
    )


# Story 4.2 - PDF Document Parsing Models


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    job_id: UUID = Field(..., description="Unique identifier for the parse job")
    status: JobStatusEnum = Field(
        default=JobStatusEnum.QUEUED, description="Initial job status"
    )
    filename: str = Field(..., description="Uploaded filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")

    model_config = {"json_schema_extra": {"examples": [
        {
            "job_id": "123e4567-e89b-12d3-a456-426614174000",
            "status": "queued",
            "filename": "document.pdf",
            "file_size": 1024567,
        }
    ]}}


class ParseProgress(BaseModel):
    """Progress metrics for a parse job."""

    pages_parsed: int = Field(default=0, ge=0, description="Number of pages parsed")
    total_pages: int = Field(default=0, ge=0, description="Total pages in document")
    tables_extracted: int = Field(default=0, ge=0, description="Number of tables extracted")
    sections_extracted: int = Field(default=0, ge=0, description="Number of sections extracted")
    current_page: Optional[int] = Field(
        default=None, ge=1, description="Current page being processed"
    )
    processing_time_ms: Optional[int] = Field(
        default=None, ge=0, description="Processing time in milliseconds"
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "pages_parsed": 25,
            "total_pages": 50,
            "tables_extracted": 5,
            "sections_extracted": 12,
            "current_page": 26,
            "processing_time_ms": 45000,
        }
    ]}}
