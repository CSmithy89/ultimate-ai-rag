"""Ingestion API endpoints for URL crawling and document processing."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

import aiofiles
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from agentic_rag_backend.config import get_settings
from agentic_rag_backend.core.errors import (
    AppError,
    FileTooLargeError,
    InvalidPdfError,
    InvalidUrlError,
    JobNotFoundError,
    StorageError,
)
from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.db.redis import (
    CRAWL_JOBS_STREAM,
    PARSE_JOBS_STREAM,
    RedisClient,
)
from agentic_rag_backend.indexing.crawler import is_valid_url
from agentic_rag_backend.models.ingest import (
    CrawlRequest,
    CrawlResponse,
    DocumentUploadResponse,
    JobStatusEnum,
    JobType,
)

logger = structlog.get_logger(__name__)

# Rate limiter instance - key function extracts client IP
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


# Response wrapper models
class Meta(BaseModel):
    """Response metadata."""

    requestId: str
    timestamp: str


class SuccessResponse(BaseModel):
    """Standard success response wrapper."""

    data: Any
    meta: Meta


def success_response(data: Any) -> dict[str, Any]:
    """
    Wrap data in standard success response format.

    Args:
        data: Response data

    Returns:
        Dictionary with data and meta fields
    """
    return {
        "data": data,
        "meta": {
            "requestId": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }


# Dependency injection for database clients from app.state
async def get_redis(request: Request) -> RedisClient:
    """Get Redis client from app.state."""
    return request.app.state.redis


async def get_postgres(request: Request) -> PostgresClient:
    """Get PostgreSQL client from app.state."""
    return request.app.state.postgres


@router.post(
    "/url",
    response_model=SuccessResponse,
    summary="Start URL crawl job",
    description="Trigger autonomous crawling of a documentation website.",
)
@limiter.limit("10/minute")
async def create_crawl_job(
    request: Request,
    crawl_request: CrawlRequest,
    redis: RedisClient = Depends(get_redis),
    postgres: PostgresClient = Depends(get_postgres),
) -> dict[str, Any]:
    """
    Create a new URL crawl job.

    This endpoint accepts a URL and queues it for crawling via the
    ingestion pipeline. The crawl runs asynchronously and progress
    can be tracked via the job status endpoint.

    Args:
        request: FastAPI request object (used by rate limiter)
        crawl_request: Crawl request with URL and options
        redis: Redis client for job queue
        postgres: PostgreSQL client for job tracking

    Returns:
        Success response with job_id and status

    Raises:
        HTTPException: If URL is invalid or processing fails
    """
    url_str = str(crawl_request.url)

    # Validate URL
    if not is_valid_url(url_str):
        raise InvalidUrlError(url_str, "URL format is not valid")

    logger.info(
        "creating_crawl_job",
        url=url_str,
        tenant_id=str(crawl_request.tenant_id),
        max_depth=crawl_request.max_depth,
    )

    try:
        # Create job in database
        job_id = await postgres.create_job(
            tenant_id=crawl_request.tenant_id,
            job_type=JobType.CRAWL,
        )

        # Queue job for processing
        await redis.publish_job(
            stream=CRAWL_JOBS_STREAM,
            job_data={
                "job_id": str(job_id),
                "tenant_id": str(crawl_request.tenant_id),
                "url": url_str,
                "max_depth": crawl_request.max_depth,
                "options": crawl_request.options.model_dump(),
            },
        )

        logger.info(
            "crawl_job_created",
            job_id=str(job_id),
            url=url_str,
        )

        return success_response(
            CrawlResponse(
                job_id=job_id,
                status=JobStatusEnum.QUEUED,
            ).model_dump()
        )

    except AppError:
        raise
    except Exception as e:
        logger.error("create_crawl_job_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create crawl job") from e


@router.post(
    "/document",
    response_model=SuccessResponse,
    summary="Upload and parse PDF document",
    description="Upload a PDF document for parsing via the Docling pipeline.",
)
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="PDF file to upload"),
    tenant_id: UUID = Form(..., description="Tenant identifier"),
    metadata: Optional[str] = Form(None, description="JSON metadata string"),
    redis: RedisClient = Depends(get_redis),
    postgres: PostgresClient = Depends(get_postgres),
) -> dict[str, Any]:
    """
    Upload and queue a PDF document for parsing.

    This endpoint:
    1. Validates the uploaded file (PDF format, size limit)
    2. Saves the file to temporary storage
    3. Creates document and job records
    4. Queues the job for async parsing via Redis Streams

    Args:
        request: FastAPI request object (used by rate limiter)
        file: Uploaded PDF file
        tenant_id: Tenant identifier for multi-tenancy
        metadata: Optional JSON string with additional metadata
        redis: Redis client for job queue
        postgres: PostgreSQL client for records

    Returns:
        Success response with job_id, status, filename, and file_size

    Raises:
        InvalidPdfError: If file is not a valid PDF
        FileTooLargeError: If file exceeds size limit
        StorageError: If file storage fails
    """
    settings = get_settings()
    max_file_size = settings.max_upload_size_mb * 1024 * 1024

    logger.info(
        "document_upload_started",
        filename=file.filename,
        content_type=file.content_type,
        tenant_id=str(tenant_id),
    )

    # Validate content type
    if file.content_type != "application/pdf":
        raise InvalidPdfError(
            file.filename or "unknown",
            "File must be a PDF document (application/pdf)",
        )

    try:
        # Read file content
        contents = await file.read()
        file_size = len(contents)

        # Validate file size
        if file_size > max_file_size:
            raise FileTooLargeError(
                file.filename or "unknown",
                settings.max_upload_size_mb,
            )

        # Validate PDF magic bytes
        if not contents.startswith(b"%PDF"):
            raise InvalidPdfError(
                file.filename or "unknown",
                "File does not appear to be a valid PDF document",
            )

        # Compute content hash for deduplication
        content_hash = hashlib.sha256(contents).hexdigest()

        # Create document record
        doc_id = await postgres.create_document(
            tenant_id=tenant_id,
            source_type="pdf",
            content_hash=content_hash,
            filename=file.filename,
            file_size=file_size,
        )

        # Create parse job
        job_id = await postgres.create_job(
            tenant_id=tenant_id,
            job_type=JobType.PARSE,
            document_id=doc_id,
        )

        # Save to temp storage
        temp_dir = Path(settings.temp_upload_dir) / str(tenant_id)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{job_id}.pdf"

        try:
            async with aiofiles.open(temp_path, "wb") as f:
                await f.write(contents)
        except Exception as e:
            raise StorageError("save_file", str(e)) from e

        # Queue for parsing
        await redis.publish_job(
            stream=PARSE_JOBS_STREAM,
            job_data={
                "job_id": str(job_id),
                "tenant_id": str(tenant_id),
                "document_id": str(doc_id),
                "file_path": str(temp_path),
                "filename": file.filename or "document.pdf",
            },
        )

        logger.info(
            "document_upload_completed",
            job_id=str(job_id),
            document_id=str(doc_id),
            filename=file.filename,
            file_size=file_size,
            content_hash=content_hash[:16] + "...",
        )

        return success_response(
            DocumentUploadResponse(
                job_id=job_id,
                status=JobStatusEnum.QUEUED,
                filename=file.filename or "document.pdf",
                file_size=file_size,
            ).model_dump()
        )

    except AppError:
        raise
    except Exception as e:
        logger.error(
            "document_upload_failed",
            filename=file.filename,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Failed to upload document") from e


@router.get(
    "/jobs/{job_id}",
    response_model=SuccessResponse,
    summary="Get job status",
    description="Retrieve the current status and progress of an ingestion job.",
)
async def get_job_status(
    job_id: UUID,
    tenant_id: UUID = Query(..., description="Tenant ID for access control"),
    postgres: PostgresClient = Depends(get_postgres),
) -> dict[str, Any]:
    """
    Get the status of an ingestion job.

    Returns the current status, progress metrics, and any error messages
    for the specified job.

    Args:
        job_id: UUID of the job to query
        tenant_id: Tenant ID for multi-tenancy access control
        postgres: PostgreSQL client for job lookup

    Returns:
        Success response with job status details

    Raises:
        HTTPException: If job is not found
    """
    logger.debug(
        "getting_job_status",
        job_id=str(job_id),
        tenant_id=str(tenant_id),
    )

    job = await postgres.get_job(job_id=job_id, tenant_id=tenant_id)

    if job is None:
        raise JobNotFoundError(str(job_id))

    return success_response(job.model_dump(mode="json"))


@router.get(
    "/jobs",
    response_model=SuccessResponse,
    summary="List jobs",
    description="List ingestion jobs for a tenant with optional filtering.",
)
async def list_jobs(
    tenant_id: UUID = Query(..., description="Tenant ID for access control"),
    status: Optional[JobStatusEnum] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    postgres: PostgresClient = Depends(get_postgres),
) -> dict[str, Any]:
    """
    List ingestion jobs for a tenant.

    Returns a paginated list of jobs with optional status filtering.

    Args:
        tenant_id: Tenant ID for multi-tenancy access control
        status: Optional filter by job status
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip for pagination
        postgres: PostgreSQL client for job lookup

    Returns:
        Success response with list of jobs
    """
    logger.debug(
        "listing_jobs",
        tenant_id=str(tenant_id),
        status=status.value if status else None,
        limit=limit,
        offset=offset,
    )

    jobs, total = await postgres.list_jobs(
        tenant_id=tenant_id,
        status=status,
        limit=limit,
        offset=offset,
    )

    return success_response({
        "jobs": [job.model_dump(mode="json") for job in jobs],
        "total": total,
        "limit": limit,
        "offset": offset,
    })
