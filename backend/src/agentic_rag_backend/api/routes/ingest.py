"""Ingestion API endpoints for URL crawling and document processing."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from agentic_rag_backend.core.errors import (
    AppError,
    InvalidUrlError,
    JobNotFoundError,
)
from agentic_rag_backend.db.postgres import PostgresClient, get_postgres_client
from agentic_rag_backend.db.redis import (
    CRAWL_JOBS_STREAM,
    RedisClient,
    get_redis_client,
)
from agentic_rag_backend.indexing.crawler import is_valid_url
from agentic_rag_backend.models.ingest import (
    CrawlRequest,
    CrawlResponse,
    JobStatus,
    JobStatusEnum,
    JobType,
)

logger = structlog.get_logger(__name__)

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
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }


# Dependency injection for database clients
async def get_redis() -> RedisClient:
    """Get Redis client dependency."""
    from agentic_rag_backend.config import load_settings

    settings = load_settings()
    return await get_redis_client(settings.redis_url)


async def get_postgres() -> PostgresClient:
    """Get PostgreSQL client dependency."""
    from agentic_rag_backend.config import load_settings

    settings = load_settings()
    return await get_postgres_client(settings.database_url)


@router.post(
    "/url",
    response_model=SuccessResponse,
    summary="Start URL crawl job",
    description="Trigger autonomous crawling of a documentation website.",
)
async def create_crawl_job(
    request: CrawlRequest,
    redis: RedisClient = Depends(get_redis),
    postgres: PostgresClient = Depends(get_postgres),
) -> dict[str, Any]:
    """
    Create a new URL crawl job.

    This endpoint accepts a URL and queues it for crawling via the
    ingestion pipeline. The crawl runs asynchronously and progress
    can be tracked via the job status endpoint.

    Args:
        request: Crawl request with URL and options
        redis: Redis client for job queue
        postgres: PostgreSQL client for job tracking

    Returns:
        Success response with job_id and status

    Raises:
        HTTPException: If URL is invalid or processing fails
    """
    url_str = str(request.url)

    # Validate URL
    if not is_valid_url(url_str):
        raise InvalidUrlError(url_str, "URL format is not valid")

    logger.info(
        "creating_crawl_job",
        url=url_str,
        tenant_id=str(request.tenant_id),
        max_depth=request.max_depth,
    )

    try:
        # Create job in database
        job_id = await postgres.create_job(
            tenant_id=request.tenant_id,
            job_type=JobType.CRAWL,
        )

        # Queue job for processing
        await redis.publish_job(
            stream=CRAWL_JOBS_STREAM,
            job_data={
                "job_id": str(job_id),
                "tenant_id": str(request.tenant_id),
                "url": url_str,
                "max_depth": request.max_depth,
                "options": request.options.model_dump(),
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
        raise HTTPException(status_code=500, detail="Failed to create crawl job")


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

    jobs = await postgres.list_jobs(
        tenant_id=tenant_id,
        status=status,
        limit=limit,
        offset=offset,
    )

    return success_response({
        "jobs": [job.model_dump(mode="json") for job in jobs],
        "total": len(jobs),
        "limit": limit,
        "offset": offset,
    })
