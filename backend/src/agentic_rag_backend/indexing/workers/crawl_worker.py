"""Async crawl worker consuming from Redis Streams."""

import asyncio
import json
import signal
from typing import Any
from uuid import UUID

import structlog

from agentic_rag_backend.db.postgres import PostgresClient, get_postgres_client
from agentic_rag_backend.db.redis import (
    CRAWL_CONSUMER_GROUP,
    CRAWL_JOBS_STREAM,
    PARSE_JOBS_STREAM,
    RedisClient,
    get_redis_client,
)
from agentic_rag_backend.indexing.crawler import CrawlerService
from agentic_rag_backend.models.documents import CrawledPage
from agentic_rag_backend.models.ingest import CrawlOptions, JobStatusEnum

logger = structlog.get_logger(__name__)


class CrawlWorker:
    """
    Async worker that processes crawl jobs from Redis Streams.

    Flow:
    1. Consume job from crawl.jobs stream
    2. Execute crawl using CrawlerService
    3. Update job progress in PostgreSQL
    4. Queue crawled content to parse.jobs stream
    """

    def __init__(
        self,
        redis_client: RedisClient,
        postgres_client: PostgresClient,
        consumer_name: str = "crawl-worker-1",
    ) -> None:
        """
        Initialize crawl worker.

        Args:
            redis_client: Redis client for stream operations
            postgres_client: PostgreSQL client for job updates
            consumer_name: Unique name for this consumer instance
        """
        self.redis = redis_client
        self.postgres = postgres_client
        self.consumer_name = consumer_name
        self.crawler = CrawlerService()
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def process_job(self, job_data: dict[str, Any]) -> None:
        """
        Process a single crawl job.

        Args:
            job_data: Job data from Redis stream
        """
        job_id = UUID(job_data["job_id"])
        tenant_id = UUID(job_data["tenant_id"])
        url = job_data["url"]
        max_depth = int(job_data.get("max_depth", 3))

        # Parse options if present
        options_data = job_data.get("options")
        if isinstance(options_data, str):
            options_data = json.loads(options_data)
        options = CrawlOptions(**options_data) if options_data else CrawlOptions()

        logger.info(
            "processing_crawl_job",
            job_id=str(job_id),
            url=url,
            max_depth=max_depth,
        )

        # Update job status to running
        await self.postgres.update_job_status(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatusEnum.RUNNING,
            progress={"pages_crawled": 0, "pages_discovered": 0, "pages_failed": 0},
        )

        pages_crawled = 0
        pages_discovered = 0
        pages_failed = 0
        crawled_pages: list[CrawledPage] = []

        try:
            async for page in self.crawler.crawl(url, max_depth=max_depth, options=options):
                pages_crawled += 1
                pages_discovered += len(page.links)
                crawled_pages.append(page)

                # Update progress periodically
                if pages_crawled % 5 == 0:
                    await self.postgres.update_job_status(
                        job_id=job_id,
                        tenant_id=tenant_id,
                        status=JobStatusEnum.RUNNING,
                        progress={
                            "pages_crawled": pages_crawled,
                            "pages_discovered": pages_discovered,
                            "pages_failed": pages_failed,
                            "current_url": page.url,
                        },
                    )

                logger.debug(
                    "page_crawled",
                    job_id=str(job_id),
                    url=page.url,
                    pages_crawled=pages_crawled,
                )

            # Queue crawled content for parsing
            for page in crawled_pages:
                await self._queue_for_parsing(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    page=page,
                )

            # Mark job as completed
            await self.postgres.update_job_status(
                job_id=job_id,
                tenant_id=tenant_id,
                status=JobStatusEnum.COMPLETED,
                progress={
                    "pages_crawled": pages_crawled,
                    "pages_discovered": pages_discovered,
                    "pages_failed": pages_failed,
                },
            )

            logger.info(
                "crawl_job_completed",
                job_id=str(job_id),
                pages_crawled=pages_crawled,
            )

        except Exception as e:
            logger.error(
                "crawl_job_failed",
                job_id=str(job_id),
                error=str(e),
            )
            await self.postgres.update_job_status(
                job_id=job_id,
                tenant_id=tenant_id,
                status=JobStatusEnum.FAILED,
                progress={
                    "pages_crawled": pages_crawled,
                    "pages_discovered": pages_discovered,
                    "pages_failed": pages_failed + 1,
                },
                error_message=str(e),
            )

    async def _queue_for_parsing(
        self,
        job_id: UUID,
        tenant_id: UUID,
        page: CrawledPage,
    ) -> None:
        """
        Queue a crawled page for the parsing pipeline.

        Args:
            job_id: Parent crawl job ID
            tenant_id: Tenant identifier
            page: Crawled page data
        """
        await self.redis.publish_job(
            stream=PARSE_JOBS_STREAM,
            job_data={
                "parent_job_id": str(job_id),
                "tenant_id": str(tenant_id),
                "source_type": "url",
                "source_url": page.url,
                "content": page.content,
                "content_hash": page.content_hash,
                "title": page.title or "",
                "crawl_timestamp": page.crawl_timestamp.isoformat(),
            },
        )
        logger.debug(
            "page_queued_for_parsing",
            job_id=str(job_id),
            url=page.url,
        )

    async def run(self) -> None:
        """
        Run the worker, consuming jobs from Redis Streams.

        This method runs until shutdown is signaled.
        """
        self._running = True
        logger.info("crawl_worker_started", consumer=self.consumer_name)

        try:
            async for job_data in self.redis.consume_jobs(
                stream=CRAWL_JOBS_STREAM,
                group=CRAWL_CONSUMER_GROUP,
                consumer=self.consumer_name,
                count=1,
                block_ms=5000,
            ):
                if not self._running:
                    break

                try:
                    await self.process_job(job_data)
                except Exception as e:
                    logger.error(
                        "job_processing_error",
                        error=str(e),
                        job_data=job_data,
                    )

        except asyncio.CancelledError:
            logger.info("crawl_worker_cancelled")
        finally:
            self._running = False
            logger.info("crawl_worker_stopped")

    async def shutdown(self) -> None:
        """Signal the worker to shutdown gracefully."""
        logger.info("crawl_worker_shutdown_requested")
        self._running = False
        self._shutdown_event.set()


async def run_crawl_worker(
    redis_url: str,
    database_url: str,
    consumer_name: str = "crawl-worker-1",
) -> None:
    """
    Run a crawl worker instance.

    This is the main entry point for running the worker as a standalone process.

    Args:
        redis_url: Redis connection URL
        database_url: PostgreSQL connection URL
        consumer_name: Unique name for this consumer
    """
    # Initialize clients
    redis_client = await get_redis_client(redis_url)
    postgres_client = await get_postgres_client(database_url)

    worker = CrawlWorker(
        redis_client=redis_client,
        postgres_client=postgres_client,
        consumer_name=consumer_name,
    )

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    _shutdown_task = None

    def signal_handler() -> None:
        nonlocal _shutdown_task
        _shutdown_task = loop.create_task(worker.shutdown())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await worker.run()
    finally:
        # Cleanup
        from agentic_rag_backend.db.redis import close_redis_client
        from agentic_rag_backend.db.postgres import close_postgres_client

        await close_redis_client()
        await close_postgres_client()


if __name__ == "__main__":
    import os

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost/agentic_rag")
    consumer_name = os.getenv("CONSUMER_NAME", "crawl-worker-1")

    asyncio.run(run_crawl_worker(redis_url, database_url, consumer_name))
