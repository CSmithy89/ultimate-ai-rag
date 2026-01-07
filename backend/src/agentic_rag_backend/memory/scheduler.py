"""Memory Consolidation Scheduler for Epic 20 (Story 20-A2).

This module provides APScheduler-based scheduling for automatic memory
consolidation. The scheduler runs consolidation at configurable intervals
(default: daily at 2 AM) to maintain memory quality.

Features:
- Cron-based scheduling using APScheduler
- Configurable schedule via MEMORY_CONSOLIDATION_SCHEDULE
- Graceful startup/shutdown integration with FastAPI lifespan
- Background execution to avoid blocking the main event loop
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

import structlog

if TYPE_CHECKING:
    from .consolidation import MemoryConsolidator

logger = structlog.get_logger(__name__)


# Check APScheduler availability
APSCHEDULER_AVAILABLE = False
AsyncIOScheduler = None
CronTrigger = None

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler as _AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger as _CronTrigger

    AsyncIOScheduler = _AsyncIOScheduler
    CronTrigger = _CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    logger.warning(
        "apscheduler_not_installed",
        hint="Install apscheduler for scheduled memory consolidation: uv add apscheduler",
    )


def parse_cron_schedule(cron_expr: str) -> dict:
    """Parse a cron expression into APScheduler CronTrigger kwargs.

    Supports standard 5-field cron format: minute hour day month day_of_week
    Example: "0 2 * * *" = daily at 2:00 AM

    Args:
        cron_expr: Cron expression string (5 fields)

    Returns:
        Dictionary of kwargs for CronTrigger

    Raises:
        ValueError: If cron expression is invalid
    """
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        raise ValueError(
            f"Invalid cron expression '{cron_expr}'. "
            "Expected 5 fields: minute hour day month day_of_week"
        )

    return {
        "minute": parts[0],
        "hour": parts[1],
        "day": parts[2],
        "month": parts[3],
        "day_of_week": parts[4],
    }


class MemoryConsolidationScheduler:
    """Scheduler for periodic memory consolidation.

    Uses APScheduler's AsyncIOScheduler to run consolidation jobs
    at configurable intervals. The scheduler integrates with FastAPI's
    lifespan for proper startup/shutdown handling.

    Attributes:
        consolidator: MemoryConsolidator instance
        schedule: Cron expression for consolidation schedule
        enabled: Whether the scheduler is enabled
    """

    def __init__(
        self,
        consolidator: "MemoryConsolidator",
        schedule: str = "0 2 * * *",  # Default: 2 AM daily
        enabled: bool = True,
    ) -> None:
        """Initialize the scheduler.

        Args:
            consolidator: MemoryConsolidator instance
            schedule: Cron expression for consolidation schedule
            enabled: Whether to enable scheduled consolidation
        """
        self.consolidator = consolidator
        self.schedule = schedule
        self.enabled = enabled
        self._scheduler: Optional[object] = None
        self._job_id = "memory_consolidation"
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if the scheduler is currently running."""
        return self._running

    def get_next_run_time(self) -> Optional[datetime]:
        """Get the next scheduled run time.

        Returns:
            Next run time as datetime, or None if scheduler not running
        """
        if not self._scheduler or not self._running:
            return None

        try:
            job = self._scheduler.get_job(self._job_id)  # type: ignore
            if job and job.next_run_time:
                return job.next_run_time
        except Exception:
            pass

        return None

    async def start(self) -> bool:
        """Start the scheduler.

        Returns:
            True if started successfully, False otherwise
        """
        if not APSCHEDULER_AVAILABLE:
            logger.warning(
                "scheduler_start_skipped",
                reason="APScheduler not available",
            )
            return False

        if not self.enabled:
            logger.info("scheduler_disabled")
            return False

        if self._running:
            logger.warning("scheduler_already_running")
            return True

        try:
            # Parse cron schedule
            cron_kwargs = parse_cron_schedule(self.schedule)

            # Create scheduler
            self._scheduler = AsyncIOScheduler()

            # Add consolidation job
            self._scheduler.add_job(  # type: ignore
                self._run_consolidation,
                trigger=CronTrigger(**cron_kwargs),  # type: ignore
                id=self._job_id,
                name="Memory Consolidation",
                replace_existing=True,
            )

            # Start scheduler
            self._scheduler.start()  # type: ignore
            self._running = True

            next_run = self.get_next_run_time()
            logger.info(
                "scheduler_started",
                schedule=self.schedule,
                next_run=next_run.isoformat() if next_run else None,
            )

            return True

        except Exception as e:
            logger.error("scheduler_start_failed", error=str(e))
            return False

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if not self._scheduler or not self._running:
            return

        try:
            # Use non-blocking shutdown to avoid stalling the event loop
            # APScheduler's AsyncIOScheduler.shutdown() is a synchronous call
            self._scheduler.shutdown(wait=False)  # type: ignore
            self._running = False
            logger.info("scheduler_stopped")
        except Exception as e:
            logger.error("scheduler_stop_failed", error=str(e))

    async def _run_consolidation(self) -> None:
        """Execute the consolidation job.

        This is called by the scheduler at each scheduled time.
        It processes all tenants with full isolation.
        """
        logger.info("scheduled_consolidation_started")

        try:
            results = await self.consolidator.consolidate_all_tenants()

            total_processed = sum(r.memories_processed for r in results)
            total_merged = sum(r.duplicates_merged for r in results)
            total_removed = sum(r.memories_removed for r in results)

            logger.info(
                "scheduled_consolidation_complete",
                tenants_processed=len(results),
                total_memories_processed=total_processed,
                total_duplicates_merged=total_merged,
                total_memories_removed=total_removed,
            )

        except Exception as e:
            logger.error("scheduled_consolidation_failed", error=str(e))

    async def trigger_now(self, tenant_id: Optional[str] = None) -> None:
        """Trigger consolidation immediately (outside of schedule).

        Args:
            tenant_id: Optional specific tenant to consolidate (None for all)
        """
        logger.info(
            "manual_consolidation_triggered",
            tenant_id=tenant_id or "all",
        )

        try:
            if tenant_id:
                await self.consolidator.consolidate(tenant_id=tenant_id)
            else:
                await self.consolidator.consolidate_all_tenants()
        except Exception as e:
            logger.error(
                "manual_consolidation_failed",
                tenant_id=tenant_id or "all",
                error=str(e),
            )
            raise


def create_consolidation_scheduler(
    consolidator: "MemoryConsolidator",
    schedule: str = "0 2 * * *",
    enabled: bool = True,
) -> MemoryConsolidationScheduler:
    """Factory function to create a consolidation scheduler.

    Args:
        consolidator: MemoryConsolidator instance
        schedule: Cron expression for schedule (default: 2 AM daily)
        enabled: Whether scheduling is enabled

    Returns:
        Configured MemoryConsolidationScheduler instance
    """
    return MemoryConsolidationScheduler(
        consolidator=consolidator,
        schedule=schedule,
        enabled=enabled,
    )
