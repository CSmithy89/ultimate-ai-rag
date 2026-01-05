"""Tests for Memory Consolidation Scheduler (Story 20-A2).

This module tests:
- Cron schedule parsing
- Scheduler lifecycle (start/stop)
- Manual trigger functionality
- APScheduler availability handling
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag_backend.memory.scheduler import (
    MemoryConsolidationScheduler,
    create_consolidation_scheduler,
    parse_cron_schedule,
    APSCHEDULER_AVAILABLE,
)
from agentic_rag_backend.memory.models import ConsolidationResult


# Test fixtures


@pytest.fixture
def mock_consolidator():
    """Create a mock memory consolidator."""
    consolidator = MagicMock()
    consolidator.consolidate = AsyncMock(
        return_value=ConsolidationResult(
            memories_processed=10,
            duplicates_merged=1,
            memories_decayed=5,
            memories_removed=0,
            processing_time_ms=100.0,
        )
    )
    consolidator.consolidate_all_tenants = AsyncMock(
        return_value=[
            ConsolidationResult(
                memories_processed=10,
                duplicates_merged=1,
                memories_decayed=5,
                memories_removed=0,
                processing_time_ms=100.0,
            )
        ]
    )
    consolidator.last_run_at = None
    consolidator.last_result = None
    return consolidator


# Test parse_cron_schedule function


class TestParseCronSchedule:
    """Tests for cron expression parsing."""

    def test_standard_five_field_cron(self):
        """Test parsing standard 5-field cron expression."""
        result = parse_cron_schedule("0 2 * * *")

        assert result == {
            "minute": "0",
            "hour": "2",
            "day": "*",
            "month": "*",
            "day_of_week": "*",
        }

    def test_complex_cron_expression(self):
        """Test parsing complex cron with specific values."""
        result = parse_cron_schedule("30 4 1,15 * 0,6")

        assert result == {
            "minute": "30",
            "hour": "4",
            "day": "1,15",
            "month": "*",
            "day_of_week": "0,6",
        }

    def test_hourly_cron(self):
        """Test hourly cron expression."""
        result = parse_cron_schedule("0 * * * *")

        assert result["minute"] == "0"
        assert result["hour"] == "*"

    def test_invalid_cron_too_few_fields(self):
        """Test error for too few fields."""
        with pytest.raises(ValueError, match="Expected 5 fields"):
            parse_cron_schedule("0 2 * *")

    def test_invalid_cron_too_many_fields(self):
        """Test error for too many fields."""
        with pytest.raises(ValueError, match="Expected 5 fields"):
            parse_cron_schedule("0 2 * * * *")

    def test_invalid_cron_empty_string(self):
        """Test error for empty string."""
        with pytest.raises(ValueError, match="Expected 5 fields"):
            parse_cron_schedule("")

    def test_whitespace_handling(self):
        """Test that extra whitespace is handled."""
        result = parse_cron_schedule("  0   2   *   *   *  ")

        assert result == {
            "minute": "0",
            "hour": "2",
            "day": "*",
            "month": "*",
            "day_of_week": "*",
        }


# Test MemoryConsolidationScheduler


class TestMemoryConsolidationScheduler:
    """Tests for MemoryConsolidationScheduler class."""

    def test_scheduler_initialization(self, mock_consolidator):
        """Test scheduler initializes with correct parameters."""
        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 3 * * *",
            enabled=True,
        )

        assert scheduler.consolidator == mock_consolidator
        assert scheduler.schedule == "0 3 * * *"
        assert scheduler.enabled is True
        assert scheduler.is_running is False

    def test_scheduler_disabled_by_default(self, mock_consolidator):
        """Test scheduler can be disabled."""
        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=False,
        )

        assert scheduler.enabled is False

    @pytest.mark.asyncio
    async def test_start_disabled_scheduler(self, mock_consolidator):
        """Test that disabled scheduler doesn't start."""
        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=False,
        )

        result = await scheduler.start()

        assert result is False
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_stop_not_running_scheduler(self, mock_consolidator):
        """Test stopping a scheduler that's not running doesn't error."""
        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=True,
        )

        # Should not raise
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_get_next_run_time_not_running(self, mock_consolidator):
        """Test next run time is None when not running."""
        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=True,
        )

        assert scheduler.get_next_run_time() is None

    @pytest.mark.asyncio
    async def test_trigger_now_single_tenant(self, mock_consolidator):
        """Test manual trigger for single tenant."""
        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=True,
        )

        await scheduler.trigger_now(tenant_id="test-tenant")

        mock_consolidator.consolidate.assert_called_once_with(tenant_id="test-tenant")

    @pytest.mark.asyncio
    async def test_trigger_now_all_tenants(self, mock_consolidator):
        """Test manual trigger for all tenants."""
        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=True,
        )

        await scheduler.trigger_now(tenant_id=None)

        mock_consolidator.consolidate_all_tenants.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_now_error_handling(self, mock_consolidator):
        """Test that errors in manual trigger are propagated."""
        mock_consolidator.consolidate.side_effect = Exception("Database error")

        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=True,
        )

        with pytest.raises(Exception, match="Database error"):
            await scheduler.trigger_now(tenant_id="test-tenant")

    @pytest.mark.asyncio
    async def test_run_consolidation_job(self, mock_consolidator):
        """Test the internal consolidation job execution."""
        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=True,
        )

        await scheduler._run_consolidation()

        mock_consolidator.consolidate_all_tenants.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_consolidation_job_error_handled(self, mock_consolidator):
        """Test that errors in job execution are logged but don't crash."""
        mock_consolidator.consolidate_all_tenants.side_effect = Exception("Test error")

        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=True,
        )

        # Should not raise
        await scheduler._run_consolidation()


class TestSchedulerWithAPScheduler:
    """Tests that require APScheduler to be available."""

    @pytest.mark.skipif(
        not APSCHEDULER_AVAILABLE,
        reason="APScheduler not installed",
    )
    @pytest.mark.asyncio
    async def test_start_and_stop_scheduler(self, mock_consolidator):
        """Test full scheduler lifecycle with APScheduler."""
        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=True,
        )

        # Start scheduler
        result = await scheduler.start()
        assert result is True
        assert scheduler.is_running is True

        # Verify next run time is available
        next_run = scheduler.get_next_run_time()
        assert next_run is not None
        assert isinstance(next_run, datetime)

        # Stop scheduler
        await scheduler.stop()
        assert scheduler.is_running is False

    @pytest.mark.skipif(
        not APSCHEDULER_AVAILABLE,
        reason="APScheduler not installed",
    )
    @pytest.mark.asyncio
    async def test_start_already_running(self, mock_consolidator):
        """Test that starting an already running scheduler returns True."""
        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=True,
        )

        await scheduler.start()
        result = await scheduler.start()  # Second start

        assert result is True
        assert scheduler.is_running is True

        # Cleanup
        await scheduler.stop()


class TestSchedulerWithoutAPScheduler:
    """Tests for behavior when APScheduler is not available."""

    @pytest.mark.asyncio
    async def test_start_without_apscheduler(self, mock_consolidator):
        """Test that scheduler gracefully handles missing APScheduler."""
        scheduler = MemoryConsolidationScheduler(
            consolidator=mock_consolidator,
            schedule="0 2 * * *",
            enabled=True,
        )

        # Temporarily disable APScheduler availability
        with patch(
            "agentic_rag_backend.memory.scheduler.APSCHEDULER_AVAILABLE",
            False,
        ):
            result = await scheduler.start()

        assert result is False


# Test create_consolidation_scheduler factory


class TestCreateConsolidationScheduler:
    """Tests for the factory function."""

    def test_factory_creates_scheduler(self, mock_consolidator):
        """Test factory creates scheduler with correct parameters."""
        scheduler = create_consolidation_scheduler(
            consolidator=mock_consolidator,
            schedule="0 4 * * *",
            enabled=True,
        )

        assert isinstance(scheduler, MemoryConsolidationScheduler)
        assert scheduler.schedule == "0 4 * * *"
        assert scheduler.enabled is True

    def test_factory_default_parameters(self, mock_consolidator):
        """Test factory uses correct defaults."""
        scheduler = create_consolidation_scheduler(
            consolidator=mock_consolidator,
        )

        assert scheduler.schedule == "0 2 * * *"  # Default 2 AM daily
        assert scheduler.enabled is True  # Default enabled
