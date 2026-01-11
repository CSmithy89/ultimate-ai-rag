"""Unit tests for A2A Resource Limits.

Story 22-A2: Implement A2A Session Resource Limits

Tests cover:
- Configuration models (A2AResourceLimits)
- InMemoryA2AResourceManager operations
- Rate limiting with sliding window
- Session and message limits
- TTL-based cleanup
- Factory creation
- Exception behavior
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from time import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag_backend.protocols.a2a_resource_limits import (
    A2AResourceLimits,
    A2AResourceMetrics,
    A2AResourceManager,
    A2AResourceManagerFactory,
    A2ASessionLimitExceeded,
    A2AMessageLimitExceeded,
    A2ARateLimitExceeded,
    InMemoryA2AResourceManager,
    RedisA2AResourceManager,
    TenantUsage,
    SessionUsage,
)


class TestA2AResourceLimitsConfig:
    """Tests for A2AResourceLimits configuration model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        limits = A2AResourceLimits()

        assert limits.session_limit_per_tenant == 100
        assert limits.message_limit_per_session == 1000
        assert limits.session_ttl_hours == 24
        assert limits.message_rate_limit == 60
        assert limits.cleanup_interval_minutes == 15

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        limits = A2AResourceLimits(
            session_limit_per_tenant=50,
            message_limit_per_session=500,
            session_ttl_hours=12,
            message_rate_limit=30,
            cleanup_interval_minutes=5,
        )

        assert limits.session_limit_per_tenant == 50
        assert limits.message_limit_per_session == 500
        assert limits.session_ttl_hours == 12
        assert limits.message_rate_limit == 30
        assert limits.cleanup_interval_minutes == 5

    def test_validation_minimum_values(self) -> None:
        """Test that minimum value validation works."""
        # Should not raise for minimum valid values
        limits = A2AResourceLimits(
            session_limit_per_tenant=1,
            message_limit_per_session=1,
            session_ttl_hours=1,
            message_rate_limit=1,
            cleanup_interval_minutes=1,
        )
        assert limits.session_limit_per_tenant == 1


class TestTenantUsageModel:
    """Tests for TenantUsage model."""

    def test_default_values(self) -> None:
        """Test default values for tenant usage."""
        usage = TenantUsage(tenant_id="tenant-123")

        assert usage.tenant_id == "tenant-123"
        assert usage.active_sessions == 0
        assert usage.total_messages == 0
        assert usage.last_activity is not None


class TestSessionUsageModel:
    """Tests for SessionUsage model."""

    def test_default_values(self) -> None:
        """Test default values for session usage."""
        usage = SessionUsage(
            session_id="session-abc",
            tenant_id="tenant-123",
        )

        assert usage.session_id == "session-abc"
        assert usage.tenant_id == "tenant-123"
        assert usage.message_count == 0
        assert usage.message_timestamps == []


class TestA2AResourceMetrics:
    """Tests for A2AResourceMetrics model."""

    def test_creation(self) -> None:
        """Test metrics model creation."""
        metrics = A2AResourceMetrics(
            tenant_id="tenant-123",
            active_sessions=5,
            total_messages=100,
            session_limit=100,
            message_limit_per_session=1000,
            message_rate_limit=60,
        )

        assert metrics.tenant_id == "tenant-123"
        assert metrics.active_sessions == 5
        assert metrics.total_messages == 100


class TestExceptions:
    """Tests for custom exceptions."""

    def test_session_limit_exceeded(self) -> None:
        """Test A2ASessionLimitExceeded exception."""
        exc = A2ASessionLimitExceeded("tenant-123", 100)

        assert exc.tenant_id == "tenant-123"
        assert exc.limit == 100
        assert "tenant-123" in str(exc)
        assert "100" in str(exc)

    def test_message_limit_exceeded(self) -> None:
        """Test A2AMessageLimitExceeded exception."""
        exc = A2AMessageLimitExceeded("session-abc", 1000)

        assert exc.session_id == "session-abc"
        assert exc.limit == 1000
        assert "session-abc" in str(exc)
        assert "1000" in str(exc)

    def test_rate_limit_exceeded(self) -> None:
        """Test A2ARateLimitExceeded exception."""
        exc = A2ARateLimitExceeded("session-abc", 60, retry_after=30)

        assert exc.session_id == "session-abc"
        assert exc.limit == 60
        assert exc.retry_after == 30
        assert "session-abc" in str(exc)
        assert "60" in str(exc)


class TestInMemoryA2AResourceManager:
    """Tests for InMemoryA2AResourceManager."""

    @pytest.fixture
    def limits(self) -> A2AResourceLimits:
        """Create test limits."""
        return A2AResourceLimits(
            session_limit_per_tenant=3,
            message_limit_per_session=5,
            session_ttl_hours=1,
            message_rate_limit=100,  # High rate limit to test message limit first
            cleanup_interval_minutes=1,
        )

    @pytest.fixture
    def rate_limited_limits(self) -> A2AResourceLimits:
        """Create limits with low rate limit for rate limiting tests."""
        return A2AResourceLimits(
            session_limit_per_tenant=3,
            message_limit_per_session=100,  # High message limit to test rate limit first
            session_ttl_hours=1,
            message_rate_limit=3,  # 3 per minute for testing
            cleanup_interval_minutes=1,
        )

    @pytest.fixture
    def manager(self, limits: A2AResourceLimits) -> InMemoryA2AResourceManager:
        """Create a manager for testing."""
        return InMemoryA2AResourceManager(limits=limits)

    @pytest.mark.asyncio
    async def test_register_session_success(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test successful session registration."""
        await manager.register_session("session-1", "tenant-123")

        assert "session-1" in manager._session_usage
        assert "tenant-123" in manager._tenant_usage
        assert manager._tenant_usage["tenant-123"].active_sessions == 1

    @pytest.mark.asyncio
    async def test_register_session_multiple(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test registering multiple sessions for same tenant."""
        await manager.register_session("session-1", "tenant-123")
        await manager.register_session("session-2", "tenant-123")

        assert manager._tenant_usage["tenant-123"].active_sessions == 2

    @pytest.mark.asyncio
    async def test_register_session_limit_exceeded(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test that session limit is enforced."""
        await manager.register_session("session-1", "tenant-123")
        await manager.register_session("session-2", "tenant-123")
        await manager.register_session("session-3", "tenant-123")

        with pytest.raises(A2ASessionLimitExceeded) as exc_info:
            await manager.register_session("session-4", "tenant-123")

        assert exc_info.value.tenant_id == "tenant-123"
        assert exc_info.value.limit == 3

    @pytest.mark.asyncio
    async def test_close_session(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test closing a session."""
        await manager.register_session("session-1", "tenant-123")
        await manager.close_session("session-1")

        assert "session-1" not in manager._session_usage
        assert manager._tenant_usage["tenant-123"].active_sessions == 0

    @pytest.mark.asyncio
    async def test_close_session_not_found(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test closing a non-existent session (should not raise)."""
        await manager.close_session("nonexistent")
        # Should complete without error

    @pytest.mark.asyncio
    async def test_record_message_success(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test recording a message."""
        await manager.register_session("session-1", "tenant-123")
        await manager.record_message("session-1")

        assert manager._session_usage["session-1"].message_count == 1
        assert manager._tenant_usage["tenant-123"].total_messages == 1

    @pytest.mark.asyncio
    async def test_record_message_limit_exceeded(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test that message limit is enforced."""
        await manager.register_session("session-1", "tenant-123")

        # Record up to limit
        for _ in range(5):
            await manager.record_message("session-1")

        # Next message should fail
        with pytest.raises(A2AMessageLimitExceeded) as exc_info:
            await manager.record_message("session-1")

        assert exc_info.value.session_id == "session-1"
        assert exc_info.value.limit == 5

    @pytest.mark.asyncio
    async def test_record_message_rate_limit_exceeded(
        self,
        rate_limited_limits: A2AResourceLimits,
    ) -> None:
        """Test that rate limit is enforced."""
        rate_limited_manager = InMemoryA2AResourceManager(limits=rate_limited_limits)
        await rate_limited_manager.register_session("session-1", "tenant-123")

        # Record up to rate limit (3 per minute)
        for _ in range(3):
            await rate_limited_manager.record_message("session-1")

        # Next message should fail due to rate limit
        with pytest.raises(A2ARateLimitExceeded) as exc_info:
            await rate_limited_manager.record_message("session-1")

        assert exc_info.value.session_id == "session-1"
        assert exc_info.value.limit == 3

    @pytest.mark.asyncio
    async def test_record_message_unknown_session(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test recording message for unknown session."""
        # Should not raise, just log warning
        await manager.record_message("nonexistent")

    @pytest.mark.asyncio
    async def test_check_session_limit_true(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test check_session_limit returns True when under limit."""
        result = await manager.check_session_limit("tenant-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_session_limit_false(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test check_session_limit returns False when at limit."""
        await manager.register_session("session-1", "tenant-123")
        await manager.register_session("session-2", "tenant-123")
        await manager.register_session("session-3", "tenant-123")

        result = await manager.check_session_limit("tenant-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_message_limit_true(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test check_message_limit returns True when under limit."""
        await manager.register_session("session-1", "tenant-123")
        result = await manager.check_message_limit("session-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_message_limit_false(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test check_message_limit returns False when at limit."""
        await manager.register_session("session-1", "tenant-123")
        for _ in range(5):
            await manager.record_message("session-1")

        result = await manager.check_message_limit("session-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_rate_limit_true(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test check_rate_limit returns True when under limit."""
        await manager.register_session("session-1", "tenant-123")
        result = await manager.check_rate_limit("session-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_tenant_metrics_empty(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test getting metrics for tenant with no activity."""
        metrics = await manager.get_tenant_metrics("tenant-123")

        assert metrics.tenant_id == "tenant-123"
        assert metrics.active_sessions == 0
        assert metrics.total_messages == 0
        assert metrics.session_limit == 3
        assert metrics.message_limit_per_session == 5
        assert metrics.message_rate_limit == 100  # High rate limit from fixture

    @pytest.mark.asyncio
    async def test_get_tenant_metrics_with_activity(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test getting metrics for tenant with activity."""
        await manager.register_session("session-1", "tenant-123")
        await manager.record_message("session-1")
        await manager.record_message("session-1")

        metrics = await manager.get_tenant_metrics("tenant-123")

        assert metrics.active_sessions == 1
        assert metrics.total_messages == 2

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test starting and stopping the manager."""
        await manager.start()
        assert manager._running is True
        assert manager._cleanup_task is not None

        await manager.stop()
        assert manager._running is False
        assert manager._cleanup_task is None

    @pytest.mark.asyncio
    async def test_tenant_isolation(
        self,
        manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test that tenants are isolated from each other."""
        # Register sessions for two different tenants
        await manager.register_session("session-1", "tenant-A")
        await manager.register_session("session-2", "tenant-B")

        # Each tenant should have their own count
        assert manager._tenant_usage["tenant-A"].active_sessions == 1
        assert manager._tenant_usage["tenant-B"].active_sessions == 1

        # Closing one tenant's session shouldn't affect the other
        await manager.close_session("session-1")
        assert manager._tenant_usage["tenant-A"].active_sessions == 0
        assert manager._tenant_usage["tenant-B"].active_sessions == 1


class TestA2AResourceManagerFactory:
    """Tests for A2AResourceManagerFactory."""

    def test_create_memory_backend(self) -> None:
        """Test creating in-memory manager."""
        limits = A2AResourceLimits()
        manager = A2AResourceManagerFactory.create(
            backend="memory",
            limits=limits,
        )

        assert isinstance(manager, InMemoryA2AResourceManager)

    def test_create_redis_backend(self) -> None:
        """Test creating Redis manager."""
        limits = A2AResourceLimits()
        mock_redis = MagicMock()

        manager = A2AResourceManagerFactory.create(
            backend="redis",
            limits=limits,
            redis_client=mock_redis,
        )

        assert isinstance(manager, RedisA2AResourceManager)

    def test_create_redis_backend_missing_client(self) -> None:
        """Test that Redis backend requires client."""
        limits = A2AResourceLimits()

        with pytest.raises(ValueError) as exc_info:
            A2AResourceManagerFactory.create(
                backend="redis",
                limits=limits,
            )

        assert "Redis client required" in str(exc_info.value)

    def test_create_postgres_fallback(self) -> None:
        """Test that postgres backend falls back to memory."""
        limits = A2AResourceLimits()
        manager = A2AResourceManagerFactory.create(
            backend="postgres",
            limits=limits,
        )

        # Postgres not implemented yet, should fall back to memory
        assert isinstance(manager, InMemoryA2AResourceManager)

    def test_create_unknown_backend_fallback(self) -> None:
        """Test that unknown backend falls back to memory."""
        limits = A2AResourceLimits()
        manager = A2AResourceManagerFactory.create(
            backend="unknown",
            limits=limits,
        )

        assert isinstance(manager, InMemoryA2AResourceManager)


class TestRedisA2AResourceManager:
    """Tests for RedisA2AResourceManager."""

    @pytest.fixture
    def limits(self) -> A2AResourceLimits:
        """Create test limits."""
        return A2AResourceLimits(
            session_limit_per_tenant=3,
            message_limit_per_session=5,
            session_ttl_hours=1,
            message_rate_limit=3,
        )

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        """Create a mock Redis client."""
        mock = MagicMock()
        mock.hget = AsyncMock(return_value=None)
        mock.hset = AsyncMock()
        mock.hincrby = AsyncMock()
        mock.hgetall = AsyncMock(return_value={})
        mock.delete = AsyncMock()
        mock.expire = AsyncMock()
        mock.pipeline = MagicMock()
        mock.zcount = AsyncMock(return_value=0)
        mock.zadd = AsyncMock()
        mock.zremrangebyscore = AsyncMock()

        # Mock pipeline
        pipeline_mock = MagicMock()
        pipeline_mock.hincrby = MagicMock(return_value=pipeline_mock)
        pipeline_mock.hset = MagicMock(return_value=pipeline_mock)
        pipeline_mock.zadd = MagicMock(return_value=pipeline_mock)
        pipeline_mock.zremrangebyscore = MagicMock(return_value=pipeline_mock)
        pipeline_mock.expire = MagicMock(return_value=pipeline_mock)
        pipeline_mock.delete = MagicMock(return_value=pipeline_mock)
        pipeline_mock.execute = AsyncMock(return_value=[])

        mock.pipeline.return_value = pipeline_mock

        return mock

    @pytest.fixture
    def manager(
        self,
        limits: A2AResourceLimits,
        mock_redis: MagicMock,
    ) -> RedisA2AResourceManager:
        """Create a manager for testing."""
        return RedisA2AResourceManager(
            limits=limits,
            redis_client=mock_redis,
        )

    def test_key_generation(
        self,
        manager: RedisA2AResourceManager,
    ) -> None:
        """Test Redis key generation."""
        assert manager._tenant_key("tenant-123") == "a2a:tenant:tenant-123"
        assert manager._session_key("session-abc") == "a2a:session:session-abc:info"
        assert manager._rate_key("session-abc") == "a2a:session:session-abc:rate"

    @pytest.mark.asyncio
    async def test_check_session_limit_under(
        self,
        manager: RedisA2AResourceManager,
        mock_redis: MagicMock,
    ) -> None:
        """Test check_session_limit when under limit."""
        mock_redis.hget.return_value = "1"
        result = await manager.check_session_limit("tenant-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_session_limit_at(
        self,
        manager: RedisA2AResourceManager,
        mock_redis: MagicMock,
    ) -> None:
        """Test check_session_limit when at limit."""
        mock_redis.hget.return_value = "3"
        result = await manager.check_session_limit("tenant-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_message_limit(
        self,
        manager: RedisA2AResourceManager,
        mock_redis: MagicMock,
    ) -> None:
        """Test check_message_limit."""
        mock_redis.hget.return_value = "3"
        result = await manager.check_message_limit("session-abc")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_rate_limit(
        self,
        manager: RedisA2AResourceManager,
        mock_redis: MagicMock,
    ) -> None:
        """Test check_rate_limit."""
        mock_redis.zcount.return_value = 1
        result = await manager.check_rate_limit("session-abc")
        assert result is True

    @pytest.mark.asyncio
    async def test_register_session_success(
        self,
        manager: RedisA2AResourceManager,
        mock_redis: MagicMock,
    ) -> None:
        """Test successful session registration."""
        # Under limit
        mock_redis.hget.return_value = None

        await manager.register_session("session-abc", "tenant-123")

        # Verify pipeline was executed
        mock_redis.pipeline.return_value.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_session_limit_exceeded(
        self,
        manager: RedisA2AResourceManager,
        mock_redis: MagicMock,
    ) -> None:
        """Test session limit exceeded."""
        # At limit
        mock_redis.hget.return_value = "3"

        with pytest.raises(A2ASessionLimitExceeded):
            await manager.register_session("session-abc", "tenant-123")

    @pytest.mark.asyncio
    async def test_get_tenant_metrics_empty(
        self,
        manager: RedisA2AResourceManager,
        mock_redis: MagicMock,
    ) -> None:
        """Test getting metrics for tenant with no activity."""
        mock_redis.hgetall.return_value = {}

        metrics = await manager.get_tenant_metrics("tenant-123")

        assert metrics.tenant_id == "tenant-123"
        assert metrics.active_sessions == 0
        assert metrics.total_messages == 0

    @pytest.mark.asyncio
    async def test_get_tenant_metrics_with_data(
        self,
        manager: RedisA2AResourceManager,
        mock_redis: MagicMock,
    ) -> None:
        """Test getting metrics with data."""
        mock_redis.hgetall.return_value = {
            b"active_sessions": b"5",
            b"total_messages": b"100",
        }

        metrics = await manager.get_tenant_metrics("tenant-123")

        assert metrics.active_sessions == 5
        assert metrics.total_messages == 100

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(
        self,
        manager: RedisA2AResourceManager,
    ) -> None:
        """Test starting and stopping the manager."""
        await manager.start()
        assert manager._running is True
        assert manager._cleanup_task is not None

        await manager.stop()
        assert manager._running is False
        assert manager._cleanup_task is None
