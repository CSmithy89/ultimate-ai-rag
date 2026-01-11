"""A2A Resource Limits for per-tenant session and message caps.

Story 22-A2: Implement A2A Session Resource Limits

This module provides:
- A2AResourceLimits: Configuration model for limits
- A2AResourceManager: Abstract base class
- InMemoryA2AResourceManager: In-memory implementation for dev/testing
- RedisA2AResourceManager: Redis-backed implementation for production
- A2AResourceManagerFactory: Factory for creating managers

Limit types enforced:
- Per-tenant session limits (default: 100 concurrent sessions)
- Per-session message limits (default: 1000 messages)
- Rate limiting (default: 60 messages/minute per session)
- Session TTL with automatic cleanup (default: 24 hours)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from threading import Lock
from time import time
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from pydantic import BaseModel, Field
import structlog

if TYPE_CHECKING:
    import redis.asyncio as redis

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
# Configuration Models
# -----------------------------------------------------------------------------


class A2AResourceLimits(BaseModel):
    """Configuration for A2A resource limits.

    Attributes:
        session_limit_per_tenant: Max concurrent sessions per tenant
        message_limit_per_session: Max messages per session
        session_ttl_hours: Session expiration time in hours
        message_rate_limit: Max messages per minute per session
        cleanup_interval_minutes: Interval for cleanup background task
    """

    session_limit_per_tenant: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent sessions allowed per tenant",
    )
    message_limit_per_session: int = Field(
        default=1000,
        ge=1,
        description="Maximum messages allowed within a single session",
    )
    session_ttl_hours: int = Field(
        default=24,
        ge=1,
        description="Session expiration time in hours",
    )
    message_rate_limit: int = Field(
        default=60,
        ge=1,
        description="Maximum messages per minute per session",
    )
    cleanup_interval_minutes: int = Field(
        default=15,
        ge=1,
        description="Interval for cleanup background task in minutes",
    )


class TenantUsage(BaseModel):
    """Tracks resource usage for a tenant.

    Attributes:
        tenant_id: Unique identifier for the tenant
        active_sessions: Current number of active sessions
        total_messages: Total messages across all sessions
        last_activity: Timestamp of last activity
    """

    tenant_id: str
    active_sessions: int = 0
    total_messages: int = 0
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionUsage(BaseModel):
    """Tracks resource usage for a session.

    Attributes:
        session_id: Unique identifier for the session
        tenant_id: Tenant that owns this session
        message_count: Total messages in this session
        created_at: When the session was created
        last_message_at: Timestamp of last message
        message_timestamps: List of recent message timestamps for rate limiting
    """

    session_id: str
    tenant_id: str
    message_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message_timestamps: list[float] = Field(default_factory=list)


class A2AResourceMetrics(BaseModel):
    """Resource usage metrics returned by get_tenant_metrics().

    Attributes:
        tenant_id: The tenant these metrics are for
        active_sessions: Current number of active sessions
        total_messages: Total messages sent across all sessions
        session_limit: Configured session limit
        message_limit_per_session: Configured message limit per session
        message_rate_limit: Configured rate limit (messages/minute)
    """

    tenant_id: str
    active_sessions: int
    total_messages: int
    session_limit: int
    message_limit_per_session: int
    message_rate_limit: int


# -----------------------------------------------------------------------------
# Exceptions (also added to core/errors.py for HTTP 429 mapping)
# -----------------------------------------------------------------------------


class A2ASessionLimitExceeded(Exception):
    """Raised when tenant has reached their session limit."""

    def __init__(self, tenant_id: str, limit: int) -> None:
        self.tenant_id = tenant_id
        self.limit = limit
        super().__init__(
            f"Tenant '{tenant_id}' has reached maximum concurrent sessions ({limit})"
        )


class A2AMessageLimitExceeded(Exception):
    """Raised when session has reached its message limit."""

    def __init__(self, session_id: str, limit: int) -> None:
        self.session_id = session_id
        self.limit = limit
        super().__init__(
            f"Session '{session_id}' has reached maximum messages ({limit})"
        )


class A2ARateLimitExceeded(Exception):
    """Raised when session has exceeded the rate limit."""

    def __init__(self, session_id: str, limit: int, retry_after: int = 60) -> None:
        self.session_id = session_id
        self.limit = limit
        self.retry_after = retry_after
        super().__init__(
            f"Session '{session_id}' has exceeded message rate limit ({limit}/minute)"
        )


# -----------------------------------------------------------------------------
# Abstract Base Class
# -----------------------------------------------------------------------------


class A2AResourceManager(ABC):
    """Abstract base class for A2A resource management.

    Implementations must handle:
    - Session registration and limits
    - Message counting and limits
    - Rate limiting per session
    - TTL-based session cleanup
    - Metrics reporting
    """

    def __init__(self, limits: A2AResourceLimits) -> None:
        """Initialize the resource manager.

        Args:
            limits: Configuration for resource limits
        """
        self.limits = limits

    @abstractmethod
    async def check_session_limit(self, tenant_id: str) -> bool:
        """Check if tenant can create a new session.

        Args:
            tenant_id: The tenant to check

        Returns:
            True if tenant can create a new session, False otherwise
        """
        ...

    @abstractmethod
    async def check_message_limit(self, session_id: str) -> bool:
        """Check if session can send another message.

        Args:
            session_id: The session to check

        Returns:
            True if session can send another message, False otherwise
        """
        ...

    @abstractmethod
    async def check_rate_limit(self, session_id: str) -> bool:
        """Check if session is within rate limit.

        Args:
            session_id: The session to check

        Returns:
            True if session is within rate limit, False otherwise
        """
        ...

    @abstractmethod
    async def register_session(self, session_id: str, tenant_id: str) -> None:
        """Register a new session.

        Args:
            session_id: Unique identifier for the session
            tenant_id: Tenant that owns this session

        Raises:
            A2ASessionLimitExceeded: If tenant has reached session limit
        """
        ...

    @abstractmethod
    async def record_message(self, session_id: str) -> None:
        """Record a message for a session.

        Args:
            session_id: The session sending the message

        Raises:
            A2AMessageLimitExceeded: If session has reached message limit
            A2ARateLimitExceeded: If session has exceeded rate limit
        """
        ...

    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        """Close a session and update limits.

        Args:
            session_id: The session to close
        """
        ...

    @abstractmethod
    async def get_tenant_metrics(self, tenant_id: str) -> A2AResourceMetrics:
        """Get resource usage metrics for a tenant.

        Args:
            tenant_id: The tenant to get metrics for

        Returns:
            A2AResourceMetrics with current usage and limits
        """
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the cleanup background task."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the cleanup background task."""
        ...


# -----------------------------------------------------------------------------
# In-Memory Implementation
# -----------------------------------------------------------------------------


class InMemoryA2AResourceManager(A2AResourceManager):
    """In-memory implementation for development and testing.

    Note: In multi-worker deployments, each worker maintains its own state.
    Use RedisA2AResourceManager for production multi-worker deployments.
    """

    def __init__(self, limits: A2AResourceLimits) -> None:
        """Initialize the in-memory resource manager.

        Args:
            limits: Configuration for resource limits
        """
        super().__init__(limits)
        self._lock = Lock()
        self._tenant_usage: dict[str, TenantUsage] = {}
        self._session_usage: dict[str, SessionUsage] = {}
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._running = False

    async def check_session_limit(self, tenant_id: str) -> bool:
        """Check if tenant can create a new session."""
        with self._lock:
            usage = self._tenant_usage.get(tenant_id)
            if usage is None:
                return True
            return usage.active_sessions < self.limits.session_limit_per_tenant

    async def check_message_limit(self, session_id: str) -> bool:
        """Check if session can send another message."""
        with self._lock:
            usage = self._session_usage.get(session_id)
            if usage is None:
                return True
            return usage.message_count < self.limits.message_limit_per_session

    async def check_rate_limit(self, session_id: str) -> bool:
        """Check if session is within rate limit using sliding window."""
        now = time()
        minute_ago = now - 60

        with self._lock:
            usage = self._session_usage.get(session_id)
            if usage is None:
                return True

            # Count messages in last minute
            recent_count = sum(1 for ts in usage.message_timestamps if ts > minute_ago)
            return recent_count < self.limits.message_rate_limit

    async def register_session(self, session_id: str, tenant_id: str) -> None:
        """Register a new session."""
        if not await self.check_session_limit(tenant_id):
            logger.warning(
                "a2a_session_limit_exceeded",
                tenant_id=tenant_id,
                limit=self.limits.session_limit_per_tenant,
            )
            raise A2ASessionLimitExceeded(tenant_id, self.limits.session_limit_per_tenant)

        with self._lock:
            # Create or update tenant usage
            if tenant_id not in self._tenant_usage:
                self._tenant_usage[tenant_id] = TenantUsage(tenant_id=tenant_id)

            self._tenant_usage[tenant_id].active_sessions += 1
            self._tenant_usage[tenant_id].last_activity = datetime.now(timezone.utc)

            # Create session usage
            self._session_usage[session_id] = SessionUsage(
                session_id=session_id,
                tenant_id=tenant_id,
            )

        logger.info(
            "a2a_session_registered",
            session_id=session_id,
            tenant_id=tenant_id,
            active_sessions=self._tenant_usage[tenant_id].active_sessions,
        )

    async def record_message(self, session_id: str) -> None:
        """Record a message for a session."""
        # Check message limit first
        if not await self.check_message_limit(session_id):
            logger.warning(
                "a2a_message_limit_exceeded",
                session_id=session_id,
                limit=self.limits.message_limit_per_session,
            )
            raise A2AMessageLimitExceeded(session_id, self.limits.message_limit_per_session)

        # Check rate limit
        if not await self.check_rate_limit(session_id):
            logger.warning(
                "a2a_rate_limit_exceeded",
                session_id=session_id,
                limit=self.limits.message_rate_limit,
            )
            raise A2ARateLimitExceeded(session_id, self.limits.message_rate_limit)

        now = time()
        now_dt = datetime.now(timezone.utc)
        five_minutes_ago = now - 300

        with self._lock:
            usage = self._session_usage.get(session_id)
            if usage is None:
                logger.warning(
                    "a2a_record_message_unknown_session",
                    session_id=session_id,
                )
                return

            # Increment message count
            usage.message_count += 1
            usage.last_message_at = now_dt

            # Add timestamp for rate limiting
            usage.message_timestamps.append(now)

            # Clean old timestamps (keep only last 5 minutes)
            usage.message_timestamps = [
                ts for ts in usage.message_timestamps if ts > five_minutes_ago
            ]

            # Update tenant total messages
            tenant_id = usage.tenant_id
            if tenant_id in self._tenant_usage:
                self._tenant_usage[tenant_id].total_messages += 1
                self._tenant_usage[tenant_id].last_activity = now_dt

        logger.debug(
            "a2a_message_recorded",
            session_id=session_id,
            message_count=usage.message_count,
        )

    async def close_session(self, session_id: str) -> None:
        """Close a session and update limits."""
        with self._lock:
            usage = self._session_usage.pop(session_id, None)
            if usage is None:
                logger.debug(
                    "a2a_close_session_not_found",
                    session_id=session_id,
                )
                return

            tenant_id = usage.tenant_id
            if tenant_id in self._tenant_usage:
                self._tenant_usage[tenant_id].active_sessions = max(
                    0, self._tenant_usage[tenant_id].active_sessions - 1
                )
                self._tenant_usage[tenant_id].last_activity = datetime.now(timezone.utc)

        logger.info(
            "a2a_session_closed",
            session_id=session_id,
            tenant_id=usage.tenant_id,
        )

    async def get_tenant_metrics(self, tenant_id: str) -> A2AResourceMetrics:
        """Get resource usage metrics for a tenant."""
        with self._lock:
            usage = self._tenant_usage.get(tenant_id)
            if usage is None:
                return A2AResourceMetrics(
                    tenant_id=tenant_id,
                    active_sessions=0,
                    total_messages=0,
                    session_limit=self.limits.session_limit_per_tenant,
                    message_limit_per_session=self.limits.message_limit_per_session,
                    message_rate_limit=self.limits.message_rate_limit,
                )

            return A2AResourceMetrics(
                tenant_id=tenant_id,
                active_sessions=usage.active_sessions,
                total_messages=usage.total_messages,
                session_limit=self.limits.session_limit_per_tenant,
                message_limit_per_session=self.limits.message_limit_per_session,
                message_rate_limit=self.limits.message_rate_limit,
            )

    async def start(self) -> None:
        """Start the cleanup background task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(
            "a2a_resource_manager_started",
            backend="memory",
            cleanup_interval_minutes=self.limits.cleanup_interval_minutes,
        )

    async def stop(self) -> None:
        """Stop the cleanup background task."""
        self._running = False
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("a2a_resource_manager_stopped", backend="memory")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired sessions."""
        interval = self.limits.cleanup_interval_minutes * 60
        while self._running:
            try:
                await asyncio.sleep(interval)
                if self._running:
                    await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(
                    "a2a_cleanup_loop_error",
                    error=str(e),
                )

    async def _cleanup_expired_sessions(self) -> None:
        """Remove sessions that have exceeded TTL."""
        now = datetime.now(timezone.utc)
        ttl = timedelta(hours=self.limits.session_ttl_hours)
        expired_sessions: list[str] = []

        with self._lock:
            for session_id, usage in list(self._session_usage.items()):
                if now - usage.created_at > ttl:
                    expired_sessions.append(session_id)

        # Close expired sessions outside the lock
        for session_id in expired_sessions:
            await self.close_session(session_id)
            logger.info(
                "a2a_session_expired",
                session_id=session_id,
            )

        if expired_sessions:
            logger.info(
                "a2a_cleanup_completed",
                expired_count=len(expired_sessions),
            )


# -----------------------------------------------------------------------------
# Redis Implementation
# -----------------------------------------------------------------------------


class RedisA2AResourceManager(A2AResourceManager):
    """Redis-backed implementation for production multi-worker deployments.

    Uses Redis for cross-worker session and message tracking:
    - HSET for tenant and session metadata
    - ZADD for rate limiting with sorted sets
    - EXPIRE for automatic TTL enforcement
    """

    def __init__(
        self,
        limits: A2AResourceLimits,
        redis_client: "redis.Redis",
        key_prefix: str = "a2a",
    ) -> None:
        """Initialize the Redis resource manager.

        Args:
            limits: Configuration for resource limits
            redis_client: Async Redis client
            key_prefix: Prefix for all Redis keys
        """
        super().__init__(limits)
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._running = False

    def _tenant_key(self, tenant_id: str) -> str:
        """Get Redis key for tenant usage."""
        return f"{self._key_prefix}:tenant:{tenant_id}"

    def _session_key(self, session_id: str) -> str:
        """Get Redis key for session info."""
        return f"{self._key_prefix}:session:{session_id}:info"

    def _rate_key(self, session_id: str) -> str:
        """Get Redis key for session rate limiting."""
        return f"{self._key_prefix}:session:{session_id}:rate"

    async def check_session_limit(self, tenant_id: str) -> bool:
        """Check if tenant can create a new session."""
        key = self._tenant_key(tenant_id)
        active = await self._redis.hget(key, "active_sessions")
        if active is None:
            return True
        return int(active) < self.limits.session_limit_per_tenant

    async def check_message_limit(self, session_id: str) -> bool:
        """Check if session can send another message."""
        key = self._session_key(session_id)
        count = await self._redis.hget(key, "message_count")
        if count is None:
            return True
        return int(count) < self.limits.message_limit_per_session

    async def check_rate_limit(self, session_id: str) -> bool:
        """Check if session is within rate limit using sorted set."""
        key = self._rate_key(session_id)
        now = time()
        minute_ago = now - 60

        # Count messages in last minute
        count = await self._redis.zcount(key, minute_ago, now)
        return count < self.limits.message_rate_limit

    async def register_session(self, session_id: str, tenant_id: str) -> None:
        """Register a new session with atomic Redis operations."""
        if not await self.check_session_limit(tenant_id):
            logger.warning(
                "a2a_session_limit_exceeded",
                tenant_id=tenant_id,
                limit=self.limits.session_limit_per_tenant,
            )
            raise A2ASessionLimitExceeded(tenant_id, self.limits.session_limit_per_tenant)

        tenant_key = self._tenant_key(tenant_id)
        session_key = self._session_key(session_id)
        now = datetime.now(timezone.utc).isoformat()
        ttl_seconds = self.limits.session_ttl_hours * 3600

        # Use pipeline for atomic operations
        pipeline = self._redis.pipeline()
        pipeline.hincrby(tenant_key, "active_sessions", 1)
        pipeline.hset(tenant_key, "last_activity", now)
        pipeline.hset(
            session_key,
            mapping={
                "session_id": session_id,
                "tenant_id": tenant_id,
                "message_count": "0",
                "created_at": now,
                "last_message_at": now,
            },
        )
        pipeline.expire(session_key, ttl_seconds)
        await pipeline.execute()

        logger.info(
            "a2a_session_registered",
            session_id=session_id,
            tenant_id=tenant_id,
        )

    async def record_message(self, session_id: str) -> None:
        """Record a message with atomic Redis operations."""
        # Check message limit first
        if not await self.check_message_limit(session_id):
            logger.warning(
                "a2a_message_limit_exceeded",
                session_id=session_id,
                limit=self.limits.message_limit_per_session,
            )
            raise A2AMessageLimitExceeded(session_id, self.limits.message_limit_per_session)

        # Check rate limit
        if not await self.check_rate_limit(session_id):
            logger.warning(
                "a2a_rate_limit_exceeded",
                session_id=session_id,
                limit=self.limits.message_rate_limit,
            )
            raise A2ARateLimitExceeded(session_id, self.limits.message_rate_limit)

        session_key = self._session_key(session_id)
        rate_key = self._rate_key(session_id)
        now = time()
        now_iso = datetime.now(timezone.utc).isoformat()
        five_minutes_ago = now - 300

        # Get tenant_id from session
        tenant_id = await self._redis.hget(session_key, "tenant_id")
        if tenant_id is None:
            logger.warning(
                "a2a_record_message_unknown_session",
                session_id=session_id,
            )
            return

        # Decode tenant_id if bytes
        if isinstance(tenant_id, bytes):
            tenant_id = tenant_id.decode("utf-8")

        tenant_key = self._tenant_key(tenant_id)

        # Use pipeline for atomic operations
        pipeline = self._redis.pipeline()
        pipeline.hincrby(session_key, "message_count", 1)
        pipeline.hset(session_key, "last_message_at", now_iso)
        pipeline.zadd(rate_key, {str(uuid4()): now})
        pipeline.zremrangebyscore(rate_key, 0, five_minutes_ago)
        pipeline.expire(rate_key, 300)  # 5 minute TTL
        pipeline.hincrby(tenant_key, "total_messages", 1)
        pipeline.hset(tenant_key, "last_activity", now_iso)
        await pipeline.execute()

        logger.debug(
            "a2a_message_recorded",
            session_id=session_id,
        )

    async def close_session(self, session_id: str) -> None:
        """Close a session and update limits."""
        session_key = self._session_key(session_id)
        rate_key = self._rate_key(session_id)

        # Get tenant_id before deleting
        tenant_id = await self._redis.hget(session_key, "tenant_id")
        if tenant_id is None:
            logger.debug(
                "a2a_close_session_not_found",
                session_id=session_id,
            )
            return

        # Decode tenant_id if bytes
        if isinstance(tenant_id, bytes):
            tenant_id = tenant_id.decode("utf-8")

        tenant_key = self._tenant_key(tenant_id)
        now = datetime.now(timezone.utc).isoformat()

        # Use pipeline for atomic operations
        pipeline = self._redis.pipeline()
        pipeline.delete(session_key)
        pipeline.delete(rate_key)
        pipeline.hincrby(tenant_key, "active_sessions", -1)
        pipeline.hset(tenant_key, "last_activity", now)
        await pipeline.execute()

        logger.info(
            "a2a_session_closed",
            session_id=session_id,
            tenant_id=tenant_id,
        )

    async def get_tenant_metrics(self, tenant_id: str) -> A2AResourceMetrics:
        """Get resource usage metrics for a tenant."""
        tenant_key = self._tenant_key(tenant_id)
        data = await self._redis.hgetall(tenant_key)

        if not data:
            return A2AResourceMetrics(
                tenant_id=tenant_id,
                active_sessions=0,
                total_messages=0,
                session_limit=self.limits.session_limit_per_tenant,
                message_limit_per_session=self.limits.message_limit_per_session,
                message_rate_limit=self.limits.message_rate_limit,
            )

        # Decode bytes if necessary
        active_sessions = data.get(b"active_sessions") or data.get("active_sessions") or 0
        total_messages = data.get(b"total_messages") or data.get("total_messages") or 0

        return A2AResourceMetrics(
            tenant_id=tenant_id,
            active_sessions=max(0, int(active_sessions)),
            total_messages=int(total_messages),
            session_limit=self.limits.session_limit_per_tenant,
            message_limit_per_session=self.limits.message_limit_per_session,
            message_rate_limit=self.limits.message_rate_limit,
        )

    async def start(self) -> None:
        """Start the cleanup background task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(
            "a2a_resource_manager_started",
            backend="redis",
            cleanup_interval_minutes=self.limits.cleanup_interval_minutes,
        )

    async def stop(self) -> None:
        """Stop the cleanup background task."""
        self._running = False
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("a2a_resource_manager_stopped", backend="redis")

    async def _cleanup_loop(self) -> None:
        """Background task for cleanup (Redis TTL handles most expiry)."""
        interval = self.limits.cleanup_interval_minutes * 60
        while self._running:
            try:
                await asyncio.sleep(interval)
                if self._running:
                    # Redis TTL handles session expiry, but we can clean up
                    # any orphaned rate limit keys or fix negative session counts
                    await self._cleanup_negative_counts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(
                    "a2a_cleanup_loop_error",
                    error=str(e),
                )

    async def _cleanup_negative_counts(self) -> None:
        """Fix any negative active session counts."""
        # This is a safety measure; normally counts should never go negative
        # We scan for tenant keys and fix any negative values
        pattern = f"{self._key_prefix}:tenant:*"
        async for key in self._redis.scan_iter(match=pattern):
            active = await self._redis.hget(key, "active_sessions")
            if active is not None and int(active) < 0:
                await self._redis.hset(key, "active_sessions", 0)
                logger.warning(
                    "a2a_fixed_negative_session_count",
                    key=key,
                )


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


class A2AResourceManagerFactory:
    """Factory for creating A2A resource managers based on configuration."""

    @staticmethod
    def create(
        backend: str,
        limits: A2AResourceLimits,
        redis_client: Optional["redis.Redis"] = None,
    ) -> A2AResourceManager:
        """Create an A2A resource manager.

        Args:
            backend: Backend type ("memory" or "redis")
            limits: Resource limits configuration
            redis_client: Redis client (required if backend="redis")

        Returns:
            A2AResourceManager implementation

        Raises:
            ValueError: If redis backend selected but no client provided
        """
        if backend == "redis":
            if redis_client is None:
                raise ValueError(
                    "Redis client required when A2A_LIMITS_BACKEND=redis. "
                    "Ensure REDIS_URL is configured."
                )
            logger.info(
                "a2a_resource_manager_created",
                backend="redis",
            )
            return RedisA2AResourceManager(limits=limits, redis_client=redis_client)

        if backend == "postgres":
            # Future implementation - for now, fall back to memory
            logger.warning(
                "a2a_postgres_backend_not_implemented",
                fallback="memory",
            )

        logger.info(
            "a2a_resource_manager_created",
            backend="memory",
        )
        return InMemoryA2AResourceManager(limits=limits)
