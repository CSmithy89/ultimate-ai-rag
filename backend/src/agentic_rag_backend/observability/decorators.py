"""Decorators for instrumentation and timing of retrieval operations.

This module provides decorators that can be applied to async functions
to automatically record metrics for retrieval operations.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    ParamSpec,
    TypeVar,
)

import structlog

from .metrics import (
    record_retrieval_request,
    record_retrieval_latency,
    track_active_retrieval,
)

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def track_retrieval_operation(
    strategy: str,
    phase: str = "search",
    tenant_id_param: str = "tenant_id",
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to track retrieval operations with full metrics.

    This decorator:
    1. Increments the retrieval request counter
    2. Tracks active retrieval operations
    3. Records latency for the specified phase

    Args:
        strategy: Retrieval strategy (vector|graph|hybrid)
        phase: Operation phase (embed|search|rerank|grade)
        tenant_id_param: Name of the parameter containing tenant_id

    Returns:
        Decorated async function

    Example:
        @track_retrieval_operation(strategy="vector", phase="search")
        async def vector_search(query: str, tenant_id: str) -> list[VectorHit]:
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Extract tenant_id from kwargs or use default
            tenant_id = kwargs.get(tenant_id_param, "unknown")
            if not isinstance(tenant_id, str):
                tenant_id = str(tenant_id) if tenant_id is not None else "unknown"

            # Record the request
            record_retrieval_request(strategy=strategy, tenant_id=tenant_id)

            # Track active operations and measure latency
            start_time = time.perf_counter()
            with track_active_retrieval(tenant_id):
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start_time
                    record_retrieval_latency(
                        strategy=strategy,
                        phase=phase,
                        tenant_id=tenant_id,
                        duration_seconds=duration,
                    )

        return wrapper

    return decorator


def measure_latency(
    strategy: str,
    phase: str,
    tenant_id_param: str = "tenant_id",
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to measure and record latency only (no request counting).

    This is useful for sub-operations within a retrieval flow where
    you want to measure latency but not count as a separate request.

    Args:
        strategy: Retrieval strategy (vector|graph|hybrid)
        phase: Operation phase (embed|search|rerank|grade)
        tenant_id_param: Name of the parameter containing tenant_id

    Returns:
        Decorated async function

    Example:
        @measure_latency(strategy="hybrid", phase="embed")
        async def embed_query(query: str, tenant_id: str) -> list[float]:
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Extract tenant_id from kwargs or use default
            tenant_id = kwargs.get(tenant_id_param, "unknown")
            if not isinstance(tenant_id, str):
                tenant_id = str(tenant_id) if tenant_id is not None else "unknown"

            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                record_retrieval_latency(
                    strategy=strategy,
                    phase=phase,
                    tenant_id=tenant_id,
                    duration_seconds=duration,
                )

        return wrapper

    return decorator


class LatencyTimer:
    """Context manager for measuring operation latency.

    This is useful when you need more control over when and how
    latency is recorded, or when working with synchronous code.

    Example:
        with LatencyTimer(strategy="vector", phase="search", tenant_id="tenant-1") as timer:
            results = await search(query)
        # Latency is automatically recorded when the context exits
    """

    def __init__(
        self,
        strategy: str,
        phase: str,
        tenant_id: str,
    ) -> None:
        """Initialize the latency timer.

        Args:
            strategy: Retrieval strategy (vector|graph|hybrid)
            phase: Operation phase (embed|search|rerank|grade)
            tenant_id: Tenant identifier
        """
        self.strategy = strategy
        self.phase = phase
        self.tenant_id = tenant_id
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self) -> "LatencyTimer":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Stop timing and record latency."""
        if self.start_time is not None:
            self.duration = time.perf_counter() - self.start_time
            record_retrieval_latency(
                strategy=self.strategy,
                phase=self.phase,
                tenant_id=self.tenant_id,
                duration_seconds=self.duration,
            )

    async def __aenter__(self) -> "LatencyTimer":
        """Start timing (async version)."""
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Stop timing and record latency (async version)."""
        if self.start_time is not None:
            self.duration = time.perf_counter() - self.start_time
            record_retrieval_latency(
                strategy=self.strategy,
                phase=self.phase,
                tenant_id=self.tenant_id,
                duration_seconds=self.duration,
            )
