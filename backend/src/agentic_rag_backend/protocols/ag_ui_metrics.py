"""Prometheus metrics for AG-UI stream monitoring.

This module defines all Prometheus metrics for tracking AG-UI stream health,
performance, and throughput. All metrics include tenant_id label for multi-tenant
analysis with optional cardinality control via bucket normalization.

Metrics defined:
- agui_stream_started_total: Counter for stream starts
- agui_stream_completed_total: Counter for stream completions (success/error)
- agui_event_emitted_total: Counter for events by type
- agui_stream_bytes_total: Counter for bytes streamed
- agui_active_streams: Gauge for currently active streams
- agui_stream_duration_seconds: Histogram for stream duration
- agui_event_latency_seconds: Histogram for inter-event latency
- agui_stream_event_count: Histogram for events per stream

Usage:
    async with track_agui_stream(tenant_id) as metrics:
        for event in generate_events():
            metrics.event_emitted(event.event.value, len(event.to_sse()))
            yield event

Story: 22-B1 - Implement AG-UI Stream Metrics
Epic: 22 - Advanced Protocol Integration
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from prometheus_client import Counter, Histogram, Gauge

from ..observability.metrics import normalize_tenant_label, get_metrics_registry

logger = structlog.get_logger(__name__)

# Get the shared registry for consistency with other metrics
_registry = get_metrics_registry()

# =============================================================================
# Issue #2 Fix: Known event types to prevent label cardinality explosion
# =============================================================================

KNOWN_EVENT_TYPES: frozenset[str] = frozenset({
    "RUN_STARTED",
    "RUN_FINISHED",
    "RUN_ERROR",  # Story 22-B2: Extended error events
    "TEXT_MESSAGE_START",
    "TEXT_MESSAGE_CONTENT",
    "TEXT_MESSAGE_END",
    "TOOL_CALL_START",
    "TOOL_CALL_ARGS",
    "TOOL_CALL_END",
    "STATE_SNAPSHOT",
    "STATE_DELTA",
    "MESSAGES_SNAPSHOT",
})
"""Known AG-UI event types for cardinality control.

Any event type not in this set will be mapped to 'OTHER' to prevent
Prometheus label cardinality explosion from unexpected event types.
"""

# =============================================================================
# Counter Metrics
# =============================================================================

STREAM_STARTED = Counter(
    "agui_stream_started_total",
    "Total AG-UI streams started",
    labelnames=["tenant_id"],
    registry=_registry,
)
"""Counter for AG-UI stream starts.

Labels:
    tenant_id: Tenant identifier for multi-tenancy
"""

STREAM_COMPLETED = Counter(
    "agui_stream_completed_total",
    "Total AG-UI streams completed",
    labelnames=["tenant_id", "status"],  # status: success, error
    registry=_registry,
)
"""Counter for AG-UI stream completions.

Labels:
    tenant_id: Tenant identifier for multi-tenancy
    status: Completion status (success or error)
"""

EVENT_EMITTED = Counter(
    "agui_event_emitted_total",
    "Total AG-UI events emitted",
    labelnames=["tenant_id", "event_type"],
    registry=_registry,
)
"""Counter for AG-UI events emitted by type.

Labels:
    tenant_id: Tenant identifier for multi-tenancy
    event_type: AG-UI event type (e.g., RUN_STARTED, TEXT_MESSAGE_CONTENT)
"""

STREAM_BYTES = Counter(
    "agui_stream_bytes_total",
    "Total bytes streamed via AG-UI",
    labelnames=["tenant_id"],
    registry=_registry,
)
"""Counter for total bytes streamed.

Labels:
    tenant_id: Tenant identifier for multi-tenancy
"""

# =============================================================================
# Gauge Metrics
# =============================================================================

ACTIVE_STREAMS = Gauge(
    "agui_active_streams",
    "Currently active AG-UI streams",
    labelnames=["tenant_id"],
    registry=_registry,
)
"""Gauge for currently active AG-UI streams.

Labels:
    tenant_id: Tenant identifier for multi-tenancy
"""

# =============================================================================
# Histogram Metrics
# =============================================================================

# Stream duration buckets: 0.1s to 600s (typical stream durations + long-tail)
# Issue #5 Fix: Extended buckets to 120, 300, 600 for long-running streams
STREAM_DURATION_BUCKETS = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)

STREAM_DURATION = Histogram(
    "agui_stream_duration_seconds",
    "AG-UI stream duration in seconds",
    labelnames=["tenant_id"],
    buckets=STREAM_DURATION_BUCKETS,
    registry=_registry,
)
"""Histogram for AG-UI stream duration.

Labels:
    tenant_id: Tenant identifier for multi-tenancy

Bucket rationale:
    Streams typically complete in 1-30 seconds. Extended buckets (60-600s)
    cover long operations like complex queries or slow LLM responses.
"""

# Event latency buckets: 10ms to 1s (token-level latency)
EVENT_LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)

EVENT_LATENCY = Histogram(
    "agui_event_latency_seconds",
    "Time between AG-UI events in seconds",
    labelnames=["tenant_id"],
    buckets=EVENT_LATENCY_BUCKETS,
    registry=_registry,
)
"""Histogram for inter-event latency.

Labels:
    tenant_id: Tenant identifier for multi-tenancy

Bucket rationale:
    Token-level latency typically ranges from 10-100ms.
"""

# Event count buckets: 1 to 250 events per stream
STREAM_EVENT_COUNT_BUCKETS = (1, 5, 10, 25, 50, 100, 250)

STREAM_EVENT_COUNT = Histogram(
    "agui_stream_event_count",
    "Number of events per AG-UI stream",
    labelnames=["tenant_id"],
    buckets=STREAM_EVENT_COUNT_BUCKETS,
    registry=_registry,
)
"""Histogram for events per stream.

Labels:
    tenant_id: Tenant identifier for multi-tenancy

Bucket rationale:
    Simple queries have few events; complex operations may have many.
"""


# =============================================================================
# Metrics Collector Class
# =============================================================================


class AGUIMetricsCollector:
    """Collects metrics for AG-UI streams.

    This class tracks metrics throughout a stream's lifecycle:
    - stream_started(): Called when stream begins (increments started counter and active gauge)
    - event_emitted(): Called for each event (tracks event type, bytes, and latency)
    - stream_completed(): Called when stream ends (records duration, event count, status)

    Example:
        collector = AGUIMetricsCollector(tenant_id)
        collector.stream_started()
        for event in events:
            collector.event_emitted(event.event.value, len(event.to_sse()))
        collector.stream_completed("success")

    Attributes:
        tenant_id: Normalized tenant identifier for metric labeling
        start_time: Timestamp when stream started
        last_event_time: Timestamp of last event (for latency calculation)
        event_count: Total number of events emitted
        total_bytes: Total bytes of event data emitted
    """

    def __init__(self, tenant_id: str) -> None:
        """Initialize collector with tenant ID.

        Args:
            tenant_id: Tenant identifier (will be normalized for cardinality control).
                      If empty or None, defaults to "unknown".
        """
        self.tenant_id = normalize_tenant_label(tenant_id or "")
        self.start_time: float = 0.0
        self.last_event_time: float = 0.0
        self.event_count: int = 0
        self.total_bytes: int = 0
        self._stream_started: bool = False  # Issue #3 Fix: Track if stream_started was called

    def stream_started(self) -> None:
        """Record stream start.

        Increments the started counter and active streams gauge.
        Initializes timing for duration and latency tracking.
        """
        self.start_time = time.time()
        self.last_event_time = self.start_time
        self._stream_started = True  # Issue #3 Fix: Track if stream_started was called
        STREAM_STARTED.labels(tenant_id=self.tenant_id).inc()
        ACTIVE_STREAMS.labels(tenant_id=self.tenant_id).inc()

    def event_emitted(self, event_type: str, event_bytes: int = 0) -> None:
        """Record event emission.

        Args:
            event_type: The AG-UI event type (e.g., "TEXT_MESSAGE_CONTENT", "RUN_STARTED")
            event_bytes: Size of the SSE-serialized event in bytes (optional)

        Note:
            Unknown event types are mapped to 'OTHER' to prevent label cardinality
            explosion (Issue #2 fix).
        """
        now = time.time()

        # Issue #2 Fix: Normalize unknown event types to prevent cardinality explosion
        normalized_event_type = event_type if event_type in KNOWN_EVENT_TYPES else "OTHER"

        # Increment event counter
        EVENT_EMITTED.labels(
            tenant_id=self.tenant_id,
            event_type=normalized_event_type,
        ).inc()

        # Record inter-event latency (skip first event as it has no previous)
        if self.last_event_time > 0 and self.event_count > 0:
            latency = now - self.last_event_time
            EVENT_LATENCY.labels(tenant_id=self.tenant_id).observe(latency)

        self.last_event_time = now
        self.event_count += 1
        self.total_bytes += event_bytes

        # Emit bytes counter
        if event_bytes > 0:
            STREAM_BYTES.labels(tenant_id=self.tenant_id).inc(event_bytes)

    def stream_completed(self, status: str = "success") -> None:
        """Record stream completion.

        Args:
            status: Completion status ("success" or "error")

        Note:
            If stream_started() was never called, this method logs a warning
            and skips gauge decrement to prevent underflow (Issue #3 fix).
            Duration histogram is also skipped to prevent recording invalid
            values (Issue #4 fix).
        """
        # Issue #3 & #4 Fix: Guard against completion without start
        if not self._stream_started:
            logger.warning(
                "stream_completed_without_start",
                tenant_id=self.tenant_id,
                status=status,
            )
            # Still record completion counter for observability
            STREAM_COMPLETED.labels(
                tenant_id=self.tenant_id,
                status=status,
            ).inc()
            return

        duration = time.time() - self.start_time

        # Increment completion counter
        STREAM_COMPLETED.labels(
            tenant_id=self.tenant_id,
            status=status,
        ).inc()

        # Record duration histogram
        STREAM_DURATION.labels(tenant_id=self.tenant_id).observe(duration)

        # Record event count histogram
        STREAM_EVENT_COUNT.labels(tenant_id=self.tenant_id).observe(self.event_count)

        # Decrement active streams gauge
        ACTIVE_STREAMS.labels(tenant_id=self.tenant_id).dec()


# =============================================================================
# Context Manager
# =============================================================================


@asynccontextmanager
async def track_agui_stream(tenant_id: str) -> AsyncIterator[AGUIMetricsCollector]:
    """Context manager for tracking AG-UI stream metrics.

    Automatically records stream start on entry, and completion on exit.
    Handles both success and error cases.

    Args:
        tenant_id: Tenant identifier for metric labeling

    Yields:
        AGUIMetricsCollector instance for tracking events during the stream

    Example:
        async with track_agui_stream(tenant_id) as metrics:
            async for event in generate_events():
                metrics.event_emitted(event.event.value, len(event.to_sse()))
                yield event
    """
    collector = AGUIMetricsCollector(tenant_id)
    collector.stream_started()

    try:
        yield collector
        collector.stream_completed("success")
    except Exception:
        collector.stream_completed("error")
        raise


# =============================================================================
# Convenience Functions
# =============================================================================


def record_stream_started(tenant_id: str) -> None:
    """Record a stream start event.

    This is a convenience function for one-off metric recording without
    using the full AGUIMetricsCollector class.

    Args:
        tenant_id: Tenant identifier
    """
    tenant_label = normalize_tenant_label(tenant_id or "")
    STREAM_STARTED.labels(tenant_id=tenant_label).inc()
    ACTIVE_STREAMS.labels(tenant_id=tenant_label).inc()


def record_stream_completed(tenant_id: str, status: str = "success") -> None:
    """Record a stream completion event.

    This is a convenience function for one-off metric recording without
    using the full AGUIMetricsCollector class.

    Args:
        tenant_id: Tenant identifier
        status: Completion status ("success" or "error")
    """
    tenant_label = normalize_tenant_label(tenant_id or "")
    STREAM_COMPLETED.labels(tenant_id=tenant_label, status=status).inc()
    ACTIVE_STREAMS.labels(tenant_id=tenant_label).dec()


def record_event_emitted(tenant_id: str, event_type: str, event_bytes: int = 0) -> None:
    """Record an event emission.

    This is a convenience function for one-off metric recording without
    using the full AGUIMetricsCollector class.

    Args:
        tenant_id: Tenant identifier
        event_type: The AG-UI event type
        event_bytes: Size of the event in bytes (optional)

    Note:
        Unknown event types are mapped to 'OTHER' to prevent label cardinality
        explosion (Issue #2 fix).
    """
    tenant_label = normalize_tenant_label(tenant_id or "")
    # Issue #2 Fix: Normalize unknown event types
    normalized_event_type = event_type if event_type in KNOWN_EVENT_TYPES else "OTHER"
    EVENT_EMITTED.labels(tenant_id=tenant_label, event_type=normalized_event_type).inc()
    if event_bytes > 0:
        STREAM_BYTES.labels(tenant_id=tenant_label).inc(event_bytes)
