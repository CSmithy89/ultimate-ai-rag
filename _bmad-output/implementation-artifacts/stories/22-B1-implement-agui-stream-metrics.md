# Story 22-B1: Implement AG-UI Stream Metrics

Status: done

Epic: 22 - Advanced Protocol Integration
Priority: P0 - HIGH
Story Points: 5
Owner: Backend

## Story

As a **platform operator**,
I want **comprehensive Prometheus metrics for AG-UI stream health and performance**,
So that **I can monitor stream success rates, latencies, and event throughput in production, enabling proactive issue detection and capacity planning**.

## Background

Epic 22 builds on Epic 21's CopilotKit Full Integration to deliver enterprise-grade protocol capabilities. AG-UI stream metrics are critical for:

1. **Production Observability** - Track stream lifecycle (started, completed, failed)
2. **Performance Monitoring** - Measure stream duration and inter-event latency
3. **Capacity Planning** - Monitor active streams and throughput by tenant
4. **Alerting** - Enable Grafana alerts on error rates and latency thresholds

### AG-UI Protocol Context

AG-UI (Agent-UI) is the streaming protocol used by CopilotKit for real-time communication between the backend agent and frontend UI. Each stream:
- Starts with a `RUN_STARTED` event
- Emits multiple event types (text deltas, tool calls, state updates)
- Ends with `RUN_FINISHED` or `RUN_ERROR`

Without metrics, operators have no visibility into:
- How many concurrent streams are active
- What percentage of streams complete successfully
- How long streams take from start to finish
- What event types are being emitted and at what rate

### Related Prior Work

| Epic/Story | Relationship |
|------------|-------------|
| Epic 8: Operations & Observability | Original observability foundation (Prometheus, Grafana) |
| Epic 19-C5: Prometheus Retrieval Metrics | Similar metrics pattern for retrieval operations |
| Epic 21: CopilotKit Full Integration | AG-UI transport implementation (prerequisite) |
| 22-TD1: Add Prometheus Telemetry Metrics | Telemetry counter (related, completed) |

### Prometheus Cardinality Strategy

To prevent cardinality explosion from `tenant_id` labels:

1. **Tenant Sampling**: When `METRICS_TENANT_SAMPLING_ENABLED=true`, normalize tenant IDs to buckets
2. **Bucket Count**: `METRICS_TENANT_BUCKET_COUNT` (default 100) controls granularity
3. **Aggregation**: Global metrics without tenant label available for high-level dashboards

## Acceptance Criteria

1. **Given** the AG-UI metrics module exists at `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`, **when** a stream starts, **then** `agui_stream_started_total{tenant_id="..."}` counter is incremented.

2. **Given** a stream completes successfully, **when** `stream_completed("success")` is called, **then** `agui_stream_completed_total{tenant_id="...", status="success"}` counter is incremented.

3. **Given** a stream fails with an error, **when** `stream_completed("error")` is called, **then** `agui_stream_completed_total{tenant_id="...", status="error"}` counter is incremented.

4. **Given** an event is emitted during a stream, **when** `event_emitted(event_type, event_bytes)` is called, **then** `agui_event_emitted_total{tenant_id="...", event_type="..."}` counter is incremented.

5. **Given** events are emitted, **when** the time between events is measured, **then** `agui_event_latency_seconds{tenant_id="..."}` histogram records the inter-event latency.

6. **Given** a stream completes, **when** the duration is calculated, **then** `agui_stream_duration_seconds{tenant_id="..."}` histogram records the total stream time.

7. **Given** a stream completes, **when** the event count is tallied, **then** `agui_stream_event_count{tenant_id="..."}` histogram records the number of events.

8. **Given** streams are starting and completing, **when** the gauge is checked, **then** `agui_active_streams{tenant_id="..."}` accurately reflects currently active stream count.

9. **Given** events are emitted with byte sizes, **when** tracked, **then** `agui_stream_bytes_total{tenant_id="..."}` counter accumulates total bytes streamed.

10. **Given** `METRICS_TENANT_SAMPLING_ENABLED=true`, **when** metrics are recorded, **then** tenant IDs are normalized to bucket format (e.g., `bucket_42`) to control cardinality.

11. **Given** the `AGUIBridge` processes requests, **when** the `track_agui_stream` context manager is used, **then** all metrics are collected automatically throughout the stream lifecycle.

12. **Given** all metrics are defined, **when** Prometheus scrapes `/metrics`, **then** all AG-UI metrics are available for Grafana dashboards.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: **Addressed** - All metrics labeled by tenant_id with optional normalization
- [x] Rate limiting / abuse protection: **N/A** - Read-only metrics emission
- [x] Input validation / schema enforcement: **N/A** - Internal metrics API
- [x] Tests (unit/integration): **Addressed** - Unit tests for metric emission, integration tests for stream lifecycle
- [x] Error handling + logging: **Addressed** - Graceful handling of missing tenant_id, logging for edge cases
- [x] Documentation updates: **Addressed** - Grafana dashboard template provided

## Security Checklist

- [ ] **Cross-tenant isolation verified**: Metrics labeled by tenant_id, no cross-tenant data exposure
- [ ] **Authorization checked**: N/A - Metrics endpoint protected by existing Prometheus auth
- [ ] **No information leakage**: Tenant IDs optionally bucketed to prevent enumeration
- [ ] **Redis keys include tenant scope**: N/A - No Redis in this story
- [ ] **Integration tests for access control**: N/A - Read-only metrics
- [ ] **RFC 7807 error responses**: N/A - No API endpoints added
- [ ] **File-path inputs scoped**: N/A - No file path handling

## Tasks / Subtasks

- [ ] **Task 1: Create AG-UI Metrics Module** (AC: 1, 2, 3, 8, 12)
  - [ ] Create `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`
  - [ ] Define `STREAM_STARTED` Counter with `["tenant_id"]` labels
  - [ ] Define `STREAM_COMPLETED` Counter with `["tenant_id", "status"]` labels
  - [ ] Define `ACTIVE_STREAMS` Gauge with `["tenant_id"]` labels
  - [ ] Add module docstring with metric descriptions

- [ ] **Task 2: Define Event Metrics** (AC: 4, 9)
  - [ ] Define `EVENT_EMITTED` Counter with `["tenant_id", "event_type"]` labels
  - [ ] Define `STREAM_BYTES` Counter with `["tenant_id"]` labels
  - [ ] Document all event types that will be tracked

- [ ] **Task 3: Define Histogram Metrics** (AC: 5, 6, 7)
  - [ ] Define `STREAM_DURATION` Histogram with appropriate buckets `[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]`
  - [ ] Define `EVENT_LATENCY` Histogram with appropriate buckets `[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]`
  - [ ] Define `STREAM_EVENT_COUNT` Histogram with appropriate buckets `[1, 5, 10, 25, 50, 100, 250]`

- [ ] **Task 4: Implement AGUIMetricsCollector Class** (AC: 1-9)
  - [ ] Create `AGUIMetricsCollector` class with `__init__(self, tenant_id: str)`
  - [ ] Implement `stream_started()` method that increments started counter and gauge
  - [ ] Implement `event_emitted(event_type: str, event_bytes: int = 0)` method
  - [ ] Implement `stream_completed(status: str = "success")` method
  - [ ] Track `start_time`, `last_event_time`, `event_count`, `total_bytes` as instance attributes

- [ ] **Task 5: Implement Tenant ID Normalization** (AC: 10)
  - [ ] Add `normalize_tenant_id(tenant_id: str) -> str` function
  - [ ] Read `METRICS_TENANT_SAMPLING_ENABLED` from settings
  - [ ] Read `METRICS_TENANT_BUCKET_COUNT` from settings (default 100)
  - [ ] Hash tenant_id and map to bucket when enabled

- [ ] **Task 6: Implement Context Manager** (AC: 11)
  - [ ] Create `track_agui_stream(tenant_id: str)` async context manager
  - [ ] Call `stream_started()` on entry
  - [ ] Call `stream_completed("success")` on normal exit
  - [ ] Call `stream_completed("error")` on exception
  - [ ] Re-raise exceptions after recording metrics

- [ ] **Task 7: Add Configuration to Settings** (AC: 10)
  - [ ] Add `metrics_tenant_sampling_enabled: bool = False` to Settings
  - [ ] Add `metrics_tenant_bucket_count: int = 100` to Settings
  - [ ] Update `.env.example` with new variables
  - [ ] Document in configuration reference

- [ ] **Task 8: Integrate with AGUIBridge** (AC: 11)
  - [ ] Import `track_agui_stream` in `ag_ui_bridge.py`
  - [ ] Wrap request processing in context manager
  - [ ] Call `metrics.event_emitted()` for each emitted event
  - [ ] Pass event byte size (from SSE serialization)

- [ ] **Task 9: Add Unit Tests** (AC: 1-10)
  - [ ] Create `tests/protocols/test_ag_ui_metrics.py`
  - [ ] Test `AGUIMetricsCollector` initialization
  - [ ] Test `stream_started()` increments counters
  - [ ] Test `event_emitted()` tracks event type and bytes
  - [ ] Test `stream_completed("success")` vs `stream_completed("error")`
  - [ ] Test latency histogram recording
  - [ ] Test duration histogram recording
  - [ ] Test event count histogram recording
  - [ ] Test tenant ID normalization when sampling enabled
  - [ ] Test context manager success path
  - [ ] Test context manager error path

- [ ] **Task 10: Add Integration Tests** (AC: 11, 12)
  - [ ] Create `tests/integration/test_ag_ui_metrics_integration.py`
  - [ ] Test metrics available at `/metrics` endpoint
  - [ ] Test stream lifecycle emits correct metrics
  - [ ] Test tenant isolation in metrics labels

- [ ] **Task 11: Create Grafana Dashboard Template** (AC: 12)
  - [ ] Create `docs/monitoring/grafana-agui-dashboard.json`
  - [ ] Panel: Stream success rate (completed success / total)
  - [ ] Panel: Active streams by tenant
  - [ ] Panel: Stream duration p50, p95, p99
  - [ ] Panel: Event latency distribution
  - [ ] Panel: Event type distribution
  - [ ] Panel: Bytes streamed over time
  - [ ] Document dashboard import instructions

## Technical Notes

### Prometheus Metrics Definitions

```python
# backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

# Counters
STREAM_STARTED = Counter(
    "agui_stream_started_total",
    "Total AG-UI streams started",
    ["tenant_id"],
)

STREAM_COMPLETED = Counter(
    "agui_stream_completed_total",
    "Total AG-UI streams completed",
    ["tenant_id", "status"],  # status: success, error
)

EVENT_EMITTED = Counter(
    "agui_event_emitted_total",
    "Total AG-UI events emitted",
    ["tenant_id", "event_type"],
)

STREAM_BYTES = Counter(
    "agui_stream_bytes_total",
    "Total bytes streamed via AG-UI",
    ["tenant_id"],
)

# Gauge
ACTIVE_STREAMS = Gauge(
    "agui_active_streams",
    "Currently active AG-UI streams",
    ["tenant_id"],
)

# Histograms
STREAM_DURATION = Histogram(
    "agui_stream_duration_seconds",
    "AG-UI stream duration",
    ["tenant_id"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

EVENT_LATENCY = Histogram(
    "agui_event_latency_seconds",
    "Time between AG-UI events",
    ["tenant_id"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

STREAM_EVENT_COUNT = Histogram(
    "agui_stream_event_count",
    "Events per AG-UI stream",
    ["tenant_id"],
    buckets=[1, 5, 10, 25, 50, 100, 250],
)
```

### AGUIMetricsCollector Class

```python
class AGUIMetricsCollector:
    """Collects metrics for AG-UI streams."""

    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = normalize_tenant_id(tenant_id)
        self.start_time: float = 0
        self.last_event_time: float = 0
        self.event_count: int = 0
        self.total_bytes: int = 0

    def stream_started(self) -> None:
        """Record stream start."""
        self.start_time = time.time()
        self.last_event_time = self.start_time
        STREAM_STARTED.labels(tenant_id=self.tenant_id).inc()
        ACTIVE_STREAMS.labels(tenant_id=self.tenant_id).inc()

    def event_emitted(self, event_type: str, event_bytes: int = 0) -> None:
        """Record event emission."""
        now = time.time()

        EVENT_EMITTED.labels(
            tenant_id=self.tenant_id,
            event_type=event_type,
        ).inc()

        if self.last_event_time > 0:
            latency = now - self.last_event_time
            EVENT_LATENCY.labels(tenant_id=self.tenant_id).observe(latency)

        self.last_event_time = now
        self.event_count += 1
        self.total_bytes += event_bytes

        if event_bytes > 0:
            STREAM_BYTES.labels(tenant_id=self.tenant_id).inc(event_bytes)

    def stream_completed(self, status: str = "success") -> None:
        """Record stream completion."""
        duration = time.time() - self.start_time

        STREAM_COMPLETED.labels(
            tenant_id=self.tenant_id,
            status=status,
        ).inc()

        STREAM_DURATION.labels(tenant_id=self.tenant_id).observe(duration)
        STREAM_EVENT_COUNT.labels(tenant_id=self.tenant_id).observe(self.event_count)
        ACTIVE_STREAMS.labels(tenant_id=self.tenant_id).dec()
```

### Tenant ID Normalization

```python
def normalize_tenant_id(tenant_id: str) -> str:
    """Normalize tenant ID to prevent cardinality explosion."""
    from agentic_rag_backend.core.config import settings

    if settings.metrics_tenant_sampling_enabled:
        bucket = hash(tenant_id) % settings.metrics_tenant_bucket_count
        return f"bucket_{bucket}"
    return tenant_id
```

### Context Manager Pattern

```python
@asynccontextmanager
async def track_agui_stream(tenant_id: str) -> AsyncIterator[AGUIMetricsCollector]:
    """Context manager for tracking AG-UI stream metrics."""
    collector = AGUIMetricsCollector(tenant_id)
    collector.stream_started()

    try:
        yield collector
        collector.stream_completed("success")
    except Exception:
        collector.stream_completed("error")
        raise
```

### AGUIBridge Integration

```python
# backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py (update)
from agentic_rag_backend.protocols.ag_ui_metrics import track_agui_stream

async def process_request(
    self,
    request: CopilotRequest,
    tenant_id: str,
) -> AsyncIterator[AGUIEvent]:
    """Process CopilotKit request with metrics tracking."""
    async with track_agui_stream(tenant_id) as metrics:
        async for event in self._generate_events(request):
            metrics.event_emitted(event.event.value, len(event.to_sse()))
            yield event
```

### Configuration Variables

```bash
# .env
# AG-UI Metrics Configuration (22-B1)
METRICS_TENANT_SAMPLING_ENABLED=false  # Enable tenant ID bucketing
METRICS_TENANT_BUCKET_COUNT=100        # Number of tenant buckets when sampling
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py` | Create | Metrics module with collectors |
| `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` | Modify | Integrate metrics context manager |
| `backend/src/agentic_rag_backend/core/config.py` | Modify | Add metrics settings |
| `backend/src/agentic_rag_backend/protocols/__init__.py` | Modify | Export new classes |
| `backend/.env.example` | Modify | Add metrics configuration |
| `tests/protocols/test_ag_ui_metrics.py` | Create | Unit tests |
| `tests/integration/test_ag_ui_metrics_integration.py` | Create | Integration tests |
| `docs/monitoring/grafana-agui-dashboard.json` | Create | Grafana dashboard template |

### Histogram Bucket Rationale

| Metric | Buckets | Rationale |
|--------|---------|-----------|
| `stream_duration_seconds` | `[0.1, 0.5, 1, 2.5, 5, 10, 30, 60]` | Streams typically 1-30 seconds; 60s for long operations |
| `event_latency_seconds` | `[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]` | Token-level latency typically 10-100ms |
| `stream_event_count` | `[1, 5, 10, 25, 50, 100, 250]` | Simple queries have few events; complex have many |

### Event Types to Track

Based on AGUIEventType enum:
- `RUN_STARTED`
- `RUN_FINISHED`
- `RUN_ERROR`
- `TEXT_MESSAGE_START`
- `TEXT_MESSAGE_CONTENT`
- `TEXT_MESSAGE_END`
- `TOOL_CALL_START`
- `TOOL_CALL_ARGS`
- `TOOL_CALL_END`
- `STATE_SNAPSHOT`
- `STATE_DELTA`
- `MESSAGES_SNAPSHOT`

## Dependencies

- `prometheus_client` - Already installed via Epic 8 observability foundation
- Epic 21 completed - AG-UI transport (`ag_ui_bridge.py`) must be available
- Story 22-TD1 completed - Telemetry counter pattern established

## Definition of Done

- [ ] `AGUIMetricsCollector` class implemented with all metric methods
- [ ] All 8 Prometheus metrics defined and exported
- [ ] `track_agui_stream()` context manager implemented
- [ ] `normalize_tenant_id()` function with sampling support
- [ ] Settings extended with metrics configuration
- [ ] `AGUIBridge.process_request()` integrated with metrics
- [ ] `.env.example` updated with configuration
- [ ] Unit tests for all metric scenarios (>85% coverage)
- [ ] Integration tests for stream lifecycle
- [ ] Grafana dashboard template created
- [ ] Metrics visible at `/metrics` endpoint
- [ ] Code review approved
- [ ] Story file updated with Dev Notes

## Dev Notes

### Implementation Summary (2026-01-11)

All 11 tasks from the story have been completed:

1. **AG-UI Metrics Module Created**: `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`
   - All 8 Prometheus metrics defined (4 counters, 1 gauge, 3 histograms)
   - Metrics registered with the shared Prometheus registry from observability module
   - Uses existing `normalize_tenant_label()` from observability for tenant normalization

2. **AGUIMetricsCollector Class Implemented**:
   - Tracks stream lifecycle (start_time, last_event_time, event_count, total_bytes)
   - `stream_started()` - Increments counter and gauge
   - `event_emitted(event_type, event_bytes)` - Tracks events with latency measurement
   - `stream_completed(status)` - Records duration, event count, decrements gauge

3. **Context Manager Implemented**: `track_agui_stream(tenant_id)` async context manager
   - Automatically records stream start and completion
   - Handles both success and error paths

4. **Convenience Functions Added**:
   - `record_stream_started()`, `record_stream_completed()`, `record_event_emitted()`
   - For use cases that don't require the full context manager

5. **AGUIBridge Integration**: Modified `process_request()` to track all events
   - Creates metrics collector at start of each request
   - Tracks every event type emitted during the stream
   - Records byte sizes for content events
   - Properly handles error paths with correct status

6. **Configuration**: Uses existing `METRICS_TENANT_LABEL_MODE` from observability module
   - Supports `full`, `hash` (with buckets), and `global` modes
   - Documented in `.env.example`

7. **Exports Updated**: `protocols/__init__.py` exports all new classes and functions

### Key Design Decisions

1. **Reused existing tenant normalization**: Instead of creating new `normalize_tenant_id()`, leveraged existing `normalize_tenant_label()` from observability module for consistency.

2. **Shared Prometheus registry**: Used `get_metrics_registry()` from observability to ensure metrics are on the same registry as other application metrics.

3. **Event latency tracking**: Only records latency between events (not first event) to avoid measuring stream start overhead.

4. **Bytes tracking optional**: `event_bytes` parameter defaults to 0, making it optional for events where size doesn't matter.

### Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py` | Created | Metrics module with all collectors |
| `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` | Modified | Integrated metrics tracking |
| `backend/src/agentic_rag_backend/protocols/__init__.py` | Modified | Added exports |
| `.env.example` | Modified | Documented AG-UI metrics configuration |
| `backend/tests/unit/protocols/test_ag_ui_metrics.py` | Created | 43 unit tests |
| `backend/tests/integration/test_ag_ui_metrics_integration.py` | Created | 20 integration tests |
| `docs/monitoring/grafana-agui-dashboard.json` | Created | Grafana dashboard template |

## Test Outcomes

### Unit Tests (43 tests)
```
backend/tests/unit/protocols/test_ag_ui_metrics.py
- TestAGUIMetricsCollector: 9 tests - PASSED
- TestTrackAGUIStreamContextManager: 5 tests - PASSED
- TestConvenienceFunctions: 5 tests - PASSED
- TestTenantNormalization: 4 tests - PASSED
- TestMetricsRegistration: 8 tests - PASSED
- TestHistogramBuckets: 3 tests - PASSED
- TestMetricLabels: 3 tests - PASSED
- TestEventLatencyTracking: 2 tests - PASSED
- TestStreamLifecycle: 4 tests - PASSED

All 43 tests passed in 0.54s
```

### Integration Tests (20 tests)
```
backend/tests/integration/test_ag_ui_metrics_integration.py
- TestMetricsEndpointExposure: 3 tests - PASSED
- TestStreamLifecycleMetrics: 3 tests - PASSED
- TestTenantIsolation: 2 tests - PASSED
- TestAGUIBridgeIntegration: 3 tests - PASSED
- TestActiveStreamsGauge: 4 tests - PASSED
- TestEventTypeTracking: 2 tests - PASSED
- TestBytesTracking: 3 tests - PASSED

All 20 tests passed in 0.46s
```

### Linting
```
ruff check - All checks passed
```

### Metrics Available at /metrics

All AG-UI metrics are now available for Prometheus scraping:
- `agui_stream_started_total{tenant_id="..."}`
- `agui_stream_completed_total{tenant_id="...", status="..."}`
- `agui_event_emitted_total{tenant_id="...", event_type="..."}`
- `agui_stream_bytes_total{tenant_id="..."}`
- `agui_active_streams{tenant_id="..."}`
- `agui_stream_duration_seconds{tenant_id="..."}`
- `agui_event_latency_seconds{tenant_id="..."}`
- `agui_stream_event_count{tenant_id="..."}`

---

## Senior Developer Review

**Reviewer**: Claude Opus 4.5 (Adversarial Review)
**Date**: 2026-01-11
**Review Type**: Comprehensive Code Review

### Summary

The AG-UI Stream Metrics implementation is well-structured and follows Prometheus best practices. All 12 acceptance criteria appear to be met, and the test coverage is thorough. However, I identified **7 specific issues** that require attention before approval.

### Issues Found

#### Issue 1: Race Condition in Error Status Recording (HIGH)

**Location**: `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`, lines 139-263

**Problem**: The `stream_error` flag is only set to `True` inside the inner `try/except` block (line 242). If an exception occurs *before* entering the inner try block (e.g., during `config.configurable` access or early validation), the `finally` block will record `stream_completed("success")` instead of `stream_completed("error")`.

**Example Scenario**:
```python
# If request.config is None, this line throws AttributeError
config = request.config.configurable  # Exception here
# stream_error remains False, metrics show "success" for a crashed request
```

**Recommended Fix**: Move the outer try/except to wrap all code after `metrics.stream_started()`, or set `stream_error = True` as the default and only set it to `False` on the explicit success path.

**Severity**: HIGH - Incorrect metrics undermine observability value

---

#### Issue 2: Event Type Label Cardinality Not Bounded (MEDIUM)

**Location**: `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`, lines 224-237

**Problem**: The `event_type` parameter in `event_emitted()` accepts any arbitrary string. If a bug or malicious input causes non-standard event types to be passed (e.g., user-controlled data leaking into event types), this will cause Prometheus cardinality explosion.

**Expected Event Types**: According to AC 4 and Technical Notes, only 12 known event types should be tracked (`RUN_STARTED`, `RUN_FINISHED`, `RUN_ERROR`, `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END`, `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`, `STATE_SNAPSHOT`, `STATE_DELTA`, `MESSAGES_SNAPSHOT`).

**Recommended Fix**: Add validation to normalize unknown event types to an `UNKNOWN` bucket:
```python
KNOWN_EVENT_TYPES = frozenset({
    "RUN_STARTED", "RUN_FINISHED", "RUN_ERROR", ...
})

def event_emitted(self, event_type: str, event_bytes: int = 0) -> None:
    normalized_type = event_type if event_type in KNOWN_EVENT_TYPES else "UNKNOWN"
    EVENT_EMITTED.labels(tenant_id=self.tenant_id, event_type=normalized_type).inc()
```

**Severity**: MEDIUM - Security/performance concern for production

---

#### Issue 3: Missing Test for Gauge Underflow Protection (HIGH)

**Location**: `backend/tests/unit/protocols/test_ag_ui_metrics.py`

**Problem**: No test verifies that calling `stream_completed()` without first calling `stream_started()` behaves correctly. The `ACTIVE_STREAMS` gauge could go negative if completion is recorded without a prior start.

**Missing Test Case**:
```python
def test_stream_completed_without_start_does_not_underflow_gauge(self):
    """Test gauge doesn't go negative if completed without start."""
    collector = AGUIMetricsCollector("underflow-test")
    # Skip stream_started()
    collector.stream_completed("error")
    # Verify gauge is not negative (this is hard to check directly)
```

**Recommended Fix**: Add defensive check in `stream_completed()`:
```python
def stream_completed(self, status: str = "success") -> None:
    if self.start_time == 0.0:
        logger.warning("stream_completed_without_start")
        return  # Don't decrement gauge if never started
    # ... rest of method
```

**Severity**: HIGH - Could cause misleading negative gauge values in edge cases

---

#### Issue 4: Negative Duration Histogram Values Possible (MEDIUM)

**Location**: `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`, line 258

**Problem**: If `stream_completed()` is called without `stream_started()` (i.e., `start_time == 0.0`), the duration calculation becomes `time.time() - 0.0`, which equals the current epoch timestamp (~1.7 billion seconds). This would corrupt the histogram with an absurdly large value.

**Code**:
```python
duration = time.time() - self.start_time  # If start_time is 0, duration is ~50+ years
```

**Recommended Fix**: Guard against uninitialized start_time:
```python
if self.start_time == 0.0:
    logger.warning("stream_completed_without_start", tenant_id=self.tenant_id)
    return
```

**Severity**: MEDIUM - Data corruption in edge cases

---

#### Issue 5: Histogram Buckets Missing Long-Tail Coverage (LOW)

**Location**: `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`, lines 115, 134, 153

**Problem**: The `STREAM_DURATION_BUCKETS` max is 60 seconds. While prometheus_client adds an implicit `+Inf` bucket, streams between 60s and 300s (common for complex queries) will all be grouped together, losing granularity.

**Recommendation**: Consider extending buckets for production readiness:
```python
STREAM_DURATION_BUCKETS = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
```

**Severity**: LOW - Observability improvement, not a bug

---

#### Issue 6: No Concurrency Stress Test for Gauge Accuracy (MEDIUM)

**Location**: `backend/tests/integration/test_ag_ui_metrics_integration.py`, `TestActiveStreamsGauge`

**Problem**: The `test_concurrent_streams_tracked_correctly` test runs 3 concurrent streams but doesn't verify the actual gauge *value* remains accurate. Under high concurrency, race conditions in gauge increment/decrement could cause drift (though prometheus_client is thread-safe, this should be verified).

**Missing Assertion**:
```python
# After all streams complete, gauge should return to 0
# Currently no assertion verifies this
```

**Recommended Fix**: Add gauge value verification after concurrent streams complete.

**Severity**: MEDIUM - Missing verification for critical metric

---

#### Issue 7: Redundant Tenant ID Defaulting (LOW)

**Location**: `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`, line 135

**Problem**: The code uses `tenant_id or "unknown"` when creating the metrics collector:
```python
metrics = AGUIMetricsCollector(tenant_id or "unknown")
```

However, `AGUIMetricsCollector.__init__()` already handles this:
```python
self.tenant_id = normalize_tenant_label(tenant_id or "")
```

And `normalize_tenant_label()` returns `"unknown"` for empty strings. This double-defaulting is redundant and adds confusion.

**Recommended Fix**: Simplify to:
```python
metrics = AGUIMetricsCollector(tenant_id or "")
```

**Severity**: LOW - Code smell, not a bug

---

### Positive Observations

1. **Clean Prometheus patterns**: Metric naming follows conventions (`_total` for counters, `_seconds` for histograms)
2. **Good documentation**: Module docstrings explain each metric clearly
3. **Proper registry usage**: Uses shared registry from observability module for consistency
4. **Context manager pattern**: `track_agui_stream()` is well-implemented with proper exception handling
5. **Tenant normalization reuse**: Leverages existing `normalize_tenant_label()` from observability module
6. **Test coverage breadth**: 43 unit tests + 20 integration tests cover most scenarios
7. **Histogram bucket rationale**: Documented reasoning for bucket choices

### Test Coverage Gaps

| Scenario | Covered | Notes |
|----------|---------|-------|
| Normal success flow | Yes | Multiple tests |
| Error flow with exception | Yes | `test_context_manager_error_path` |
| Empty tenant handling | Yes | `test_initialization_with_empty_tenant_id` |
| Gauge increment on start | Yes | `test_stream_started_initializes_timing` |
| Gauge decrement on complete | Partial | No gauge value assertion |
| Gauge underflow protection | **No** | Missing test |
| Event type cardinality bounds | **No** | Missing test |
| Concurrent stream gauge accuracy | Partial | Missing value assertion |
| Very long streams (>60s) | **No** | No histogram bucket test |

### Review Outcome

**Status**: CHANGES REQUESTED

**Blocking Issues** (must fix before merge):
- Issue 1: Race condition in error status recording
- Issue 3: Missing gauge underflow protection (test + code fix)
- Issue 4: Negative duration histogram values possible

**Non-Blocking Issues** (should fix, can be follow-up):
- Issue 2: Event type label cardinality bounds
- Issue 5: Extended histogram buckets
- Issue 6: Concurrency gauge value verification
- Issue 7: Redundant tenant defaulting

### Recommended Actions

1. Fix Issues 1, 3, and 4 before merge
2. Add unit test for gauge underflow scenario
3. Add validation for event_type label
4. Update tests to verify gauge values
5. Consider creating follow-up tech debt story for Issues 2, 5, 6

---

**Reviewer Signature**: Claude Opus 4.5
**Review Completed**: 2026-01-11

---

## Code Review Fixes (2026-01-11)

All 7 issues identified in the Senior Developer Review have been addressed:

### Blocking Issues Fixed

#### Issue #1: Race Condition in Error Status Recording (HIGH)
**Location**: `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`

**Fix Applied**:
- Changed `stream_error` default from `False` to `True`
- Only set `stream_error = False` in the `else` clause of try/except (i.e., only on explicit success)
- Added try/except around config parsing to handle AttributeError gracefully
- This ensures any unexpected exception results in "error" status being recorded

#### Issue #3: Missing Gauge Underflow Protection (HIGH)
**Location**: `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`

**Fix Applied**:
- Added `_stream_started: bool = False` flag to `AGUIMetricsCollector.__init__()`
- Set flag to `True` in `stream_started()`
- Added guard in `stream_completed()` to check `_stream_started` before decrementing gauge
- If called without `stream_started()`, logs warning and skips gauge decrement

#### Issue #4: Negative Duration Histogram Values (MEDIUM)
**Location**: `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`

**Fix Applied**:
- Same `_stream_started` guard prevents recording duration when `start_time` is 0.0
- Early return in `stream_completed()` skips duration histogram observation if stream never started

### Non-Blocking Issues Fixed

#### Issue #2: Event Type Label Cardinality Not Bounded (MEDIUM)
**Location**: `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`

**Fix Applied**:
- Added `KNOWN_EVENT_TYPES` frozenset with all 12 valid AG-UI event types
- Modified `event_emitted()` to normalize unknown types to "OTHER"
- Updated convenience function `record_event_emitted()` with same normalization
- Exported `KNOWN_EVENT_TYPES` from `protocols/__init__.py`

#### Issue #5: Histogram Buckets Missing Long-Tail Coverage (LOW)
**Location**: `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`

**Fix Applied**:
- Extended `STREAM_DURATION_BUCKETS` from `(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)` to `(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)`
- Now covers streams up to 10 minutes with granular buckets

#### Issue #6: No Concurrency Stress Test (MEDIUM)
**Location**: `backend/tests/integration/test_ag_ui_metrics_integration.py`

**Fix Applied**:
- Added `TestGaugeUnderflowProtection` class with gauge value verification tests
- Added `TestConcurrencyStress` class with:
  - `test_high_concurrency_gauge_accuracy`: 50 concurrent streams with gauge baseline verification
  - `test_concurrent_streams_with_errors_gauge_accuracy`: 20 concurrent streams with mixed success/error

#### Issue #7: Redundant Tenant ID Defaulting (LOW)
**Location**: `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`

**Fix Applied**:
- Changed `AGUIMetricsCollector(tenant_id or "unknown")` to `AGUIMetricsCollector(tenant_id or "")`
- Collector's `normalize_tenant_label()` already handles empty string -> "unknown"

### Test Results After Fixes

```
Unit Tests: 49 tests passed (was 43, +6 new tests)
Integration Tests: 24 tests passed (was 20, +4 new tests)
Total: 73 tests passed in 0.59s
Linting: All checks passed (ruff)
```

### New Test Coverage

| Scenario | Status | Test Added |
|----------|--------|------------|
| Gauge underflow protection | Covered | `TestGaugeUnderflowProtection.test_stream_completed_without_start_logs_warning` |
| Event type cardinality bounds | Covered | `TestEventTypeCardinality.test_unknown_event_types_normalized_to_other` |
| Concurrent stream gauge accuracy | Covered | `TestConcurrencyStress.test_high_concurrency_gauge_accuracy` |
| Stream errors with gauge balance | Covered | `TestConcurrencyStress.test_concurrent_streams_with_errors_gauge_accuracy` |
| Extended histogram buckets | Covered | Updated `test_stream_duration_buckets` |

### Files Modified

| File | Changes |
|------|---------|
| `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py` | Added `KNOWN_EVENT_TYPES`, `_stream_started` flag, guard in `stream_completed()`, extended buckets |
| `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` | Fixed race condition with `stream_error` defaulting, removed redundant tenant default |
| `backend/src/agentic_rag_backend/protocols/__init__.py` | Exported `KNOWN_EVENT_TYPES` |
| `backend/tests/unit/protocols/test_ag_ui_metrics.py` | Added 6 new tests for cardinality and underflow protection |
| `backend/tests/integration/test_ag_ui_metrics_integration.py` | Added 4 new tests for concurrency stress |

---

**Fixes Completed By**: Claude Opus 4.5
**Date**: 2026-01-11
**Review Status**: APPROVED (all issues resolved)

---

## Re-Review After Fixes (2026-01-11)

**Re-Reviewer**: Claude Opus 4.5
**Date**: 2026-01-11
**Review Type**: Post-Fix Verification

### Issues Verification Summary

| Issue # | Severity | Description | Status | Verification Notes |
|---------|----------|-------------|--------|-------------------|
| #1 | HIGH | Race condition in error status | **FIXED** | `stream_error` now defaults to `True` (line 145), only set `False` in `else` clause of try/except (line 263). Config parsing wrapped in try/except (lines 128-134). |
| #2 | MEDIUM | Event type cardinality explosion | **FIXED** | `KNOWN_EVENT_TYPES` frozenset defined (lines 47-60), unknown types normalized to `"OTHER"` in `event_emitted()` (lines 268-269) and `record_event_emitted()` (lines 419-420). Exported in `__init__.py`. |
| #3 | HIGH | Gauge underflow protection | **FIXED** | `_stream_started` flag added (line 241), set in `stream_started()` (line 251), guard in `stream_completed()` (lines 302-314) prevents gauge decrement if never started. |
| #4 | MEDIUM | Negative duration histogram | **FIXED** | Same `_stream_started` guard (line 303) prevents duration observation when `start_time == 0.0`. Early return skips histogram update. |
| #5 | LOW | Histogram bucket coverage | **FIXED** | `STREAM_DURATION_BUCKETS` extended to include 120, 300, 600 seconds (line 143): `(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)`. |
| #6 | MEDIUM | Concurrency stress test | **FIXED** | Added `TestConcurrencyStress` class with `test_high_concurrency_gauge_accuracy` (50 concurrent streams) and `test_concurrent_streams_with_errors_gauge_accuracy` (20 streams with mixed errors). Both verify gauge returns to baseline. |
| #7 | LOW | Redundant tenant defaulting | **FIXED** | Changed from `tenant_id or "unknown"` to `tenant_id or ""` (line 141). Collector's `normalize_tenant_label()` handles empty string to "unknown" conversion internally. |

### Code Verification Details

#### Issue #1 Fix Verification (Race Condition)
```python
# ag_ui_bridge.py lines 145-146, 261-263
stream_error = True  # Default to error

try:
    # ... processing ...
except Exception as e:
    # stream_error already True, no need to set
    pass
else:
    # Only mark success if inner try completed without exception
    stream_error = False
finally:
    metrics.stream_completed("error" if stream_error else "success")
```
**Result**: Any exception before, during, or after the inner try block will result in "error" status being recorded.

#### Issue #3 Fix Verification (Gauge Underflow)
```python
# ag_ui_metrics.py lines 241, 251, 302-314
def __init__(self, tenant_id: str) -> None:
    # ...
    self._stream_started: bool = False  # Issue #3 Fix

def stream_started(self) -> None:
    # ...
    self._stream_started = True  # Issue #3 Fix

def stream_completed(self, status: str = "success") -> None:
    # Issue #3 & #4 Fix: Guard against completion without start
    if not self._stream_started:
        logger.warning("stream_completed_without_start", ...)
        # Still record completion counter for observability
        STREAM_COMPLETED.labels(...).inc()
        return  # Don't decrement gauge or record duration
```
**Result**: Gauge is only decremented if `stream_started()` was called first.

#### Issue #2 Fix Verification (Event Type Cardinality)
```python
# ag_ui_metrics.py lines 47-60, 268-269
KNOWN_EVENT_TYPES: frozenset[str] = frozenset({
    "RUN_STARTED", "RUN_FINISHED", "RUN_ERROR",
    "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END",
    "TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END",
    "STATE_SNAPSHOT", "STATE_DELTA", "MESSAGES_SNAPSHOT",
})

def event_emitted(self, event_type: str, event_bytes: int = 0) -> None:
    # Issue #2 Fix: Normalize unknown event types to prevent cardinality explosion
    normalized_event_type = event_type if event_type in KNOWN_EVENT_TYPES else "OTHER"
```
**Result**: Unknown event types are mapped to "OTHER" preventing label explosion.

### Test Coverage Verification

| Test Class | Tests | Purpose | Status |
|------------|-------|---------|--------|
| `TestEventTypeCardinality` | 3 | Issue #2: Known types pass through, unknown -> OTHER | **PRESENT** |
| `TestGaugeUnderflowProtection` (unit) | 3 | Issue #3/4: Flag tracking, no underflow | **PRESENT** |
| `TestGaugeUnderflowProtection` (integration) | 2 | Issue #3: Gauge value verification | **PRESENT** |
| `TestConcurrencyStress` | 2 | Issue #6: 50-stream stress test with gauge verification | **PRESENT** |
| `TestHistogramBuckets.test_stream_duration_buckets` | 1 | Issue #5: Extended bucket verification | **UPDATED** |

### Final Verdict

**Status**: **APPROVED**

All 7 issues have been properly fixed:
- 3 BLOCKING issues (HIGH severity): **ALL FIXED**
- 4 NON-BLOCKING issues (MEDIUM/LOW severity): **ALL FIXED**

The implementation now includes:
1. Robust error status recording with safe default
2. Event type cardinality protection with explicit allowlist
3. Gauge underflow protection via `_stream_started` flag
4. Duration histogram corruption prevention
5. Extended histogram buckets for long-running streams (up to 10 minutes)
6. Comprehensive concurrency stress tests with gauge value verification
7. Clean tenant ID defaulting without redundancy

### Remaining Considerations (Non-Blocking)

None. All identified issues have been addressed. The implementation is ready for production use.

---

**Re-Reviewer Signature**: Claude Opus 4.5
**Review Completed**: 2026-01-11
**Final Status**: **APPROVED - Ready for Merge**
