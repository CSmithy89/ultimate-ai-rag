# Story 22-B1: Implement AG-UI Stream Metrics

Status: drafted

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

(To be filled in during implementation)

## Test Outcomes

(To be filled in after tests pass)
