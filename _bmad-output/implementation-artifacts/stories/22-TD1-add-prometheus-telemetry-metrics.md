# Story 22-TD1: Add Prometheus Metrics to Telemetry Endpoint

Status: backlog

Epic: 22 - Advanced Protocol Integration
Priority: P0 - HIGH (Pre-Epic 22 Blocker)
Story Points: 2
Owner: Backend
Origin: Epic 21 Retrospective (TD-21-4)

## Story

As a **platform operator**,
I want **the telemetry endpoint to emit Prometheus metrics**,
So that **I can monitor CopilotKit usage in Grafana dashboards and set up alerting**.

## Background

Story 21-B1 (Configure Observability Hooks) was marked complete, but code review identified that the telemetry endpoint (`/api/telemetry`) only logs to structlog - it doesn't increment Prometheus counters as specified in the original story context.

This is a gap that should be addressed before Epic 22 starts, as Epic 22 adds more AG-UI telemetry features.

### Current State

```python
# backend/src/agentic_rag_backend/api/routes/telemetry.py
@router.post("/telemetry")
async def receive_telemetry(payload: TelemetryPayload, ...):
    logger.info("copilot_telemetry", event=payload.event, ...)  # Logs only
    return {"status": "accepted"}
```

### Expected State

```python
from prometheus_client import Counter

TELEMETRY_EVENTS = Counter(
    "copilotkit_telemetry_events_total",
    "CopilotKit telemetry events by type",
    labelnames=["event_type", "tenant_id"],
)

@router.post("/telemetry")
async def receive_telemetry(payload: TelemetryPayload, ...):
    logger.info("copilot_telemetry", event=payload.event, ...)
    TELEMETRY_EVENTS.labels(
        event_type=payload.event,
        tenant_id=tenant_id
    ).inc()
    return {"status": "accepted"}
```

## Acceptance Criteria

1. **Given** the telemetry endpoint receives an event, **when** it processes the payload, **then** a Prometheus counter `copilotkit_telemetry_events_total` is incremented.

2. **Given** the counter is incremented, **when** labels are applied, **then** `event_type` and `tenant_id` labels are populated correctly.

3. **Given** high-cardinality risk, **when** event_type is set, **then** it is validated against an allowlist of known event types (or normalized to "unknown").

4. **Given** the metrics are exposed, **when** scraping `/metrics`, **then** the new counter appears with correct labels.

5. **Given** tenant_id may be sensitive, **when** labeling metrics, **then** tenant_id is normalized/hashed if required by privacy policy.

## Tasks

- [ ] **Task 1: Add Prometheus Counter** (AC: 1, 4)
  - [ ] Import Counter from prometheus_client
  - [ ] Define `COPILOTKIT_TELEMETRY_EVENTS` counter with appropriate labels
  - [ ] Register with existing metrics registry

- [ ] **Task 2: Validate Event Types** (AC: 3)
  - [ ] Create allowlist of known CopilotKit event types
  - [ ] Normalize unknown events to "other" to prevent cardinality explosion
  - [ ] Log warning for unknown event types

- [ ] **Task 3: Normalize Tenant Labels** (AC: 5)
  - [ ] Use existing `normalize_tenant_label()` utility if available
  - [ ] Hash or truncate tenant_id if needed for privacy

- [ ] **Task 4: Wire to Endpoint** (AC: 1, 2)
  - [ ] Increment counter after logging
  - [ ] Apply labels correctly

- [ ] **Task 5: Add Tests** (AC: 1, 2, 3)
  - [ ] Test counter increment on valid event
  - [ ] Test label values are correct
  - [ ] Test unknown event type normalization

## Technical Notes

### Known CopilotKit Event Types

Based on CopilotKit observability hooks:
- `onStart` - Chat started
- `onStop` - Chat stopped
- `onResponse` - Response received
- `onError` - Error occurred
- `onToolCall` - Tool called
- `onStateChange` - State changed

### Cardinality Concern

The code review (PR #20) flagged that `event_type` populated from arbitrary input creates high-cardinality risk. Solution: validate against allowlist.

## Definition of Done

- [ ] Prometheus counter defined and wired to telemetry endpoint
- [ ] Event types validated against allowlist
- [ ] Tests pass
- [ ] `/metrics` endpoint shows new counter
- [ ] Code review approved

## Files to Modify

1. `backend/src/agentic_rag_backend/api/routes/telemetry.py` - Add counter
2. `backend/tests/api/test_telemetry.py` - Add tests

## Dependencies

- prometheus_client (already installed)
- Existing metrics registry from `observability/metrics.py`
