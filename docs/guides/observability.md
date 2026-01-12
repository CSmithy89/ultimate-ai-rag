# Observability Guide

This guide provides comprehensive documentation for operators managing the Agentic RAG + GraphRAG platform. It covers all metrics, logging patterns, dashboard configuration, and alerting strategies.

## Table of Contents

1. [Overview](#overview)
2. [Key Metrics](#key-metrics)
   - [LLM & Cost Metrics](#llm--cost-metrics)
   - [Retrieval Metrics](#retrieval-metrics)
   - [Ingestion Metrics](#ingestion-metrics)
   - [Cache Metrics](#cache-metrics)
   - [AG-UI Stream Metrics](#ag-ui-stream-metrics)
   - [Telemetry Metrics](#telemetry-metrics)
3. [Logging](#logging)
   - [Structlog Configuration](#structlog-configuration)
   - [Correlation IDs](#correlation-ids)
   - [Trajectory Logging](#trajectory-logging)
4. [Dashboards](#dashboards)
   - [Grafana Panel Recommendations](#grafana-panel-recommendations)
   - [Dashboard JSON Templates](#dashboard-json-templates)
5. [Alert Thresholds](#alert-thresholds)
   - [Critical Alerts](#critical-alerts)
   - [Warning Alerts](#warning-alerts)
   - [Alert Configuration](#alert-configuration)
6. [Troubleshooting](#troubleshooting)
7. [Configuration Reference](#configuration-reference)

---

## Overview

The platform exposes Prometheus metrics via the `/metrics` endpoint (configurable via `PROMETHEUS_PATH`). Metrics are designed for multi-tenant environments with configurable tenant label cardinality control.

### Enabling Prometheus Metrics

```bash
# Enable metrics endpoint
PROMETHEUS_ENABLED=true
PROMETHEUS_PATH=/metrics

# Tenant label cardinality control (production recommendation)
METRICS_TENANT_LABEL_MODE=hash  # Options: full | hash | global
METRICS_TENANT_LABEL_BUCKETS=100  # Hash bucket count (when mode=hash)
```

**Cardinality Modes:**
- `full`: Use exact tenant_id (highest cardinality, not recommended for production)
- `hash`: Hash tenant_id into N buckets (balanced cardinality)
- `global`: Single "global" label (lowest cardinality, aggregated view)

---

## Key Metrics

### LLM & Cost Metrics

| Metric | Type | Labels | Description | Alert Threshold |
|--------|------|--------|-------------|-----------------|
| `llm_api_calls_total` | Counter | `model`, `operation`, `tenant_id` | Total LLM API calls | N/A |
| `llm_api_cost_total` | Counter | `model`, `tenant_id` | Cumulative LLM cost in USD | Rate > $10/hour |
| `contextual_enrichment_tokens_total` | Counter | `type`, `model`, `tenant_id` | Tokens used for contextual retrieval | Rate > 100k tokens/hour |
| `contextual_enrichment_cost_usd_total` | Counter | `model`, `tenant_id` | Contextual retrieval cost in USD | N/A |

**Operation Labels for `llm_api_calls_total`:**
- `summary`: Document summarization
- `synthesis`: Answer synthesis
- `embedding`: Embedding generation
- `chat`: Conversational interaction

**Token Type Labels:**
- `input`: Input tokens consumed
- `output`: Output tokens generated

### Retrieval Metrics

| Metric | Type | Labels | Description | Alert Threshold |
|--------|------|--------|-------------|-----------------|
| `retrieval_requests_total` | Counter | `strategy`, `tenant_id` | Total retrieval requests | N/A |
| `retrieval_latency_seconds` | Histogram | `strategy`, `phase`, `tenant_id` | Retrieval operation latency | p95 > 2s |
| `retrieval_fallback_triggered_total` | Counter | `reason`, `tenant_id` | Fallback triggers | Rate > 10/minute |
| `retrieval_precision` | Gauge | `strategy`, `k`, `tenant_id` | Precision@k score | < 0.6 |
| `retrieval_recall` | Gauge | `strategy`, `k`, `tenant_id` | Recall@k score | < 0.5 |
| `active_retrieval_operations` | Gauge | `tenant_id` | Currently active retrievals | > 50 concurrent |
| `reranking_improvement_ratio` | Histogram | `tenant_id` | Post-rerank/pre-rerank score ratio | N/A |
| `grader_evaluations_total` | Counter | `result`, `tenant_id` | Grader evaluation outcomes | N/A |
| `grader_score` | Histogram | `model`, `tenant_id` | Grader relevance scores | N/A |

**Grader Result Labels:**
- `pass`: Document passed relevance threshold
- `fail`: Document failed relevance threshold
- `fallback`: Fallback strategy triggered

**Strategy Labels:**
- `vector`: Pure vector/semantic search
- `graph`: Graph traversal search
- `hybrid`: Combined vector + graph

**Phase Labels:**
- `embed`: Query embedding generation
- `search`: Database search execution
- `rerank`: Result reranking
- `grade`: Quality grading

**Latency Buckets:** 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s

**Fallback Reason Labels:**
- `low_score`: Grader score below threshold
- `empty_results`: No results returned
- `timeout`: Operation timed out

### Ingestion Metrics

| Metric | Type | Labels | Description | Alert Threshold |
|--------|------|--------|-------------|-----------------|
| `contextual_enrichment_chunks_total` | Counter | `model`, `tenant_id` | Chunks enriched with context | N/A |
| `contextual_enrichment_latency_seconds` | Histogram | `model`, `tenant_id` | Enrichment operation latency | p95 > 5s |

### Cache Metrics

| Metric | Type | Labels | Description | Alert Threshold |
|--------|------|--------|-------------|-----------------|
| `redis_cache_hits_total` | Counter | `type`, `tenant_id` | Redis cache hits | N/A |
| `redis_cache_misses_total` | Counter | `type`, `tenant_id` | Redis cache misses | Hit ratio < 80% |
| `contextual_enrichment_cache_hits_total` | Counter | `model`, `tenant_id` | Prompt cache hits (Anthropic) | N/A |
| `contextual_enrichment_cache_misses_total` | Counter | `model`, `tenant_id` | Prompt cache misses | N/A |
| `reranker_cache_hits_total` | Counter | `tenant_id` | Reranker result cache hits | N/A |
| `reranker_cache_misses_total` | Counter | `tenant_id` | Reranker result cache misses | Hit ratio < 70% |
| `reranker_cache_size` | Gauge | - | Current reranker cache entries | > 10000 entries |

**Cache Type Labels:**
- `memory`: Memory store cache
- `other`: Other cache types

### AG-UI Stream Metrics

| Metric | Type | Labels | Description | Alert Threshold |
|--------|------|--------|-------------|-----------------|
| `agui_stream_started_total` | Counter | `tenant_id` | AG-UI streams started | N/A |
| `agui_stream_completed_total` | Counter | `tenant_id`, `status` | Stream completions | Error rate > 5% |
| `agui_event_emitted_total` | Counter | `tenant_id`, `event_type` | Events emitted by type | N/A |
| `agui_stream_bytes_total` | Counter | `tenant_id` | Total bytes streamed | N/A |
| `agui_active_streams` | Gauge | `tenant_id` | Currently active streams | > 100 concurrent |
| `agui_stream_duration_seconds` | Histogram | `tenant_id` | Stream duration | p95 > 60s |
| `agui_event_latency_seconds` | Histogram | `tenant_id` | Inter-event latency | p95 > 500ms |
| `agui_stream_event_count` | Histogram | `tenant_id` | Events per stream | N/A |

**Stream Status Labels:**
- `success`: Stream completed successfully
- `error`: Stream terminated with error

**Known Event Types (cardinality-controlled):**
- `RUN_STARTED`, `RUN_FINISHED`, `RUN_ERROR`
- `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END`
- `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`
- `STATE_SNAPSHOT`, `STATE_DELTA`, `MESSAGES_SNAPSHOT`
- `OTHER` (catch-all for unknown types)

**Duration Buckets:** 0.1s, 0.5s, 1s, 2.5s, 5s, 10s, 30s, 60s, 120s, 300s, 600s

### Telemetry Metrics

| Metric | Type | Labels | Description | Alert Threshold |
|--------|------|--------|-------------|-----------------|
| `telemetry_events_total` | Counter | `event`, `tenant_id` | Frontend telemetry events | N/A |
| `metrics_tenant_label_cardinality` | Gauge | `mode` | Unique tenant labels observed | > 500 (mode=full) |

**Allowlisted Event Types:**
- `page_view`, `search_query`, `message_sent`, `tool_call`
- `tool_result`, `button_click`, `login`
- `other` (catch-all for non-allowlisted events)

---

## Logging

### Structlog Configuration

The platform uses `structlog` for structured JSON logging. All log entries include standard context fields.

**Standard Log Fields:**
```json
{
  "timestamp": "2026-01-12T10:30:00.000000Z",
  "level": "info",
  "logger": "agentic_rag_backend.retrieval.pipeline",
  "event": "retrieval_completed",
  "tenant_id": "tenant-123",
  "request_id": "req-abc-123",
  "duration_ms": 245,
  "strategy": "hybrid",
  "result_count": 10
}
```

**Log Levels:**
- `debug`: Detailed diagnostic information (disabled in production)
- `info`: Operational events (requests, completions, state changes)
- `warning`: Recoverable issues (fallbacks triggered, cache misses)
- `error`: Errors requiring attention (database failures, API errors)
- `exception`: Errors with full stack traces

### Correlation IDs

Correlation IDs enable request tracing across services and log aggregation.

**Request ID Pattern:**
- Generated at API gateway/middleware layer
- Format: UUID v4 (`xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx`)
- Propagated via `X-Request-ID` header
- Included in all log entries as `request_id`

**Tenant ID Pattern:**
- Extracted from authentication/session context
- Validated against pattern: `^[a-zA-Z0-9_-]{1,64}$`
- Required for all authenticated endpoints
- Used for metrics labeling and log filtering

**Log Entry Example with Correlation:**
```json
{
  "timestamp": "2026-01-12T10:30:00.000000Z",
  "level": "info",
  "event": "query_processed",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "tenant_id": "acme-corp",
  "session_id": "sess-789",
  "trajectory_id": "traj-456",
  "query": "What is the project architecture?",
  "duration_ms": 1250,
  "retrieval_strategy": "hybrid",
  "sources_used": 5
}
```

### Trajectory Logging

Trajectory logging captures agent decision-making for debugging and compliance. Events are stored in PostgreSQL with optional encryption.

**Event Types:**
- `thought`: Agent reasoning/planning steps
- `action`: Tool calls and external interactions
- `observation`: Results and observations from actions

**Database Schema:**
```sql
-- trajectories table
CREATE TABLE trajectories (
  id UUID PRIMARY KEY,
  tenant_id VARCHAR NOT NULL,
  session_id VARCHAR,
  agent_type VARCHAR,
  has_error BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- trajectory_events table
CREATE TABLE trajectory_events (
  id UUID PRIMARY KEY,
  trajectory_id UUID REFERENCES trajectories(id),
  tenant_id VARCHAR NOT NULL,
  event_type VARCHAR NOT NULL,  -- thought | action | observation
  content TEXT,                  -- Optionally encrypted
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Logging Pattern:**
```python
from agentic_rag_backend.trajectory import TrajectoryLogger, EventType

# Start a trajectory
trajectory_id = await logger.start_trajectory(
    tenant_id="tenant-123",
    session_id="sess-456",
    agent_type="orchestrator"
)

# Log events
await logger.log_thought(tenant_id, trajectory_id, "Planning retrieval strategy...")
await logger.log_action(tenant_id, trajectory_id, "Calling vector_search tool")
await logger.log_observation(tenant_id, trajectory_id, "Found 10 relevant documents")

# Bulk logging (single transaction)
await logger.log_events(tenant_id, trajectory_id, [
    (EventType.THOUGHT, "Analyzing results..."),
    (EventType.ACTION, "Synthesizing answer"),
    (EventType.OBSERVATION, "Answer generated successfully"),
])
```

**Error Detection:**
Events containing "error" (case-insensitive) automatically flag the trajectory's `has_error` field.

**Encryption:**
Configure `TRACE_ENCRYPTION_KEY` (64-char hex) for AES-256 encryption of trajectory content:
```bash
TRACE_ENCRYPTION_KEY=<64-char-hex-key>
```

---

## Dashboards

### Grafana Panel Recommendations

#### Overview Dashboard

| Panel | Visualization | Query | Purpose |
|-------|--------------|-------|---------|
| Request Rate | Stat + Sparkline | `rate(retrieval_requests_total[5m])` | Current throughput |
| Error Rate | Gauge | `rate(agui_stream_completed_total{status="error"}[5m]) / rate(agui_stream_completed_total[5m])` | Stream health |
| p95 Latency | Stat | `histogram_quantile(0.95, rate(retrieval_latency_seconds_bucket[5m]))` | Performance |
| Active Streams | Gauge | `agui_active_streams` | Current load |
| LLM Cost (24h) | Stat | `increase(llm_api_cost_total[24h])` | Daily spend |

#### Retrieval Performance Dashboard

| Panel | Visualization | Query | Purpose |
|-------|--------------|-------|---------|
| Latency by Phase | Heatmap | `retrieval_latency_seconds_bucket` by `phase` | Bottleneck identification |
| Strategy Distribution | Pie Chart | `sum by(strategy) (retrieval_requests_total)` | Strategy usage |
| Fallback Rate | Time Series | `rate(retrieval_fallback_triggered_total[5m])` | Quality issues |
| Precision@K | Gauge | `retrieval_precision{k="10"}` | Retrieval quality |
| Reranking Impact | Histogram | `reranking_improvement_ratio_bucket` | Reranker effectiveness |

#### Cost Management Dashboard

| Panel | Visualization | Query | Purpose |
|-------|--------------|-------|---------|
| Cost by Model | Bar Chart | `sum by(model) (increase(llm_api_cost_total[24h]))` | Model cost breakdown |
| Token Usage | Time Series | `rate(contextual_enrichment_tokens_total[5m])` | Token consumption |
| Cache Savings | Stat | `contextual_enrichment_cache_hits_total / (cache_hits + cache_misses)` | Cache efficiency |
| Daily Trend | Time Series | `increase(llm_api_cost_total[1h])` | Cost trajectory |

#### Cache Performance Dashboard

| Panel | Visualization | Query | Purpose |
|-------|--------------|-------|---------|
| Redis Hit Ratio | Gauge | `redis_cache_hits_total / (hits + misses)` | Cache effectiveness |
| Reranker Cache Size | Gauge | `reranker_cache_size` | Memory usage |
| Cache Hit Rate | Time Series | `rate(redis_cache_hits_total[5m])` | Cache utilization |

### Dashboard JSON Templates

#### Basic Retrieval Dashboard

```json
{
  "dashboard": {
    "title": "Agentic RAG - Retrieval Overview",
    "uid": "agentic-rag-retrieval",
    "panels": [
      {
        "title": "Request Rate",
        "type": "stat",
        "gridPos": { "x": 0, "y": 0, "w": 6, "h": 4 },
        "targets": [
          {
            "expr": "sum(rate(retrieval_requests_total[5m]))",
            "legendFormat": "requests/sec"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "thresholds": {
              "steps": [
                { "color": "green", "value": null },
                { "color": "yellow", "value": 100 },
                { "color": "red", "value": 500 }
              ]
            }
          }
        }
      },
      {
        "title": "p95 Latency",
        "type": "gauge",
        "gridPos": { "x": 6, "y": 0, "w": 6, "h": 4 },
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(retrieval_latency_seconds_bucket[5m])) by (le))",
            "legendFormat": "p95"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "min": 0,
            "max": 5,
            "thresholds": {
              "steps": [
                { "color": "green", "value": null },
                { "color": "yellow", "value": 1 },
                { "color": "red", "value": 2 }
              ]
            }
          }
        }
      },
      {
        "title": "Latency by Phase",
        "type": "heatmap",
        "gridPos": { "x": 0, "y": 4, "w": 12, "h": 8 },
        "targets": [
          {
            "expr": "sum(rate(retrieval_latency_seconds_bucket[5m])) by (le, phase)",
            "format": "heatmap"
          }
        ]
      },
      {
        "title": "Fallback Events",
        "type": "timeseries",
        "gridPos": { "x": 12, "y": 0, "w": 12, "h": 8 },
        "targets": [
          {
            "expr": "sum by(reason) (rate(retrieval_fallback_triggered_total[5m]))",
            "legendFormat": "{{reason}}"
          }
        ]
      }
    ]
  }
}
```

#### Cost Monitoring Dashboard

```json
{
  "dashboard": {
    "title": "Agentic RAG - LLM Cost Monitoring",
    "uid": "agentic-rag-costs",
    "panels": [
      {
        "title": "Daily LLM Cost",
        "type": "stat",
        "gridPos": { "x": 0, "y": 0, "w": 6, "h": 4 },
        "targets": [
          {
            "expr": "sum(increase(llm_api_cost_total[24h]))",
            "legendFormat": "24h cost"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "thresholds": {
              "steps": [
                { "color": "green", "value": null },
                { "color": "yellow", "value": 50 },
                { "color": "red", "value": 100 }
              ]
            }
          }
        }
      },
      {
        "title": "Cost by Model",
        "type": "piechart",
        "gridPos": { "x": 6, "y": 0, "w": 6, "h": 8 },
        "targets": [
          {
            "expr": "sum by(model) (increase(llm_api_cost_total[24h]))",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "title": "Hourly Cost Trend",
        "type": "timeseries",
        "gridPos": { "x": 0, "y": 4, "w": 12, "h": 8 },
        "targets": [
          {
            "expr": "sum(increase(llm_api_cost_total[1h]))",
            "legendFormat": "Cost (USD)"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD"
          }
        }
      },
      {
        "title": "Token Usage Rate",
        "type": "timeseries",
        "gridPos": { "x": 12, "y": 0, "w": 12, "h": 8 },
        "targets": [
          {
            "expr": "sum by(type) (rate(contextual_enrichment_tokens_total[5m]))",
            "legendFormat": "{{type}} tokens/sec"
          }
        ]
      }
    ]
  }
}
```

---

## Alert Thresholds

### Critical Alerts

| Alert Name | Condition | Duration | Action |
|------------|-----------|----------|--------|
| HighErrorRate | `agui_stream_completed_total{status="error"}` > 10% | 5m | Page on-call |
| DatabaseUnavailable | `up{job="postgres"}` == 0 | 1m | Page on-call |
| HighLatency | p95 `retrieval_latency_seconds` > 5s | 5m | Page on-call |
| CostSpike | `increase(llm_api_cost_total[1h])` > $20 | immediate | Notify + investigate |

### Warning Alerts

| Alert Name | Condition | Duration | Action |
|------------|-----------|----------|--------|
| ElevatedFallbackRate | `retrieval_fallback_triggered_total` rate > 10/min | 10m | Investigate quality |
| LowCacheHitRate | Redis hit ratio < 70% | 15m | Review cache config |
| HighActiveStreams | `agui_active_streams` > 100 | 5m | Monitor capacity |
| PrecisionDegraded | `retrieval_precision{k="10"}` < 0.6 | 30m | Review retrieval |
| RerankerCacheFull | `reranker_cache_size` > 900 | 5m | Increase cache size |

### Alert Configuration

**Prometheus Alerting Rules (alerting_rules.yml):**

```yaml
groups:
  - name: agentic-rag-critical
    rules:
      - alert: HighStreamErrorRate
        expr: |
          sum(rate(agui_stream_completed_total{status="error"}[5m])) /
          sum(rate(agui_stream_completed_total[5m])) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High AG-UI stream error rate"
          description: "Stream error rate is {{ $value | humanizePercentage }} (threshold: 10%)"

      - alert: RetrievalLatencyHigh
        expr: |
          histogram_quantile(0.95, sum(rate(retrieval_latency_seconds_bucket[5m])) by (le)) > 5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Retrieval p95 latency exceeds 5 seconds"
          description: "p95 latency is {{ $value | humanizeDuration }}"

      - alert: LLMCostSpike
        expr: increase(llm_api_cost_total[1h]) > 20
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM cost spike detected"
          description: "Hourly cost is ${{ $value | printf \"%.2f\" }}"

  - name: agentic-rag-warnings
    rules:
      - alert: ElevatedFallbackRate
        expr: sum(rate(retrieval_fallback_triggered_total[5m])) > 0.17
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Elevated retrieval fallback rate"
          description: "Fallback rate is {{ $value | printf \"%.2f\" }}/sec"

      - alert: LowCacheHitRatio
        expr: |
          sum(rate(redis_cache_hits_total[5m])) /
          (sum(rate(redis_cache_hits_total[5m])) + sum(rate(redis_cache_misses_total[5m]))) < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Redis cache hit ratio below 70%"
          description: "Hit ratio is {{ $value | humanizePercentage }}"

      - alert: HighActiveStreams
        expr: sum(agui_active_streams) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High number of active AG-UI streams"
          description: "{{ $value }} active streams"
```

### LLM Cost Alerts (Database-Backed)

The platform also supports database-backed cost alerts via the `/api/v1/ops/costs/alerts` endpoint:

```bash
# Configure per-tenant cost alerts
curl -X POST http://localhost:8000/api/v1/ops/costs/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme-corp",
    "daily_threshold_usd": 50.00,
    "monthly_threshold_usd": 1000.00,
    "enabled": true
  }'

# Check alert status
curl "http://localhost:8000/api/v1/ops/costs/summary?tenant_id=acme-corp&window=day"
```

Response includes alert status:
```json
{
  "data": {
    "total_cost_usd": 45.50,
    "alerts": {
      "enabled": true,
      "daily_threshold_usd": 50.00,
      "monthly_threshold_usd": 1000.00,
      "daily_total_usd": 45.50,
      "monthly_total_usd": 890.00,
      "daily_exceeded": false,
      "monthly_exceeded": false
    }
  }
}
```

---

## Troubleshooting

### High Latency

1. **Check phase breakdown:**
   ```promql
   histogram_quantile(0.95, sum by(phase, le) (rate(retrieval_latency_seconds_bucket[5m])))
   ```

2. **Common causes by phase:**
   - `embed`: Slow embedding provider, network latency
   - `search`: Database performance, missing indexes
   - `rerank`: Large result sets, slow reranker model
   - `grade`: Complex queries, slow grader model

3. **Solutions:**
   - Enable reranker caching (`RERANKER_CACHE_ENABLED=true`)
   - Reduce `RERANKER_TOP_K` (default: 10)
   - Use faster grader model (`GRADER_MODEL`)

### High Fallback Rate

1. **Check fallback reasons:**
   ```promql
   sum by(reason) (rate(retrieval_fallback_triggered_total[5m]))
   ```

2. **Solutions by reason:**
   - `low_score`: Lower `GRADER_THRESHOLD` (default: 0.5)
   - `empty_results`: Check index population, query analysis
   - `timeout`: Increase timeouts, optimize queries

### Cache Miss Rate High

1. **Check Redis connectivity:**
   ```bash
   redis-cli ping
   ```

2. **Review cache configuration:**
   ```bash
   RERANKER_CACHE_ENABLED=true
   RERANKER_CACHE_TTL_SECONDS=3600
   RERANKER_CACHE_MAX_SIZE=10000
   ```

3. **Analyze cache patterns:**
   ```promql
   rate(redis_cache_misses_total[5m]) / rate(redis_cache_hits_total[5m])
   ```

### Trajectory Query Performance

1. **Slow trajectory listing:**
   - Add indexes on `tenant_id`, `created_at`, `has_error`
   - Use pagination (`limit`, `offset` parameters)

2. **Large trajectory events:**
   - Enable encryption compression
   - Archive old trajectories

### Cost Anomalies

1. **Identify high-cost models:**
   ```promql
   topk(5, sum by(model) (increase(llm_api_cost_total[24h])))
   ```

2. **Check token consumption patterns:**
   ```promql
   rate(contextual_enrichment_tokens_total[5m])
   ```

3. **Solutions:**
   - Enable intelligent model routing (`ROUTING_ENABLED=true`)
   - Adjust complexity thresholds (`ROUTING_SIMPLE_MAX_SCORE`, `ROUTING_COMPLEX_MIN_SCORE`)

---

## Configuration Reference

### Prometheus Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_ENABLED` | `false` | Enable `/metrics` endpoint |
| `PROMETHEUS_PATH` | `/metrics` | Metrics endpoint path |
| `METRICS_TENANT_LABEL_MODE` | `global` | Tenant label cardinality mode |
| `METRICS_TENANT_LABEL_BUCKETS` | `100` | Hash bucket count (mode=hash) |

### Reranker Cache Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_CACHE_ENABLED` | `false` | Enable reranker result caching |
| `RERANKER_CACHE_TTL_SECONDS` | `300` | Cache entry TTL (5 minutes) |
| `RERANKER_CACHE_MAX_SIZE` | `1000` | Maximum cache entries |

### Trajectory Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACE_ENCRYPTION_KEY` | (generated) | 64-char hex AES-256 key |
| `DB_POOL_MIN` | `1` | Minimum pool connections |
| `DB_POOL_MAX` | `50` | Maximum pool connections |

### Cost Tracking Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PRICING_JSON` | (built-in) | Custom model pricing JSON |
| `ROUTING_SIMPLE_MODEL` | `gpt-4o-mini` | Model for simple queries |
| `ROUTING_MEDIUM_MODEL` | `gpt-4o` | Model for medium queries |
| `ROUTING_COMPLEX_MODEL` | `gpt-4o` | Model for complex queries |
| `ROUTING_BASELINE_MODEL` | `gpt-4o` | Baseline for cost comparison |
| `ROUTING_SIMPLE_MAX_SCORE` | `2` | Max complexity score for simple (integer) |
| `ROUTING_COMPLEX_MIN_SCORE` | `5` | Min complexity score for complex (integer) |

---

## Related Documentation

- [Advanced Retrieval Configuration](/docs/guides/advanced-retrieval-configuration.md)
- [Voice I/O Configuration](/docs/guides/voice-io-configuration.md)
- [Protocol Integration Guide](/docs/guides/protocol-integration/)
- [MCP Wrapper Architecture](/docs/guides/mcp-wrapper-architecture.md)
