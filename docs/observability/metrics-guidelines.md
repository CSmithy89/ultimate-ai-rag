# Metrics Design Guidelines

This document defines guardrails for Prometheus metrics to keep cardinality under control and avoid performance issues.

## Cardinality Budget

- Prefer low-cardinality labels (small, bounded sets).
- Avoid labels with unbounded values (user IDs, raw URLs, free-form strings).
- Keep label combinations under control; be cautious with multiple labels on a single metric.

## Approved Label Patterns

- **strategy**: Small fixed set (e.g., `vector`, `graph`, `hybrid`).
- **phase**: Small fixed set (e.g., `embed`, `search`, `rerank`, `grade`).
- **result**: Small fixed set (e.g., `pass`, `fail`, `fallback`).
- **tenant_id**: Use normalized labels (see below).

## Tenant Label Normalization

Tenant labels can explode cardinality. Use the configured normalization mode:

- `METRICS_TENANT_LABEL_MODE=global` (default): label all metrics as `tenant_id="global"`.
- `METRICS_TENANT_LABEL_MODE=hash`: bucket tenant IDs into a bounded number of hashes.
- `METRICS_TENANT_LABEL_MODE=full`: emit full tenant IDs (use only in controlled environments).

If `hash` is enabled, use `METRICS_TENANT_LABEL_BUCKETS` to cap the number of buckets.

## Do / Don't

**Do:**
- Use small enumerations for labels.
- Document any new label values in the relevant guide or story notes.
- Prefer counters and histograms over high-cardinality gauges.

**Don't:**
- Add labels for request IDs, URLs, queries, or user input.
- Introduce new labels without reviewing cardinality impact.

## Review Checklist (Metrics Changes)

- Labels are bounded and documented.
- Tenant labels use normalization mode.
- No raw user input is emitted as labels.
- Metrics naming follows existing conventions.
