# Crawl Profile Defaults

This document explains the default values used in crawl profiles and why they were chosen.

## Defaults and Rationale

### fast

- `max_concurrent=10`: balances throughput with common documentation-site limits.
- `rate_limit=5.0`: stays under typical 10 requests/sec per-host limits.
- `wait_timeout=5.0`: short wait to avoid slowing down static sites.

### thorough

- `max_concurrent=5`: reduces load for JS-heavy pages and avoids resource spikes.
- `rate_limit=2.0`: slower rate for stability while waiting on render.
- `wait_timeout=15.0`: allows SPA hydration before capture.

### stealth

- `max_concurrent=3`: conservative parallelism for bot-protected targets.
- `rate_limit=0.5`: slows requests to reduce detection risk.
- `wait_timeout=30.0`: extra buffer for sites with heavy defenses.

## Where Defaults Live

Defaults are defined in:
- `backend/src/agentic_rag_backend/indexing/crawl_profiles.py`

## Tuning Guidance

- Increase `rate_limit` only if target sites permit higher throughput.
- Increase `max_concurrent` when CPU/memory capacity allows and target sites are stable.
- Adjust `wait_timeout` for SPAs that render late or require additional resources.
