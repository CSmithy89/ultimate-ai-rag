# Persistence + Usage Runbook

## HITL Checkpoints
- Stored in Redis when available.
- Keys use the `hitl:checkpoint:<checkpoint_id>` prefix with tenant list keys `hitl:tenant:<tenant_id>`.
- TTL is enforced via `HITL_CHECKPOINT_TTL_SECONDS` (defaults set in app settings).

### HITL API Endpoints
- `POST /api/v1/copilot/validation-response`
  - Headers: `X-Tenant-ID` (optional, required for tenant enforcement)
  - Body: `{ "checkpoint_id": "<uuid4>", "approved_source_ids": ["..."] }`
- `GET /api/v1/copilot/hitl/checkpoints/{checkpoint_id}`
  - Headers: `X-Tenant-ID` (required when checkpoint has a tenant)
- `GET /api/v1/copilot/hitl/checkpoints?limit=20`
  - Headers: `X-Tenant-ID` (required)

Example:
```bash
curl -X POST https://api.example.com/api/v1/copilot/validation-response \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: <tenant-uuid>" \
  -d '{"checkpoint_id":"<uuid4>","approved_source_ids":["source-1"]}'
```

## Workspace Persistence
- Stored in Postgres tables:
  - `workspace_items` (save/load)
  - `workspace_shares` (share links)
  - `workspace_bookmarks` (bookmarks)
- Share links expire after `SHARE_LINK_TTL_HOURS` (default 24 hours).
- Content size is enforced by byte length to avoid payload abuse.
- Share secret rotation requires an app restart to refresh cached settings.

## A2A Session Persistence
- Sessions are cached in memory and persisted to Redis when available.
- Redis key prefix: `a2a:sessions:<session_id>`
- TTL is controlled by `A2A_SESSION_TTL_SECONDS`.
- Cleanup is run every `A2A_CLEANUP_INTERVAL_SECONDS`.
- Sessions with corrupted timestamps are discarded on load to avoid reviving expired data.

### A2A Recovery Notes
- If Redis is unavailable at startup, sessions remain in memory only and are not restored.
- When Redis reconnects, new sessions will persist; previously missing sessions are not rehydrated.
- If session payloads are corrupted, they are skipped and must be recreated by clients.

## Embedding Usage Tracking
- Embedding usage is recorded through `CostTracker`.
- Usage events include tenant IDs and are exposed via existing ops endpoints.

### Embedding Usage Endpoints
- `GET /api/v1/ops/costs/summary?tenant_id=<tenant-uuid>&window=24h`
- `GET /api/v1/ops/costs/events?tenant_id=<tenant-uuid>&limit=50`
- `GET /api/v1/ops/costs/alerts?tenant_id=<tenant-uuid>`

## Operational Notes
- When Redis is unavailable, persistence falls back to in-memory behavior with warnings logged.
- Verify TTL settings align with expected session and checkpoint lifetimes.
