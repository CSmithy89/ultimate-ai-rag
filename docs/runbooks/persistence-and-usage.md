# Persistence + Usage Runbook

## HITL Checkpoints
- Stored in Redis when available.
- Keys use the `hitl:checkpoint:<checkpoint_id>` prefix with tenant list keys `hitl:tenant:<tenant_id>`.
- TTL is enforced via `HITL_CHECKPOINT_TTL_SECONDS` (defaults set in app settings).

## Workspace Persistence
- Stored in Postgres tables:
  - `workspace_items` (save/load)
  - `workspace_shares` (share links)
  - `workspace_bookmarks` (bookmarks)
- Share links expire after `SHARE_LINK_TTL_HOURS` (default 24 hours).
- Content size is enforced by byte length to avoid payload abuse.

## A2A Session Persistence
- Sessions are cached in memory and persisted to Redis when available.
- Redis key prefix: `a2a:sessions:<session_id>`
- TTL is controlled by `A2A_SESSION_TTL_SECONDS`.
- Cleanup is run every `A2A_CLEANUP_INTERVAL_SECONDS`.

## Embedding Usage Tracking
- Embedding usage is recorded through `CostTracker`.
- Usage events include tenant IDs and are exposed via existing ops endpoints.

## Operational Notes
- When Redis is unavailable, persistence falls back to in-memory behavior with warnings logged.
- Verify TTL settings align with expected session and checkpoint lifetimes.
