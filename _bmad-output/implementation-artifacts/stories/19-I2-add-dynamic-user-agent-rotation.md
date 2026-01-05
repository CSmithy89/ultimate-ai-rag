# Story 19-I2: Add dynamic user-agent rotation

Status: done

## Story

As a developer crawling web content,
I want configurable user-agent rotation strategies,
so that crawls avoid bot detection and are easier to tune per environment.

## Acceptance Criteria

1. User-agent strategy is configurable via `CRAWLER_USER_AGENT_STRATEGY` (rotate|static|random), default rotate.
2. Custom user-agent list can be provided via `CRAWLER_USER_AGENT_LIST_PATH`.
3. Default rotation list contains at least 10 realistic user agents.
4. Selected user-agent is logged for each crawl session.
5. Optional fake-useragent integration is supported (fallback if unavailable).

## Tasks / Subtasks

- [x] Add a default user-agent list file (>=10 entries).
- [x] Implement strategy selection (rotate/static/random).
- [x] Log selected user-agent per crawl session.
- [x] Wire new config/env vars and document usage.
- [x] Update tests or add coverage for strategy parsing.

## Dev Notes

- Primary change in `backend/src/agentic_rag_backend/indexing/crawler.py`.
- Config/env parsing should remain backward compatible with `CRAWL4AI_USER_AGENT`.
- Use a deterministic rotation order for `rotate` strategy.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Group I: 19-I2)
- `backend/src/agentic_rag_backend/indexing/crawler.py`

## Dev Agent Record

### Agent Model Used

GPT-5

### Completion Notes List

- Added configurable user-agent strategy with rotate/random/static modes.
- Added default user-agent list (10+ entries) and optional fake-useragent support.
- Logged selected user-agent per crawl session and documented configuration.

### File List

- `backend/src/agentic_rag_backend/indexing/crawler.py`
- `config/user-agents.txt`
- `.env.example`
- `docs/guides/crawler-user-agent-rotation.md`

## Senior Developer Review

Outcome: APPROVE

- Strategy parsing and selection are clear with safe fallbacks.
- Default UA list meets the minimum size requirement.
- Logging covers strategy, source, and selected UA per crawl session.
