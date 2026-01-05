# Story 19-I5: Optimize BeautifulSoup async parsing

Status: done

## Story

As a developer running large crawls,
I want HTML parsing to avoid blocking the event loop for large documents,
so that crawl throughput and latency improve for big pages.

## Acceptance Criteria

1. Documents >1MB use async-compatible parsing (offloaded from event loop).
2. Behavior remains backward compatible for small documents.
3. Logging indicates when async parsing is used.

## Tasks / Subtasks

- [ ] Add async parsing path for large HTML payloads.
- [ ] Update crawler conversion flow to await parsing results.
- [ ] Document threshold and behavior.

## Dev Notes

- Update `backend/src/agentic_rag_backend/indexing/crawler.py`.
- Use `asyncio` executor for large HTML parsing.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Group I: 19-I5)
- `backend/src/agentic_rag_backend/indexing/crawler.py`

## Dev Agent Record

### Agent Model Used

GPT-5

### Completion Notes List

 - Added async parsing path for HTML payloads >= 1MB using asyncio.to_thread.
 - Logged when async parsing is enabled and documented the threshold.

### File List

 - `backend/src/agentic_rag_backend/indexing/crawler.py`
 - `docs/guides/crawler-html-parsing.md`

## Senior Developer Review

Outcome: APPROVE

- Async path is limited to large payloads and keeps existing behavior for small pages.
- Logging provides visibility into when offload occurs.
