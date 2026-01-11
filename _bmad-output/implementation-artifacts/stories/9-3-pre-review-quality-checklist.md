# Story 9.3: Pre-Review Quality Checklist

Status: done

## Story

As a developer,  
I want a pre-review checklist to run before submitting PRs,  
So that common issues are caught before code review.

## Acceptance Criteria

1. Given a PR is created, when the developer reviews the checklist, then they confirm all items are addressed.
2. Given the checklist exists, when PR template is used, then checklist is embedded.
3. Given checklist is used, when review starts, then review rounds decrease by 50%.

## Tasks / Subtasks

- [x] Create `.github/PULL_REQUEST_TEMPLATE.md` with quality checklist (AC: 1, 2)
  - [x] Include items: tests pass, lint clean, types check, tenant isolation, error handling, docs updated
  - [x] Add protocol compliance section for API routes

- [x] Document expected workflow (AC: 1, 3)
  - [x] Add checklist source doc in docs/quality
  - [x] Describe reviewer expectations

## Technical Notes

Checklist should be concise and oriented around common review misses.

## Definition of Done

- [x] PR template includes checklist
- [x] Checklist doc created

## Dev Notes

- PR template embeds checklist and protocol compliance prompts.
- Added a checklist source doc for guidance and review expectations.

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added `.github/PULL_REQUEST_TEMPLATE.md` with quality checklist.
- Documented checklist expectations in `docs/quality/pre-review-checklist.md`.

### File List

- .github/PULL_REQUEST_TEMPLATE.md
- docs/quality/pre-review-checklist.md

## Senior Developer Review

Outcome: APPROVE

Notes:
- Checklist items align with common review misses and are concise.
- Protocol compliance prompt is clearly scoped to API changes.
