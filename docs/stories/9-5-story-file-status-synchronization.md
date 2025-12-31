# Story 9.5: Story File Status Synchronization

Status: done

## Story

As a Scrum Master,  
I want story file status to always match sprint-status.yaml,  
So that project state is reliable and unambiguous.

## Acceptance Criteria

1. Given a story is marked done in sprint-status.yaml, when the story file is checked, then Status field shows "done".
2. Given a story transitions status, when the change is made, then both files are updated atomically.
3. Given a discrepancy exists, when detected, then it is flagged for resolution.

## Tasks / Subtasks

- [x] Document story status update process (AC: 2)
- [x] Create script to validate story file vs sprint-status.yaml alignment (AC: 3)
- [x] Add validation to CI pipeline or pre-commit hook (AC: 3)
- [x] Fix existing discrepancies (5-2 through 5-6, 4-4) (AC: 1)

## Technical Notes

Validation should read `_bmad-output/implementation-artifacts/sprint-status.yaml`
and compare to `docs/stories/*.md` status lines.

## Definition of Done

- [x] Validation script flags status mismatches
- [x] CI or pre-commit runs validation
- [x] Known mismatches resolved

## Dev Notes

- Added a lightweight status validation script that compares sprint tracking with story files.
- CI now runs the validation to prevent future drift.
- Fixed known mismatches in Epic 4 and 5 story files.

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Created `scripts/validate-story-status.py` for status alignment checks.
- Documented the status update process in `docs/stories/README.md`.
- Updated CI to run the validation step.
- Fixed status lines for 4-4 and 5-2 through 5-6 story files.

### File List

- scripts/validate-story-status.py
- docs/stories/README.md
- .github/workflows/ci-backend.yml
- docs/stories/4-4-knowledge-graph-visualization.md
- docs/stories/5-2-episode-ingestion-pipeline.md
- docs/stories/5-3-hybrid-retrieval-integration.md
- docs/stories/5-4-temporal-query-capabilities.md
- docs/stories/5-5-legacy-code-removal-migration.md
- docs/stories/5-6-test-suite-adaptation.md

## Senior Developer Review

Outcome: APPROVE

Notes:
- Script provides clear mismatch reporting and avoids false positives on templates.
- CI integration ensures status drift is caught early.
