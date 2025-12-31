# Story 9.6: Story Completion Dev Notes Requirement

Status: done

## Story

As a developer,  
I want story files to capture dev notes and test outcomes,  
So that retrospectives are evidence-based.

## Acceptance Criteria

1. Given a story is implemented, when completion is marked, then Dev Notes section is filled.
2. Given tests were run, when story is closed, then test outcomes are documented.
3. Given challenges occurred, when story is closed, then challenges are captured.

## Tasks / Subtasks

- [x] Add Dev Notes section to story template (AC: 1)
  - [x] Include Agent Model, Debug Log, Completion Notes, File List
- [x] Add Test Outcomes section (AC: 2)
  - [x] Capture tests run, coverage, failures
- [x] Add Challenges Encountered section (AC: 3)
- [x] Document as mandatory for story completion (AC: 1-3)

## Technical Notes

Template updates should align with existing story file patterns.

## Definition of Done

- [x] Story template includes Dev Notes and Test Outcomes sections
- [x] Completion requirements documented

## Dev Notes

- Extended the story template to capture test outcomes and challenges.
- Updated story authoring guide to require these sections at completion.

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added Test Outcomes and Challenges sections to `docs/stories/_template.md`.
- Documented completion requirements in `docs/stories/README.md`.

### File List

- docs/stories/_template.md
- docs/stories/README.md

## Test Outcomes

- Tests run: Not run (documentation-only changes)
- Coverage: N/A
- Failures: None

## Challenges Encountered

- None

## Senior Developer Review

Outcome: APPROVE

Notes:
- Template updates are clear and align with existing story patterns.
- Completion requirements are documented and actionable.
