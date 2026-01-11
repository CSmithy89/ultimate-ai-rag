# Story 9.1: Retrospective Action Item Tracking System

Status: done

## Story

As a Scrum Master,  
I want a system to track retrospective action items,  
So that commitments from retrospectives are actually executed.

## Acceptance Criteria

1. Given a retrospective is completed, when action items are defined, then each item is added to project tracker with owner and due date.
2. Given action items exist, when the next retrospective runs, then completion status of previous items is reviewed first.
3. Given an action item is overdue, when standup runs, then it is surfaced automatically.

## Tasks / Subtasks

- [x] Define tracking mechanism and storage format (AC: 1)
  - [x] Create `docs/retrospectives/action-items.yaml` with required fields
  - [x] Document the schema and update workflow notes

- [x] Add action item template and retro section (AC: 1, 2)
  - [x] Add "Previous Retro Follow-Through" section to retrospective template
  - [x] Provide action item entry template (owner, due date, success criteria)

- [x] Add overdue surfacing workflow (AC: 3)
  - [x] Add a script to list overdue action items
  - [x] Document standup usage (run script before standup)

- [x] Document process (AC: 1-3)
  - [x] Update README or docs with retro tracking process

## Technical Notes

Suggested YAML schema:

```yaml
action_items:
  - id: retro-2025-01-07-01
    title: "Add PR checklist"
    owner: "name"
    due_date: "2025-01-14"
    status: "open|in-progress|done|blocked"
    success_criteria: "Checklist added to PR template"
    source_retro: "epic-8-retrospective"
```

## Definition of Done

- [x] Action items are tracked in a structured file with required fields
- [x] Retrospective template includes prior action item review section
- [x] Overdue items can be surfaced via script
- [x] Process documented for team use

## Dev Notes

- Use file-based tracking in `docs/retrospectives` to keep the process auditable.
- The overdue check script parses a constrained YAML schema without external dependencies.
- Standup workflow calls the script to surface overdue items.

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added retro action item registry with required fields and status guidance.
- Created a retrospective template with a mandatory follow-through section.
- Implemented a lightweight overdue check script and documented standup usage.

### File List

- docs/retrospectives/action-items.yaml
- docs/retrospectives/retrospective-template.md
- docs/retrospectives/README.md
- scripts/check-retro-action-items.py

## Senior Developer Review

Outcome: APPROVE

Notes:
- File-based tracking and template updates meet ACs without external dependencies.
- Overdue check script is clear and avoids extra tooling requirements.
