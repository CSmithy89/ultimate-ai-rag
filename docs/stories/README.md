# Story Authoring Guide

Use `docs/stories/_template.md` for all new stories.

## Standards Coverage

Every story must include a "Standards Coverage" section and mark each standard as:

- **Addressed**: Implemented in this story
- **N/A**: Not applicable (include a short rationale)
- **Planned**: Explicitly deferred with follow-up plan

Reviewers should confirm the standards coverage is filled out before approval.

## Updating Existing Stories

If a story is still in backlog or drafted status, update it to include the
Standards Coverage section before implementation begins.

## Status Synchronization

Story status must match `_bmad-output/implementation-artifacts/sprint-status.yaml`.
When you update a story status, update both files in the same change.

Validate with:

```bash
python3 scripts/validate-story-status.py
```

## Completion Requirements

When marking a story as done, fill out:

- Dev Notes
- Dev Agent Record (Agent Model, Debug Log, Completion Notes, File List)
- Test Outcomes
- Challenges Encountered (or explicitly note none)
