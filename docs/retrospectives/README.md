# Retrospective Process

This folder contains retrospective templates and action item tracking.

## Action Item Tracking

Action items are stored in `docs/retrospectives/action-items.yaml`.
Required fields per item:

- `id`: Unique ID, e.g. `retro-2025-01-07-01`
- `title`: Short action item description
- `owner`: Responsible person
- `due_date`: `YYYY-MM-DD`
- `status`: `open | in-progress | done | blocked`
- `success_criteria`: What success looks like
- `source_retro`: Retrospective identifier

## Standup Overdue Check

Before standup, run:

```bash
python3 scripts/check-retro-action-items.py
```

Use `--fail-on-overdue` if you want the script to return a non-zero exit code
when overdue items exist.

## Retrospective Template

Use `docs/retrospectives/retrospective-template.md` for new retrospectives.
The template includes a required "Previous Retro Follow-Through" section to
review action items before creating new ones.
