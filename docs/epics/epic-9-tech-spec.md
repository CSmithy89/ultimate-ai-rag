# Epic 9 Tech Spec: Process & Quality Foundation

**Version:** 1.0  
**Created:** 2025-12-31  
**Status:** Ready for Implementation

---

## Overview

Epic 9 hardens delivery quality by making process requirements explicit, enforceable, and auditable.
It introduces action item tracking for retrospectives, standardizes story templates, adds pre-review and
protocol compliance checklists, and prevents story status drift between files and sprint tracking.

### Business Value

- Reduce review cycles by enforcing pre-review checks and protocol compliance.
- Ensure retrospective commitments are tracked and completed.
- Make story completion evidence-based with dev notes and test outcomes.
- Prevent status inconsistencies that cause planning and reporting errors.

### Functional Requirements Covered

| FR | Description | Story |
|----|-------------|-------|
| FR30 | Retrospective action item tracking | 9-1 |
| FR31 | Standards-aware story templates | 9-2 |
| FR32 | Pre-review quality checklist | 9-3 |
| FR33 | Protocol compliance checklist | 9-4 |
| FR34 | Story status synchronization | 9-5 |
| FR35 | Dev notes & test outcomes required | 9-6 |

### NFRs Addressed

| NFR | Requirement | Implementation |
|-----|-------------|----------------|
| NFR1 | Auditability | File-based tracking + CI validation for status drift |
| NFR2 | Consistency | Standard templates and checklist-driven reviews |

---

## Architecture Decisions

### 1. File-Based Retrospective Action Item Registry

**Decision:** Track retrospective action items in versioned files under `docs/retrospectives/`,
using a structured YAML or Markdown table format with required fields (owner, due date, success criteria, status).

**Rationale:** Aligns with existing file-system tracking, keeps the workflow local and auditable, and avoids
external dependencies or API access.

### 2. Single Source of Truth for Story Status

**Decision:** Treat `_bmad-output/implementation-artifacts/sprint-status.yaml` as the source of truth,
but enforce parity in story files via a validation script.

**Rationale:** The tracking file is already used by workflows; enforcing parity eliminates inconsistent
reporting without changing current tooling.

### 3. Checklist-Driven Quality Gates

**Decision:** Add a PR template with a required quality checklist and a protocol compliance checklist
section for API changes.

**Rationale:** Embedding checks at PR creation time improves early detection of issues and standardizes
review expectations.

---

## Component Changes

### New/Updated Documents

| Path | Purpose |
|------|---------|
| `docs/retrospectives/action-items.yaml` | Registry of retrospective action items |
| `docs/quality/protocol-compliance-checklist.md` | Protocol compliance checklist |
| `docs/quality/pre-review-checklist.md` | Pre-review checklist (source of truth) |
| `docs/stories/_template.md` | Story template with Standards Coverage + Dev Notes |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR checklist enforcement |

### New Tooling

| Path | Purpose |
|------|---------|
| `scripts/validate-story-status.ts` (or `.py`) | Detect status mismatches between story files and sprint-status |

### Modified Files

| Path | Change |
|------|--------|
| `docs/stories/*` | Apply updated template fields for new stories |
| CI config (if applicable) | Add status validation to CI or pre-commit |

---

## API Contracts

No API changes in this epic.

---

## Story Breakdown

1. **9-1 Retrospective Action Item Tracking**
   - Add structured action item registry and update retro workflow docs.

2. **9-2 Story Template Standards Section**
   - Introduce Standards Coverage in template and documentation.

3. **9-3 Pre-Review Quality Checklist**
   - Add PR template and checklist source doc.

4. **9-4 Protocol Compliance Checklist**
   - Publish checklist and integrate into PR workflow.

5. **9-5 Story Status Synchronization**
   - Add validation script and fix existing discrepancies.

6. **9-6 Story Completion Dev Notes Requirement**
   - Add Dev Notes, Test Outcomes, Challenges to template.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Checklist fatigue | Medium | Keep lists concise and avoid duplication |
| Enforcement gaps | Medium | Add CI validation and pre-commit hooks where possible |
| Retro tracking ignored | Medium | Require review of prior action items in retrospectives |

---

## Deployment Notes

No infrastructure changes. New scripts and templates should be documented in README or developer docs.

---

## Testing Strategy

- Add unit tests for status validation (if script is in TS/JS).
- Verify checklist files and template usage in story creation.
- Manual validation: introduce a status mismatch and ensure detection.

---

## Out of Scope

- Automated metrics dashboards for process compliance
- External issue tracker integration
