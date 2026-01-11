# Epic 9: Process & Quality Foundation

**Status:** Ready for Development
**Priority:** Critical
**Estimated Effort:** 1 Sprint
**Created:** 2025-12-31
**Source:** Tech Debt Audit (Epics 1-8)

---

## Overview

This epic addresses the systemic process issues identified across 8 retrospectives. The same problems recur because we lack enforcement mechanisms and tracking. This epic establishes the foundation to prevent future debt accumulation.

### Business Value

- Reduce code review rounds from 7 to ≤2
- Ensure 100% retrospective action item completion
- Eliminate story status discrepancies
- Prevent late-cycle quality fixes

### Success Criteria

1. Retrospective action items tracked with 90% completion rate
2. All new stories use updated template with standards section
3. Pre-review checklist reduces review rounds by 50%
4. Story file status matches sprint-status.yaml 100%

---

## Stories

### Story 9.1: Retrospective Action Item Tracking System

**As a** Scrum Master,
**I want** a system to track retrospective action items,
**So that** commitments from retrospectives are actually executed.

**Acceptance Criteria:**
1. Given a retrospective is completed, when action items are defined, then each item is added to project tracker with owner and due date
2. Given action items exist, when the next retrospective runs, then completion status of previous items is reviewed first
3. Given an action item is overdue, when standup runs, then it is surfaced automatically

**Tasks:**
- [ ] Define tracking mechanism (GitHub Issues, project board, or docs)
- [ ] Create action item template with required fields (owner, due date, success criteria)
- [ ] Add "Previous Retro Follow-Through" as mandatory retrospective section
- [ ] Document the tracking process

**Story Points:** 3

---

### Story 9.2: Story Template Standards Section

**As a** Developer,
**I want** story templates to include a standards coverage section,
**So that** I know which quality standards apply before implementation.

**Acceptance Criteria:**
1. Given a new story is created, when the template is used, then it includes a "Standards Coverage" section
2. Given standards are listed, when the story is implemented, then each standard is explicitly addressed or marked N/A
3. Given the section exists, when code review runs, then reviewers check standards completion

**Tasks:**
- [ ] Update story template in `_bmad-output/implementation-artifacts/stories/` with Standards Coverage section
- [ ] Define standard checklist items (multi-tenancy, rate limiting, validation, tests, error handling)
- [ ] Add template usage documentation
- [ ] Migrate existing backlog story templates

**Story Points:** 2

---

### Story 9.3: Pre-Review Quality Checklist

**As a** Developer,
**I want** a pre-review checklist to run before submitting PRs,
**So that** common issues are caught before code review.

**Acceptance Criteria:**
1. Given a PR is created, when the developer reviews the checklist, then they confirm all items are addressed
2. Given the checklist exists, when PR template is used, then checklist is embedded
3. Given checklist is used, when review starts, then review rounds decrease by 50%

**Tasks:**
- [ ] Create `.github/PULL_REQUEST_TEMPLATE.md` with quality checklist
- [ ] Include items: tests pass, lint clean, types check, tenant isolation, error handling, docs updated
- [ ] Add protocol compliance section for API routes
- [ ] Document expected workflow

**Story Points:** 2

---

### Story 9.4: Protocol Compliance Checklist

**As a** Developer,
**I want** a protocol compliance checklist for API routes,
**So that** every endpoint meets security and quality standards.

**Acceptance Criteria:**
1. Given a new API route is created, when the checklist is applied, then rate limiting is configured
2. Given a route handles data, when the checklist is applied, then tenant_id filtering is enforced
3. Given a route returns errors, when the checklist is applied, then RFC 7807 format is used
4. Given a route accepts input, when the checklist is applied, then Pydantic validation is configured

**Tasks:**
- [ ] Create protocol compliance checklist document
- [ ] Include: rate limiting, tenant isolation, Pydantic validation, RFC 7807 errors, timeout handling
- [ ] Add to PR template as required section for API changes
- [ ] Create automated linting rule if feasible

**Story Points:** 3

---

### Story 9.5: Story File Status Synchronization

**As a** Scrum Master,
**I want** story file status to always match sprint-status.yaml,
**So that** project state is reliable and unambiguous.

**Acceptance Criteria:**
1. Given a story is marked done in sprint-status.yaml, when the story file is checked, then Status field shows "done"
2. Given a story transitions status, when the change is made, then both files are updated atomically
3. Given a discrepancy exists, when detected, then it is flagged for resolution

**Tasks:**
- [ ] Document story status update process
- [ ] Create script to validate story file vs sprint-status.yaml alignment
- [ ] Add validation to CI pipeline or pre-commit hook
- [ ] Fix existing discrepancies (5-2 through 5-6, 4-4)

**Story Points:** 3

---

### Story 9.6: Story Completion Dev Notes Requirement

**As a** Developer,
**I want** story files to capture dev notes and test outcomes,
**So that** retrospectives are evidence-based.

**Acceptance Criteria:**
1. Given a story is implemented, when completion is marked, then Dev Notes section is filled
2. Given tests were run, when story is closed, then test outcomes are documented
3. Given challenges occurred, when story is closed, then challenges are captured

**Tasks:**
- [ ] Add Dev Notes section to story template (Agent Model, Debug Log, Completion Notes, File List)
- [ ] Add Test Outcomes section (tests run, coverage, failures)
- [ ] Add Challenges Encountered section
- [ ] Document as mandatory for story completion

**Story Points:** 2

---

## Dependencies

- None (foundational epic)

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Team resistance to new process | Medium | Automate where possible, demonstrate value |
| Checklist fatigue | Medium | Keep checklists concise, remove redundancy |
| Enforcement gaps | Low | Add CI validation where feasible |

---

## Definition of Done

- [ ] All 6 stories completed and reviewed
- [ ] Templates updated and documented
- [ ] CI/automation in place where specified
- [ ] Team trained on new processes
- [ ] Existing discrepancies fixed

---

## References

- Tech Debt Audit: `_bmad-output/implementation-artifacts/tech-debt-audit-2025-12-31.md`
- Recurring patterns: E2-01, E2-02, E3-02, E5-01, E5-08, E6-04, E6-05
