# Story [X.Y]: [Title]

Status: backlog

## Story

As a [role],  
I want [capability],  
So that [benefit].

## Acceptance Criteria

1. Given [context], when [action], then [outcome].
2. [Additional criteria]

## Standards Coverage

Mark each standard as Addressed, N/A, or Planned, with a brief note.

- [ ] Multi-tenancy / tenant isolation: [Addressed | N/A | Planned] - [note]
- [ ] Rate limiting / abuse protection: [Addressed | N/A | Planned] - [note]
- [ ] Input validation / schema enforcement: [Addressed | N/A | Planned] - [note]
- [ ] Tests (unit/integration): [Addressed | N/A | Planned] - [note]
- [ ] Error handling + logging: [Addressed | N/A | Planned] - [note]
- [ ] Documentation updates: [Addressed | N/A | Planned] - [note]

## Security Checklist

For data-access operations, verify each item. Mark N/A if not applicable.

- [ ] **Cross-tenant isolation verified**: After fetching data, verify `tenant_id` matches before returning
- [ ] **Authorization checked**: User/agent has permission to access the resource
- [ ] **No information leakage**: Error messages don't reveal existence of resources in other tenants
- [ ] **Redis keys include tenant scope**: Keys are prefixed/scoped to prevent cross-tenant access
- [ ] **Integration tests for access control**: Tests attempt cross-tenant access and verify denial
- [ ] **RFC 7807 error responses**: All errors follow standard format with proper codes

## Tasks / Subtasks

- [ ] Task (AC: 1)
  - [ ] Subtask

## Technical Notes

- [Note]

## Definition of Done

- [ ] Acceptance criteria met
- [ ] Standards coverage updated
- [ ] Tests run and documented

## Dev Notes

- [Implementation notes]

## Dev Agent Record

### Agent Model Used

[model]

### Debug Log References

### Completion Notes List

- [Note]

### File List

- [path]

## Test Outcomes

- Tests run: [list]
- Coverage: [percentage or N/A]
- Failures: [none | details]

## Challenges Encountered

- [Challenge and resolution]

## Senior Developer Review

Outcome: [APPROVE | Changes Requested | Blocked]

Notes:
- [Review notes]
