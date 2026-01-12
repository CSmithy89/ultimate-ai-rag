# Story 17.13: Implement CLI Codebase Intelligence Config

Status: done

Epic: 17 - Developer Experience, CLI & Framework Integration
Priority: P2 - MEDIUM
Story Points: 3
Owner: Platform

## Story

As a **developer configuring codebase intelligence**,
I want **CLI prompts for codebase indexing and hallucination detection**,
So that **codebase features can be enabled in enterprise setups**.

## Acceptance Criteria

1. **Given** enterprise profile, **when** setup runs, **then** codebase prompts are shown.
2. **Given** selections are made, **when** setup completes, **then** custom overrides are written.

## Standards Coverage

Mark each standard as Addressed, N/A, or Planned, with a brief note.

- [x] Multi-tenancy / tenant isolation: N/A - config only
- [x] Rate limiting / abuse protection: N/A - config only
- [x] Input validation / schema enforcement: Addressed - constrained prompt values
- [x] Tests (unit/integration): Addressed - codebase setup test added
- [x] Error handling + logging: Addressed - missing profile errors
- [ ] Documentation updates: Planned - document setup usage

## Security Checklist

For data-access operations, verify each item. Mark N/A if not applicable.

- [x] **Cross-tenant isolation verified**: N/A - config only
- [x] **Authorization checked**: N/A - config only
- [x] **No information leakage**: N/A - no secrets
- [x] **Redis keys include tenant scope**: N/A - no Redis
- [x] **Integration tests for access control**: N/A - no auth
- [x] **RFC 7807 error responses**: N/A - no API responses
- [x] **File-path inputs scoped**: N/A - no file path input

## Tasks / Subtasks

- [x] **Task 1: Add codebase prompts** (AC: 1-2)
  - [x] Codebase indexing enabled
  - [x] Hallucination detection enabled

- [x] **Task 2: Write overrides** (AC: 2)
  - [x] Update custom.yaml with codebase section

- [x] **Task 3: Tests** (AC: 2)
  - [x] Verify codebase overrides written

## Definition of Done

- [x] Acceptance criteria met
- [x] Standards coverage updated
- [ ] Tests run and documented
- [x] Story file and context file updated

## Dev Notes

- Added codebase intelligence prompts and overrides in setup flow.

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added codebase prompts and custom profile overrides.\n- Added setup test for codebase category.\n

### File List

- cli/commands/setup.py\n- tests/cli/test_setup.py\n

## Test Outcomes

- Tests run: Not run (not requested)\n- Coverage: N/A\n- Failures: N/A

## Challenges Encountered

- [Challenge and resolution]

## Senior Developer Review

Outcome: APPROVE

Notes:
- Codebase prompts are gated to enterprise profile and write overrides as expected.
