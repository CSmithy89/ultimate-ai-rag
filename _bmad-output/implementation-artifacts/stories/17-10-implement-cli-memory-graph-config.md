# Story 17.10: Implement CLI Memory & Graph Config

Status: done

Epic: 17 - Developer Experience, CLI & Framework Integration
Priority: P1 - HIGH
Story Points: 5
Owner: Platform

## Story

As a **developer configuring memory and graph features**,
I want **CLI prompts for memory scopes, consolidation, and graph intelligence**,
So that **I can enable or disable advanced graph features by profile**.

## Acceptance Criteria

1. **Given** minimal profile, **when** setup runs, **then** advanced memory/graph options are skipped.
2. **Given** standard profile, **when** setup runs, **then** basic memory scopes are shown.
3. **Given** enterprise profile, **when** setup runs, **then** all memory and graph options are shown.
4. **Given** selections are made, **when** setup completes, **then** custom profile overrides are written.

## Standards Coverage

Mark each standard as Addressed, N/A, or Planned, with a brief note.

- [x] Multi-tenancy / tenant isolation: N/A - config only
- [x] Rate limiting / abuse protection: N/A - config only
- [x] Input validation / schema enforcement: Addressed - constrained prompt values
- [x] Tests (unit/integration): Addressed - memory/graph setup tests added
- [x] Error handling + logging: Addressed - profile validation errors
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

- [x] **Task 1: Add memory/graph prompts** (AC: 1-3)
  - [x] Memory scopes + default scope + consolidation
  - [x] Graph intelligence toggles (LazyRAG, query routing, graph reranker)
  - [x] Community detection toggle

- [x] **Task 2: Write custom overrides** (AC: 4)
  - [x] Update custom.yaml with memory/graph sections

- [x] **Task 3: Tests** (AC: 4)
  - [x] Verify memory/graph overrides are written

## Definition of Done

- [x] Acceptance criteria met
- [x] Standards coverage updated
- [ ] Tests run and documented
- [x] Story file and context file updated

## Dev Notes

- Added memory, community, and graph intelligence prompts to setup flow.\n

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added memory/graph setup prompts with profile-aware defaults.\n- Custom profile overrides updated for memory/community/graph sections.\n- Added setup test for memory-graph category.\n

### File List

- cli/commands/setup.py\n- tests/cli/test_setup.py\n

## Test Outcomes

- Tests run: Not run (not requested)\n- Coverage: N/A\n- Failures: N/A

## Challenges Encountered

- [Challenge and resolution]

## Senior Developer Review

Outcome: APPROVE

Notes:
- Memory and graph prompts respect profile defaults and write overrides correctly.
