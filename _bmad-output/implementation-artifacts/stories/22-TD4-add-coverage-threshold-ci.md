# Story 22-TD4: Add Test Coverage Threshold to CI

Status: backlog

Epic: 22 - Advanced Protocol Integration
Priority: P1 - MEDIUM
Story Points: 1
Owner: DevOps
Origin: Epic 21 Retrospective (Action Item 2)

## Story

As a **code reviewer**,
I want **CI to enforce a minimum test coverage threshold**,
So that **test coverage gaps are caught before PRs are submitted rather than during code review**.

## Background

Epic 21 code reviews identified several test coverage gaps:
- VectorSearchCard `extractResults` helper untested
- Tool renderer integration tests missing
- QuickActions test mock not properly applied

These gaps were discovered during review cycles. Enforcing coverage thresholds in CI would catch them earlier.

## Acceptance Criteria

1. **Given** a PR is submitted, **when** CI runs tests, **then** coverage is measured and reported.

2. **Given** coverage falls below 80%, **when** CI completes, **then** the check fails with a clear message.

3. **Given** coverage passes, **when** CI completes, **then** a coverage summary is posted to the PR.

4. **Given** new files are added, **when** coverage is calculated, **then** new files are included in the calculation.

5. **Given** specific files should be excluded, **when** coverage runs, **then** test files, configs, and generated files are excluded.

## Tasks

### Backend (pytest)

- [ ] **Task 1: Configure pytest-cov**
  - [ ] Add `--cov-fail-under=80` to pytest configuration
  - [ ] Configure coverage exclusions in `pyproject.toml`

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src/agentic_rag_backend"]
omit = ["*/tests/*", "*/__pycache__/*", "*/migrations/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

- [ ] **Task 2: Update CI Workflow**
  - [ ] Add coverage reporting step to GitHub Actions
  - [ ] Upload coverage to Codecov or similar (optional)

### Frontend (Jest)

- [ ] **Task 3: Configure Jest Coverage**
  - [ ] Add `coverageThreshold` to jest.config.js

```javascript
// jest.config.js
module.exports = {
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
  collectCoverageFrom: [
    "components/**/*.{ts,tsx}",
    "hooks/**/*.{ts,tsx}",
    "lib/**/*.{ts,tsx}",
    "!**/*.d.ts",
    "!**/node_modules/**",
  ],
};
```

- [ ] **Task 4: Update CI Workflow**
  - [ ] Add `--coverage` flag to test command
  - [ ] Fail CI if threshold not met

### Documentation

- [ ] **Task 5: Update Contributing Guidelines**
  - [ ] Document coverage requirements
  - [ ] Add "check coverage locally" instructions

## Definition of Done

- [ ] Backend coverage threshold set to 80%
- [ ] Frontend coverage threshold set to 80%
- [ ] CI fails on coverage below threshold
- [ ] Coverage report generated in CI
- [ ] Contributing guidelines updated

## Files to Modify

1. `backend/pyproject.toml` - Add coverage config
2. `frontend/jest.config.js` - Add coverage threshold
3. `.github/workflows/ci.yml` - Update test commands
4. `CONTRIBUTING.md` - Document coverage requirements
