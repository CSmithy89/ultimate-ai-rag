# Story 22-TD2: Create Frontend Security Review Checklist

Status: done

Epic: 22 - Advanced Protocol Integration
Priority: P0 - HIGH
Story Points: 1
Owner: QA / Documentation
Origin: Epic 21 Retrospective (Action Item 1)

## Story

As a **code reviewer**,
I want **a comprehensive security checklist for frontend components**,
So that **security issues are caught systematically before code review rather than discovered during review cycles**.

## Background

Epic 21 code reviews identified security issues in every PR:
- Incomplete SENSITIVE_PATTERNS regex (missing jwt, bearer, session)
- Key redaction false positives (`monkey` matches `key`)
- Value-embedded credentials not detected
- Markdown XSS risk in A2UI renderer
- External image URLs leaking metadata

These issues required multiple review rounds to fix. A proactive checklist would catch these earlier.

## Acceptance Criteria

1. **Given** the checklist exists, **when** a developer reviews it before submitting a PR, **then** common security issues are addressed proactively.

2. **Given** the checklist covers data handling, **when** reviewed, **then** sensitive data redaction, localStorage security, and API credential handling are verified.

3. **Given** the checklist covers UI rendering, **when** reviewed, **then** XSS prevention, markdown sanitization, and external resource loading are verified.

4. **Given** the checklist covers network requests, **when** reviewed, **then** CORS, credential exposure, and error message leakage are verified.

5. **Given** the checklist is documented, **when** a new team member joins, **then** they can quickly understand frontend security requirements.

## Checklist Categories

### 1. Sensitive Data Handling
- [ ] All sensitive keys match comprehensive regex pattern (password, secret, token, key, auth, bearer, jwt, session, cookie, oauth, credential, api_key, private_key, access_token, refresh_token, client_secret, signature, ssn)
- [ ] Regex uses word boundaries to avoid false positives
- [ ] Value-embedded credentials detected (e.g., `password=secret` in connection strings)
- [ ] localStorage/sessionStorage data validated with Zod before use
- [ ] No sensitive data logged to console in production

### 2. UI Rendering Security
- [ ] Markdown rendered with `rehype-sanitize` plugin
- [ ] User-provided HTML escaped or sanitized
- [ ] External images proxied through backend or allowlisted domains
- [ ] iframe content sandboxed with appropriate restrictions
- [ ] Dynamic component rendering uses allowlist of known components

### 3. Network & API Security
- [ ] API errors don't expose stack traces or internal details in production
- [ ] CORS headers verified for API endpoints
- [ ] Credentials not included in URLs (use headers/body)
- [ ] Rate limiting applied to user-facing endpoints
- [ ] Tenant isolation verified for multi-tenant operations

### 4. React-Specific Security
- [ ] No `dangerouslySetInnerHTML` without sanitization
- [ ] User input not interpolated into event handlers
- [ ] External links use `rel="noopener noreferrer"`
- [ ] Error boundaries prevent full app crashes from revealing state

### 5. Third-Party Dependencies
- [ ] npm audit shows no high/critical vulnerabilities
- [ ] External scripts loaded from trusted CDNs or self-hosted
- [ ] Dependency versions pinned in package.json

## Tasks

- [ ] **Task 1: Create Checklist Document**
  - [ ] Create `docs/checklists/frontend-security-checklist.md`
  - [ ] Organize by category with checkboxes
  - [ ] Include code examples for each item

- [ ] **Task 2: Add to PR Template**
  - [ ] Update `.github/PULL_REQUEST_TEMPLATE.md` to reference checklist
  - [ ] Add abbreviated checklist section to PR template

- [ ] **Task 3: Document in CONTRIBUTING.md**
  - [ ] Reference checklist in contributing guidelines
  - [ ] Add to "Before Submitting" section

## Definition of Done

- [ ] Checklist document created in `docs/checklists/`
- [ ] PR template updated with checklist reference
- [ ] Team notified of new checklist
- [ ] Checklist reviewed by security-conscious team member

## Files to Create/Modify

1. **Create:** `docs/checklists/frontend-security-checklist.md`
2. **Modify:** `.github/PULL_REQUEST_TEMPLATE.md` (if exists)
3. **Modify:** `CONTRIBUTING.md` (if exists)
