## Summary

## Changes

## Pre-Review Checklist

- [ ] Tests pass locally (or scope documented)
- [ ] Lint passes with no new warnings
- [ ] Type check passes (if applicable)
- [ ] Tenant isolation enforced where data is scoped
- [ ] Input validation present for external inputs
- [ ] File-path inputs are restricted to allowed directories (path traversal prevented)
- [ ] Error handling and logging follow project conventions
- [ ] Docs updated (README, API docs, or story notes)
- [ ] Story artifacts complete (story file + context file for completed stories)
- [ ] AI review comments triaged (addressed or explicitly waived)
- [ ] Test outcomes recorded (story notes or epic test report)

## Frontend Security (UI Changes Only)

Complete this section if you modified frontend components.
Reference: `docs/checklists/frontend-security-checklist.md`

- [ ] Sensitive data patterns comprehensive (password, token, key, jwt, bearer, etc.)
- [ ] No dangerouslySetInnerHTML without sanitization
- [ ] External links use rel="noopener noreferrer"
- [ ] User content sanitized before rendering (markdown, HTML)
- [ ] No credentials in URLs (use headers/body)

## Protocol Compliance (API Routes Only)

Complete this section if you added or modified API endpoints.
Reference: `docs/quality/protocol-compliance-checklist.md`

- [ ] Rate limiting configured
- [ ] Tenant isolation enforced in queries
- [ ] RFC 7807 error format used
- [ ] Pydantic validation in place
- [ ] Timeout handling documented

## AI Review Triage (If applicable)

- [ ] AI review comments reviewed and addressed
- [ ] Any waived items documented with rationale

## Testing

- [ ] Unit tests
- [ ] Integration tests
- [ ] E2E tests (if applicable)
