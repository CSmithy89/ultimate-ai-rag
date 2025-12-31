## Summary

## Changes

## Pre-Review Checklist

- [ ] Tests pass locally (or scope documented)
- [ ] Lint passes with no new warnings
- [ ] Type check passes (if applicable)
- [ ] Tenant isolation enforced where data is scoped
- [ ] Input validation present for external inputs
- [ ] Error handling and logging follow project conventions
- [ ] Docs updated (README, API docs, or story notes)

## Protocol Compliance (API Routes Only)

Complete this section if you added or modified API endpoints.
Reference: `docs/quality/protocol-compliance-checklist.md`

- [ ] Rate limiting configured
- [ ] Tenant isolation enforced in queries
- [ ] RFC 7807 error format used
- [ ] Pydantic validation in place
- [ ] Timeout handling documented

## Testing

- [ ] Unit tests
- [ ] Integration tests
- [ ] E2E tests (if applicable)
