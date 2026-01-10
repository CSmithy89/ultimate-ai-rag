# Pre-Review Quality Checklist

Use this checklist before requesting review. Keep entries concise and honest.

## Checklist

- [ ] Tests pass locally (or relevant test scope documented)
- [ ] Lint passes with no new warnings
- [ ] Type check passes (if applicable)
- [ ] Tenant isolation enforced where data is scoped
- [ ] Input validation present for external inputs
- [ ] File-path inputs are restricted to allowed directories (path traversal prevented)
- [ ] Error handling and logging follow project conventions
- [ ] Docs updated (README, API docs, or story notes)
- [ ] Story artifacts complete (story file + context file for completed stories)
- [ ] AI review comments triaged (addressed or explicitly waived with rationale)
- [ ] Test outcomes recorded (story notes or epic test report)

## Protocol Compliance (API Changes Only)

If this PR adds or modifies API routes, complete the Protocol Compliance section in
`.github/PULL_REQUEST_TEMPLATE.md` and reference `docs/quality/protocol-compliance-checklist.md`.
