# Protocol Compliance Checklist (API Routes)

Use this checklist for any new or modified API endpoints.

## Checklist

- [ ] **Rate limiting** configured for the route
- [ ] **Tenant isolation** enforced in queries (tenant_id scoped)
- [ ] **Pydantic validation** used for request bodies/params
- [ ] **RFC 7807 error format** returned for error responses
- [ ] **Timeout handling** documented or enforced

## Notes

- If a checklist item is not applicable, document the reason in the PR.
- Include references to the code paths where each item is handled.
