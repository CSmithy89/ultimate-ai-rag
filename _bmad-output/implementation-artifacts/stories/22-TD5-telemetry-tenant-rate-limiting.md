# Story 22-TD5: Enhance Telemetry Rate Limiting with Tenant Awareness

Status: done

Epic: 22 - Advanced Protocol Integration
Priority: P1 - MEDIUM
Story Points: 1
Owner: Backend
Origin: Epic 21 Retrospective (TD-21-3)

## Story

As a **platform operator**,
I want **telemetry rate limiting to use a composite key of tenant_id and IP**,
So that **users behind shared NAT gateways aren't incorrectly rate-limited due to IP collisions**.

## Background

The current telemetry endpoint rate limits by IP address only:

```python
client_ip = request.client.host
if not await limiter.allow(f"telemetry:{client_ip}"):
    raise rate_limit_exceeded()
```

In multi-tenant scenarios where many users share the same NAT gateway IP (e.g., corporate networks, cloud functions), legitimate requests from different tenants can be incorrectly rate-limited.

## Acceptance Criteria

1. **Given** a rate limit check occurs, **when** the key is constructed, **then** it includes both tenant_id and IP.

2. **Given** tenant_id is available from request headers, **when** rate limiting, **then** it is included in the composite key.

3. **Given** tenant_id is not available, **when** rate limiting, **then** "anonymous" is used as the tenant component.

4. **Given** the composite key is used, **when** different tenants share an IP, **then** each tenant has its own rate limit bucket.

## Tasks

- [ ] **Task 1: Update Rate Limit Key Construction**
  - [ ] Get tenant_id from request state or header
  - [ ] Construct composite key: `telemetry:{tenant_id}:{ip}`
  - [ ] Fall back to "anonymous" if no tenant_id

```python
# Before
client_ip = request.client.host
if not await limiter.allow(f"telemetry:{client_ip}"):
    raise rate_limit_exceeded()

# After
client_ip = request.client.host
tenant_id = getattr(request.state, "tenant_id", None) or "anonymous"
rate_key = f"telemetry:{tenant_id}:{client_ip}"
if not await limiter.allow(rate_key):
    raise rate_limit_exceeded()
```

- [ ] **Task 2: Add Tests**
  - [ ] Test rate limiting uses composite key
  - [ ] Test different tenants on same IP are independent
  - [ ] Test anonymous fallback works

## Definition of Done

- [ ] Rate limit key includes tenant_id
- [ ] Tests verify tenant isolation
- [ ] Code review approved

## Files to Modify

1. `backend/src/agentic_rag_backend/api/routes/telemetry.py`
2. `backend/tests/api/test_telemetry.py`
