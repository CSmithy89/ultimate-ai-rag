# Story 22-A2: Implement A2A Session Resource Limits

Status: drafted

Epic: 22 - Advanced Protocol Integration
Priority: P0 - HIGH
Story Points: 5
Owner: Backend

## Story

As a **platform operator**,
I want **per-tenant session and message limits enforced for A2A operations**,
So that **multi-tenant safety is guaranteed with fair resource allocation and abuse prevention**.

## Background

Epic 22 builds on Epic 21's CopilotKit Full Integration to deliver enterprise-grade protocol capabilities. Story 22-A2 implements resource limits that ensure:

1. **Per-Tenant Session Caps** - Limit concurrent A2A sessions per tenant
2. **Per-Session Message Caps** - Limit messages within a single session
3. **Rate Limiting** - Throttle message frequency per session
4. **Session TTL & Cleanup** - Automatic expiration of idle sessions
5. **Cross-Worker Enforcement** - Redis backend for distributed deployments

### Why Resource Limits Matter

Without resource limits, a single tenant could:
- Exhaust server resources with unlimited concurrent sessions
- Overwhelm the system with message floods
- Prevent other tenants from accessing A2A capabilities
- Accumulate stale sessions indefinitely

### Persistence Strategy

| Environment | Backend | Configuration |
|-------------|---------|---------------|
| Development | In-memory | Default, no config needed |
| Production | Redis | `A2A_LIMITS_BACKEND=redis`, `REDIS_URL` required |
| High-scale | PostgreSQL | `A2A_LIMITS_BACKEND=postgres` for audit trail |

### Related Prior Work

| Epic/Story | Relationship |
|------------|-------------|
| Story 22-A1: A2AMiddlewareAgent | Provides middleware infrastructure this story extends |
| Epic 14-2: Robust A2A Protocol | Session/message lifecycle patterns |
| Story 22-TD5: Telemetry Tenant Rate Limiting | Similar rate limiting pattern (composite keys) |

## Acceptance Criteria

1. **Given** the `A2AResourceManager` class exists in `backend/src/agentic_rag_backend/protocols/a2a_limits.py`, **when** it is instantiated with `A2AResourceLimits` config, **then** it initializes with configurable session/message limits, TTL, and cleanup interval.

2. **Given** a tenant has reached their session limit (default 100), **when** `register_session()` is called, **then** an `A2ASessionLimitExceeded` exception is raised.

3. **Given** a session has reached its message limit (default 1000), **when** `record_message()` is called, **then** an `A2AMessageLimitExceeded` exception is raised.

4. **Given** a session has exceeded the rate limit (default 60 messages/minute), **when** `record_message()` is called, **then** an `A2ARateLimitExceeded` exception is raised.

5. **Given** sessions exist with age exceeding TTL (default 24 hours), **when** the cleanup task runs, **then** expired sessions are automatically closed and tenant active session counts are decremented.

6. **Given** a session is registered and actively used, **when** `get_tenant_metrics(tenant_id)` is called, **then** current usage metrics are returned including `active_sessions`, `total_messages`, `session_limit`, and `message_rate_limit`.

7. **Given** limit violations occur on A2A endpoints, **when** errors are returned, **then** they use RFC 7807 format with HTTP 429 status and include `retry_after` hint where applicable.

8. **Given** `A2A_LIMITS_BACKEND=redis` is configured, **when** the `A2AResourceManagerFactory` creates the manager, **then** it returns a `RedisA2AResourceManager` that enforces limits across multiple worker processes.

9. **Given** `A2A_LIMITS_BACKEND=memory` (or not set), **when** the manager is created, **then** an `InMemoryA2AResourceManager` is returned for development/testing.

10. **Given** the resource manager is started via `start()`, **when** running, **then** a background cleanup task runs at the configured interval to remove expired sessions.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: **Addressed** - All limits scoped to tenant_id, no cross-tenant visibility
- [x] Rate limiting / abuse protection: **Addressed** - Per-session rate limiting with configurable messages/minute
- [x] Input validation / schema enforcement: **Addressed** - Pydantic models for limits configuration
- [x] Tests (unit/integration): **Addressed** - Unit tests for all limit scenarios, integration tests for endpoints
- [x] Error handling + logging: **Addressed** - RFC 7807 errors, structlog for all operations
- [x] Documentation updates: **Addressed** - Configuration documented in .env.example

## Security Checklist

- [ ] **Cross-tenant isolation verified**: Tenant metrics only visible to requesting tenant
- [ ] **Authorization checked**: API key validation on metrics endpoint
- [ ] **No information leakage**: Cannot enumerate other tenant session counts
- [ ] **Redis keys include tenant scope**: Keys prefixed with `a2a:sessions:{tenant_id}:`
- [ ] **Integration tests for access control**: Cross-tenant metrics access rejected
- [ ] **RFC 7807 error responses**: All limit violations use AppError pattern
- [ ] **Background task safety**: Cleanup task handles exceptions gracefully

## Configuration

```bash
# .env
# A2A Resource Limits (Story 22-A2)
A2A_LIMITS_BACKEND=memory|redis|postgres  # Default: memory
A2A_SESSION_LIMIT_PER_TENANT=100          # Max concurrent sessions per tenant
A2A_MESSAGE_LIMIT_PER_SESSION=1000        # Max messages per session
A2A_SESSION_TTL_HOURS=24                  # Session expiry in hours
A2A_MESSAGE_RATE_LIMIT=60                 # Messages per minute per session
A2A_CLEANUP_INTERVAL_MINUTES=15           # Cleanup task interval
```

## Tasks / Subtasks

- [ ] **Task 1: Create A2A Resource Limits Configuration** (AC: 1)
  - [ ] Create `backend/src/agentic_rag_backend/protocols/a2a_limits.py`
  - [ ] Define `A2AResourceLimits` Pydantic model with all configurable limits
  - [ ] Define `TenantUsage` Pydantic model (tenant_id, active_sessions, total_messages, last_activity)
  - [ ] Define `SessionUsage` Pydantic model (session_id, tenant_id, message_count, created_at, last_message_at, message_timestamps)
  - [ ] Add limit configuration to Settings class in `core/config.py`
  - [ ] Update `.env.example` with all A2A limit environment variables

- [ ] **Task 2: Implement A2AResourceManager Base Class** (AC: 1, 6)
  - [ ] Create abstract base class `A2AResourceManager` with interface methods
  - [ ] Define `check_session_limit(tenant_id)` -> bool
  - [ ] Define `check_message_limit(session_id)` -> bool
  - [ ] Define `check_rate_limit(session_id)` -> bool
  - [ ] Define `register_session(session_id, tenant_id)` -> None
  - [ ] Define `record_message(session_id)` -> None
  - [ ] Define `close_session(session_id)` -> None
  - [ ] Define `get_tenant_metrics(tenant_id)` -> dict
  - [ ] Define `start()` and `stop()` lifecycle methods

- [ ] **Task 3: Implement InMemoryA2AResourceManager** (AC: 2, 3, 4, 5, 6, 9, 10)
  - [ ] Implement `InMemoryA2AResourceManager` class
  - [ ] Maintain `_tenant_usage: dict[str, TenantUsage]` in memory
  - [ ] Maintain `_session_usage: dict[str, SessionUsage]` in memory
  - [ ] Implement session limit check (compare active_sessions vs limit)
  - [ ] Implement message limit check (compare message_count vs limit)
  - [ ] Implement rate limit check using sliding window (messages in last minute)
  - [ ] Implement `_cleanup_loop()` asyncio background task
  - [ ] Implement `_cleanup_expired_sessions()` for TTL enforcement
  - [ ] Clean old timestamps from `message_timestamps` (keep only last 5 minutes)
  - [ ] Add structlog logging for all operations

- [ ] **Task 4: Implement RedisA2AResourceManager** (AC: 8)
  - [ ] Implement `RedisA2AResourceManager` class for production use
  - [ ] Use Redis HSET for tenant and session usage tracking
  - [ ] Use Redis ZADD with timestamps for rate limiting (sorted sets)
  - [ ] Key pattern: `a2a:tenant:{tenant_id}:sessions` for session count
  - [ ] Key pattern: `a2a:session:{session_id}:messages` for message count
  - [ ] Key pattern: `a2a:session:{session_id}:rate` for rate limiting timestamps
  - [ ] Use Redis EXPIRE for automatic TTL enforcement
  - [ ] Implement atomic operations for cross-worker safety

- [ ] **Task 5: Implement A2AResourceManagerFactory** (AC: 8, 9)
  - [ ] Create factory class `A2AResourceManagerFactory`
  - [ ] Read `A2A_LIMITS_BACKEND` from config
  - [ ] Return `InMemoryA2AResourceManager` for "memory" or default
  - [ ] Return `RedisA2AResourceManager` for "redis"
  - [ ] Future: Return `PostgresA2AResourceManager` for "postgres" (stub only)
  - [ ] Validate REDIS_URL is set when redis backend selected

- [ ] **Task 6: Define Exception Classes** (AC: 2, 3, 4, 7)
  - [ ] Define `A2ASessionLimitExceeded` exception
  - [ ] Define `A2AMessageLimitExceeded` exception
  - [ ] Define `A2ARateLimitExceeded` exception
  - [ ] Register exceptions in `core/errors.py` error handler
  - [ ] Map to HTTP 429 with RFC 7807 format
  - [ ] Include `retry_after` in rate limit error response

- [ ] **Task 7: Add Metrics API Endpoint** (AC: 6, 7)
  - [ ] Extend `backend/src/agentic_rag_backend/api/routes/a2a.py`
  - [ ] Add `GET /a2a/metrics/{tenant_id}` endpoint
  - [ ] Add tenant authorization check (can only view own metrics)
  - [ ] Return usage metrics dict from resource manager
  - [ ] Return 403 if attempting to view other tenant's metrics

- [ ] **Task 8: Wire Resource Manager to Application Lifecycle**
  - [ ] Add `get_a2a_resource_manager` dependency in `api/dependencies.py`
  - [ ] Wire resource manager to `app.state` in `main.py` startup event
  - [ ] Call `await app.state.a2a_resource_manager.start()` on startup
  - [ ] Call `await app.state.a2a_resource_manager.stop()` on shutdown
  - [ ] Export new classes in `protocols/__init__.py`

- [ ] **Task 9: Integrate Limits with A2A Session Operations**
  - [ ] Update A2A session creation to call `register_session()`
  - [ ] Update A2A message handling to call `record_message()`
  - [ ] Update A2A session close to call `close_session()`
  - [ ] Handle limit exceptions in API endpoints with proper error responses

- [ ] **Task 10: Add Unit Tests** (AC: 1, 2, 3, 4, 5, 6, 9, 10)
  - [ ] Create `tests/protocols/test_a2a_limits.py`
  - [ ] Test `A2AResourceLimits` model validation
  - [ ] Test `InMemoryA2AResourceManager` initialization
  - [ ] Test `register_session()` success and limit exceeded
  - [ ] Test `record_message()` success, limit exceeded, and rate limited
  - [ ] Test `close_session()` decrements active session count
  - [ ] Test `get_tenant_metrics()` returns correct values
  - [ ] Test `_cleanup_expired_sessions()` removes old sessions
  - [ ] Test cleanup task starts and stops correctly
  - [ ] Test `A2AResourceManagerFactory` returns correct implementation

- [ ] **Task 11: Add Integration Tests** (AC: 6, 7, 8)
  - [ ] Create `tests/integration/test_a2a_limits.py`
  - [ ] Test `GET /a2a/metrics/{tenant_id}` returns metrics
  - [ ] Test `GET /a2a/metrics/{tenant_id}` returns 403 for other tenant
  - [ ] Test session limit enforcement on registration endpoint
  - [ ] Test rate limit enforcement returns 429 with retry_after
  - [ ] Test Redis backend integration (requires Redis test container)

## Technical Notes

### A2AResourceManager Class Structure

```python
# backend/src/agentic_rag_backend/protocols/a2a_limits.py
from datetime import datetime, timedelta
from pydantic import BaseModel
import asyncio
import structlog

logger = structlog.get_logger(__name__)

class A2AResourceLimits(BaseModel):
    """Configuration for A2A resource limits."""
    session_limit_per_tenant: int = 100
    message_limit_per_session: int = 1000
    session_ttl_hours: int = 24
    message_rate_limit: int = 60  # per minute
    cleanup_interval_minutes: int = 15

class TenantUsage(BaseModel):
    """Tracks resource usage for a tenant."""
    tenant_id: str
    active_sessions: int = 0
    total_messages: int = 0
    last_activity: datetime = datetime.utcnow()

class SessionUsage(BaseModel):
    """Tracks resource usage for a session."""
    session_id: str
    tenant_id: str
    message_count: int = 0
    created_at: datetime = datetime.utcnow()
    last_message_at: datetime = datetime.utcnow()
    message_timestamps: list[datetime] = []  # For rate limiting
```

### Rate Limiting Algorithm

The rate limiting uses a sliding window approach:
1. Store timestamp of each message in `message_timestamps`
2. On each `record_message()`, count timestamps within last 60 seconds
3. If count >= `message_rate_limit`, raise `A2ARateLimitExceeded`
4. Clean timestamps older than 5 minutes to bound memory usage

### Redis Key Schema

```
a2a:tenant:{tenant_id}:sessions       # Hash: session counts
a2a:session:{session_id}:info         # Hash: session metadata
a2a:session:{session_id}:rate         # Sorted Set: message timestamps
```

### Error Response Format

```json
{
  "type": "/errors/rate-limited",
  "title": "Rate Limit Exceeded",
  "status": 429,
  "detail": "Session has exceeded message rate limit (60/minute)",
  "instance": "/a2a/messages",
  "retry_after": 60
}
```

## Dependencies

- **Story 22-A1**: A2AMiddlewareAgent provides the session/message lifecycle this story limits
- **redis-py**: Required for Redis backend (already in dependencies)
- **asyncio**: For background cleanup task

## Definition of Done

- [ ] All acceptance criteria met and verified
- [ ] Unit tests passing with >90% coverage for a2a_limits.py
- [ ] Integration tests passing for metrics endpoint
- [ ] Redis backend tested with docker-compose Redis container
- [ ] Configuration documented in .env.example
- [ ] RFC 7807 error responses verified for all limit violations
- [ ] Code reviewed and approved
- [ ] No new linting errors or type check failures

## Dev Notes

_To be filled in during implementation_

## Completion Notes

_To be filled in when story is marked done_
