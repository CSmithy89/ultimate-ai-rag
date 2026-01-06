# Epic 22 Tech Spec: Advanced Protocol Integration

**Date:** 2026-01-06
**Status:** Backlog
**Epic Owner:** Platform Engineering
**Origin:** Party Mode Deep Dive Analysis (2026-01-06)
**Depends On:** Epic 21 (CopilotKit Full Integration)

---

## Overview

Epic 22 completes the protocol integration layer by implementing advanced A2A middleware, AG-UI telemetry, multi-tenant resource limits, and cross-platform UI specifications (MCP-UI, Open-JSON-UI). This epic builds on Epic 21's foundation to deliver enterprise-grade protocol capabilities.

### Strategic Context

Analysis of the CopilotKit Feature Gap Roadmap identified several advanced protocol features that extend beyond basic integration:

1. **A2A Middleware Agent**: CopilotKit's A2AMiddlewareAgent enables agent-to-agent collaboration
2. **AG-UI Telemetry**: Stream metrics, error events, and performance monitoring
3. **A2A Resource Limits**: Per-tenant session/message caps for multi-tenant safety
4. **MCP-UI**: Iframe-based rendering for embedding external interactive tools
5. **Open-JSON-UI**: OpenAI-style declarative UI for cross-platform interoperability

### Research Sources

| Source | Usage |
|--------|-------|
| CopilotKit Feature Gap Roadmap | Gap identification and implementation plans |
| DeepWiki: CopilotKit/CopilotKit | A2A middleware patterns, AG-UI events |
| Context7: /copilotkit/copilotkit | A2AMiddlewareAgent examples |
| A2A Protocol Specification | Google's agent-to-agent protocol |
| MCP-UI Specification | Model Context Protocol UI extension |

### Goals

- Enable agent-to-agent collaboration via A2AMiddlewareAgent
- Add comprehensive AG-UI stream telemetry and error handling
- Implement per-tenant resource limits for safe multi-tenancy
- Support MCP-UI iframe-based external tool embedding
- Add Open-JSON-UI rendering for OpenAI ecosystem interoperability
- Achieve full protocol compliance across all supported specs

### Related Epics

| Epic | Relationship |
|------|-------------|
| Epic 7: Protocol Integration | Original MCP server, A2A foundation (completed) |
| Epic 14: Connectivity | Robust A2A protocol (completed) |
| Epic 21: CopilotKit Full Integration | Modern hooks, MCP client, A2UI (prerequisite) |

---

## Current Implementation Audit

### What We Have (After Epic 21)

| Feature | Location | Status |
|---------|----------|--------|
| A2A Protocol | `backend/src/agentic_rag_backend/api/routes/a2a.py` | Basic session/message lifecycle |
| AG-UI Transport | `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` | Core events only |
| MCP Server | `backend/src/agentic_rag_backend/mcp_server/` | Full implementation |
| MCP Client | Epic 21-C (after completion) | Outbound connections |
| A2UI Rendering | Epic 21-D (after completion) | Widget rendering |

### What This Epic Adds

| Feature | Gap |
|---------|-----|
| A2AMiddlewareAgent | No agent delegation/collaboration pattern |
| AG-UI Stream Metrics | No latency, event count, drop rate metrics |
| A2A Session Caps | No per-tenant limits (session/message counts) |
| A2A Message Caps | No rate limiting on messages |
| MCP-UI Renderer | No iframe sandbox for external tools |
| Open-JSON-UI | No OpenAI-style declarative UI support |
| AG-UI Error Events | Limited error event taxonomy |

---

## Story Groups

### Group A: A2A Middleware & Collaboration

*Focus: Enable agent-to-agent delegation patterns*
*Priority: P0 - Core capability*

#### Story 22-A1: Implement A2AMiddlewareAgent Foundation

**Priority:** P0 - HIGH
**Story Points:** 8
**Owner:** Backend

**Objective:** Create A2AMiddlewareAgent for agent delegation and collaboration.

**What A2AMiddlewareAgent Provides:**
- Agent-to-agent message routing
- Capability discovery across agents
- Task delegation with context preservation
- Async agent invocation with response handling

**Implementation:**

```python
# backend/src/agentic_rag_backend/protocols/a2a_middleware.py (new)
from typing import Any, AsyncIterator
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)

class A2AAgentCapability(BaseModel):
    """Advertised capability of an A2A agent."""
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

class A2AAgentInfo(BaseModel):
    """Registered A2A agent information."""
    agent_id: str
    name: str
    description: str
    capabilities: list[A2AAgentCapability]
    endpoint: str  # AG-UI endpoint for this agent

class A2AMiddlewareAgent:
    """Middleware agent for A2A protocol collaboration."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        capabilities: list[A2AAgentCapability],
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self._registered_agents: dict[str, A2AAgentInfo] = {}

    def register_agent(self, agent_info: A2AAgentInfo) -> None:
        """Register an agent for collaboration."""
        self._registered_agents[agent_info.agent_id] = agent_info
        logger.info(
            "a2a_agent_registered",
            agent_id=agent_info.agent_id,
            capabilities=[c.name for c in agent_info.capabilities],
        )

    def discover_capabilities(
        self,
        capability_filter: str | None = None,
    ) -> list[tuple[str, A2AAgentCapability]]:
        """Discover capabilities across all registered agents."""
        results = []
        for agent_id, agent_info in self._registered_agents.items():
            for cap in agent_info.capabilities:
                if capability_filter is None or capability_filter in cap.name:
                    results.append((agent_id, cap))
        return results

    async def delegate_task(
        self,
        target_agent_id: str,
        capability_name: str,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Delegate a task to another agent."""
        agent_info = self._registered_agents.get(target_agent_id)
        if not agent_info:
            raise A2AAgentNotFoundError(f"Agent not found: {target_agent_id}")

        capability = next(
            (c for c in agent_info.capabilities if c.name == capability_name),
            None,
        )
        if not capability:
            raise A2ACapabilityNotFoundError(
                f"Capability {capability_name} not found on agent {target_agent_id}"
            )

        logger.info(
            "a2a_task_delegated",
            from_agent=self.agent_id,
            to_agent=target_agent_id,
            capability=capability_name,
        )

        # Execute via AG-UI endpoint
        async for event in self._invoke_agent(
            agent_info.endpoint,
            capability_name,
            input_data,
            context,
        ):
            yield event

    async def _invoke_agent(
        self,
        endpoint: str,
        capability: str,
        input_data: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Invoke an agent's capability via AG-UI protocol."""
        import httpx

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                endpoint,
                json={
                    "capability": capability,
                    "input": input_data,
                    "context": context or {},
                },
                headers={"Accept": "text/event-stream"},
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        import json
                        yield json.loads(line[6:])


class A2AAgentNotFoundError(Exception):
    """Raised when target agent is not registered."""
    pass


class A2ACapabilityNotFoundError(Exception):
    """Raised when capability is not available."""
    pass
```

**Agent Registration API:**
```python
# backend/src/agentic_rag_backend/api/routes/a2a.py (extend)
from agentic_rag_backend.protocols.a2a_middleware import (
    A2AMiddlewareAgent,
    A2AAgentInfo,
)

@router.post("/a2a/agents/register")
async def register_agent(
    agent_info: A2AAgentInfo,
    middleware: A2AMiddlewareAgent = Depends(get_a2a_middleware),
) -> dict[str, str]:
    """Register an agent for A2A collaboration."""
    middleware.register_agent(agent_info)
    return {"status": "registered", "agent_id": agent_info.agent_id}

@router.get("/a2a/agents")
async def list_agents(
    middleware: A2AMiddlewareAgent = Depends(get_a2a_middleware),
) -> list[A2AAgentInfo]:
    """List all registered A2A agents."""
    return list(middleware._registered_agents.values())

@router.get("/a2a/capabilities")
async def discover_capabilities(
    filter: str | None = None,
    middleware: A2AMiddlewareAgent = Depends(get_a2a_middleware),
) -> list[dict]:
    """Discover available capabilities across agents."""
    caps = middleware.discover_capabilities(filter)
    return [
        {"agent_id": agent_id, "capability": cap.model_dump()}
        for agent_id, cap in caps
    ]
```

**Acceptance Criteria:**
1. A2AMiddlewareAgent class implemented with registration, discovery, delegation
2. Agent registration API endpoint functional
3. Capability discovery across registered agents
4. Task delegation with AG-UI streaming response
5. Context preservation during delegation
6. Proper error handling for missing agents/capabilities
7. Logging for all A2A operations
8. Tests for registration, discovery, delegation flows

---

#### Story 22-A2: Implement A2A Session Resource Limits

**Priority:** P0 - HIGH
**Story Points:** 5
**Owner:** Backend

**Objective:** Add per-tenant session and message caps for multi-tenant safety.

**Configuration:**
```bash
# .env
# A2A Resource Limits
A2A_SESSION_LIMIT_PER_TENANT=100  # Max concurrent sessions
A2A_MESSAGE_LIMIT_PER_SESSION=1000  # Max messages per session
A2A_SESSION_TTL_HOURS=24  # Session expiry
A2A_MESSAGE_RATE_LIMIT=60  # Messages per minute per session
A2A_CLEANUP_INTERVAL_MINUTES=15  # Cleanup interval
```

**Implementation:**

```python
# backend/src/agentic_rag_backend/protocols/a2a_limits.py (new)
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

class A2AResourceManager:
    """Manages A2A resource limits per tenant."""

    def __init__(self, limits: A2AResourceLimits) -> None:
        self.limits = limits
        self._tenant_usage: dict[str, TenantUsage] = {}
        self._session_usage: dict[str, SessionUsage] = {}
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the cleanup background task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the cleanup background task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()

    async def check_session_limit(self, tenant_id: str) -> bool:
        """Check if tenant can create a new session."""
        usage = self._tenant_usage.get(tenant_id)
        if not usage:
            return True
        return usage.active_sessions < self.limits.session_limit_per_tenant

    async def check_message_limit(self, session_id: str) -> bool:
        """Check if session can send another message."""
        usage = self._session_usage.get(session_id)
        if not usage:
            return True
        return usage.message_count < self.limits.message_limit_per_session

    async def check_rate_limit(self, session_id: str) -> bool:
        """Check if session is within rate limit."""
        usage = self._session_usage.get(session_id)
        if not usage:
            return True

        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        # Count messages in last minute
        recent_messages = [
            ts for ts in usage.message_timestamps
            if ts > minute_ago
        ]

        return len(recent_messages) < self.limits.message_rate_limit

    async def register_session(
        self,
        session_id: str,
        tenant_id: str,
    ) -> None:
        """Register a new session."""
        if not await self.check_session_limit(tenant_id):
            raise A2ASessionLimitExceeded(
                f"Tenant {tenant_id} has reached session limit"
            )

        # Update tenant usage
        if tenant_id not in self._tenant_usage:
            self._tenant_usage[tenant_id] = TenantUsage(tenant_id=tenant_id)
        self._tenant_usage[tenant_id].active_sessions += 1

        # Create session usage
        self._session_usage[session_id] = SessionUsage(
            session_id=session_id,
            tenant_id=tenant_id,
        )

        logger.info(
            "a2a_session_registered",
            session_id=session_id,
            tenant_id=tenant_id,
            active_sessions=self._tenant_usage[tenant_id].active_sessions,
        )

    async def record_message(self, session_id: str) -> None:
        """Record a message for a session."""
        if not await self.check_message_limit(session_id):
            raise A2AMessageLimitExceeded(
                f"Session {session_id} has reached message limit"
            )

        if not await self.check_rate_limit(session_id):
            raise A2ARateLimitExceeded(
                f"Session {session_id} is rate limited"
            )

        usage = self._session_usage.get(session_id)
        if usage:
            now = datetime.utcnow()
            usage.message_count += 1
            usage.last_message_at = now
            usage.message_timestamps.append(now)

            # Clean old timestamps (keep only last 5 minutes)
            cutoff = now - timedelta(minutes=5)
            usage.message_timestamps = [
                ts for ts in usage.message_timestamps if ts > cutoff
            ]

            # Update tenant total
            tenant = self._tenant_usage.get(usage.tenant_id)
            if tenant:
                tenant.total_messages += 1
                tenant.last_activity = now

    async def close_session(self, session_id: str) -> None:
        """Close a session and update limits."""
        usage = self._session_usage.pop(session_id, None)
        if usage:
            tenant = self._tenant_usage.get(usage.tenant_id)
            if tenant:
                tenant.active_sessions = max(0, tenant.active_sessions - 1)

            logger.info(
                "a2a_session_closed",
                session_id=session_id,
                tenant_id=usage.tenant_id,
                message_count=usage.message_count,
            )

    async def get_tenant_metrics(self, tenant_id: str) -> dict[str, Any]:
        """Get resource usage metrics for a tenant."""
        usage = self._tenant_usage.get(tenant_id)
        if not usage:
            return {"active_sessions": 0, "total_messages": 0}

        return {
            "active_sessions": usage.active_sessions,
            "total_messages": usage.total_messages,
            "session_limit": self.limits.session_limit_per_tenant,
            "message_rate_limit": self.limits.message_rate_limit,
        }

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired sessions."""
        while True:
            await asyncio.sleep(self.limits.cleanup_interval_minutes * 60)
            await self._cleanup_expired_sessions()

    async def _cleanup_expired_sessions(self) -> None:
        """Clean up sessions that have expired."""
        now = datetime.utcnow()
        ttl = timedelta(hours=self.limits.session_ttl_hours)
        expired = []

        for session_id, usage in self._session_usage.items():
            if now - usage.created_at > ttl:
                expired.append(session_id)

        for session_id in expired:
            await self.close_session(session_id)

        if expired:
            logger.info("a2a_sessions_expired", count=len(expired))


class A2ASessionLimitExceeded(Exception):
    """Raised when tenant session limit is exceeded."""
    pass


class A2AMessageLimitExceeded(Exception):
    """Raised when session message limit is exceeded."""
    pass


class A2ARateLimitExceeded(Exception):
    """Raised when message rate limit is exceeded."""
    pass
```

**Metrics Endpoint:**
```python
# backend/src/agentic_rag_backend/api/routes/a2a.py (extend)
@router.get("/a2a/metrics/{tenant_id}")
async def get_tenant_metrics(
    tenant_id: str,
    resource_manager: A2AResourceManager = Depends(get_a2a_resource_manager),
) -> dict[str, Any]:
    """Get A2A resource usage metrics for a tenant."""
    return await resource_manager.get_tenant_metrics(tenant_id)
```

**Acceptance Criteria:**
1. Per-tenant session limits enforced
2. Per-session message limits enforced
3. Rate limiting (messages per minute) enforced
4. Session TTL with automatic cleanup
5. Metrics endpoint for tenant usage
6. Proper error responses (429) for limit violations
7. Background cleanup task for expired sessions
8. Tests for all limit scenarios

---

### Group B: AG-UI Telemetry & Error Handling

*Focus: Production-grade observability for AG-UI streams*
*Priority: P0 - Critical for operations*

#### Story 22-B1: Implement AG-UI Stream Metrics

**Priority:** P0 - HIGH
**Story Points:** 5
**Owner:** Backend

**Objective:** Add comprehensive metrics for AG-UI stream health and performance.

**Metrics to Track:**

| Metric | Type | Description |
|--------|------|-------------|
| `agui_stream_started_total` | Counter | Total streams started |
| `agui_stream_completed_total` | Counter | Total streams completed successfully |
| `agui_stream_failed_total` | Counter | Total streams that failed |
| `agui_event_emitted_total` | Counter | Events emitted by type |
| `agui_stream_duration_seconds` | Histogram | Stream duration |
| `agui_event_latency_seconds` | Histogram | Time between events |
| `agui_stream_event_count` | Histogram | Events per stream |
| `agui_stream_bytes_total` | Counter | Total bytes streamed |

**Implementation:**

```python
# backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py (new)
from prometheus_client import Counter, Histogram, Gauge
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any

# Prometheus metrics
STREAM_STARTED = Counter(
    "agui_stream_started_total",
    "Total AG-UI streams started",
    ["tenant_id"],
)

STREAM_COMPLETED = Counter(
    "agui_stream_completed_total",
    "Total AG-UI streams completed",
    ["tenant_id", "status"],
)

EVENT_EMITTED = Counter(
    "agui_event_emitted_total",
    "Total AG-UI events emitted",
    ["tenant_id", "event_type"],
)

STREAM_DURATION = Histogram(
    "agui_stream_duration_seconds",
    "AG-UI stream duration",
    ["tenant_id"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

EVENT_LATENCY = Histogram(
    "agui_event_latency_seconds",
    "Time between AG-UI events",
    ["tenant_id"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

STREAM_EVENT_COUNT = Histogram(
    "agui_stream_event_count",
    "Events per AG-UI stream",
    ["tenant_id"],
    buckets=[1, 5, 10, 25, 50, 100, 250],
)

ACTIVE_STREAMS = Gauge(
    "agui_active_streams",
    "Currently active AG-UI streams",
    ["tenant_id"],
)


class AGUIMetricsCollector:
    """Collects metrics for AG-UI streams."""

    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id
        self.start_time: float = 0
        self.last_event_time: float = 0
        self.event_count: int = 0
        self.total_bytes: int = 0

    def stream_started(self) -> None:
        """Record stream start."""
        self.start_time = time.time()
        self.last_event_time = self.start_time
        STREAM_STARTED.labels(tenant_id=self.tenant_id).inc()
        ACTIVE_STREAMS.labels(tenant_id=self.tenant_id).inc()

    def event_emitted(self, event_type: str, event_bytes: int = 0) -> None:
        """Record event emission."""
        now = time.time()

        EVENT_EMITTED.labels(
            tenant_id=self.tenant_id,
            event_type=event_type,
        ).inc()

        if self.last_event_time > 0:
            latency = now - self.last_event_time
            EVENT_LATENCY.labels(tenant_id=self.tenant_id).observe(latency)

        self.last_event_time = now
        self.event_count += 1
        self.total_bytes += event_bytes

    def stream_completed(self, status: str = "success") -> None:
        """Record stream completion."""
        duration = time.time() - self.start_time

        STREAM_COMPLETED.labels(
            tenant_id=self.tenant_id,
            status=status,
        ).inc()

        STREAM_DURATION.labels(tenant_id=self.tenant_id).observe(duration)
        STREAM_EVENT_COUNT.labels(tenant_id=self.tenant_id).observe(self.event_count)
        ACTIVE_STREAMS.labels(tenant_id=self.tenant_id).dec()


@asynccontextmanager
async def track_agui_stream(tenant_id: str) -> AsyncIterator[AGUIMetricsCollector]:
    """Context manager for tracking AG-UI stream metrics."""
    collector = AGUIMetricsCollector(tenant_id)
    collector.stream_started()

    try:
        yield collector
        collector.stream_completed("success")
    except Exception as e:
        collector.stream_completed("error")
        raise
```

**Integration with AGUIBridge:**
```python
# backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py (update)
from agentic_rag_backend.protocols.ag_ui_metrics import track_agui_stream

async def process_request(
    self,
    request: CopilotRequest,
    tenant_id: str,
) -> AsyncIterator[AGUIEvent]:
    """Process CopilotKit request with metrics tracking."""
    async with track_agui_stream(tenant_id) as metrics:
        async for event in self._generate_events(request):
            metrics.event_emitted(event.event.value, len(event.to_sse()))
            yield event
```

**Acceptance Criteria:**
1. All metrics exported to Prometheus
2. Metrics labeled by tenant_id
3. Stream duration tracked with histogram
4. Event latency tracked between events
5. Event counts tracked per stream
6. Active streams gauge accurate
7. Grafana dashboard template provided
8. Tests verify metric emission

---

#### Story 22-B2: Implement Extended AG-UI Error Events

**Priority:** P1 - MEDIUM
**Story Points:** 3
**Owner:** Backend

**Objective:** Add comprehensive error event taxonomy for AG-UI streams.

**Error Event Types:**

| Error Code | HTTP Status | When |
|------------|-------------|------|
| `AGENT_EXECUTION_ERROR` | 500 | Agent throws unhandled exception |
| `TENANT_REQUIRED` | 401 | Missing tenant_id |
| `TENANT_UNAUTHORIZED` | 403 | Invalid tenant_id |
| `SESSION_NOT_FOUND` | 404 | Invalid session reference |
| `RATE_LIMITED` | 429 | Request rate limit exceeded |
| `TIMEOUT` | 504 | Request/response timeout |
| `INVALID_REQUEST` | 400 | Malformed request |
| `CAPABILITY_NOT_FOUND` | 404 | Requested capability unavailable |
| `UPSTREAM_ERROR` | 502 | External service failure |
| `SERVICE_UNAVAILABLE` | 503 | System overloaded |

**Implementation:**

```python
# backend/src/agentic_rag_backend/models/copilot.py (extend)
from enum import Enum

class AGUIErrorCode(str, Enum):
    """Standardized AG-UI error codes."""
    AGENT_EXECUTION_ERROR = "AGENT_EXECUTION_ERROR"
    TENANT_REQUIRED = "TENANT_REQUIRED"
    TENANT_UNAUTHORIZED = "TENANT_UNAUTHORIZED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    INVALID_REQUEST = "INVALID_REQUEST"
    CAPABILITY_NOT_FOUND = "CAPABILITY_NOT_FOUND"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class AGUIErrorEvent(AGUIEvent):
    """Extended error event with standardized codes."""

    event: AGUIEventType = AGUIEventType.RUN_ERROR

    def __init__(
        self,
        code: AGUIErrorCode,
        message: str,
        http_status: int = 500,
        details: dict[str, Any] | None = None,
        retry_after: int | None = None,  # For rate limiting
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data={
                "code": code.value,
                "message": message,
                "http_status": http_status,
                "details": details or {},
                "retry_after": retry_after,
            },
            **kwargs,
        )


def create_error_event(
    exception: Exception,
    is_debug: bool = False,
) -> AGUIErrorEvent:
    """Create appropriate error event from exception."""
    from agentic_rag_backend.protocols.a2a_limits import (
        A2ASessionLimitExceeded,
        A2AMessageLimitExceeded,
        A2ARateLimitExceeded,
    )

    if isinstance(exception, A2ARateLimitExceeded):
        return AGUIErrorEvent(
            code=AGUIErrorCode.RATE_LIMITED,
            message="Request rate limit exceeded. Please wait before retrying.",
            http_status=429,
            retry_after=60,
        )
    elif isinstance(exception, (A2ASessionLimitExceeded, A2AMessageLimitExceeded)):
        return AGUIErrorEvent(
            code=AGUIErrorCode.RATE_LIMITED,
            message="Resource limit exceeded.",
            http_status=429,
        )
    elif isinstance(exception, TimeoutError):
        return AGUIErrorEvent(
            code=AGUIErrorCode.TIMEOUT,
            message="Request timed out. Please try again.",
            http_status=504,
        )
    else:
        return AGUIErrorEvent(
            code=AGUIErrorCode.AGENT_EXECUTION_ERROR,
            message="An error occurred processing your request.",
            http_status=500,
            details={"error_type": type(exception).__name__} if is_debug else None,
        )
```

**Frontend Handling:**
```tsx
// frontend/components/copilot/ErrorHandler.tsx (new)
import { useEffect } from "react";
import { toast } from "@/components/ui/use-toast";

interface AGUIErrorData {
  code: string;
  message: string;
  http_status: number;
  retry_after?: number;
}

export function useAGUIErrorHandler() {
  // Subscribe to AG-UI error events
  // Display appropriate toast/notification
  // Handle retry_after for rate limiting
  // Log errors for debugging
}
```

**Acceptance Criteria:**
1. All error codes defined with HTTP status mapping
2. Error events include retry_after for rate limits
3. create_error_event maps exceptions to codes
4. Frontend displays appropriate error messages
5. Debug mode reveals additional details
6. Tests for each error code

---

### Group C: MCP-UI & Open-JSON-UI Rendering

*Focus: Cross-platform UI interoperability*
*Priority: P1 - Ecosystem expansion*

#### Story 22-C1: Implement MCP-UI Renderer

**Priority:** P1 - MEDIUM
**Story Points:** 8
**Owner:** Frontend + Backend

**Objective:** Add iframe-based rendering for MCP tool UIs.

**What MCP-UI Provides:**
- Embed external interactive tools in chat
- Sandboxed iframe execution
- PostMessage bridge for sizing/events
- CSP and origin allowlisting

**Backend Implementation:**

```python
# backend/src/agentic_rag_backend/protocols/mcp_ui.py (new)
from pydantic import BaseModel, HttpUrl
from typing import Any
import hashlib
import hmac
import time

class MCPUIPayload(BaseModel):
    """MCP-UI iframe payload."""
    type: str = "mcp_ui"
    tool_name: str
    ui_url: HttpUrl
    ui_type: str = "iframe"
    sandbox: list[str] = ["allow-scripts", "allow-same-origin"]
    size: dict[str, int] = {"width": 600, "height": 400}
    allow: list[str] = []  # Permissions policy
    data: dict[str, Any] = {}

class MCPUISignedURL:
    """Generate signed URLs for MCP-UI iframes."""

    def __init__(self, secret_key: str) -> None:
        self.secret_key = secret_key

    def sign_url(
        self,
        base_url: str,
        tool_name: str,
        data: dict[str, Any],
        ttl_seconds: int = 300,
    ) -> str:
        """Generate a signed URL with expiry."""
        expires = int(time.time()) + ttl_seconds
        payload = f"{base_url}|{tool_name}|{expires}"
        signature = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        return f"{base_url}?tool={tool_name}&expires={expires}&sig={signature}"

    def verify_url(self, url: str, tool_name: str, expires: int, signature: str) -> bool:
        """Verify a signed URL."""
        if int(time.time()) > expires:
            return False

        expected_payload = f"{url.split('?')[0]}|{tool_name}|{expires}"
        expected_sig = hmac.new(
            self.secret_key.encode(),
            expected_payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature, expected_sig)
```

**Frontend Implementation:**

```tsx
// frontend/components/copilot/MCPUIRenderer.tsx (new)
import { useRef, useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface MCPUIPayload {
  type: "mcp_ui";
  tool_name: string;
  ui_url: string;
  ui_type: "iframe";
  sandbox: string[];
  size: { width: number; height: number };
  allow: string[];
  data: Record<string, unknown>;
}

interface MCPUIRendererProps {
  payload: MCPUIPayload;
  onResult?: (result: unknown) => void;
}

// Allowed origins for MCP-UI iframes
const ALLOWED_ORIGINS = new Set([
  "https://mcp-ui.example.com",
  "https://tools.copilotkit.ai",
  // Add trusted origins here
]);

export function MCPUIRenderer({ payload, onResult }: MCPUIRendererProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [dimensions, setDimensions] = useState(payload.size);

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      // Verify origin
      if (!ALLOWED_ORIGINS.has(event.origin)) {
        console.warn("MCP-UI: Blocked message from untrusted origin", event.origin);
        return;
      }

      const { type, ...data } = event.data;

      switch (type) {
        case "mcp_ui_resize":
          setDimensions({ width: data.width, height: data.height });
          break;
        case "mcp_ui_result":
          onResult?.(data.result);
          break;
        case "mcp_ui_error":
          console.error("MCP-UI error:", data.error);
          break;
      }
    };

    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, [onResult]);

  // Validate URL is in allowed origins
  const url = new URL(payload.ui_url);
  if (!ALLOWED_ORIGINS.has(url.origin)) {
    return (
      <Card className="border-destructive">
        <CardContent className="p-4 text-destructive">
          MCP-UI blocked: Untrusted origin {url.origin}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="my-2">
      <CardHeader className="p-3">
        <CardTitle className="text-sm font-mono">{payload.tool_name}</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <iframe
          ref={iframeRef}
          src={payload.ui_url}
          sandbox={payload.sandbox.join(" ")}
          allow={payload.allow.join("; ")}
          style={{
            width: dimensions.width,
            height: dimensions.height,
            border: "none",
          }}
          title={`MCP-UI: ${payload.tool_name}`}
        />
      </CardContent>
    </Card>
  );
}
```

**CSP Configuration:**
```typescript
// next.config.js (update)
const ContentSecurityPolicy = `
  frame-src 'self' https://mcp-ui.example.com https://tools.copilotkit.ai;
  child-src 'self' https://mcp-ui.example.com https://tools.copilotkit.ai;
`;
```

**Acceptance Criteria:**
1. MCP-UI payloads parsed and validated
2. Iframe rendered with proper sandbox attributes
3. Origin allowlist enforced
4. PostMessage bridge for resize events
5. Signed URLs for secure tool invocation
6. CSP headers configured
7. Tests for origin validation
8. Tests for postMessage handling

---

#### Story 22-C2: Implement Open-JSON-UI Renderer

**Priority:** P2 - MEDIUM
**Story Points:** 5
**Owner:** Frontend

**Objective:** Add support for OpenAI-style declarative UI payloads.

**Open-JSON-UI Component Types:**

| Type | Description | Mapping |
|------|-------------|---------|
| `text` | Plain text block | `<p>` |
| `heading` | Heading (h1-h6) | `<h1>`-`<h6>` |
| `code` | Code block | `<pre><code>` |
| `list` | Ordered/unordered list | `<ol>/<ul>` |
| `table` | Data table | shadcn Table |
| `image` | Image with alt text | Next.js Image |
| `link` | Hyperlink | `<a>` |
| `button` | Action button | shadcn Button |
| `card` | Content card | shadcn Card |
| `divider` | Horizontal rule | `<hr>` |

**Implementation:**

```tsx
// frontend/components/copilot/OpenJSONUIRenderer.tsx (new)
import { z } from "zod";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import Image from "next/image";

// Schema for Open-JSON-UI components
const OpenJSONUIComponentSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("text"),
    content: z.string(),
    style: z.enum(["normal", "muted", "error", "success"]).optional(),
  }),
  z.object({
    type: z.literal("heading"),
    level: z.number().min(1).max(6),
    content: z.string(),
  }),
  z.object({
    type: z.literal("code"),
    language: z.string().optional(),
    content: z.string(),
  }),
  z.object({
    type: z.literal("list"),
    ordered: z.boolean().default(false),
    items: z.array(z.string()),
  }),
  z.object({
    type: z.literal("table"),
    headers: z.array(z.string()),
    rows: z.array(z.array(z.string())),
    caption: z.string().optional(),
  }),
  z.object({
    type: z.literal("image"),
    src: z.string().url(),
    alt: z.string(),
    width: z.number().optional(),
    height: z.number().optional(),
  }),
  z.object({
    type: z.literal("link"),
    href: z.string().url(),
    text: z.string(),
    target: z.enum(["_self", "_blank"]).default("_blank"),
  }),
  z.object({
    type: z.literal("button"),
    label: z.string(),
    action: z.string(),
    variant: z.enum(["default", "destructive", "outline", "ghost"]).optional(),
  }),
  z.object({
    type: z.literal("card"),
    title: z.string().optional(),
    content: z.string(),
    footer: z.string().optional(),
  }),
  z.object({
    type: z.literal("divider"),
  }),
]);

type OpenJSONUIComponent = z.infer<typeof OpenJSONUIComponentSchema>;

interface OpenJSONUIPayload {
  type: "open_json_ui";
  components: OpenJSONUIComponent[];
}

interface OpenJSONUIRendererProps {
  payload: OpenJSONUIPayload;
  onAction?: (action: string) => void;
}

export function OpenJSONUIRenderer({ payload, onAction }: OpenJSONUIRendererProps) {
  return (
    <div className="space-y-2 my-2">
      {payload.components.map((component, idx) => (
        <OpenJSONUIComponent
          key={idx}
          component={component}
          onAction={onAction}
        />
      ))}
    </div>
  );
}

function OpenJSONUIComponent({
  component,
  onAction,
}: {
  component: OpenJSONUIComponent;
  onAction?: (action: string) => void;
}) {
  switch (component.type) {
    case "text":
      return (
        <p className={getTextStyle(component.style)}>
          {component.content}
        </p>
      );

    case "heading":
      const HeadingTag = `h${component.level}` as keyof JSX.IntrinsicElements;
      return <HeadingTag>{component.content}</HeadingTag>;

    case "code":
      return (
        <pre className="bg-muted p-3 rounded overflow-auto">
          <code className={component.language ? `language-${component.language}` : ""}>
            {component.content}
          </code>
        </pre>
      );

    case "list":
      const ListTag = component.ordered ? "ol" : "ul";
      return (
        <ListTag className={component.ordered ? "list-decimal" : "list-disc"}>
          {component.items.map((item, idx) => (
            <li key={idx}>{item}</li>
          ))}
        </ListTag>
      );

    case "table":
      return (
        <Table>
          {component.caption && <caption>{component.caption}</caption>}
          <TableHeader>
            <TableRow>
              {component.headers.map((header, idx) => (
                <TableHead key={idx}>{header}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {component.rows.map((row, ridx) => (
              <TableRow key={ridx}>
                {row.map((cell, cidx) => (
                  <TableCell key={cidx}>{cell}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      );

    case "image":
      return (
        <Image
          src={component.src}
          alt={component.alt}
          width={component.width || 600}
          height={component.height || 400}
          className="rounded"
        />
      );

    case "link":
      return (
        <a
          href={component.href}
          target={component.target}
          rel={component.target === "_blank" ? "noopener noreferrer" : undefined}
          className="text-primary underline"
        >
          {component.text}
        </a>
      );

    case "button":
      return (
        <Button
          variant={component.variant || "default"}
          onClick={() => onAction?.(component.action)}
        >
          {component.label}
        </Button>
      );

    case "card":
      return (
        <Card>
          {component.title && (
            <CardHeader>
              <CardTitle>{component.title}</CardTitle>
            </CardHeader>
          )}
          <CardContent>{component.content}</CardContent>
        </Card>
      );

    case "divider":
      return <hr className="my-4" />;

    default:
      return (
        <div className="text-muted-foreground text-sm">
          Unsupported component type
        </div>
      );
  }
}

function getTextStyle(style?: string): string {
  switch (style) {
    case "muted":
      return "text-muted-foreground";
    case "error":
      return "text-destructive";
    case "success":
      return "text-green-600";
    default:
      return "";
  }
}
```

**Backend Emission:**
```python
# backend/src/agentic_rag_backend/protocols/open_json_ui.py (new)
from pydantic import BaseModel
from typing import Literal, Any

class OpenJSONUIText(BaseModel):
    type: Literal["text"] = "text"
    content: str
    style: str | None = None

class OpenJSONUITable(BaseModel):
    type: Literal["table"] = "table"
    headers: list[str]
    rows: list[list[str]]
    caption: str | None = None

class OpenJSONUIPayload(BaseModel):
    type: str = "open_json_ui"
    components: list[dict[str, Any]]

def create_open_json_ui(components: list[BaseModel]) -> OpenJSONUIPayload:
    """Create Open-JSON-UI payload from component models."""
    return OpenJSONUIPayload(
        components=[c.model_dump() for c in components]
    )
```

**Acceptance Criteria:**
1. All component types mapped to React components
2. Zod schema validation for payloads
3. Button actions trigger callbacks
4. Images use Next.js Image optimization
5. Tables use shadcn Table components
6. Fallback for unsupported types
7. Tests for each component type

---

### Group D: Protocol Polish & Documentation

*Focus: Final polish and documentation*
*Priority: P2 - Quality*

#### Story 22-D1: Create Protocol Integration Guide

**Priority:** P2 - MEDIUM
**Story Points:** 3
**Owner:** Tech Writing

**Objective:** Document all protocol integrations comprehensively.

**Documentation Structure:**

```
docs/guides/
├── protocol-integration/
│   ├── overview.md           # High-level architecture
│   ├── ag-ui-protocol.md     # AG-UI events, metrics, errors
│   ├── a2a-protocol.md       # A2A middleware, limits
│   ├── mcp-integration.md    # MCP server + client
│   ├── a2ui-widgets.md       # A2UI widget rendering
│   ├── mcp-ui-rendering.md   # MCP-UI iframe embedding
│   └── open-json-ui.md       # Open-JSON-UI components
```

**Each Document Includes:**
- Architecture diagram
- Configuration reference
- Code examples
- Troubleshooting guide
- Security considerations

**Acceptance Criteria:**
1. All protocol docs written
2. Architecture diagrams created (Mermaid)
3. Configuration tables complete
4. Code examples tested
5. Security considerations documented
6. Troubleshooting guides for common issues

---

#### Story 22-D2: Implement Protocol Compliance Tests

**Priority:** P2 - MEDIUM
**Story Points:** 5
**Owner:** QA

**Objective:** Add compliance test suite for all protocols.

**Test Categories:**

| Protocol | Tests |
|----------|-------|
| AG-UI | Event format, stream lifecycle, error codes |
| A2A | Agent registration, delegation, limits |
| MCP | Tool discovery, invocation, error handling |
| A2UI | Widget validation, rendering |
| MCP-UI | Origin validation, CSP, postMessage |
| Open-JSON-UI | Schema validation, component rendering |

**Implementation:**

```python
# tests/protocols/test_ag_ui_compliance.py
import pytest
from agentic_rag_backend.models.copilot import AGUIEventType

class TestAGUIEventCompliance:
    """Verify AG-UI events match specification."""

    def test_run_started_format(self):
        """RUN_STARTED must include run_id."""
        ...

    def test_run_error_includes_code(self):
        """RUN_ERROR must include code from AGUIErrorCode."""
        ...

    def test_state_delta_json_patch(self):
        """STATE_DELTA must use valid JSON Patch operations."""
        ...


class TestAGUIStreamLifecycle:
    """Verify AG-UI stream follows correct lifecycle."""

    async def test_stream_starts_with_run_started(self):
        """First event must be RUN_STARTED."""
        ...

    async def test_stream_ends_with_run_finished_or_error(self):
        """Last event must be RUN_FINISHED or RUN_ERROR."""
        ...
```

```typescript
// tests/protocols/ag-ui-compliance.test.ts
describe("AG-UI Frontend Compliance", () => {
  it("handles RUN_ERROR with retry_after", async () => {
    // Verify rate limit handling
  });

  it("applies STATE_DELTA operations correctly", async () => {
    // Verify JSON Patch application
  });
});
```

**Acceptance Criteria:**
1. Compliance tests for all protocols
2. Tests run in CI/CD pipeline
3. Coverage report for protocol code
4. Contract tests for external interfaces
5. Regression tests for fixed issues

---

## Testing Requirements

### Unit Tests

| Story | Test Focus | Coverage Target |
|-------|------------|-----------------|
| 22-A1 | A2AMiddlewareAgent methods | 90% |
| 22-A2 | A2AResourceManager limits | 95% |
| 22-B1 | Prometheus metrics emission | 85% |
| 22-B2 | Error event creation | 90% |
| 22-C1 | MCP-UI rendering, origin validation | 90% |
| 22-C2 | Open-JSON-UI component rendering | 85% |
| 22-D2 | Protocol compliance | 95% |

### Integration Tests

| Test Scenario | Stories |
|---------------|---------|
| A2A agent registration → discovery → delegation | 22-A1 |
| Session limit → rejection → cleanup | 22-A2 |
| AG-UI stream → metrics → Prometheus | 22-B1 |
| MCP-UI iframe → postMessage → result | 22-C1 |
| Open-JSON-UI payload → render → action | 22-C2 |

### E2E Tests (Playwright)

| Test | Purpose |
|------|---------|
| MCP-UI iframe loads and resizes | Verify iframe integration |
| Open-JSON-UI button triggers action | Verify action callbacks |
| Rate limit toast appears | Verify error handling |

---

## Configuration Summary

### New Environment Variables

```bash
# Epic 22 - Advanced Protocol Integration

# --- A2A Resource Limits (22-A2) ---
A2A_SESSION_LIMIT_PER_TENANT=100
A2A_MESSAGE_LIMIT_PER_SESSION=1000
A2A_SESSION_TTL_HOURS=24
A2A_MESSAGE_RATE_LIMIT=60
A2A_CLEANUP_INTERVAL_MINUTES=15

# --- MCP-UI (22-C1) ---
MCP_UI_ENABLED=true|false
MCP_UI_SIGNING_SECRET=secret-for-signed-urls
MCP_UI_ALLOWED_ORIGINS=https://mcp-ui.example.com,https://tools.copilotkit.ai

# --- Open-JSON-UI (22-C2) ---
OPEN_JSON_UI_ENABLED=true|false
```

---

## Dependencies

### External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `prometheus_client` | ^0.x | Metrics export |
| `jsonpatch` | ^1.x | JSON Patch operations |
| `zod` | ^3.x | Schema validation |

### Internal Dependencies

| Module | Dependency |
|--------|-----------|
| 22-A1 | Epic 7 A2A foundation |
| 22-A2 | 22-A1 (middleware) |
| 22-B1 | Epic 8 observability foundation |
| 22-C1 | Epic 21-C (MCP client) |
| 22-C2 | Epic 21-D (A2UI patterns) |

---

## Sprint Allocation

### Sprint 1 (Stories: 22-A1, 22-A2, 22-B1)
- Focus: A2A middleware + resource limits + telemetry
- Points: 18
- Goal: Multi-agent collaboration with safety limits

### Sprint 2 (Stories: 22-B2, 22-C1, 22-C2)
- Focus: Error handling + MCP-UI + Open-JSON-UI
- Points: 16
- Goal: Cross-platform UI rendering

### Sprint 3 (Stories: 22-D1, 22-D2)
- Focus: Documentation + compliance tests
- Points: 8
- Goal: Production readiness

**Total: 42 story points across 3 sprints**

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| A2A delegation latency | N/A | <500ms p95 | Prometheus histogram |
| Session limit violations | N/A | <1% of requests | Error rate |
| AG-UI stream success rate | Unknown | >99% | Completion rate |
| MCP-UI render success | 0% | 100% | Iframe loads |
| Open-JSON-UI components | 0 | 10 | Supported types |
| Protocol test coverage | 0% | 95% | Test coverage |

---

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| A2A latency overhead | High | Medium | Optimize delegation path |
| iframe security bypass | Critical | Low | Strict CSP, origin validation |
| Open-JSON-UI spec changes | Medium | Low | Pin version, fallback rendering |
| Prometheus cardinality | Medium | Medium | Label cardinality limits |

---

## References

- [A2A Protocol Specification](https://github.com/google/a2a-protocol)
- [MCP-UI Extension](https://modelcontextprotocol.io/ui)
- [AG-UI Protocol](https://github.com/copilotkit/ag-ui)
- [Open-JSON-UI Draft](https://platform.openai.com/docs/api-reference)
- CopilotKit Feature Gap Roadmap (internal)
- Epic 21 Tech Spec (prerequisite)
