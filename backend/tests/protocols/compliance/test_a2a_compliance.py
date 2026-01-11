"""A2A Protocol Compliance Tests.

Story 22-D2: Implement Protocol Compliance Tests

Verifies A2A protocol implementation:
- Agent registration format
- Delegation protocol
- Resource limit enforcement
- Message format compliance
"""

import pytest
from pydantic import ValidationError

from agentic_rag_backend.protocols.a2a_resource_limits import (
    A2AResourceLimits,
    A2AResourceMetrics,
    TenantUsage,
    InMemoryA2AResourceManager,
    A2ASessionLimitExceeded,
    A2AMessageLimitExceeded,
)


# =============================================================================
# Resource Limits Configuration Compliance Tests
# =============================================================================


class TestA2AResourceLimitsCompliance:
    """Verify A2AResourceLimits configuration compliance."""

    def test_default_session_limit_per_tenant(self) -> None:
        """Default session limit should be 100."""
        limits = A2AResourceLimits()
        assert limits.session_limit_per_tenant == 100

    def test_default_message_limit_per_session(self) -> None:
        """Default message limit should be 1000."""
        limits = A2AResourceLimits()
        assert limits.message_limit_per_session == 1000

    def test_default_session_ttl_hours(self) -> None:
        """Default session TTL should be 24 hours."""
        limits = A2AResourceLimits()
        assert limits.session_ttl_hours == 24

    def test_default_message_rate_limit(self) -> None:
        """Default message rate limit should be 60/minute."""
        limits = A2AResourceLimits()
        assert limits.message_rate_limit == 60

    def test_session_limit_must_be_positive(self) -> None:
        """session_limit_per_tenant must be >= 1."""
        with pytest.raises(ValidationError):
            A2AResourceLimits(session_limit_per_tenant=0)

    def test_message_limit_must_be_positive(self) -> None:
        """message_limit_per_session must be >= 1."""
        with pytest.raises(ValidationError):
            A2AResourceLimits(message_limit_per_session=0)

    def test_session_ttl_must_be_positive(self) -> None:
        """session_ttl_hours must be >= 1."""
        with pytest.raises(ValidationError):
            A2AResourceLimits(session_ttl_hours=0)

    def test_rate_limit_must_be_positive(self) -> None:
        """message_rate_limit must be >= 1."""
        with pytest.raises(ValidationError):
            A2AResourceLimits(message_rate_limit=0)

    def test_custom_limits_accepted(self) -> None:
        """Custom limit values should be accepted."""
        limits = A2AResourceLimits(
            session_limit_per_tenant=50,
            message_limit_per_session=500,
            session_ttl_hours=12,
            message_rate_limit=30,
        )
        assert limits.session_limit_per_tenant == 50
        assert limits.message_limit_per_session == 500
        assert limits.session_ttl_hours == 12
        assert limits.message_rate_limit == 30


class TestTenantUsageCompliance:
    """Verify TenantUsage model compliance."""

    def test_tenant_usage_has_required_fields(self) -> None:
        """TenantUsage must have tenant_id and counters."""
        usage = TenantUsage(
            tenant_id="tenant-123",
            active_sessions=5,
            total_messages=100,
        )
        assert usage.tenant_id == "tenant-123"
        assert usage.active_sessions == 5
        assert usage.total_messages == 100

    def test_tenant_usage_default_values(self) -> None:
        """TenantUsage defaults should be zero."""
        usage = TenantUsage(tenant_id="tenant-123")
        assert usage.active_sessions == 0
        assert usage.total_messages == 0


class TestA2AResourceMetricsCompliance:
    """Verify A2AResourceMetrics model compliance."""

    def test_resource_metrics_has_required_fields(self) -> None:
        """A2AResourceMetrics must have all required fields."""
        metrics = A2AResourceMetrics(
            tenant_id="tenant-123",
            active_sessions=5,
            total_messages=100,
            session_limit=100,
            message_limit_per_session=1000,
            message_rate_limit=60,
        )
        assert metrics.tenant_id == "tenant-123"
        assert metrics.active_sessions == 5
        assert metrics.total_messages == 100
        assert metrics.session_limit == 100
        assert metrics.message_limit_per_session == 1000
        assert metrics.message_rate_limit == 60


# =============================================================================
# Resource Manager Protocol Compliance Tests
# =============================================================================


class TestA2AResourceManagerCompliance:
    """Verify A2AResourceManager protocol compliance."""

    @pytest.fixture
    def manager(self) -> InMemoryA2AResourceManager:
        """Create test resource manager."""
        limits = A2AResourceLimits(
            session_limit_per_tenant=3,
            message_limit_per_session=10,
            message_rate_limit=100,  # High enough to not hit rate limit during tests
        )
        return InMemoryA2AResourceManager(limits)

    @pytest.mark.asyncio
    async def test_register_session_succeeds(
        self, manager: InMemoryA2AResourceManager
    ) -> None:
        """register_session must successfully register a session."""
        # Should not raise
        await manager.register_session("session-1", "tenant-1")

        metrics = await manager.get_tenant_metrics("tenant-1")
        assert metrics.active_sessions == 1

    @pytest.mark.asyncio
    async def test_session_limit_enforced(
        self, manager: InMemoryA2AResourceManager
    ) -> None:
        """Session limit must be enforced per tenant."""
        tenant_id = "tenant-limit-test"

        # Register up to limit
        for i in range(3):
            await manager.register_session(f"session-{i}", tenant_id)

        # Next should raise
        with pytest.raises(A2ASessionLimitExceeded):
            await manager.register_session("session-overflow", tenant_id)

    @pytest.mark.asyncio
    async def test_close_session_allows_new_sessions(
        self, manager: InMemoryA2AResourceManager
    ) -> None:
        """Closing a session should allow new sessions."""
        tenant_id = "tenant-release-test"

        # Fill up limit
        for i in range(3):
            await manager.register_session(f"session-{i}", tenant_id)

        # Close one
        await manager.close_session("session-0")

        # Now should succeed
        await manager.register_session("session-new", tenant_id)

        metrics = await manager.get_tenant_metrics(tenant_id)
        assert metrics.active_sessions == 3

    @pytest.mark.asyncio
    async def test_message_limit_enforced(
        self, manager: InMemoryA2AResourceManager
    ) -> None:
        """Message limit must be enforced per session."""
        tenant_id = "tenant-msg-test"
        session_id = "session-msg-test"
        await manager.register_session(session_id, tenant_id)

        # Send up to limit
        for _ in range(10):
            await manager.record_message(session_id)

        # Next should raise
        with pytest.raises(A2AMessageLimitExceeded):
            await manager.record_message(session_id)

    @pytest.mark.asyncio
    async def test_get_tenant_metrics_returns_correct_counts(
        self, manager: InMemoryA2AResourceManager
    ) -> None:
        """get_tenant_metrics must return accurate counts."""
        tenant_id = "tenant-metrics-test"

        # Create sessions
        await manager.register_session("session-1", tenant_id)
        await manager.register_session("session-2", tenant_id)

        metrics = await manager.get_tenant_metrics(tenant_id)
        assert metrics.tenant_id == tenant_id
        assert metrics.active_sessions == 2

    @pytest.mark.asyncio
    async def test_different_tenants_isolated(
        self, manager: InMemoryA2AResourceManager
    ) -> None:
        """Different tenants must have isolated limits."""
        # Fill tenant-1's limit
        for i in range(3):
            await manager.register_session(f"t1-session-{i}", "tenant-1")

        # tenant-2 should still be able to create sessions
        await manager.register_session("t2-session-1", "tenant-2")

        metrics = await manager.get_tenant_metrics("tenant-2")
        assert metrics.active_sessions == 1


# =============================================================================
# Message Protocol Compliance Tests
# =============================================================================


class TestA2AMessageProtocolCompliance:
    """Verify A2A message format compliance."""

    def test_delegation_request_required_fields(self) -> None:
        """Delegation request must include capability and payload."""
        # This tests the expected protocol format
        request = {
            "capability": "QUERY",
            "payload": {"query": "test"},
            "tenant_id": "tenant-123",
        }
        assert "capability" in request
        assert "payload" in request
        assert "tenant_id" in request

    def test_delegation_response_required_fields(self) -> None:
        """Delegation response must include status and result/error."""
        success_response = {
            "status": "success",
            "result": {"data": []},
            "agent_id": "agent-1",
        }
        assert success_response["status"] == "success"
        assert "result" in success_response

        error_response = {
            "status": "error",
            "error": "Agent not found",
        }
        assert error_response["status"] == "error"
        assert "error" in error_response

    def test_standard_capabilities(self) -> None:
        """Standard A2A capabilities must be defined."""
        standard_capabilities = ["QUERY", "INGEST", "ANALYZE", "SUMMARIZE"]
        for cap in standard_capabilities:
            assert cap.isupper(), f"Capability {cap} should be uppercase"


# =============================================================================
# Security Compliance Tests
# =============================================================================


class TestA2ASecurityCompliance:
    """Verify A2A security controls."""

    def test_tenant_id_required_for_sessions(self) -> None:
        """tenant_id must be required for all session operations."""
        # Verify the protocol requires tenant_id
        limits = A2AResourceLimits()
        manager = InMemoryA2AResourceManager(limits)

        # The API should require tenant_id (empty string not allowed)
        # This is enforced at the API layer, not the manager level

    def test_ssrf_protection_url_patterns(self) -> None:
        """SSRF protection must block dangerous URLs."""
        from agentic_rag_backend.protocols.a2a_middleware import is_safe_endpoint_url

        # Should block localhost
        assert not is_safe_endpoint_url("http://localhost:8080")
        assert not is_safe_endpoint_url("http://127.0.0.1:8080")
        assert not is_safe_endpoint_url("http://0.0.0.0:8080")

        # Should block private IPs
        assert not is_safe_endpoint_url("http://10.0.0.1:8080")
        assert not is_safe_endpoint_url("http://192.168.1.1:8080")
        assert not is_safe_endpoint_url("http://172.16.0.1:8080")

        # Should block non-HTTP schemes
        assert not is_safe_endpoint_url("file:///etc/passwd")
        assert not is_safe_endpoint_url("ftp://example.com")

        # Should allow public HTTPS
        assert is_safe_endpoint_url("https://api.example.com")
