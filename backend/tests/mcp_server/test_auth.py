"""Tests for MCP server authentication.

Story 14-1: Expose RAG Engine via MCP Server
"""

import pytest
import time

from agentic_rag_backend.mcp_server.auth import (
    MCPAPIKeyAuth,
    MCPAuthContext,
    MCPRateLimiter,
    generate_api_key,
)
from agentic_rag_backend.mcp_server.types import MCPError, MCPErrorCode


class TestMCPAuthContext:
    """Tests for MCPAuthContext."""

    def test_context_creation(self):
        """Test creating auth context."""
        ctx = MCPAuthContext(
            tenant_id="test-tenant",
            api_key_id="key-123",
            scopes=["tools:read", "tools:write"],
            authenticated_at=time.time(),
        )
        assert ctx.tenant_id == "test-tenant"
        assert ctx.api_key_id == "key-123"
        assert len(ctx.scopes) == 2

    def test_has_scope_with_scopes(self):
        """Test scope checking with explicit scopes."""
        ctx = MCPAuthContext(
            tenant_id="test",
            scopes=["tools:read"],
        )
        assert ctx.has_scope("tools:read")
        assert not ctx.has_scope("tools:write")

    def test_has_scope_without_scopes(self):
        """Test scope checking without scopes (admin)."""
        ctx = MCPAuthContext(
            tenant_id="test",
            scopes=None,  # Admin - all access
        )
        assert ctx.has_scope("tools:read")
        assert ctx.has_scope("anything")


class TestMCPAPIKeyAuth:
    """Tests for MCPAPIKeyAuth."""

    def test_register_key(self):
        """Test registering an API key."""
        auth = MCPAPIKeyAuth()
        key = generate_api_key()
        key_hash = auth.register_key(
            api_key=key,
            tenant_id="tenant-123",
            scopes=["tools:read"],
        )
        assert key_hash is not None
        assert len(key_hash) == 64  # SHA-256 hash

    @pytest.mark.asyncio
    async def test_authenticate_valid_key(self):
        """Test authenticating with a valid key."""
        auth = MCPAPIKeyAuth()
        key = generate_api_key()
        auth.register_key(
            api_key=key,
            tenant_id="tenant-123",
            scopes=["tools:read"],
        )

        ctx = await auth.authenticate({"api_key": key})
        assert ctx.tenant_id == "tenant-123"
        assert "tools:read" in ctx.scopes

    @pytest.mark.asyncio
    async def test_authenticate_bearer_token(self):
        """Test authenticating with Bearer token format."""
        auth = MCPAPIKeyAuth()
        key = generate_api_key()
        auth.register_key(api_key=key, tenant_id="tenant-456")

        ctx = await auth.authenticate({"authorization": f"Bearer {key}"})
        assert ctx.tenant_id == "tenant-456"

    @pytest.mark.asyncio
    async def test_authenticate_invalid_key(self):
        """Test authenticating with invalid key."""
        auth = MCPAPIKeyAuth()
        with pytest.raises(MCPError) as exc_info:
            await auth.authenticate({"api_key": "invalid-key"})
        assert exc_info.value.code == MCPErrorCode.AUTHENTICATION_FAILED

    @pytest.mark.asyncio
    async def test_authenticate_missing_key(self):
        """Test authenticating without key."""
        auth = MCPAPIKeyAuth()
        with pytest.raises(MCPError) as exc_info:
            await auth.authenticate({})
        assert exc_info.value.code == MCPErrorCode.AUTHENTICATION_REQUIRED

    @pytest.mark.asyncio
    async def test_admin_key(self):
        """Test admin key has null scopes."""
        auth = MCPAPIKeyAuth()
        key = generate_api_key()
        auth.register_key(
            api_key=key,
            tenant_id="admin-tenant",
            is_admin=True,
        )

        ctx = await auth.authenticate({"api_key": key})
        assert ctx.scopes is None  # Admin has all access
        assert ctx.has_scope("anything")

    @pytest.mark.asyncio
    async def test_validate_tenant_access_same_tenant(self):
        """Test tenant access validation for same tenant."""
        auth = MCPAPIKeyAuth()
        ctx = MCPAuthContext(tenant_id="tenant-1", scopes=["tools:read"])
        assert await auth.validate_tenant_access(ctx, "tenant-1")

    @pytest.mark.asyncio
    async def test_validate_tenant_access_different_tenant(self):
        """Test tenant access validation for different tenant."""
        auth = MCPAPIKeyAuth()
        ctx = MCPAuthContext(tenant_id="tenant-1", scopes=["tools:read"])
        assert not await auth.validate_tenant_access(ctx, "tenant-2")

    @pytest.mark.asyncio
    async def test_validate_tenant_access_admin(self):
        """Test admin can access any tenant."""
        auth = MCPAPIKeyAuth()
        ctx = MCPAuthContext(tenant_id="admin", scopes=None)  # Admin
        assert await auth.validate_tenant_access(ctx, "tenant-1")
        assert await auth.validate_tenant_access(ctx, "tenant-2")

    def test_revoke_key(self):
        """Test revoking an API key."""
        auth = MCPAPIKeyAuth()
        key = generate_api_key()
        auth.register_key(api_key=key, tenant_id="tenant")

        assert auth.revoke_key(key)
        assert not auth.revoke_key(key)  # Already revoked


class TestMCPRateLimiter:
    """Tests for MCPRateLimiter."""

    @pytest.mark.asyncio
    async def test_allow_under_limit(self):
        """Test requests under limit are allowed."""
        limiter = MCPRateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert await limiter.allow("tenant-1")

    @pytest.mark.asyncio
    async def test_deny_over_limit(self):
        """Test requests over limit are denied."""
        limiter = MCPRateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            await limiter.allow("tenant-1")
        assert not await limiter.allow("tenant-1")

    @pytest.mark.asyncio
    async def test_separate_buckets(self):
        """Test different keys have separate buckets."""
        limiter = MCPRateLimiter(max_requests=2, window_seconds=60)

        assert await limiter.allow("tenant-1")
        assert await limiter.allow("tenant-1")
        assert not await limiter.allow("tenant-1")

        # Different tenant should have its own bucket
        assert await limiter.allow("tenant-2")

    @pytest.mark.asyncio
    async def test_get_remaining(self):
        """Test getting remaining requests."""
        limiter = MCPRateLimiter(max_requests=5, window_seconds=60)
        await limiter.allow("tenant-1")
        await limiter.allow("tenant-1")

        remaining, _ = await limiter.get_remaining("tenant-1")
        assert remaining == 3


class TestGenerateAPIKey:
    """Tests for API key generation."""

    def test_generate_key_format(self):
        """Test generated key format."""
        key = generate_api_key()
        assert key.startswith("mcp_")
        assert len(key) > 40  # Prefix + random part

    def test_generate_key_custom_prefix(self):
        """Test generating key with custom prefix."""
        key = generate_api_key(prefix="custom")
        assert key.startswith("custom_")

    def test_generate_key_uniqueness(self):
        """Test generated keys are unique."""
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100
