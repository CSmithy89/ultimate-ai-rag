"""MCP Server authentication.

Provides authentication mechanisms for MCP server access,
supporting API key authentication with tenant isolation.

Story 14-1: Expose RAG Engine via MCP Server
"""

from __future__ import annotations

import hashlib
import secrets
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import structlog

from .types import MCPError, MCPErrorCode

logger = structlog.get_logger(__name__)


@dataclass
class MCPAuthContext:
    """Authentication context for MCP requests."""

    tenant_id: str
    api_key_id: Optional[str] = None
    scopes: list[str] | None = None
    authenticated_at: float = 0.0

    def has_scope(self, scope: str) -> bool:
        """Check if context has a specific scope."""
        if self.scopes is None:
            return True  # No scopes = all access
        return scope in self.scopes


class MCPAuthenticator(ABC):
    """Abstract base class for MCP authentication."""

    @abstractmethod
    async def authenticate(
        self,
        credentials: dict[str, str],
    ) -> MCPAuthContext:
        """Authenticate credentials and return auth context.

        Args:
            credentials: Authentication credentials dict

        Returns:
            MCPAuthContext with tenant and scope info

        Raises:
            MCPError: If authentication fails
        """
        pass

    @abstractmethod
    async def validate_tenant_access(
        self,
        context: MCPAuthContext,
        tenant_id: str,
    ) -> bool:
        """Validate that the context has access to a tenant.

        Args:
            context: Authentication context
            tenant_id: Target tenant ID

        Returns:
            True if access is allowed
        """
        pass


class MCPAPIKeyAuth(MCPAuthenticator):
    """API key-based authentication for MCP.

    Supports two modes:
    1. Tenant-bound keys: API key is associated with a specific tenant
    2. Admin keys: Can access any tenant (for cross-tenant operations)

    API keys are stored as SHA-256 hashes for security.
    """

    def __init__(
        self,
        api_keys: Optional[dict[str, dict[str, str | list[str]]]] = None,
        admin_keys: Optional[set[str]] = None,
    ) -> None:
        """Initialize API key authenticator.

        Args:
            api_keys: Mapping of API key hash -> {tenant_id, scopes}
            admin_keys: Set of admin API key hashes
        """
        self._api_keys = api_keys or {}
        self._admin_keys = admin_keys or set()
        self._lock = Lock()

    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for secure comparison."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def register_key(
        self,
        api_key: str,
        tenant_id: str,
        scopes: Optional[list[str]] = None,
        is_admin: bool = False,
    ) -> str:
        """Register a new API key.

        Args:
            api_key: The API key to register
            tenant_id: Tenant ID for this key
            scopes: Optional list of scopes
            is_admin: Whether this is an admin key

        Returns:
            The key hash for reference
        """
        key_hash = self._hash_key(api_key)
        with self._lock:
            self._api_keys[key_hash] = {
                "tenant_id": tenant_id,
                "scopes": scopes or [],
            }
            if is_admin:
                self._admin_keys.add(key_hash)
        logger.info(
            "mcp_api_key_registered",
            key_hash=key_hash[:16] + "...",
            tenant_id=tenant_id,
            is_admin=is_admin,
        )
        return key_hash

    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key.

        Args:
            api_key: The API key to revoke

        Returns:
            True if key was found and revoked
        """
        key_hash = self._hash_key(api_key)
        with self._lock:
            if key_hash in self._api_keys:
                del self._api_keys[key_hash]
                self._admin_keys.discard(key_hash)
                logger.info("mcp_api_key_revoked", key_hash=key_hash[:16] + "...")
                return True
        return False

    async def authenticate(
        self,
        credentials: dict[str, str],
    ) -> MCPAuthContext:
        """Authenticate an API key.

        Args:
            credentials: Dict with "api_key" or "authorization" header

        Returns:
            MCPAuthContext with tenant info

        Raises:
            MCPError: If authentication fails
        """
        # Extract API key from various sources
        api_key = credentials.get("api_key")
        if not api_key:
            auth_header = credentials.get("authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]

        if not api_key:
            raise MCPError(
                code=MCPErrorCode.AUTHENTICATION_REQUIRED,
                message="API key is required",
            )

        key_hash = self._hash_key(api_key)

        with self._lock:
            key_info = self._api_keys.get(key_hash)
            if not key_info:
                logger.warning(
                    "mcp_auth_invalid_key",
                    key_hash=key_hash[:16] + "...",
                )
                raise MCPError(
                    code=MCPErrorCode.AUTHENTICATION_FAILED,
                    message="Invalid API key",
                )

            is_admin = key_hash in self._admin_keys

        tenant_id = str(key_info.get("tenant_id", ""))
        scopes = key_info.get("scopes")
        if isinstance(scopes, list):
            scopes_list: list[str] = [str(s) for s in scopes]
        else:
            scopes_list = []

        # Admin keys have all scopes
        if is_admin:
            scopes_list = None  # type: ignore[assignment]

        logger.info(
            "mcp_auth_success",
            tenant_id=tenant_id,
            is_admin=is_admin,
        )

        return MCPAuthContext(
            tenant_id=tenant_id,
            api_key_id=key_hash[:16],
            scopes=scopes_list,
            authenticated_at=time.time(),
        )

    async def validate_tenant_access(
        self,
        context: MCPAuthContext,
        tenant_id: str,
    ) -> bool:
        """Validate tenant access.

        Args:
            context: Authentication context
            tenant_id: Target tenant ID

        Returns:
            True if access is allowed
        """
        # Admin keys (scopes=None) can access any tenant
        if context.scopes is None:
            return True
        # Regular keys can only access their own tenant
        return context.tenant_id == tenant_id


class MCPRateLimiter:
    """Rate limiter for MCP operations.

    Uses a sliding window algorithm to limit requests per tenant.
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = Lock()
        self._requests: dict[str, deque[float]] = {}
        self._last_cleanup = time.time()

    async def allow(self, key: str) -> bool:
        """Check if request is allowed.

        Args:
            key: Rate limit key (typically tenant_id)

        Returns:
            True if request is allowed
        """
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            # Periodic cleanup of stale buckets
            if now - self._last_cleanup > self.window_seconds:
                self._cleanup(now)
                self._last_cleanup = now

            bucket = self._requests.setdefault(key, deque())

            # Remove old entries
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                logger.warning("mcp_rate_limit_exceeded", key=key)
                return False

            bucket.append(now)
            return True

    def _cleanup(self, now: float) -> None:
        """Clean up stale buckets.

        Thread-safety note: This method MUST only be called while holding self._lock.
        It modifies self._requests directly and is not safe to call concurrently.
        The current implementation ensures this by only calling _cleanup from within
        the lock context in the allow() method.
        """
        stale_before = now - (self.window_seconds * 2)
        stale_keys = [
            key
            for key, bucket in self._requests.items()
            if not bucket or bucket[-1] < stale_before
        ]
        for key in stale_keys:
            del self._requests[key]

    async def get_remaining(self, key: str) -> tuple[int, float]:
        """Get remaining requests and reset time.

        Args:
            key: Rate limit key

        Returns:
            Tuple of (remaining_requests, seconds_until_reset)
        """
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            bucket = self._requests.get(key, deque())

            # Remove old entries
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            count = len(bucket)
            remaining = max(0, self.max_requests - count)

            # Time until oldest request expires
            if bucket:
                reset_at = bucket[0] + self.window_seconds
                seconds_until_reset = max(0.0, reset_at - now)
            else:
                seconds_until_reset = 0.0

            return remaining, seconds_until_reset


def generate_api_key(prefix: str = "mcp") -> str:
    """Generate a secure API key.

    Args:
        prefix: Key prefix for identification

    Returns:
        Secure API key string
    """
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}_{random_part}"
