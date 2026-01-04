"""MCP Server tool registry.

Manages registration and execution of MCP tools with
tenant isolation and rate limiting.

Story 14-1: Expose RAG Engine via MCP Server
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import structlog

from .types import (
    MCPError,
    MCPErrorCode,
    MCPToolSpec,
    MCPToolResult,
)
from .auth import MCPAuthContext, MCPRateLimiter

logger = structlog.get_logger(__name__)


class MCPServerRegistry:
    """Registry for MCP tools with tenant-isolated execution.

    Manages tool registration, discovery, and execution with:
    - Tenant isolation enforcement
    - Rate limiting per tenant
    - Timeout handling
    - Comprehensive logging
    """

    def __init__(
        self,
        default_timeout_seconds: float = 30.0,
        max_timeout_seconds: float = 300.0,
        rate_limiter: Optional[MCPRateLimiter] = None,
    ) -> None:
        """Initialize the registry.

        Args:
            default_timeout_seconds: Default timeout for tool execution
            max_timeout_seconds: Maximum allowed timeout
            rate_limiter: Optional rate limiter instance
        """
        self._tools: dict[str, MCPToolSpec] = {}
        self._default_timeout = default_timeout_seconds
        self._max_timeout = max_timeout_seconds
        self._rate_limiter = rate_limiter

    def register(self, tool: MCPToolSpec) -> None:
        """Register a tool.

        Args:
            tool: Tool specification to register

        Raises:
            ValueError: If tool name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool
        logger.info(
            "mcp_tool_registered",
            name=tool.name,
            category=tool.category,
            requires_auth=tool.requires_auth,
        )

    def unregister(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was found and removed
        """
        if name in self._tools:
            del self._tools[name]
            logger.info("mcp_tool_unregistered", name=name)
            return True
        return False

    def get_tool(self, name: str) -> Optional[MCPToolSpec]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool specification or None
        """
        return self._tools.get(name)

    def list_tools(
        self,
        category: Optional[str] = None,
        auth_context: Optional[MCPAuthContext] = None,
    ) -> list[dict[str, Any]]:
        """List available tools.

        Args:
            category: Optional category filter
            auth_context: Optional auth context for scope filtering

        Returns:
            List of tool specifications in MCP format
        """
        tools = []
        for tool in self._tools.values():
            # Filter by category
            if category and tool.category != category:
                continue

            # Filter by scope if auth context provided
            if auth_context and tool.requires_auth:
                scope = f"tools:{tool.name}"
                if not auth_context.has_scope(scope):
                    continue

            tools.append(tool.to_dict())

        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        auth_context: Optional[MCPAuthContext] = None,
    ) -> MCPToolResult:
        """Execute a tool.

        Args:
            name: Tool name
            arguments: Tool arguments
            auth_context: Optional authentication context

        Returns:
            MCPToolResult with execution results

        Raises:
            MCPError: If tool not found or execution fails
        """
        tool = self._tools.get(name)
        if not tool:
            raise MCPError(
                code=MCPErrorCode.TOOL_NOT_FOUND,
                message=f"Tool '{name}' not found",
            )

        # Check authentication requirement
        if tool.requires_auth and not auth_context:
            raise MCPError(
                code=MCPErrorCode.AUTHENTICATION_REQUIRED,
                message=f"Tool '{name}' requires authentication",
            )

        # Extract tenant_id from arguments for multi-tenancy
        tenant_id = arguments.get("tenant_id")
        if not tenant_id and auth_context:
            # Use tenant from auth context if not in arguments
            tenant_id = auth_context.tenant_id
            arguments["tenant_id"] = tenant_id

        if not tenant_id:
            raise MCPError(
                code=MCPErrorCode.TENANT_REQUIRED,
                message="tenant_id is required",
            )

        # Validate tenant access
        if auth_context:
            # Check scope
            scope = f"tools:{name}"
            if not auth_context.has_scope(scope):
                raise MCPError(
                    code=MCPErrorCode.AUTHENTICATION_FAILED,
                    message=f"Access denied to tool '{name}'",
                )

            # CRITICAL: Validate cross-tenant access
            # If tenant_id is provided in arguments and differs from auth context tenant,
            # only admin users (scopes=None) can access other tenants
            if tenant_id != auth_context.tenant_id:
                if auth_context.scopes is not None:
                    # Non-admin user attempting cross-tenant access
                    logger.warning(
                        "cross_tenant_access_denied",
                        auth_tenant=auth_context.tenant_id,
                        target_tenant=tenant_id,
                        tool=name,
                    )
                    raise MCPError(
                        code=MCPErrorCode.AUTHENTICATION_FAILED,
                        message="Access denied to tenant",
                    )

        # Rate limiting
        if self._rate_limiter:
            rate_key = f"{tenant_id}:{name}"
            if not await self._rate_limiter.allow(rate_key):
                remaining, reset_in = await self._rate_limiter.get_remaining(rate_key)
                raise MCPError(
                    code=MCPErrorCode.RATE_LIMIT_EXCEEDED,
                    message="Rate limit exceeded",
                    data={
                        "retry_after_seconds": int(reset_in),
                        "remaining": remaining,
                    },
                )

        # Calculate timeout
        timeout = tool.timeout_seconds or self._default_timeout
        timeout = min(timeout, self._max_timeout)

        start_time = time.perf_counter()
        logger.info(
            "mcp_tool_call_started",
            tool=name,
            tenant_id=tenant_id,
            timeout_seconds=timeout,
        )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool.handler(arguments),
                timeout=timeout,
            )

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                "mcp_tool_call_completed",
                tool=name,
                tenant_id=tenant_id,
                elapsed_ms=elapsed_ms,
            )

            # Convert dict result to MCPToolResult
            if isinstance(result, MCPToolResult):
                return result
            elif isinstance(result, dict):
                return MCPToolResult.json(result, {"elapsed_ms": elapsed_ms})
            else:
                return MCPToolResult.text(str(result), {"elapsed_ms": elapsed_ms})

        except asyncio.TimeoutError:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.warning(
                "mcp_tool_call_timeout",
                tool=name,
                tenant_id=tenant_id,
                timeout_seconds=timeout,
                elapsed_ms=elapsed_ms,
            )
            raise MCPError(
                code=MCPErrorCode.TIMEOUT,
                message=f"Tool execution timed out after {timeout} seconds",
                data={"timeout_seconds": timeout},
            )

        except MCPError:
            raise

        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(
                "mcp_tool_call_failed",
                tool=name,
                tenant_id=tenant_id,
                error=str(e),
                elapsed_ms=elapsed_ms,
            )
            raise MCPError(
                code=MCPErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Tool execution failed: {str(e)}",
            ) from e

    def get_categories(self) -> list[str]:
        """Get all registered tool categories.

        Returns:
            Sorted list of unique categories
        """
        categories = set()
        for tool in self._tools.values():
            categories.add(tool.category)
        return sorted(categories)

    def get_tools_by_category(self) -> dict[str, list[str]]:
        """Get tools grouped by category.

        Returns:
            Dict of category -> list of tool names
        """
        result: dict[str, list[str]] = {}
        for tool in self._tools.values():
            if tool.category not in result:
                result[tool.category] = []
            result[tool.category].append(tool.name)
        return result
