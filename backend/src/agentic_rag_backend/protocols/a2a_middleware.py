"""A2A Middleware Agent for agent delegation and collaboration.

This module provides the A2AMiddlewareAgent class that enables:
- Agent-to-agent message routing
- Capability discovery across registered agents
- Task delegation with context preservation
- Async agent invocation with response handling via AG-UI SSE streaming

Multi-tenancy: All operations are scoped by tenant_id prefix in agent IDs.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx
import structlog
from pydantic import BaseModel, Field

from agentic_rag_backend.core.errors import (
    A2AAgentNotFoundError,
    A2ACapabilityNotFoundError,
    InvalidUrlError,
)

logger = structlog.get_logger(__name__)


def is_safe_endpoint_url(url: str) -> bool:
    """Validate that a URL is safe for making outbound requests (SSRF protection).

    Rejects:
    - Non-HTTP(S) schemes
    - Missing hostnames
    - Localhost variants (127.0.0.1, ::1, localhost, 0.0.0.0)
    - Private IP ranges (10.x.x.x, 192.168.x.x, 172.16-31.x.x)
    - Link-local, loopback, and reserved IP addresses

    Args:
        url: The URL to validate

    Returns:
        True if URL is safe, False otherwise
    """
    try:
        parsed = urlparse(url)

        # Only allow HTTP and HTTPS
        if parsed.scheme not in ("http", "https"):
            logger.warning("ssrf_blocked_invalid_scheme", url=url, scheme=parsed.scheme)
            return False

        hostname = parsed.hostname
        if not hostname:
            logger.warning("ssrf_blocked_no_hostname", url=url)
            return False

        # Block localhost variants
        localhost_variants = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
        if hostname.lower() in localhost_variants:
            logger.warning("ssrf_blocked_localhost", url=url, hostname=hostname)
            return False

        # Check if hostname is an IP address and block private/reserved ranges
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                logger.warning(
                    "ssrf_blocked_private_ip",
                    url=url,
                    hostname=hostname,
                    ip_type=(
                        "private" if ip.is_private
                        else "loopback" if ip.is_loopback
                        else "link_local" if ip.is_link_local
                        else "reserved"
                    ),
                )
                return False
        except ValueError:
            # Not an IP address, likely a domain name - that's fine
            pass

        return True
    except Exception as e:
        logger.warning("ssrf_url_validation_failed", url=url, error=str(e))
        return False


class A2AAgentCapability(BaseModel):
    """Advertised capability of an A2A agent.

    Capabilities describe what tasks an agent can perform,
    including the expected input/output schemas.
    """

    name: str = Field(..., min_length=1, max_length=100, description="Capability name")
    description: str = Field(
        ..., min_length=1, max_length=500, description="Human-readable description"
    )
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for input parameters"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for output"
    )


class A2AAgentInfo(BaseModel):
    """Registered A2A agent information.

    Contains all metadata needed to discover and invoke an agent.
    The agent_id MUST be prefixed with the tenant_id for multi-tenancy.
    """

    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique agent ID (must be prefixed with tenant_id:)",
    )
    name: str = Field(
        ..., min_length=1, max_length=100, description="Human-readable agent name"
    )
    description: str = Field(
        ..., min_length=1, max_length=500, description="Agent description"
    )
    capabilities: list[A2AAgentCapability] = Field(
        default_factory=list, description="List of agent capabilities"
    )
    endpoint: str = Field(
        ..., min_length=1, max_length=500, description="AG-UI endpoint URL"
    )


class A2AMiddlewareAgent:
    """Middleware agent for A2A protocol collaboration.

    This agent acts as a broker between agents, enabling:
    - Registration of agents with their capabilities
    - Discovery of capabilities across the agent network
    - Task delegation via AG-UI SSE streaming

    All operations are tenant-scoped through agent_id prefixes.

    Example:
        middleware = A2AMiddlewareAgent(
            agent_id="system:middleware",
            name="RAG Middleware",
            capabilities=[],
        )
        middleware.register_agent(agent_info)
        capabilities = middleware.list_agents_for_tenant("tenant123")

        async for event in middleware.delegate_task(
            target_agent_id="tenant123:search-agent",
            capability_name="vector_search",
            input_data={"query": "What is RAG?"},
        ):
            print(event)

        await middleware.close()
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        capabilities: list[A2AAgentCapability] | None = None,
    ) -> None:
        """Initialize the A2A middleware agent.

        Args:
            agent_id: Unique identifier for this middleware agent
            name: Human-readable name for this middleware
            capabilities: List of capabilities this middleware itself offers
        """
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities or []
        self._registered_agents: dict[str, A2AAgentInfo] = {}
        self._http_client: httpx.AsyncClient | None = None
        self._http_client_lock = asyncio.Lock()

        logger.info(
            "a2a_middleware_initialized",
            agent_id=agent_id,
            name=name,
        )

    def register_agent(self, agent_info: A2AAgentInfo) -> None:
        """Register an agent for collaboration.

        The agent is stored in the internal registry keyed by agent_id.
        Registration is idempotent - re-registering updates the existing entry.

        Args:
            agent_info: Agent information including capabilities and endpoint
        """
        self._registered_agents[agent_info.agent_id] = agent_info
        logger.info(
            "a2a_agent_registered",
            agent_id=agent_info.agent_id,
            name=agent_info.name,
            capabilities=[c.name for c in agent_info.capabilities],
        )

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the middleware.

        Args:
            agent_id: The agent ID to unregister

        Returns:
            True if agent was unregistered, False if not found
        """
        if agent_id in self._registered_agents:
            del self._registered_agents[agent_id]
            logger.info("a2a_agent_unregistered", agent_id=agent_id)
            return True
        return False

    def get_agent(self, agent_id: str) -> A2AAgentInfo | None:
        """Get a specific agent by ID.

        Args:
            agent_id: The agent ID to retrieve

        Returns:
            Agent info if found, None otherwise
        """
        return self._registered_agents.get(agent_id)

    def list_agents_for_tenant(self, tenant_id: str) -> list[A2AAgentInfo]:
        """List all registered agents for a specific tenant.

        Filters agents by tenant_id prefix in their agent_id.

        Args:
            tenant_id: The tenant ID to filter by

        Returns:
            List of agents belonging to the tenant
        """
        prefix = f"{tenant_id}:"
        return [
            agent
            for agent in self._registered_agents.values()
            if agent.agent_id.startswith(prefix)
        ]

    def discover_capabilities(
        self,
        tenant_id: str,
        capability_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Discover capabilities across all registered agents for a tenant.

        Args:
            tenant_id: The tenant ID to filter agents by
            capability_filter: Optional substring to filter capability names

        Returns:
            List of dicts with agent_id and capability info
        """
        results: list[dict[str, Any]] = []
        prefix = f"{tenant_id}:"

        for agent_id, agent_info in self._registered_agents.items():
            if not agent_id.startswith(prefix):
                continue

            for cap in agent_info.capabilities:
                if capability_filter is None or capability_filter.lower() in cap.name.lower():
                    results.append({
                        "agent_id": agent_id,
                        "capability": cap.model_dump(),
                    })

        return results

    async def delegate_task(
        self,
        target_agent_id: str,
        capability_name: str,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Delegate a task to another agent via AG-UI SSE streaming.

        Args:
            target_agent_id: ID of the agent to delegate to
            capability_name: Name of the capability to invoke
            input_data: Input parameters for the capability
            context: Optional context to pass to the agent

        Yields:
            Events from the agent's SSE stream

        Raises:
            A2AAgentNotFoundError: If target agent is not registered
            A2ACapabilityNotFoundError: If agent doesn't have the capability
        """
        agent_info = self._registered_agents.get(target_agent_id)
        if not agent_info:
            logger.warning(
                "a2a_delegation_agent_not_found",
                target_agent_id=target_agent_id,
            )
            raise A2AAgentNotFoundError(target_agent_id)

        capability = next(
            (c for c in agent_info.capabilities if c.name == capability_name),
            None,
        )
        if not capability:
            logger.warning(
                "a2a_delegation_capability_not_found",
                target_agent_id=target_agent_id,
                capability_name=capability_name,
                available_capabilities=[c.name for c in agent_info.capabilities],
            )
            raise A2ACapabilityNotFoundError(capability_name)

        logger.info(
            "a2a_task_delegated",
            from_agent=self.agent_id,
            to_agent=target_agent_id,
            capability=capability_name,
        )

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
        """Invoke an agent's capability via AG-UI protocol.

        Uses Server-Sent Events (SSE) for streaming responses.

        Args:
            endpoint: Agent's AG-UI endpoint URL
            capability: Name of the capability to invoke
            input_data: Input parameters
            context: Optional context

        Yields:
            Parsed JSON events from the SSE stream

        Raises:
            InvalidUrlError: If the endpoint URL is not safe (SSRF protection)
        """
        # SSRF protection: Validate endpoint URL before making request
        if not is_safe_endpoint_url(endpoint):
            raise InvalidUrlError(
                endpoint,
                "Endpoint URL is not allowed (SSRF protection: private/internal IPs blocked)",
            )

        client = await self._get_http_client()

        try:
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
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            yield json.loads(line[6:])
                        except json.JSONDecodeError as e:
                            logger.warning(
                                "a2a_sse_parse_error",
                                endpoint=endpoint,
                                error=str(e),
                            )
        except httpx.HTTPStatusError as e:
            logger.error(
                "a2a_invocation_http_error",
                endpoint=endpoint,
                status_code=e.response.status_code,
            )
            raise
        except httpx.RequestError as e:
            logger.error(
                "a2a_invocation_request_error",
                endpoint=endpoint,
                error=str(e),
            )
            raise

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create pooled HTTP client with thread-safe lazy initialization.

        Uses an async lock to prevent race conditions when multiple concurrent
        calls attempt to create the client simultaneously.

        Returns:
            Shared HTTP client with connection pooling
        """
        async with self._http_client_lock:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                    timeout=httpx.Timeout(30.0, connect=5.0),
                )
            return self._http_client

    async def close(self) -> None:
        """Close the HTTP client pool.

        Should be called during application shutdown.
        """
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
            logger.info("a2a_middleware_http_client_closed")
