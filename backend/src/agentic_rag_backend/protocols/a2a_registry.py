"""A2A Agent Registry with health monitoring and heartbeat detection.

This module provides agent registration, discovery, and health monitoring
for the A2A (Agent-to-Agent) protocol. Agents register with their capabilities
and are monitored via periodic heartbeats.
"""

from __future__ import annotations

import asyncio
import json
from asyncio import Lock
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import structlog

from agentic_rag_backend.db.redis import RedisClient

if TYPE_CHECKING:
    import redis

from .a2a_messages import (
    AgentCapability,
    AgentRegistration,
    get_implemented_rag_capabilities,
)

logger = structlog.get_logger(__name__)


import re

# Constants for TTL and cleanup multipliers (documented rationale)
# Registration TTL: 2x heartbeat timeout provides buffer for network delays
REGISTRATION_TTL_MULTIPLIER = 2
# Cleanup threshold: 3x heartbeat timeout ensures we don't prematurely remove agents
# that are slow to respond but still functioning
CLEANUP_THRESHOLD_MULTIPLIER = 3

# Agent ID validation (prevents Redis key injection)
# Must be alphanumeric with hyphens, underscores, or dots, max 128 chars
MAX_AGENT_ID_LENGTH = 128
AGENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")


@dataclass
class RegistryConfig:
    """Configuration for the A2A agent registry.

    Attributes:
        heartbeat_interval_seconds: Time between expected heartbeats
        heartbeat_timeout_seconds: Time after which missing heartbeat marks unhealthy
        cleanup_interval_seconds: Time between registry cleanup runs
        redis_prefix: Prefix for Redis keys
        registration_ttl_multiplier: Multiplier for registration TTL (default 2x timeout)
        cleanup_threshold_multiplier: Multiplier for cleanup threshold (default 3x timeout)
    """

    heartbeat_interval_seconds: int = 30
    heartbeat_timeout_seconds: int = 60
    cleanup_interval_seconds: int = 60
    redis_prefix: str = "a2a:registry"
    registration_ttl_multiplier: int = REGISTRATION_TTL_MULTIPLIER
    cleanup_threshold_multiplier: int = CLEANUP_THRESHOLD_MULTIPLIER


class A2AAgentRegistry:
    """Registry for A2A agents with health monitoring and Redis persistence.

    Provides:
    - Agent registration with capability declaration
    - Heartbeat-based health monitoring
    - Agent discovery by capability or tenant
    - Automatic health status updates based on heartbeat timing
    - Redis persistence for restart recovery

    Multi-tenancy: All operations enforce tenant_id filtering to ensure
    isolation between tenants.
    """

    def __init__(
        self,
        config: Optional[RegistryConfig] = None,
        redis_client: Optional[RedisClient] = None,
    ) -> None:
        """Initialize the agent registry.

        Args:
            config: Configuration for registry behavior
            redis_client: Optional Redis client for persistence
        """
        self._config = config or RegistryConfig()
        self._agents: dict[str, AgentRegistration] = {}
        self._lock = Lock()
        self._redis_client = redis_client
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._last_cleanup = 0.0

    # ==================== Redis Persistence ====================

    def _agent_key(self, agent_id: str) -> str:
        """Generate Redis key for an agent."""
        return f"{self._config.redis_prefix}:agent:{agent_id}"

    def _tenant_index_key(self, tenant_id: str) -> str:
        """Generate Redis key for tenant's agent index."""
        return f"{self._config.redis_prefix}:tenant:{tenant_id}:agents"

    def _get_redis(self) -> "redis.Redis | None":
        """Get Redis client if available."""
        if not self._redis_client:
            return None
        try:
            return self._redis_client.client
        except Exception as exc:
            logger.warning("a2a_registry_redis_unavailable", error=str(exc))
            return None

    async def _persist_agent(self, agent: AgentRegistration) -> None:
        """Persist agent registration to Redis."""
        redis = self._get_redis()
        if not redis:
            return
        try:
            key = self._agent_key(agent.agent_id)
            payload = json.dumps(agent.to_dict())
            # Store agent with configurable TTL multiplier for automatic cleanup
            ttl = self._config.heartbeat_timeout_seconds * self._config.registration_ttl_multiplier
            await redis.set(key, payload, ex=ttl)

            # Add to tenant index
            tenant_key = self._tenant_index_key(agent.tenant_id)
            await redis.sadd(tenant_key, agent.agent_id)
            await redis.expire(tenant_key, ttl)
        except Exception as exc:
            logger.warning("a2a_agent_persist_failed", agent_id=agent.agent_id, error=str(exc))

    async def _load_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Load agent registration from Redis."""
        redis = self._get_redis()
        if not redis:
            return None
        try:
            key = self._agent_key(agent_id)
            payload = await redis.get(key)
            if not payload:
                return None
            raw = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
            data = json.loads(raw)
            return AgentRegistration.from_dict(data)
        except Exception as exc:
            logger.warning("a2a_agent_load_failed", agent_id=agent_id, error=str(exc))
            return None

    async def _remove_agent_from_redis(self, agent_id: str, tenant_id: str) -> None:
        """Remove agent from Redis."""
        redis = self._get_redis()
        if not redis:
            return
        try:
            await redis.delete(self._agent_key(agent_id))
            await redis.srem(self._tenant_index_key(tenant_id), agent_id)
        except Exception as exc:
            logger.warning("a2a_agent_remove_failed", agent_id=agent_id, error=str(exc))

    async def _load_tenant_agents(self, tenant_id: str) -> list[str]:
        """Load list of agent IDs for a tenant from Redis."""
        redis = self._get_redis()
        if not redis:
            return []
        try:
            tenant_key = self._tenant_index_key(tenant_id)
            agent_ids = await redis.smembers(tenant_key)
            return [
                aid.decode("utf-8") if isinstance(aid, (bytes, bytearray)) else aid
                for aid in agent_ids
            ]
        except Exception as exc:
            logger.warning("a2a_tenant_agents_load_failed", tenant_id=tenant_id, error=str(exc))
            return []

    # ==================== Health Monitoring ====================

    def _is_agent_healthy(self, agent: AgentRegistration) -> bool:
        """Check if agent is healthy based on heartbeat timing."""
        now = datetime.now(timezone.utc)
        elapsed = (now - agent.last_heartbeat).total_seconds()
        return elapsed <= self._config.heartbeat_timeout_seconds

    def _update_health_status(self, agent: AgentRegistration) -> None:
        """Update agent health status based on current heartbeat state."""
        agent.health_status = "healthy" if self._is_agent_healthy(agent) else "unhealthy"

    async def _cleanup_unhealthy_agents(self) -> None:
        """Remove agents that have been unhealthy beyond tolerance."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            agents_to_remove: list[str] = []

            for agent_id, agent in self._agents.items():
                self._update_health_status(agent)
                elapsed = (now - agent.last_heartbeat).total_seconds()
                # Remove if exceeds configurable cleanup threshold
                cleanup_threshold = (
                    self._config.heartbeat_timeout_seconds
                    * self._config.cleanup_threshold_multiplier
                )
                if elapsed > cleanup_threshold:
                    agents_to_remove.append(agent_id)

            for agent_id in agents_to_remove:
                agent = self._agents.pop(agent_id, None)
                if agent:
                    await self._remove_agent_from_redis(agent_id, agent.tenant_id)
                    logger.info(
                        "a2a_agent_removed_unhealthy",
                        agent_id=agent_id,
                        tenant_id=agent.tenant_id,
                    )

    async def _periodic_cleanup_task(self) -> None:
        """Background task for periodic health checks and cleanup."""
        try:
            while True:
                await asyncio.sleep(self._config.cleanup_interval_seconds)
                await self._cleanup_unhealthy_agents()
        except asyncio.CancelledError:
            return

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            return
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup_task())
        logger.info("a2a_registry_cleanup_started")

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        if not self._cleanup_task:
            return
        self._cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._cleanup_task
        self._cleanup_task = None
        logger.info("a2a_registry_cleanup_stopped")

    # ==================== Registration Operations ====================

    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        endpoint_url: str,
        capabilities: list[AgentCapability],
        tenant_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentRegistration:
        """Register a new agent or update existing registration.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type classification of the agent
            endpoint_url: HTTP endpoint for task delegation
            capabilities: List of capabilities the agent offers
            tenant_id: Tenant scope for multi-tenancy
            metadata: Optional additional metadata

        Returns:
            The created or updated AgentRegistration

        Raises:
            ValueError: If registration parameters are invalid
        """
        if not agent_id or not agent_type or not endpoint_url or not tenant_id:
            raise ValueError("agent_id, agent_type, endpoint_url, and tenant_id are required")

        # Validate agent_id format to prevent Redis key injection
        # Must be alphanumeric with hyphens, underscores, or dots
        if len(agent_id) > MAX_AGENT_ID_LENGTH:
            raise ValueError(f"agent_id exceeds maximum length of {MAX_AGENT_ID_LENGTH}")
        if not AGENT_ID_PATTERN.match(agent_id):
            raise ValueError(
                "agent_id must contain only alphanumeric characters, hyphens, underscores, or dots"
            )

        async with self._lock:
            now = datetime.now(timezone.utc)

            # Check for existing registration
            existing = self._agents.get(agent_id)
            if existing:
                # Verify tenant ownership
                if existing.tenant_id != tenant_id:
                    raise PermissionError(
                        f"Agent {agent_id} is owned by different tenant"
                    )
                # Update existing registration
                existing.agent_type = agent_type
                existing.endpoint_url = endpoint_url
                existing.capabilities = capabilities
                existing.last_heartbeat = now
                existing.health_status = "healthy"
                existing.metadata = metadata or {}
                await self._persist_agent(existing)
                logger.info(
                    "a2a_agent_updated",
                    agent_id=agent_id,
                    tenant_id=tenant_id,
                    capabilities=[c.name for c in capabilities],
                )
                return existing

            # Create new registration
            registration = AgentRegistration(
                agent_id=agent_id,
                agent_type=agent_type,
                endpoint_url=endpoint_url,
                capabilities=capabilities,
                tenant_id=tenant_id,
                registered_at=now,
                last_heartbeat=now,
                health_status="healthy",
                metadata=metadata or {},
            )
            self._agents[agent_id] = registration
            await self._persist_agent(registration)
            logger.info(
                "a2a_agent_registered",
                agent_id=agent_id,
                tenant_id=tenant_id,
                agent_type=agent_type,
                capabilities=[c.name for c in capabilities],
            )
            return registration

    async def unregister_agent(self, agent_id: str, tenant_id: str) -> bool:
        """Unregister an agent from the registry.

        Args:
            agent_id: ID of agent to unregister
            tenant_id: Tenant ID for ownership verification

        Returns:
            True if agent was unregistered, False if not found

        Raises:
            PermissionError: If tenant doesn't own the agent
        """
        async with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                # Try loading from Redis
                agent = await self._load_agent(agent_id)

            if not agent:
                return False

            if agent.tenant_id != tenant_id:
                raise PermissionError(f"Agent {agent_id} is owned by different tenant")

            self._agents.pop(agent_id, None)
            await self._remove_agent_from_redis(agent_id, tenant_id)
            logger.info("a2a_agent_unregistered", agent_id=agent_id, tenant_id=tenant_id)
            return True

    async def heartbeat(self, agent_id: str, tenant_id: str) -> bool:
        """Record heartbeat for an agent.

        Args:
            agent_id: ID of agent sending heartbeat
            tenant_id: Tenant ID for ownership verification

        Returns:
            True if heartbeat recorded, False if agent not found

        Raises:
            PermissionError: If tenant doesn't own the agent
        """
        async with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                # Try loading from Redis
                agent = await self._load_agent(agent_id)
                if agent:
                    self._agents[agent_id] = agent

            if not agent:
                return False

            if agent.tenant_id != tenant_id:
                raise PermissionError(f"Agent {agent_id} is owned by different tenant")

            agent.last_heartbeat = datetime.now(timezone.utc)
            agent.health_status = "healthy"
            await self._persist_agent(agent)
            logger.debug("a2a_heartbeat_received", agent_id=agent_id)
            return True

    # ==================== Discovery Operations ====================

    async def get_agent(
        self,
        agent_id: str,
        tenant_id: str,
    ) -> Optional[AgentRegistration]:
        """Get a specific agent by ID.

        Args:
            agent_id: ID of the agent to retrieve
            tenant_id: Tenant ID for filtering

        Returns:
            AgentRegistration if found and belongs to tenant, None otherwise
        """
        async with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                agent = await self._load_agent(agent_id)
                if agent:
                    self._agents[agent_id] = agent

            if not agent or agent.tenant_id != tenant_id:
                return None

            self._update_health_status(agent)
            return agent

    async def list_agents(
        self,
        tenant_id: str,
        healthy_only: bool = False,
    ) -> list[AgentRegistration]:
        """List all agents for a tenant.

        Args:
            tenant_id: Tenant ID to filter by
            healthy_only: If True, only return healthy agents

        Returns:
            List of AgentRegistration objects for the tenant
        """
        async with self._lock:
            # First load any agents from Redis that we don't have in memory
            redis_agent_ids = await self._load_tenant_agents(tenant_id)

            # Use asyncio.gather() for parallel Redis loads instead of sequential loop
            missing_ids = [aid for aid in redis_agent_ids if aid not in self._agents]
            if missing_ids:
                loaded_agents = await asyncio.gather(
                    *[self._load_agent(aid) for aid in missing_ids],
                    return_exceptions=True,
                )
                for agent_id, agent in zip(missing_ids, loaded_agents):
                    if isinstance(agent, AgentRegistration):
                        self._agents[agent.agent_id] = agent
                    elif isinstance(agent, Exception):
                        logger.warning(
                            "a2a_agent_parallel_load_failed",
                            agent_id=agent_id,
                            error=str(agent),
                        )

            # Filter by tenant
            tenant_agents = [
                agent for agent in self._agents.values()
                if agent.tenant_id == tenant_id
            ]

            # Update health status and filter if needed
            for agent in tenant_agents:
                self._update_health_status(agent)

            if healthy_only:
                tenant_agents = [a for a in tenant_agents if a.health_status == "healthy"]

            return tenant_agents

    async def find_agents_by_capability(
        self,
        capability_name: str,
        tenant_id: str,
        healthy_only: bool = True,
    ) -> list[AgentRegistration]:
        """Find agents that offer a specific capability.

        Args:
            capability_name: Name of the capability to search for
            tenant_id: Tenant ID to filter by
            healthy_only: If True, only return healthy agents

        Returns:
            List of agents that have the specified capability
        """
        agents = await self.list_agents(tenant_id, healthy_only=healthy_only)
        return [a for a in agents if a.has_capability(capability_name)]

    async def find_agents_by_type(
        self,
        agent_type: str,
        tenant_id: str,
        healthy_only: bool = True,
    ) -> list[AgentRegistration]:
        """Find agents of a specific type.

        Args:
            agent_type: Type of agent to search for
            tenant_id: Tenant ID to filter by
            healthy_only: If True, only return healthy agents

        Returns:
            List of agents of the specified type
        """
        agents = await self.list_agents(tenant_id, healthy_only=healthy_only)
        return [a for a in agents if a.agent_type == agent_type]

    # ==================== Self Registration ====================

    async def register_self(
        self,
        agent_id: str,
        endpoint_url: str,
        tenant_id: str,
    ) -> AgentRegistration:
        """Register this RAG system as an agent with predefined capabilities.

        This is a convenience method for registering the local RAG engine
        with its known capabilities.

        Args:
            agent_id: Unique identifier for this agent instance
            endpoint_url: The base URL where this agent receives requests
            tenant_id: Tenant scope for the registration

        Returns:
            The created AgentRegistration
        """
        # Use get_implemented_rag_capabilities() to only advertise capabilities
        # that are actually implemented in the execute endpoint (hybrid_retrieve,
        # vector_search). This prevents advertising unimplemented capabilities.
        return await self.register_agent(
            agent_id=agent_id,
            agent_type="rag_engine",
            endpoint_url=endpoint_url,
            capabilities=get_implemented_rag_capabilities(),
            tenant_id=tenant_id,
            metadata={
                "version": "0.1.0",
                "protocol": "a2a-v1",
            },
        )

    def to_dict(self, agent: AgentRegistration) -> dict[str, Any]:
        """Convert agent registration to dictionary format."""
        return agent.to_dict()
