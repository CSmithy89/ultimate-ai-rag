"""A2A Task Delegation Manager for structured task delegation.

This module provides task delegation between agents including request queuing,
result tracking, timeout handling, and retry logic.
"""

from __future__ import annotations

import asyncio
import json
import time
from asyncio import Lock
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    import redis
from uuid import uuid4

import httpx
import structlog

from agentic_rag_backend.db.redis import RedisClient

from .a2a_messages import TaskRequest, TaskResult, TaskStatus
from .a2a_registry import A2AAgentRegistry

logger = structlog.get_logger(__name__)


# Constants for TTL multipliers (documented rationale)
# Pending task TTL: 2x timeout ensures task survives execution + response time
PENDING_TASK_TTL_MULTIPLIER = 2
# Result TTL: 1 hour allows clients to poll for results after completion
RESULT_TTL_SECONDS = 3600


@dataclass
class DelegationConfig:
    """Configuration for task delegation behavior.

    Attributes:
        default_timeout_seconds: Default timeout for task execution
        max_retries: Maximum number of retry attempts
        retry_delay_seconds: Initial delay between retries (exponential backoff)
        max_concurrent_tasks: Maximum concurrent outbound delegations
        redis_prefix: Prefix for Redis keys
        http_timeout_seconds: Timeout for HTTP requests to agents
        result_ttl_seconds: TTL for completed task results in Redis
    """

    default_timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    max_concurrent_tasks: int = 50
    redis_prefix: str = "a2a:tasks"
    http_timeout_seconds: float = 30.0
    result_ttl_seconds: int = RESULT_TTL_SECONDS


@dataclass
class PendingTask:
    """Internal tracking for a pending delegated task.

    Attributes:
        request: The task request
        started_at: When task execution started
        retries: Number of retries attempted
        result_future: Future for the task result
    """

    request: TaskRequest
    started_at: datetime
    retries: int = 0
    result_future: Optional[asyncio.Future[TaskResult]] = None


class TaskDelegationManager:
    """Manager for delegating tasks to remote agents.

    Provides:
    - Task delegation with automatic agent discovery
    - Timeout handling and retry logic with exponential backoff
    - Task result tracking via Redis
    - Concurrent task execution limits
    - Multi-tenancy isolation via tenant_id

    Usage:
        delegation_manager = TaskDelegationManager(registry, config)
        result = await delegation_manager.delegate_task(
            capability_name="hybrid_retrieve",
            parameters={"query": "What is RAG?"},
            tenant_id="tenant-123",
        )
    """

    def __init__(
        self,
        registry: A2AAgentRegistry,
        config: Optional[DelegationConfig] = None,
        redis_client: Optional[RedisClient] = None,
    ) -> None:
        """Initialize the task delegation manager.

        Args:
            registry: Agent registry for discovering target agents
            config: Configuration for delegation behavior
            redis_client: Optional Redis client for persistence
        """
        self._registry = registry
        self._config = config or DelegationConfig()
        self._redis_client = redis_client
        self._pending_tasks: dict[str, PendingTask] = {}
        self._lock = Lock()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_tasks)
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        # Shared HTTP client for connection pooling
        self._http_client: httpx.AsyncClient | None = None

    # ==================== Redis Persistence ====================

    def _task_key(self, task_id: str) -> str:
        """Generate Redis key for a task."""
        return f"{self._config.redis_prefix}:task:{task_id}"

    def _result_key(self, task_id: str) -> str:
        """Generate Redis key for a task result."""
        return f"{self._config.redis_prefix}:result:{task_id}"

    def _get_redis(self) -> "redis.Redis | None":
        """Get Redis client if available."""
        if not self._redis_client:
            return None
        try:
            return self._redis_client.client
        except Exception as exc:
            logger.warning("a2a_delegation_redis_unavailable", error=str(exc))
            return None

    async def _persist_task(self, request: TaskRequest) -> None:
        """Persist task request to Redis."""
        redis = self._get_redis()
        if not redis:
            return
        try:
            key = self._task_key(request.task_id)
            payload = json.dumps(request.to_dict())
            # TTL = timeout * multiplier to survive execution + response handling
            ttl = request.timeout_seconds * PENDING_TASK_TTL_MULTIPLIER
            await redis.set(key, payload, ex=ttl)
        except Exception as exc:
            logger.warning("a2a_task_persist_failed", task_id=request.task_id, error=str(exc))

    async def _persist_result(self, result: TaskResult) -> None:
        """Persist task result to Redis."""
        redis = self._get_redis()
        if not redis:
            return
        try:
            key = self._result_key(result.task_id)
            payload = json.dumps(result.to_dict())
            await redis.set(key, payload, ex=self._config.result_ttl_seconds)
        except Exception as exc:
            logger.warning("a2a_result_persist_failed", task_id=result.task_id, error=str(exc))

    async def _load_result(self, task_id: str) -> Optional[TaskResult]:
        """Load task result from Redis."""
        redis = self._get_redis()
        if not redis:
            return None
        try:
            key = self._result_key(task_id)
            payload = await redis.get(key)
            if not payload:
                return None
            raw = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
            data = json.loads(raw)
            return TaskResult.from_dict(data)
        except Exception as exc:
            logger.warning("a2a_result_load_failed", task_id=task_id, error=str(exc))
            return None

    # ==================== HTTP Client Management ====================

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client for connection pooling."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self._config.http_timeout_seconds
            )
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    # ==================== Task Execution ====================

    async def _send_task_to_agent(
        self,
        request: TaskRequest,
        endpoint_url: str,
    ) -> TaskResult:
        """Send task request to a specific agent endpoint.

        Args:
            request: The task to send
            endpoint_url: Target agent's endpoint URL

        Returns:
            TaskResult from the agent
        """
        start_time = time.monotonic()
        try:
            client = self._get_http_client()
            url = f"{endpoint_url.rstrip('/')}/api/v1/a2a/execute"
            response = await client.post(
                url,
                json=request.to_dict(),
                headers={
                    "Content-Type": "application/json",
                    "X-Tenant-ID": request.tenant_id,
                },
            )
            response.raise_for_status()
            data = response.json()

            execution_time_ms = int((time.monotonic() - start_time) * 1000)

            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.COMPLETED,
                result=data.get("result"),
                execution_time_ms=execution_time_ms,
                tenant_id=request.tenant_id,
            )
        except httpx.TimeoutException as exc:
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error=f"Request timeout: {exc}",
                execution_time_ms=int((time.monotonic() - start_time) * 1000),
                tenant_id=request.tenant_id,
            )
        except httpx.HTTPStatusError as exc:
            # Log full error details for debugging, but don't expose in response
            # to avoid leaking sensitive info from downstream agents
            logger.warning(
                "a2a_downstream_agent_error",
                task_id=request.task_id,
                target_agent=request.target_agent,
                status_code=exc.response.status_code,
                response_text=exc.response.text[:500],  # Limit for logs
            )
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error="Downstream agent request failed",
                execution_time_ms=int((time.monotonic() - start_time) * 1000),
                tenant_id=request.tenant_id,
            )
        except Exception as exc:
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error=f"Request failed: {exc}",
                execution_time_ms=int((time.monotonic() - start_time) * 1000),
                tenant_id=request.tenant_id,
            )

    async def _execute_with_retry(
        self,
        request: TaskRequest,
        endpoint_url: str,
    ) -> TaskResult:
        """Execute task with retry logic.

        Uses exponential backoff for retries.

        Args:
            request: The task to execute
            endpoint_url: Target agent's endpoint URL

        Returns:
            Final TaskResult after retries exhausted or success
        """
        last_result: Optional[TaskResult] = None
        retry_delay = self._config.retry_delay_seconds

        for attempt in range(self._config.max_retries + 1):
            result = await self._send_task_to_agent(request, endpoint_url)

            if result.is_success:
                return result

            last_result = result

            if attempt < self._config.max_retries:
                logger.warning(
                    "a2a_task_retry",
                    task_id=request.task_id,
                    attempt=attempt + 1,
                    max_retries=self._config.max_retries,
                    delay_seconds=retry_delay,
                    error=result.error,
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

        return last_result or TaskResult(
            task_id=request.task_id,
            status=TaskStatus.FAILED,
            error="All retry attempts exhausted",
            tenant_id=request.tenant_id,
        )

    # ==================== Public API ====================

    async def delegate_task(
        self,
        capability_name: str,
        parameters: dict[str, Any],
        tenant_id: str,
        source_agent: str = "orchestrator",
        priority: int = 5,
        timeout_seconds: Optional[int] = None,
        target_agent_id: Optional[str] = None,
    ) -> TaskResult:
        """Delegate a task to an agent with the required capability.

        Args:
            capability_name: Name of the capability to invoke
            parameters: Parameters to pass to the capability
            tenant_id: Tenant ID for isolation
            source_agent: ID of the delegating agent
            priority: Task priority (1-10)
            timeout_seconds: Override default timeout
            target_agent_id: Specific agent to target (optional)

        Returns:
            TaskResult with execution outcome

        Raises:
            ValueError: If no suitable agent is found
        """
        async with self._semaphore:
            # Create task request
            request = TaskRequest(
                task_id=str(uuid4()),
                source_agent=source_agent,
                capability_name=capability_name,
                parameters=parameters,
                priority=priority,
                timeout_seconds=timeout_seconds or self._config.default_timeout_seconds,
                tenant_id=tenant_id,
            )

            # Find suitable agent
            if target_agent_id:
                agent = await self._registry.get_agent(target_agent_id, tenant_id)
                if not agent:
                    return TaskResult(
                        task_id=request.task_id,
                        status=TaskStatus.FAILED,
                        error=f"Target agent '{target_agent_id}' not found",
                        tenant_id=tenant_id,
                    )
                if not agent.has_capability(capability_name):
                    return TaskResult(
                        task_id=request.task_id,
                        status=TaskStatus.FAILED,
                        error=f"Agent '{target_agent_id}' does not have capability '{capability_name}'",
                        tenant_id=tenant_id,
                    )
                target_agents = [agent]
            else:
                target_agents = await self._registry.find_agents_by_capability(
                    capability_name,
                    tenant_id,
                    healthy_only=True,
                )

            if not target_agents:
                return TaskResult(
                    task_id=request.task_id,
                    status=TaskStatus.FAILED,
                    error=f"No healthy agent found with capability '{capability_name}'",
                    tenant_id=tenant_id,
                )

            # Select agent (could be enhanced with load balancing)
            selected_agent = target_agents[0]
            request.target_agent = selected_agent.agent_id

            logger.info(
                "a2a_task_delegating",
                task_id=request.task_id,
                capability=capability_name,
                target_agent=selected_agent.agent_id,
                tenant_id=tenant_id,
            )

            # Persist and execute
            await self._persist_task(request)

            async with self._lock:
                self._pending_tasks[request.task_id] = PendingTask(
                    request=request,
                    started_at=datetime.now(timezone.utc),
                )

            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_with_retry(request, selected_agent.endpoint_url),
                    timeout=request.timeout_seconds,
                )
            except asyncio.TimeoutError:
                result = TaskResult(
                    task_id=request.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Task timeout after {request.timeout_seconds} seconds",
                    tenant_id=tenant_id,
                )

            # Cleanup and persist result
            async with self._lock:
                self._pending_tasks.pop(request.task_id, None)

            await self._persist_result(result)

            logger.info(
                "a2a_task_completed",
                task_id=request.task_id,
                status=result.status.value,
                execution_time_ms=result.execution_time_ms,
                success=result.is_success,
            )

            return result

    async def get_task_status(self, task_id: str, tenant_id: str) -> Optional[TaskResult]:
        """Get the status/result of a delegated task.

        Args:
            task_id: ID of the task to check
            tenant_id: Tenant ID for isolation

        Returns:
            TaskResult if found, None otherwise
        """
        # Check if still pending
        async with self._lock:
            pending = self._pending_tasks.get(task_id)
            if pending:
                if pending.request.tenant_id != tenant_id:
                    return None
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    tenant_id=tenant_id,
                )

        # Check for completed result
        result = await self._load_result(task_id)
        if result:
            # Verify tenant access - result must belong to requesting tenant
            if result.tenant_id and result.tenant_id != tenant_id:
                return None
            return result

        return None

    async def cancel_task(self, task_id: str, tenant_id: str) -> bool:
        """Attempt to cancel a pending task.

        Args:
            task_id: ID of the task to cancel
            tenant_id: Tenant ID for isolation

        Returns:
            True if task was cancelled, False if not found or already completed
        """
        # Extract task info inside lock, but do Redis I/O outside to avoid blocking
        result_to_persist: TaskResult | None = None

        async with self._lock:
            pending = self._pending_tasks.get(task_id)
            if not pending:
                return False

            if pending.request.tenant_id != tenant_id:
                raise PermissionError("Task belongs to different tenant")

            # Remove from pending tasks while holding lock
            self._pending_tasks.pop(task_id, None)

            # Prepare result to persist (will do I/O outside lock)
            result_to_persist = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED,
                error="Task cancelled by request",
                tenant_id=tenant_id,
            )

        # Persist result outside the lock to avoid blocking other operations
        # Check if task already completed to avoid overwriting completion result
        if result_to_persist:
            existing = await self._load_result(task_id)
            if existing and existing.status == TaskStatus.COMPLETED:
                # Task completed between removal and cancellation - don't overwrite
                logger.info(
                    "a2a_task_cancel_skipped_completed",
                    task_id=task_id,
                    tenant_id=tenant_id,
                )
                return False
            await self._persist_result(result_to_persist)
            logger.info("a2a_task_cancelled", task_id=task_id, tenant_id=tenant_id)

        return True

    async def list_pending_tasks(self, tenant_id: str) -> list[TaskRequest]:
        """List all pending tasks for a tenant.

        Args:
            tenant_id: Tenant ID to filter by

        Returns:
            List of pending TaskRequest objects
        """
        async with self._lock:
            return [
                pt.request
                for pt in self._pending_tasks.values()
                if pt.request.tenant_id == tenant_id
            ]

    # ==================== Local Task Handling ====================

    async def handle_incoming_task(
        self,
        request_data: dict[str, Any],
        handler: Callable[[TaskRequest], Any],
    ) -> dict[str, Any]:
        """Handle an incoming task request from another agent.

        This method is called when this agent receives a delegated task.

        Args:
            request_data: The incoming task request data
            handler: Async function to execute the task

        Returns:
            Response dictionary with task result
        """
        request = TaskRequest.from_dict(request_data)

        logger.info(
            "a2a_task_received",
            task_id=request.task_id,
            capability=request.capability_name,
            source_agent=request.source_agent,
            tenant_id=request.tenant_id,
        )

        start_time = time.monotonic()
        try:
            result_data = await handler(request)
            execution_time_ms = int((time.monotonic() - start_time) * 1000)

            return {
                "task_id": request.task_id,
                "status": TaskStatus.COMPLETED.value,
                "result": result_data,
                "execution_time_ms": execution_time_ms,
            }
        except Exception as exc:
            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "a2a_task_execution_failed",
                task_id=request.task_id,
                error=str(exc),
            )
            return {
                "task_id": request.task_id,
                "status": TaskStatus.FAILED.value,
                "error": str(exc),
                "execution_time_ms": execution_time_ms,
            }
