from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

import psycopg
from psycopg_pool import AsyncConnectionPool


class EventType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"


def create_pool(database_url: str, min_size: int, max_size: int) -> AsyncConnectionPool:
    """Create an async connection pool for trajectory storage."""
    try:
        return AsyncConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            open=False,
        )
    except psycopg.OperationalError as exc:
        raise RuntimeError("Database connection failed during pool initialization.") from exc
    except psycopg.Error as exc:
        raise RuntimeError("Database error during pool initialization.") from exc


async def close_pool(pool: AsyncConnectionPool) -> None:
    """Close a connection pool."""
    await pool.close()


@dataclass
class TrajectoryLogger:
    pool: AsyncConnectionPool

    async def start_trajectory(self, tenant_id: str, session_id: Optional[str]) -> UUID:
        """Create a trajectory row and return its ID."""
        trajectory_id = uuid4()
        async with self.pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "insert into trajectories (id, tenant_id, session_id) values (%s, %s, %s)",
                    (trajectory_id, tenant_id, session_id),
                )
            await conn.commit()
        return trajectory_id

    async def log_thought(self, tenant_id: str, trajectory_id: UUID, content: str) -> None:
        """Record a thought event for a trajectory."""
        await self._log_event(tenant_id, trajectory_id, EventType.THOUGHT, content)

    async def log_action(self, tenant_id: str, trajectory_id: UUID, content: str) -> None:
        """Record an action event for a trajectory."""
        await self._log_event(tenant_id, trajectory_id, EventType.ACTION, content)

    async def log_observation(self, tenant_id: str, trajectory_id: UUID, content: str) -> None:
        """Record an observation event for a trajectory."""
        await self._log_event(tenant_id, trajectory_id, EventType.OBSERVATION, content)

    async def log_events(
        self, tenant_id: str, trajectory_id: UUID, events: list[tuple[EventType, str]]
    ) -> None:
        """Record multiple events in a single transaction."""
        if not events:
            return
        async with self.pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.executemany(
                    """
                    insert into trajectory_events (id, trajectory_id, tenant_id, event_type, content)
                    values (%s, %s, %s, %s, %s)
                    """,
                    [
                        (uuid4(), trajectory_id, tenant_id, event_type.value, content)
                        for event_type, content in events
                    ],
                )
            await conn.commit()

    async def _log_event(
        self,
        tenant_id: str,
        trajectory_id: UUID,
        event_type: EventType,
        content: str,
    ) -> None:
        """Record a single event within its own transaction."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    insert into trajectory_events (id, trajectory_id, tenant_id, event_type, content)
                    values (%s, %s, %s, %s, %s)
                    """,
                    (uuid4(), trajectory_id, tenant_id, event_type.value, content),
                )
            await conn.commit()
