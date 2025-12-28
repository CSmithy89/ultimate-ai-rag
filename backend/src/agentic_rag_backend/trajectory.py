from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID, uuid4

import psycopg
from psycopg_pool import ConnectionPool

EVENT_THOUGHT = "thought"
EVENT_ACTION = "action"
EVENT_OBSERVATION = "observation"

def create_pool(database_url: str, min_size: int, max_size: int) -> ConnectionPool:
    """Create a connection pool for trajectory storage."""
    try:
        return ConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            open=True,
        )
    except psycopg.OperationalError as exc:
        raise RuntimeError("Database connection failed during pool initialization.") from exc
    except psycopg.Error as exc:
        raise RuntimeError("Database error during pool initialization.") from exc


def close_pool(pool: ConnectionPool) -> None:
    """Close a connection pool."""
    pool.close()


@dataclass
class TrajectoryLogger:
    pool: ConnectionPool

    def start_trajectory(self, tenant_id: str, session_id: Optional[str]) -> UUID:
        """Create a trajectory row and return its ID."""
        trajectory_id = uuid4()
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "insert into trajectories (id, tenant_id, session_id) values (%s, %s, %s)",
                    (trajectory_id, tenant_id, session_id),
                )
            conn.commit()
        return trajectory_id

    def log_thought(self, tenant_id: str, trajectory_id: UUID, content: str) -> None:
        """Record a thought event for a trajectory."""
        self._log_event(tenant_id, trajectory_id, EVENT_THOUGHT, content)

    def log_action(self, tenant_id: str, trajectory_id: UUID, content: str) -> None:
        """Record an action event for a trajectory."""
        self._log_event(tenant_id, trajectory_id, EVENT_ACTION, content)

    def log_observation(self, tenant_id: str, trajectory_id: UUID, content: str) -> None:
        """Record an observation event for a trajectory."""
        self._log_event(tenant_id, trajectory_id, EVENT_OBSERVATION, content)

    def log_events(
        self, tenant_id: str, trajectory_id: UUID, events: list[tuple[str, str]]
    ) -> None:
        """Record multiple events in a single transaction."""
        if not events:
            return
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    insert into trajectory_events (id, trajectory_id, tenant_id, event_type, content)
                    values (%s, %s, %s, %s, %s)
                    """,
                    [
                        (uuid4(), trajectory_id, tenant_id, event_type, content)
                        for event_type, content in events
                    ],
                )
            conn.commit()

    def _log_event(
        self,
        tenant_id: str,
        trajectory_id: UUID,
        event_type: str,
        content: str,
    ) -> None:
        """Record a single event within its own transaction."""
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    insert into trajectory_events (id, trajectory_id, tenant_id, event_type, content)
                    values (%s, %s, %s, %s, %s)
                    """,
                    (uuid4(), trajectory_id, tenant_id, event_type, content),
                )
            conn.commit()
