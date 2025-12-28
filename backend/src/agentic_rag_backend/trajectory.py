from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID, uuid4

from psycopg_pool import ConnectionPool

EVENT_THOUGHT = "thought"
EVENT_ACTION = "action"
EVENT_OBSERVATION = "observation"

_POOL: ConnectionPool | None = None


def get_pool(database_url: str) -> ConnectionPool:
    """Return a singleton connection pool for trajectory storage."""
    global _POOL
    if _POOL is None:
        _POOL = ConnectionPool(conninfo=database_url, min_size=1, max_size=5, open=True)
    return _POOL


def close_pool() -> None:
    """Close the shared connection pool."""
    if _POOL is not None:
        _POOL.close()


def ensure_trajectory_schema(database_url: str) -> None:
    """Ensure trajectory tables and indexes exist."""
    pool = get_pool(database_url)
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                create table if not exists trajectories (
                    id uuid primary key,
                    tenant_id text not null default 'unknown',
                    session_id text,
                    created_at timestamptz not null default now()
                );
                """
            )
            cursor.execute(
                """
                create table if not exists trajectory_events (
                    id uuid primary key,
                    trajectory_id uuid not null references trajectories(id) on delete cascade,
                    tenant_id text not null default 'unknown',
                    event_type text not null,
                    content text not null,
                    created_at timestamptz not null default now()
                );
                """
            )
            cursor.execute(
                "alter table trajectories add column if not exists tenant_id text not null default 'unknown'"
            )
            cursor.execute(
                "alter table trajectory_events add column if not exists tenant_id text not null default 'unknown'"
            )
            cursor.execute(
                "create index if not exists idx_trajectory_events_trajectory_id on trajectory_events (trajectory_id)"
            )
            cursor.execute(
                "create index if not exists idx_trajectory_events_created_at on trajectory_events (created_at)"
            )
            cursor.execute(
                "create index if not exists idx_trajectories_session_id on trajectories (session_id)"
            )
            cursor.execute(
                "create index if not exists idx_trajectories_tenant_id on trajectories (tenant_id)"
            )
            cursor.execute(
                "create index if not exists idx_trajectory_events_tenant_id on trajectory_events (tenant_id)"
            )
        conn.commit()


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

    def _log_event(
        self,
        tenant_id: str,
        trajectory_id: UUID,
        event_type: str,
        content: str,
    ) -> None:
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
