from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID, uuid4

from psycopg_pool import ConnectionPool

EVENT_THOUGHT = "thought"
EVENT_ACTION = "action"
EVENT_OBSERVATION = "observation"

def create_pool(database_url: str, min_size: int, max_size: int) -> ConnectionPool:
    """Create a connection pool for trajectory storage."""
    return ConnectionPool(conninfo=database_url, min_size=min_size, max_size=max_size, open=True)


def close_pool(pool: ConnectionPool) -> None:
    """Close a connection pool."""
    pool.close()


def ensure_trajectory_schema(pool: ConnectionPool) -> None:
    """Ensure trajectory tables and indexes exist."""
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
                "create index if not exists idx_trajectory_events_event_type on trajectory_events (event_type)"
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
