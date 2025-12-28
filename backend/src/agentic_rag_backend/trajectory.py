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
    global _POOL
    if _POOL is None:
        _POOL = ConnectionPool(conninfo=database_url, min_size=1, max_size=5, open=True)
    return _POOL


def close_pool() -> None:
    if _POOL is not None:
        _POOL.close()


def ensure_trajectory_schema(database_url: str) -> None:
    pool = get_pool(database_url)
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                create table if not exists trajectories (
                    id uuid primary key,
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
                    event_type text not null,
                    content text not null,
                    created_at timestamptz not null default now()
                );
                """
            )
        conn.commit()


@dataclass
class TrajectoryLogger:
    pool: ConnectionPool

    def start_trajectory(self, session_id: Optional[str]) -> UUID:
        trajectory_id = uuid4()
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "insert into trajectories (id, session_id) values (%s, %s)",
                    (trajectory_id, session_id),
                )
            conn.commit()
        return trajectory_id

    def log_thought(self, trajectory_id: UUID, content: str) -> None:
        self._log_event(trajectory_id, EVENT_THOUGHT, content)

    def log_action(self, trajectory_id: UUID, content: str) -> None:
        self._log_event(trajectory_id, EVENT_ACTION, content)

    def log_observation(self, trajectory_id: UUID, content: str) -> None:
        self._log_event(trajectory_id, EVENT_OBSERVATION, content)

    def _log_event(self, trajectory_id: UUID, event_type: str, content: str) -> None:
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    insert into trajectory_events (id, trajectory_id, event_type, content)
                    values (%s, %s, %s, %s)
                    """,
                    (uuid4(), trajectory_id, event_type, content),
                )
            conn.commit()
