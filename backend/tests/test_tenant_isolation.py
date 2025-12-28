import os

import pytest
from psycopg import OperationalError, errors

from agentic_rag_backend.trajectory import TrajectoryLogger, create_pool


@pytest.mark.integration
def test_trajectory_events_isolated_by_tenant() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")

    pool = create_pool(database_url, min_size=1, max_size=1)
    logger = TrajectoryLogger(pool=pool)

    tenant_a = "tenant-a"
    tenant_b = "tenant-b"
    trajectory_id = logger.start_trajectory(tenant_a, session_id="session-a")
    logger.log_events(
        tenant_a,
        trajectory_id,
        [("thought", "a thought"), ("action", "an action")],
    )

    try:
        with pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    select count(*)
                    from trajectory_events
                    where trajectory_id = %s and tenant_id = %s
                    """,
                    (trajectory_id, tenant_b),
                )
                wrong_tenant_count = cursor.fetchone()[0]

                cursor.execute(
                    """
                    select count(*)
                    from trajectory_events
                    where trajectory_id = %s and tenant_id = %s
                    """,
                    (trajectory_id, tenant_a),
                )
                right_tenant_count = cursor.fetchone()[0]
    except (OperationalError, errors.UndefinedTable):
        pytest.skip("Database unavailable or migrations not applied")
    finally:
        pool.close()

    assert wrong_tenant_count == 0
    assert right_tenant_count == 2
