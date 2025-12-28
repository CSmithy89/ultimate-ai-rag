import os

import pytest
from psycopg import OperationalError, errors

from agentic_rag_backend.trajectory import EventType, TrajectoryLogger, create_pool


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trajectory_events_isolated_by_tenant() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")

    pool = create_pool(database_url, min_size=1, max_size=1)
    await pool.open()
    logger = TrajectoryLogger(pool=pool)

    tenant_a = "tenant-a"
    tenant_b = "tenant-b"
    trajectory_id = await logger.start_trajectory(tenant_a, session_id="session-a")
    await logger.log_events(
        tenant_a,
        trajectory_id,
        [(EventType.THOUGHT, "a thought"), (EventType.ACTION, "an action")],
    )

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    select count(*)
                    from trajectory_events
                    where trajectory_id = %s and tenant_id = %s
                    """,
                    (trajectory_id, tenant_b),
                )
                wrong_tenant_count = cursor.fetchone()[0]

                await cursor.execute(
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
        await pool.close()

    assert wrong_tenant_count == 0
    assert right_tenant_count == 2
