import os

import pytest
from psycopg import OperationalError

from agentic_rag_backend.trajectory import create_pool


@pytest.mark.integration
def test_create_pool_connects() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")

    try:
        pool = create_pool(database_url, min_size=1, max_size=1)
    except RuntimeError:
        pytest.skip("Database unavailable")

    try:
        with pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("select 1")
                assert cursor.fetchone()[0] == 1
    except OperationalError:
        pytest.skip("Database unavailable")
    finally:
        pool.close()
