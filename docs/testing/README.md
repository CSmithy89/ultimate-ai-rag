# Testing Notes

## Skipped Tests

The following tests are intentionally skipped when required infrastructure is missing.
Each skip includes a descriptive reason in the test file.

| Test File | Skip Reason | Follow-up |
| --- | --- | --- |
| `backend/tests/test_rate_limit_redis.py` | Requires `RATE_LIMIT_BACKEND=redis` and a running Redis instance | Add Redis service to optional local test profile |
| `backend/tests/test_db_pool.py` | Requires `DATABASE_URL` and reachable Postgres | Provide CI job to cover DB pool paths |
| `backend/tests/test_tenant_isolation.py` | Requires Postgres with migrations applied | Add migration step to integration test job |

## Follow-up Issues

- Add a dedicated CI job that runs Redis-backed rate limit tests.
- Extend integration test job to run migrations before tenant isolation tests.
- Document local test profiles (unit vs integration vs benchmark).
