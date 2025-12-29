import os
from uuid import uuid4

import pytest
import redis.asyncio as redis

from agentic_rag_backend.rate_limit import RedisRateLimiter


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_rate_limiter_allows_and_blocks() -> None:
    if os.getenv("RATE_LIMIT_BACKEND") != "redis":
        pytest.skip("RATE_LIMIT_BACKEND is not set to redis")

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        pytest.skip("REDIS_URL not set")

    client = redis.from_url(redis_url, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        await client.close()
        pytest.skip("Redis unavailable")

    limiter = RedisRateLimiter(
        client=client,
        max_requests=2,
        window_seconds=60,
        key_prefix=f"test-rate-limit:{uuid4()}",
    )

    try:
        assert await limiter.allow("tenant-1") is True
        assert await limiter.allow("tenant-1") is True
        assert await limiter.allow("tenant-1") is False
    finally:
        await client.close()
