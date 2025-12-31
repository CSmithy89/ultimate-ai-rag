"""Tests for ops API routes."""

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from agentic_rag_backend.api.routes.ops import (
    get_cost_summary,
    get_trajectory_detail,
    list_trajectories,
)
from agentic_rag_backend.ops.cost_tracker import CostSummary, CostTrendPoint


class AllowLimiter:
    async def allow(self, key: str) -> bool:
        return True


class DenyLimiter:
    async def allow(self, key: str) -> bool:
        return False


class DummyCursor:
    def __init__(self, fetchone_results=None, fetchall_results=None) -> None:
        self._fetchone_results = list(fetchone_results or [])
        self._fetchall_results = list(fetchall_results or [])
        self.queries: list[tuple[str, object]] = []

    async def execute(self, query: str, params=None) -> None:
        text = query if isinstance(query, str) else str(query)
        self.queries.append((text, params))

    async def fetchone(self):
        if self._fetchone_results:
            return self._fetchone_results.pop(0)
        return None

    async def fetchall(self):
        if self._fetchall_results:
            return self._fetchall_results.pop(0)
        return []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class DummyConnection:
    def __init__(self, cursor: DummyCursor) -> None:
        self._cursor = cursor

    def cursor(self, row_factory=None) -> DummyCursor:
        return self._cursor


class DummyConnectionManager:
    def __init__(self, connection: DummyConnection) -> None:
        self._connection = connection

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class DummyPool:
    def __init__(self, cursor: DummyCursor) -> None:
        self._cursor = cursor

    def connection(self) -> DummyConnectionManager:
        return DummyConnectionManager(DummyConnection(self._cursor))


class DummyCostTracker:
    async def get_summary(self, tenant_id: str, window: str) -> CostSummary:
        return CostSummary(
            total_cost_usd=Decimal("1.00"),
            baseline_cost_usd=Decimal("1.50"),
            total_savings_usd=Decimal("0.50"),
            total_premium_usd=Decimal("0.00"),
            total_tokens=100,
            total_requests=2,
            by_model=[
                {
                    "model_id": "gpt-4o-mini",
                    "requests": 2,
                    "total_tokens": 100,
                    "total_cost_usd": Decimal("1.00"),
                }
            ],
            trend=[
                CostTrendPoint(
                    bucket=datetime.now(timezone.utc),
                    total_cost_usd=Decimal("1.00"),
                    total_tokens=100,
                )
            ],
            alerts={"enabled": False},
        )


@pytest.mark.asyncio
async def test_get_cost_summary_rate_limited() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await get_cost_summary(
            tenant_id="11111111-1111-1111-1111-111111111111",
            window="day",
            cost_tracker=DummyCostTracker(),
            limiter=DenyLimiter(),
        )
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_get_cost_summary_serializes_values() -> None:
    response = await get_cost_summary(
        tenant_id="11111111-1111-1111-1111-111111111111",
        window="day",
        cost_tracker=DummyCostTracker(),
        limiter=AllowLimiter(),
    )

    assert response.data.total_cost_usd == 1.0
    assert response.data.baseline_cost_usd == 1.5
    assert response.data.total_savings_usd == 0.5


@pytest.mark.asyncio
async def test_list_trajectories_applies_error_filter() -> None:
    rows = [
        {
            "id": "traj-1",
            "session_id": None,
            "agent_type": "router",
            "created_at": datetime.now(timezone.utc),
            "has_error": True,
            "event_count": 1,
            "last_event_at": None,
        }
    ]
    cursor = DummyCursor(fetchall_results=[rows])
    pool = DummyPool(cursor)

    response = await list_trajectories(
        tenant_id="11111111-1111-1111-1111-111111111111",
        status="error",
        agent_type=None,
        limit=50,
        offset=0,
        pool=pool,
        limiter=AllowLimiter(),
    )

    query, _ = cursor.queries[0]
    assert "t.has_error = true" in query
    assert response.data.trajectories[0].has_error is True


@pytest.mark.asyncio
async def test_get_trajectory_detail_not_found() -> None:
    cursor = DummyCursor(fetchone_results=[None])
    pool = DummyPool(cursor)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(trace_crypto=None)))

    with pytest.raises(HTTPException) as exc_info:
        await get_trajectory_detail(
            trajectory_id="traj-1",
            request=request,
            tenant_id="11111111-1111-1111-1111-111111111111",
            pool=pool,
            limiter=AllowLimiter(),
        )
    assert exc_info.value.status_code == 404
