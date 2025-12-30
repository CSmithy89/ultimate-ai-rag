"""Ops endpoints for cost monitoring and observability."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from agentic_rag_backend.api.utils import build_meta, rate_limit_exceeded
from agentic_rag_backend.ops import CostTracker, CostSummary, TraceCrypto
from agentic_rag_backend.rate_limit import RateLimiter
from agentic_rag_backend.validation import TENANT_ID_PATTERN
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

router = APIRouter(prefix="/ops", tags=["ops"])


def success_response(data: Any) -> dict[str, Any]:
    return {
        "data": data,
        "meta": build_meta(),
    }


async def get_cost_tracker(request: Request) -> CostTracker:
    tracker = getattr(request.app.state, "cost_tracker", None)
    if tracker is None:
        raise HTTPException(status_code=503, detail="Cost tracker unavailable")
    return tracker


async def get_trajectory_pool(request: Request) -> AsyncConnectionPool:
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="Trajectory storage unavailable")
    return pool


def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter from app state."""
    return request.app.state.rate_limiter


def _decrypt_events(events: list[dict[str, Any]], crypto: TraceCrypto | None) -> None:
    if not crypto:
        return
    for event in events:
        content = event.get("content")
        if isinstance(content, str):
            event["content"] = crypto.decrypt(content)


def _decimal_to_float(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


class MetaResponse(BaseModel):
    requestId: str
    timestamp: str


class CostSummaryByModelResponse(BaseModel):
    model_id: str
    requests: int
    total_tokens: int
    total_cost_usd: float


class CostTrendPointResponse(BaseModel):
    bucket: str
    total_cost_usd: float
    total_tokens: int


class CostAlertStatusResponse(BaseModel):
    enabled: bool
    daily_threshold_usd: Optional[float] = None
    monthly_threshold_usd: Optional[float] = None
    daily_total_usd: Optional[float] = None
    monthly_total_usd: Optional[float] = None
    daily_exceeded: Optional[bool] = None
    monthly_exceeded: Optional[bool] = None


class CostSummaryDataResponse(BaseModel):
    total_cost_usd: float
    baseline_cost_usd: float
    total_savings_usd: float
    total_tokens: int
    total_requests: int
    by_model: list[CostSummaryByModelResponse]
    trend: list[CostTrendPointResponse]
    alerts: CostAlertStatusResponse


class CostSummaryResponse(BaseModel):
    data: CostSummaryDataResponse
    meta: MetaResponse


class CostEventResponse(BaseModel):
    id: str
    trajectory_id: Optional[str] = None
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    baseline_model_id: Optional[str] = None
    baseline_total_cost_usd: Optional[float] = None
    savings_usd: Optional[float] = None
    complexity: Optional[str] = None
    created_at: datetime


class CostEventsDataResponse(BaseModel):
    events: list[CostEventResponse]


class CostEventsResponse(BaseModel):
    data: CostEventsDataResponse
    meta: MetaResponse


class AlertConfigResponse(BaseModel):
    tenant_id: str
    daily_threshold_usd: Optional[float] = None
    monthly_threshold_usd: Optional[float] = None
    enabled: bool


class AlertsDataResponse(BaseModel):
    alerts: Optional[AlertConfigResponse]


class AlertsResponse(BaseModel):
    data: AlertsDataResponse
    meta: MetaResponse


class TrajectorySummaryResponse(BaseModel):
    id: str
    session_id: Optional[str] = None
    agent_type: Optional[str] = None
    created_at: datetime
    has_error: bool
    event_count: int
    last_event_at: Optional[datetime] = None


class TrajectoryListDataResponse(BaseModel):
    trajectories: list[TrajectorySummaryResponse]


class TrajectoryListResponse(BaseModel):
    data: TrajectoryListDataResponse
    meta: MetaResponse


class TrajectoryEventResponse(BaseModel):
    id: str
    event_type: str
    content: str
    created_at: datetime


class TrajectoryInfoResponse(BaseModel):
    id: str
    session_id: Optional[str] = None
    agent_type: Optional[str] = None
    created_at: datetime


class TrajectoryDetailDataResponse(BaseModel):
    trajectory: TrajectoryInfoResponse
    events: list[TrajectoryEventResponse]
    duration_ms: Optional[int] = None


class TrajectoryDetailEnvelopeResponse(BaseModel):
    data: TrajectoryDetailDataResponse
    meta: MetaResponse


class TraceCryptoHealthDataResponse(BaseModel):
    key_fingerprint: str
    key_source: str


class TraceCryptoHealthResponse(BaseModel):
    data: TraceCryptoHealthDataResponse
    meta: MetaResponse


class AlertConfigRequest(BaseModel):
    tenant_id: str = Field(..., pattern=TENANT_ID_PATTERN)
    daily_threshold_usd: Optional[Decimal] = Field(None, ge=0)
    monthly_threshold_usd: Optional[Decimal] = Field(None, ge=0)
    enabled: bool = True


@router.get("/costs/summary", response_model=CostSummaryResponse)
async def get_cost_summary(
    tenant_id: str = Query(..., pattern=TENANT_ID_PATTERN),
    window: str = Query("day", description="day|week|month"),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> CostSummaryResponse:
    """Get aggregated cost summary with trends and alert status."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()
    summary = await cost_tracker.get_summary(tenant_id, window)
    return success_response(_serialize_summary(summary))


@router.get("/costs/events", response_model=CostEventsResponse)
async def list_cost_events(
    tenant_id: str = Query(..., pattern=TENANT_ID_PATTERN),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> CostEventsResponse:
    """List recent cost usage events for a tenant."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()
    events = await cost_tracker.list_events(tenant_id, limit=limit, offset=offset)
    serialized = [
        {key: _decimal_to_float(value) for key, value in event.items()}
        for event in events
    ]
    return success_response({"events": serialized})


@router.get("/costs/alerts", response_model=AlertsResponse)
async def get_cost_alerts(
    tenant_id: str = Query(..., pattern=TENANT_ID_PATTERN),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> AlertsResponse:
    """Fetch alert threshold configuration for a tenant."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()
    config = await cost_tracker.get_alert_config(tenant_id)
    if not config:
        return success_response({"alerts": None})
    config = {key: _decimal_to_float(value) for key, value in config.items()}
    return success_response({"alerts": config})


@router.post("/costs/alerts", response_model=AlertsResponse)
async def upsert_cost_alerts(
    payload: AlertConfigRequest = Body(...),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> AlertsResponse:
    """Create or update cost alert thresholds for a tenant."""
    if not await limiter.allow(payload.tenant_id):
        raise rate_limit_exceeded()
    await cost_tracker.upsert_alerts(
        tenant_id=payload.tenant_id,
        daily_threshold_usd=payload.daily_threshold_usd,
        monthly_threshold_usd=payload.monthly_threshold_usd,
        enabled=payload.enabled,
    )
    config = await cost_tracker.get_alert_config(payload.tenant_id)
    if not config:
        return success_response({"alerts": None})
    return success_response(
        {"alerts": {key: _decimal_to_float(value) for key, value in config.items()}}
    )


@router.get("/crypto/health", response_model=TraceCryptoHealthResponse)
async def get_crypto_health(
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> TraceCryptoHealthResponse:
    """Return fingerprint metadata for the trace encryption key."""
    if not await limiter.allow("ops:crypto:health"):
        raise rate_limit_exceeded()
    fingerprint = getattr(request.app.state, "trace_key_fingerprint", "")
    source = getattr(request.app.state, "trace_key_source", "unknown")
    return success_response(
        {
            "key_fingerprint": fingerprint,
            "key_source": source,
        }
    )


def _serialize_summary(summary: CostSummary) -> dict[str, Any]:
    return {
        "total_cost_usd": _decimal_to_float(summary.total_cost_usd),
        "baseline_cost_usd": _decimal_to_float(summary.baseline_cost_usd),
        "total_savings_usd": _decimal_to_float(summary.total_savings_usd),
        "total_tokens": summary.total_tokens,
        "total_requests": summary.total_requests,
        "by_model": [
            {key: _decimal_to_float(value) for key, value in row.items()}
            for row in summary.by_model
        ],
        "trend": [
            {
                "bucket": point.bucket.isoformat().replace("+00:00", "Z"),
                "total_cost_usd": _decimal_to_float(point.total_cost_usd),
                "total_tokens": point.total_tokens,
            }
            for point in summary.trend
        ],
        "alerts": {key: _decimal_to_float(value) for key, value in summary.alerts.items()},
    }


@router.get("/trajectories", response_model=TrajectoryListResponse)
async def list_trajectories(
    tenant_id: str = Query(..., pattern=TENANT_ID_PATTERN),
    status: str | None = Query(None, description="error|ok"),
    agent_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    pool: AsyncConnectionPool = Depends(get_trajectory_pool),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> TrajectoryListResponse:
    """List trajectories with optional error status and agent-type filters."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()
    conditions = ["t.tenant_id = %s"]
    params: list[Any] = [tenant_id]

    if agent_type:
        conditions.append("t.agent_type = %s")
        params.append(agent_type)

    if status == "error":
        conditions.append("t.has_error = true")
    elif status == "ok":
        conditions.append("t.has_error = false")

    where_sql = " AND ".join(conditions)
    query = (
        "SELECT t.id, "
        "t.session_id, "
        "t.agent_type, "
        "t.created_at, "
        "t.has_error AS has_error, "
        "(SELECT COUNT(*) FROM trajectory_events e "
        "WHERE e.trajectory_id = t.id AND e.tenant_id = t.tenant_id) AS event_count, "
        "(SELECT MAX(created_at) FROM trajectory_events e "
        "WHERE e.trajectory_id = t.id AND e.tenant_id = t.tenant_id) AS last_event_at "
        "FROM trajectories t "
        "WHERE " + where_sql + " "
        "ORDER BY t.created_at DESC "
        "LIMIT %s OFFSET %s"
    )
    params.extend([limit, offset])

    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, params)
            rows = await cursor.fetchall()

    return success_response({"trajectories": rows})


@router.get("/trajectories/{trajectory_id}", response_model=TrajectoryDetailEnvelopeResponse)
async def get_trajectory_detail(
    trajectory_id: str,
    request: Request,
    tenant_id: str = Query(..., pattern=TENANT_ID_PATTERN),
    pool: AsyncConnectionPool = Depends(get_trajectory_pool),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> TrajectoryDetailEnvelopeResponse:
    """Get detailed trajectory events and timing information."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                """
                SELECT id, session_id, agent_type, created_at
                FROM trajectories
                WHERE id = %s AND tenant_id = %s
                """,
                (trajectory_id, tenant_id),
            )
            trajectory = await cursor.fetchone()
            if not trajectory:
                raise HTTPException(status_code=404, detail="Trajectory not found")

            await cursor.execute(
                """
                SELECT id, event_type, content, created_at
                FROM trajectory_events
                WHERE trajectory_id = %s AND tenant_id = %s
                ORDER BY created_at ASC
                """,
                (trajectory_id, tenant_id),
            )
            events = await cursor.fetchall()

    crypto = getattr(request.app.state, "trace_crypto", None)
    _decrypt_events(events, crypto)

    duration_ms = None
    if events:
        start = events[0]["created_at"]
        end = events[-1]["created_at"]
        if start and end:
            duration_ms = int((end - start).total_seconds() * 1000)

    return success_response(
        {
            "trajectory": trajectory,
            "events": events,
            "duration_ms": duration_ms,
        }
    )
