"""Ops endpoints for cost monitoring and observability."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from agentic_rag_backend.ops import CostTracker, CostSummary
from agentic_rag_backend.validation import TENANT_ID_PATTERN

router = APIRouter(prefix="/ops", tags=["ops"])


def success_response(data: Any) -> dict[str, Any]:
    return {
        "data": data,
        "meta": {
            "requestId": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }


async def get_cost_tracker(request: Request) -> CostTracker:
    tracker = getattr(request.app.state, "cost_tracker", None)
    if tracker is None:
        raise HTTPException(status_code=503, detail="Cost tracker unavailable")
    return tracker


def _decimal_to_float(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


class AlertConfigRequest(BaseModel):
    tenant_id: str = Field(..., pattern=TENANT_ID_PATTERN)
    daily_threshold_usd: Optional[Decimal] = Field(None, ge=0)
    monthly_threshold_usd: Optional[Decimal] = Field(None, ge=0)
    enabled: bool = True


@router.get("/costs/summary")
async def get_cost_summary(
    tenant_id: str = Query(..., pattern=TENANT_ID_PATTERN),
    window: str = Query("day", description="day|week|month"),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> dict[str, Any]:
    summary = await cost_tracker.get_summary(tenant_id, window)
    return success_response(_serialize_summary(summary))


@router.get("/costs/events")
async def list_cost_events(
    tenant_id: str = Query(..., pattern=TENANT_ID_PATTERN),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> dict[str, Any]:
    events = await cost_tracker.list_events(tenant_id, limit=limit, offset=offset)
    serialized = [
        {key: _decimal_to_float(value) for key, value in event.items()}
        for event in events
    ]
    return success_response({"events": serialized})


@router.get("/costs/alerts")
async def get_cost_alerts(
    tenant_id: str = Query(..., pattern=TENANT_ID_PATTERN),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> dict[str, Any]:
    config = await cost_tracker.get_alert_config(tenant_id)
    if not config:
        return success_response({"alerts": None})
    config = {key: _decimal_to_float(value) for key, value in config.items()}
    return success_response({"alerts": config})


@router.post("/costs/alerts")
async def upsert_cost_alerts(
    payload: AlertConfigRequest = Body(...),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> dict[str, Any]:
    await cost_tracker.upsert_alerts(
        tenant_id=payload.tenant_id,
        daily_threshold_usd=payload.daily_threshold_usd,
        monthly_threshold_usd=payload.monthly_threshold_usd,
        enabled=payload.enabled,
    )
    config = await cost_tracker.get_alert_config(payload.tenant_id)
    return success_response({"alerts": {key: _decimal_to_float(value) for key, value in config.items()}})


def _serialize_summary(summary: CostSummary) -> dict[str, Any]:
    return {
        "total_cost_usd": _decimal_to_float(summary.total_cost_usd),
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
