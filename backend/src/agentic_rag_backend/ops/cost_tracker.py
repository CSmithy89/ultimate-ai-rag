from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from typing import Any
from uuid import UUID

import asyncpg
import structlog
import tiktoken

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Pricing:
    input_per_1k: Decimal
    output_per_1k: Decimal


@dataclass(frozen=True)
class CostTrendPoint:
    bucket: datetime
    total_cost_usd: Decimal
    total_tokens: int


@dataclass(frozen=True)
class CostSummary:
    total_cost_usd: Decimal
    baseline_cost_usd: Decimal
    total_savings_usd: Decimal
    total_tokens: int
    total_requests: int
    by_model: list[dict[str, Any]]
    trend: list[CostTrendPoint]
    alerts: dict[str, Any]


DEFAULT_PRICING: dict[str, Pricing] = {
    "gpt-4o-mini": Pricing(
        input_per_1k=Decimal("0.00015"),
        output_per_1k=Decimal("0.00060"),
    ),
    "gpt-4o": Pricing(
        input_per_1k=Decimal("0.00250"),
        output_per_1k=Decimal("0.01000"),
    ),
}


def _load_pricing(raw_pricing: str | None) -> dict[str, Pricing]:
    if not raw_pricing:
        return DEFAULT_PRICING
    try:
        parsed = json.loads(raw_pricing)
    except json.JSONDecodeError:
        logger.warning("cost_pricing_invalid_json", raw=raw_pricing)
        return DEFAULT_PRICING
    if not isinstance(parsed, dict):
        logger.warning("cost_pricing_invalid_shape", raw=raw_pricing)
        return DEFAULT_PRICING

    pricing: dict[str, Pricing] = {}
    for model_id, values in parsed.items():
        if not isinstance(values, dict):
            continue
        try:
            input_per_1k = Decimal(str(values.get("input_per_1k", "")))
            output_per_1k = Decimal(str(values.get("output_per_1k", "")))
        except Exception:
            continue
        if input_per_1k < 0 or output_per_1k < 0:
            continue
        pricing[str(model_id)] = Pricing(
            input_per_1k=input_per_1k, output_per_1k=output_per_1k
        )

    return pricing or DEFAULT_PRICING


class CostTracker:
    def __init__(self, pool: asyncpg.Pool, pricing_json: str | None = None) -> None:
        self._pool = pool
        self._pricing = _load_pricing(pricing_json)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def _calculate_costs(
        self, model_id: str, prompt_tokens: int, completion_tokens: int
    ) -> tuple[Decimal, Decimal, Decimal]:
        pricing = self._pricing.get(model_id)
        if not pricing:
            logger.warning("cost_pricing_missing", model_id=model_id)
            return Decimal("0"), Decimal("0"), Decimal("0")

        input_cost = (Decimal(prompt_tokens) / Decimal(1000)) * pricing.input_per_1k
        output_cost = (Decimal(completion_tokens) / Decimal(1000)) * pricing.output_per_1k
        total_cost = input_cost + output_cost
        return input_cost, output_cost, total_cost

    async def record_usage(
        self,
        tenant_id: str,
        model_id: str,
        prompt: str,
        completion: str,
        trajectory_id: UUID | None = None,
        complexity: str | None = None,
        baseline_model_id: str | None = None,
    ) -> None:
        prompt_tokens = self._estimate_tokens(prompt)
        completion_tokens = self._estimate_tokens(completion)
        total_tokens = prompt_tokens + completion_tokens
        input_cost, output_cost, total_cost = self._calculate_costs(
            model_id, prompt_tokens, completion_tokens
        )
        baseline_total_cost = Decimal("0")
        savings = Decimal("0")
        if baseline_model_id:
            _, _, baseline_total_cost = self._calculate_costs(
                baseline_model_id,
                prompt_tokens,
                completion_tokens,
            )
            savings = baseline_total_cost - total_cost
            if savings < 0:
                savings = Decimal("0")

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_usage_events (
                    tenant_id,
                    trajectory_id,
                    model_id,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    input_cost_usd,
                    output_cost_usd,
                    total_cost_usd,
                    baseline_model_id,
                    baseline_total_cost_usd,
                    savings_usd,
                    complexity
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                tenant_id,
                trajectory_id,
                model_id,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                input_cost,
                output_cost,
                total_cost,
                baseline_model_id,
                baseline_total_cost,
                savings,
                complexity,
            )

    async def list_events(
        self, tenant_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, trajectory_id, model_id, prompt_tokens, completion_tokens,
                       total_tokens, input_cost_usd, output_cost_usd, total_cost_usd,
                       baseline_model_id, baseline_total_cost_usd, savings_usd,
                       complexity, created_at
                FROM llm_usage_events
                WHERE tenant_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                tenant_id,
                limit,
                offset,
            )
        return [dict(row) for row in rows]

    async def get_summary(
        self, tenant_id: str, window: str = "day"
    ) -> CostSummary:
        if window not in {"day", "week", "month"}:
            window = "day"

        now = datetime.now(timezone.utc)
        if window == "day":
            start = now - timedelta(days=1)
            bucket = "hour"
        elif window == "week":
            start = now - timedelta(days=7)
            bucket = "day"
        else:
            start = now - timedelta(days=30)
            bucket = "day"

        async with self._pool.acquire() as conn:
            totals = await conn.fetchrow(
                """
                SELECT COUNT(*) AS total_requests,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd,
                       COALESCE(SUM(baseline_total_cost_usd), 0) AS baseline_cost_usd,
                       COALESCE(SUM(savings_usd), 0) AS total_savings_usd
                FROM llm_usage_events
                WHERE tenant_id = $1 AND created_at >= $2
                """,
                tenant_id,
                start,
            )

            by_model = await conn.fetch(
                """
                SELECT model_id,
                       COUNT(*) AS requests,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd
                FROM llm_usage_events
                WHERE tenant_id = $1 AND created_at >= $2
                GROUP BY model_id
                ORDER BY total_cost_usd DESC
                """,
                tenant_id,
                start,
            )

            trend_rows = await conn.fetch(
                f"""
                SELECT date_trunc('{bucket}', created_at) AS bucket,
                       COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM llm_usage_events
                WHERE tenant_id = $1 AND created_at >= $2
                GROUP BY bucket
                ORDER BY bucket ASC
                """,
                tenant_id,
                start,
            )

        trend = [
            CostTrendPoint(
                bucket=row["bucket"],
                total_cost_usd=row["total_cost_usd"],
                total_tokens=row["total_tokens"],
            )
            for row in trend_rows
        ]

        alerts = await self.get_alert_status(tenant_id, now)

        return CostSummary(
            total_cost_usd=totals["total_cost_usd"],
            baseline_cost_usd=totals["baseline_cost_usd"],
            total_savings_usd=totals["total_savings_usd"],
            total_tokens=totals["total_tokens"],
            total_requests=totals["total_requests"],
            by_model=[dict(row) for row in by_model],
            trend=trend,
            alerts=alerts,
        )

    async def upsert_alerts(
        self,
        tenant_id: str,
        daily_threshold_usd: Decimal | None,
        monthly_threshold_usd: Decimal | None,
        enabled: bool = True,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_cost_alerts (
                    tenant_id,
                    daily_threshold_usd,
                    monthly_threshold_usd,
                    enabled,
                    created_at,
                    updated_at
                )
                VALUES ($1, $2, $3, $4, NOW(), NOW())
                ON CONFLICT (tenant_id) DO UPDATE SET
                    daily_threshold_usd = EXCLUDED.daily_threshold_usd,
                    monthly_threshold_usd = EXCLUDED.monthly_threshold_usd,
                    enabled = EXCLUDED.enabled,
                    updated_at = NOW()
                """,
                tenant_id,
                daily_threshold_usd,
                monthly_threshold_usd,
                enabled,
            )

    async def get_alert_config(self, tenant_id: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT tenant_id, daily_threshold_usd, monthly_threshold_usd, enabled
                FROM llm_cost_alerts
                WHERE tenant_id = $1
                """,
                tenant_id,
            )
        return dict(row) if row else None

    async def get_alert_status(self, tenant_id: str, now: datetime) -> dict[str, Any]:
        alert_config = await self.get_alert_config(tenant_id)
        if not alert_config or not alert_config.get("enabled", True):
            return {"enabled": False}

        daily_start = now - timedelta(days=1)
        month_start = now - timedelta(days=30)

        async with self._pool.acquire() as conn:
            daily_total = await conn.fetchrow(
                """
                SELECT COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd
                FROM llm_usage_events
                WHERE tenant_id = $1 AND created_at >= $2
                """,
                tenant_id,
                daily_start,
            )
            monthly_total = await conn.fetchrow(
                """
                SELECT COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd
                FROM llm_usage_events
                WHERE tenant_id = $1 AND created_at >= $2
                """,
                tenant_id,
                month_start,
            )

        daily_threshold = alert_config.get("daily_threshold_usd")
        monthly_threshold = alert_config.get("monthly_threshold_usd")

        return {
            "enabled": True,
            "daily_threshold_usd": daily_threshold,
            "monthly_threshold_usd": monthly_threshold,
            "daily_total_usd": daily_total["total_cost_usd"],
            "monthly_total_usd": monthly_total["total_cost_usd"],
            "daily_exceeded": bool(
                daily_threshold is not None and daily_total["total_cost_usd"] >= daily_threshold
            ),
            "monthly_exceeded": bool(
                monthly_threshold is not None
                and monthly_total["total_cost_usd"] >= monthly_threshold
            ),
        }

    def pricing_snapshot(self) -> dict[str, dict[str, str]]:
        return {
            model_id: {
                "input_per_1k": str(pricing.input_per_1k),
                "output_per_1k": str(pricing.output_per_1k),
            }
            for model_id, pricing in self._pricing.items()
        }

    def update_pricing(self, raw_pricing: str) -> None:
        self._pricing = _load_pricing(raw_pricing)
