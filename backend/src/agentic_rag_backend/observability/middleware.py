"""FastAPI middleware for exposing Prometheus /metrics endpoint.

This module provides the middleware and configuration for serving
Prometheus metrics from the FastAPI application.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, Response
from fastapi.routing import APIRoute
from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
)
import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for Prometheus metrics endpoint.

    Attributes:
        enabled: Whether metrics endpoint is enabled
        path: URL path for the metrics endpoint
        port: Port for the metrics endpoint (if separate from main app)
    """

    enabled: bool = False
    path: str = "/metrics"
    port: int = 9090


def create_metrics_endpoint(
    app: FastAPI,
    config: MetricsConfig,
    registry: CollectorRegistry | None = None,
) -> None:
    """Create and mount the /metrics endpoint on the FastAPI app.

    Args:
        app: FastAPI application instance
        config: Metrics configuration
        registry: Optional custom CollectorRegistry (uses default if None)
    """
    if not config.enabled:
        logger.info("prometheus_metrics_disabled")
        return

    if registry is None:
        registry = REGISTRY

    async def metrics_endpoint() -> Response:
        """Serve Prometheus metrics in text format.

        Returns:
            Response with metrics in Prometheus text format
        """
        try:
            metrics_output = generate_latest(registry)
            return Response(
                content=metrics_output,
                media_type=CONTENT_TYPE_LATEST,
            )
        except Exception as e:
            logger.error("metrics_generation_failed", error=str(e))
            return Response(
                content=f"# Error generating metrics: {e}",
                media_type="text/plain",
                status_code=500,
            )

    # Create the route directly to avoid issues with decorators
    route = APIRoute(
        path=config.path,
        endpoint=metrics_endpoint,
        methods=["GET"],
        name="prometheus_metrics",
        tags=["observability"],
        summary="Prometheus metrics endpoint",
        description="Returns metrics in Prometheus text exposition format",
    )

    # Add the route to the app's routes
    app.routes.append(route)

    logger.info(
        "prometheus_metrics_endpoint_mounted",
        path=config.path,
        enabled=config.enabled,
    )
