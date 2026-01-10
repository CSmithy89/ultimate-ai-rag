"""Feature flag adapter for feedback loop system.

Story 20-E2: Implement Self-Improving Feedback Loop

This module provides a feature flag wrapper around the FeedbackLoop
to enable/disable feedback functionality based on configuration.
"""

from typing import Optional

import structlog

from .loop import EmbeddingProvider, FeedbackLoop
from .models import (
    FeedbackRecordResult,
    FeedbackStats,
    QueryBoost,
    UserFeedback,
)

logger = structlog.get_logger(__name__)


class FeedbackLoopAdapter:
    """Feature flag adapter for FeedbackLoop.

    This adapter wraps the FeedbackLoop and only performs feedback
    operations when the feature is enabled. When disabled, all
    operations return neutral/no-op results.

    Example:
        adapter = FeedbackLoopAdapter(
            enabled=settings.feedback_loop_enabled,
            embedding_provider=embeddings,
        )

        # Works regardless of enabled state
        await adapter.record_feedback(feedback)
        boost = await adapter.get_query_boost("query", "tenant-1")
    """

    def __init__(
        self,
        enabled: bool = False,
        embedding_provider: Optional[EmbeddingProvider] = None,
        min_samples: int = 10,
        decay_days: int = 90,
        boost_max: float = 1.5,
        boost_min: float = 0.5,
    ) -> None:
        """Initialize the feedback loop adapter.

        Args:
            enabled: Whether feedback loop is enabled
            embedding_provider: Provider for generating embeddings
            min_samples: Minimum feedback count before using for boost
            decay_days: Days after which feedback starts decaying
            boost_max: Maximum boost factor
            boost_min: Minimum boost factor
        """
        self._enabled = enabled
        self._loop: Optional[FeedbackLoop] = None

        if enabled:
            self._loop = FeedbackLoop(
                embedding_provider=embedding_provider,
                min_samples=min_samples,
                decay_days=decay_days,
                boost_max=boost_max,
                boost_min=boost_min,
            )
            logger.info(
                "feedback_loop_enabled",
                min_samples=min_samples,
                decay_days=decay_days,
                boost_max=boost_max,
                boost_min=boost_min,
            )
        else:
            logger.info("feedback_loop_disabled")

    @property
    def enabled(self) -> bool:
        """Check if feedback loop is enabled."""
        return self._enabled

    async def record_feedback(
        self,
        feedback: UserFeedback,
    ) -> FeedbackRecordResult:
        """Record user feedback.

        When disabled, returns a success result without storing anything.

        Args:
            feedback: The feedback to record

        Returns:
            FeedbackRecordResult indicating success or failure
        """
        if not self._enabled or self._loop is None:
            logger.debug(
                "feedback_record_skipped_disabled",
                feedback_id=feedback.id,
            )
            return FeedbackRecordResult(
                feedback_id=feedback.id,
                success=True,
                stats_updated=False,
            )

        return await self._loop.record_feedback(feedback)

    async def get_query_boost(
        self,
        query: str,
        tenant_id: str,
    ) -> QueryBoost:
        """Get boost factors based on similar query feedback.

        When disabled, returns a neutral boost (1.0).

        Args:
            query: The query to get boost for
            tenant_id: The tenant identifier

        Returns:
            QueryBoost with boost factor and metadata
        """
        if not self._enabled or self._loop is None:
            return QueryBoost.neutral()

        return await self._loop.get_query_boost(query, tenant_id)

    async def get_feedback_stats(self, query_id: str) -> Optional[FeedbackStats]:
        """Get aggregated stats for a query.

        Args:
            query_id: The query identifier

        Returns:
            FeedbackStats or None if disabled or no feedback exists
        """
        if not self._enabled or self._loop is None:
            return None

        return await self._loop.get_feedback_stats(query_id)

    async def get_feedback_for_query(
        self,
        query_id: str,
        tenant_id: str,
    ) -> list[UserFeedback]:
        """Get all feedback for a query.

        Args:
            query_id: The query identifier
            tenant_id: The tenant identifier for filtering

        Returns:
            List of UserFeedback for this query and tenant (empty if disabled)
        """
        if not self._enabled or self._loop is None:
            return []

        return await self._loop.get_feedback_for_query(query_id, tenant_id)

    async def get_feedback_count(self, tenant_id: str) -> int:
        """Get total feedback count for a tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            Total number of feedback items (0 if disabled)
        """
        if not self._enabled or self._loop is None:
            return 0

        return await self._loop.get_feedback_count(tenant_id)

    async def store_query_embedding(
        self,
        query_id: str,
        query: str,
    ) -> bool:
        """Store embedding for a query.

        Args:
            query_id: The query identifier
            query: The query text

        Returns:
            True if stored successfully (False if disabled)
        """
        if not self._enabled or self._loop is None:
            return False

        return await self._loop.store_query_embedding(query_id, query)

    async def clear_tenant_feedback(self, tenant_id: str) -> int:
        """Clear all feedback for a tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            Number of feedback items cleared (0 if disabled)
        """
        if not self._enabled or self._loop is None:
            return 0

        return await self._loop.clear_tenant_feedback(tenant_id)
