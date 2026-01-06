"""Self-improving feedback loop for retrieval quality.

Story 20-E2: Implement Self-Improving Feedback Loop

This module provides a feedback mechanism that uses user corrections
and preferences to improve retrieval quality over time.
"""

import math
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

import structlog

from .models import (
    FeedbackRecordResult,
    FeedbackStats,
    QueryBoost,
    UserFeedback,
)

logger = structlog.get_logger(__name__)

# Default configuration values
DEFAULT_FEEDBACK_LOOP_ENABLED = False
DEFAULT_FEEDBACK_MIN_SAMPLES = 10
DEFAULT_FEEDBACK_DECAY_DAYS = 90
DEFAULT_FEEDBACK_BOOST_MAX = 1.5
DEFAULT_FEEDBACK_BOOST_MIN = 0.5


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...


class FeedbackLoop:
    """Self-improving feedback system for retrieval quality.

    This class provides methods to:
    - Record and store user feedback on retrieval results
    - Learn from user corrections to improve future results
    - Calculate boost factors for queries based on similar past feedback
    - Apply time-based decay to older feedback

    Feedback is stored per-tenant to ensure tenant isolation.

    Example:
        loop = FeedbackLoop(
            embedding_provider=embeddings,
            decay_days=90,
        )

        # Record feedback
        feedback = UserFeedback(
            query_id="q-123",
            feedback_type=FeedbackType.RELEVANCE,
            score=0.8,
            tenant_id="tenant-1",
            user_id="user-1",
        )
        await loop.record_feedback(feedback)

        # Get boost for a query
        boost = await loop.get_query_boost("similar query", "tenant-1")
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        min_samples: int = DEFAULT_FEEDBACK_MIN_SAMPLES,
        decay_days: int = DEFAULT_FEEDBACK_DECAY_DAYS,
        boost_max: float = DEFAULT_FEEDBACK_BOOST_MAX,
        boost_min: float = DEFAULT_FEEDBACK_BOOST_MIN,
    ) -> None:
        """Initialize the feedback loop.

        Args:
            embedding_provider: Provider for generating embeddings
            min_samples: Minimum feedback count before using for boost
            decay_days: Days after which feedback starts decaying
            boost_max: Maximum boost factor (default 1.5)
            boost_min: Minimum boost factor (default 0.5)
        """
        self._embeddings = embedding_provider
        self._min_samples = min_samples
        self._decay_days = decay_days
        self._boost_max = boost_max
        self._boost_min = boost_min

        # In-memory storage for feedback (can be replaced with persistent storage)
        self._feedback_store: dict[str, list[UserFeedback]] = {}  # tenant_id -> list
        self._query_feedback: dict[str, list[UserFeedback]] = {}  # query_id -> list
        self._query_stats: dict[str, FeedbackStats] = {}  # query_id -> stats
        self._query_embeddings: dict[str, list[float]] = {}  # query_id -> embedding

    async def record_feedback(
        self,
        feedback: UserFeedback,
    ) -> FeedbackRecordResult:
        """Record user feedback.

        Stores the feedback and updates aggregations. If a correction
        is provided, triggers learning from the correction.

        Args:
            feedback: The feedback to record

        Returns:
            FeedbackRecordResult indicating success or failure
        """
        try:
            # Store in tenant's feedback list
            if feedback.tenant_id not in self._feedback_store:
                self._feedback_store[feedback.tenant_id] = []
            self._feedback_store[feedback.tenant_id].append(feedback)

            # Store in query's feedback list
            if feedback.query_id not in self._query_feedback:
                self._query_feedback[feedback.query_id] = []
            self._query_feedback[feedback.query_id].append(feedback)

            # Update query stats
            if feedback.query_id not in self._query_stats:
                self._query_stats[feedback.query_id] = FeedbackStats(
                    query_id=feedback.query_id,
                    result_id=feedback.result_id,
                )
            self._query_stats[feedback.query_id].add_feedback(feedback)

            # If correction provided, learn from it
            if feedback.has_correction:
                await self._learn_from_correction(feedback)

            logger.info(
                "feedback_recorded",
                feedback_id=feedback.id,
                feedback_type=feedback.feedback_type.value,
                score=feedback.score,
                has_correction=feedback.has_correction,
                tenant_id=feedback.tenant_id,
            )

            return FeedbackRecordResult(
                feedback_id=feedback.id,
                success=True,
                stats_updated=True,
            )

        except Exception as e:
            logger.error(
                "feedback_record_failed",
                feedback_id=feedback.id,
                error=str(e),
            )
            return FeedbackRecordResult.failure(error=str(e))

    async def _learn_from_correction(
        self,
        feedback: UserFeedback,
    ) -> None:
        """Learn from a user correction.

        This method can:
        1. Store the correction as a high-quality example
        2. Generate embedding for the correction
        3. Associate correction with query for future retrieval

        Args:
            feedback: The feedback containing the correction
        """
        if not feedback.correction:
            return

        logger.debug(
            "learning_from_correction",
            feedback_id=feedback.id,
            query_id=feedback.query_id,
            correction_length=len(feedback.correction),
        )

        # Generate embedding for the correction if provider available
        if self._embeddings:
            try:
                correction_embedding = await self._embeddings.embed(feedback.correction)

                # Store for future similarity matching
                correction_key = f"{feedback.query_id}:correction:{feedback.id}"
                self._query_embeddings[correction_key] = correction_embedding

                logger.debug(
                    "correction_embedding_stored",
                    feedback_id=feedback.id,
                    embedding_dim=len(correction_embedding),
                )
            except Exception as e:
                logger.warning(
                    "correction_embedding_failed",
                    feedback_id=feedback.id,
                    error=str(e),
                )

    async def get_query_boost(
        self,
        query: str,
        tenant_id: str,
    ) -> QueryBoost:
        """Get boost factors based on similar query feedback.

        Returns adjustment factors for retrieval based on feedback
        from similar past queries.

        Args:
            query: The query to get boost for
            tenant_id: The tenant identifier

        Returns:
            QueryBoost with boost factor and metadata
        """
        # Find similar past queries with feedback
        similar_queries = await self._find_similar_queries(query, tenant_id)

        if not similar_queries:
            return QueryBoost.neutral()

        # Aggregate feedback scores with decay
        total_weighted_score = 0.0
        total_weight = 0.0
        feedback_count = 0
        now = datetime.now(timezone.utc)

        for similar in similar_queries:
            query_id = similar.get("query_id", similar.get("id"))
            if not query_id:
                continue

            feedback_list = self._query_feedback.get(query_id, [])

            for fb in feedback_list:
                # Only consider feedback from same tenant
                if fb.tenant_id != tenant_id:
                    continue

                # Calculate decay weight based on age
                age_days = (now - fb.created_at).days
                if age_days > self._decay_days:
                    # Apply exponential decay after threshold
                    decay_factor = math.exp(
                        -(age_days - self._decay_days) / self._decay_days
                    )
                else:
                    decay_factor = 1.0

                total_weighted_score += fb.score * decay_factor
                total_weight += decay_factor
                feedback_count += 1

        if feedback_count < self._min_samples:
            # Not enough samples to be confident
            return QueryBoost(
                boost=1.0,
                based_on_queries=len(similar_queries),
                feedback_count=feedback_count,
                confidence=feedback_count / self._min_samples,
                decay_applied=total_weight < feedback_count,
            )

        avg_score = total_weighted_score / total_weight if total_weight > 0 else 0

        # Convert score (-1 to 1) to boost factor
        # score = -1 -> boost = 0.5
        # score = 0 -> boost = 1.0
        # score = 1 -> boost = 1.5
        boost = 1.0 + (avg_score * 0.5)

        # Clamp to configured range
        boost = max(self._boost_min, min(self._boost_max, boost))

        # Calculate confidence (based on sample count and recency)
        confidence = min(1.0, feedback_count / (self._min_samples * 2))

        return QueryBoost(
            boost=boost,
            based_on_queries=len(similar_queries),
            feedback_count=feedback_count,
            confidence=confidence,
            decay_applied=total_weight < feedback_count,
        )

    async def _find_similar_queries(
        self,
        query: str,
        tenant_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find similar past queries with feedback.

        Args:
            query: The query to find similar queries for
            tenant_id: The tenant identifier
            limit: Maximum number of similar queries to return

        Returns:
            List of similar query metadata dicts
        """
        if not self._embeddings:
            # Without embeddings, return all queries for this tenant
            # (This is a fallback - production should use embeddings)
            tenant_feedback = self._feedback_store.get(tenant_id, [])
            unique_queries = set(fb.query_id for fb in tenant_feedback)
            return [{"query_id": qid} for qid in list(unique_queries)[:limit]]

        try:
            # Generate embedding for the query
            query_embedding = await self._embeddings.embed(query)

            # Find similar queries using cosine similarity
            similar = []
            for query_id, embedding in self._query_embeddings.items():
                # Skip correction embeddings (they have ':correction:' in the key)
                if ":correction:" in query_id:
                    continue

                # Check if this query has feedback from this tenant
                feedback_list = self._query_feedback.get(query_id, [])
                tenant_feedback = [
                    fb for fb in feedback_list if fb.tenant_id == tenant_id
                ]
                if not tenant_feedback:
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, embedding)
                similar.append({
                    "query_id": query_id,
                    "similarity": similarity,
                    "feedback_count": len(tenant_feedback),
                })

            # Sort by similarity and return top matches
            similar.sort(key=lambda x: x["similarity"], reverse=True)
            return similar[:limit]

        except Exception as e:
            logger.warning(
                "find_similar_queries_failed",
                query=query[:50],
                tenant_id=tenant_id,
                error=str(e),
            )
            return []

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_feedback_stats(self, query_id: str) -> Optional[FeedbackStats]:
        """Get aggregated stats for a query.

        Args:
            query_id: The query identifier

        Returns:
            FeedbackStats or None if no feedback exists
        """
        return self._query_stats.get(query_id)

    def get_feedback_for_query(
        self,
        query_id: str,
        tenant_id: str,
    ) -> list[UserFeedback]:
        """Get all feedback for a query.

        Args:
            query_id: The query identifier
            tenant_id: The tenant identifier for filtering

        Returns:
            List of UserFeedback for this query and tenant
        """
        feedback_list = self._query_feedback.get(query_id, [])
        return [fb for fb in feedback_list if fb.tenant_id == tenant_id]

    def get_feedback_count(self, tenant_id: str) -> int:
        """Get total feedback count for a tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            Total number of feedback items
        """
        return len(self._feedback_store.get(tenant_id, []))

    async def store_query_embedding(
        self,
        query_id: str,
        query: str,
    ) -> bool:
        """Store embedding for a query (for later similarity search).

        Args:
            query_id: The query identifier
            query: The query text

        Returns:
            True if stored successfully
        """
        if not self._embeddings:
            return False

        try:
            embedding = await self._embeddings.embed(query)
            self._query_embeddings[query_id] = embedding
            return True
        except Exception as e:
            logger.warning(
                "store_query_embedding_failed",
                query_id=query_id,
                error=str(e),
            )
            return False

    def clear_tenant_feedback(self, tenant_id: str) -> int:
        """Clear all feedback for a tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            Number of feedback items cleared
        """
        count = len(self._feedback_store.get(tenant_id, []))
        self._feedback_store[tenant_id] = []

        # Also clear from query feedback
        for query_id in list(self._query_feedback.keys()):
            self._query_feedback[query_id] = [
                fb for fb in self._query_feedback[query_id]
                if fb.tenant_id != tenant_id
            ]
            # Remove empty query feedback lists
            if not self._query_feedback[query_id]:
                del self._query_feedback[query_id]
                if query_id in self._query_stats:
                    del self._query_stats[query_id]

        logger.info(
            "tenant_feedback_cleared",
            tenant_id=tenant_id,
            count=count,
        )

        return count
