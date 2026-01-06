"""Self-improving feedback loop for retrieval quality.

Story 20-E2: Implement Self-Improving Feedback Loop

This module provides a feedback mechanism that uses user corrections
and preferences to improve retrieval quality over time.

Components:
- FeedbackType: Enum of feedback types (relevance, accuracy, etc.)
- UserFeedback: Individual feedback record with score and optional correction
- FeedbackStats: Aggregated statistics for a query
- QueryBoost: Boost factors calculated from past feedback
- FeedbackLoop: Core feedback processing logic
- FeedbackLoopAdapter: Feature flag wrapper

Example:
    from agentic_rag_backend.feedback import (
        FeedbackLoopAdapter,
        FeedbackType,
        UserFeedback,
    )

    # Create adapter with feature flag
    adapter = FeedbackLoopAdapter(
        enabled=settings.feedback_loop_enabled,
        embedding_provider=embeddings,
    )

    # Record feedback
    feedback = UserFeedback(
        query_id="q-123",
        feedback_type=FeedbackType.RELEVANCE,
        score=0.8,
        tenant_id="tenant-1",
        user_id="user-1",
    )
    await adapter.record_feedback(feedback)

    # Get boost for similar queries
    boost = await adapter.get_query_boost("my query", "tenant-1")
    if boost.confidence > 0.5:
        # Apply boost to retrieval
        pass
"""

from .adapter import FeedbackLoopAdapter
from .loop import (
    DEFAULT_FEEDBACK_BOOST_MAX,
    DEFAULT_FEEDBACK_BOOST_MIN,
    DEFAULT_FEEDBACK_DECAY_DAYS,
    DEFAULT_FEEDBACK_LOOP_ENABLED,
    DEFAULT_FEEDBACK_MIN_SAMPLES,
    EmbeddingProvider,
    FeedbackLoop,
)
from .models import (
    FeedbackRecordResult,
    FeedbackStats,
    FeedbackType,
    QueryBoost,
    UserFeedback,
)

__all__ = [
    # Models
    "FeedbackType",
    "UserFeedback",
    "FeedbackStats",
    "QueryBoost",
    "FeedbackRecordResult",
    # Core
    "FeedbackLoop",
    "EmbeddingProvider",
    # Adapter
    "FeedbackLoopAdapter",
    # Defaults
    "DEFAULT_FEEDBACK_LOOP_ENABLED",
    "DEFAULT_FEEDBACK_MIN_SAMPLES",
    "DEFAULT_FEEDBACK_DECAY_DAYS",
    "DEFAULT_FEEDBACK_BOOST_MAX",
    "DEFAULT_FEEDBACK_BOOST_MIN",
]
