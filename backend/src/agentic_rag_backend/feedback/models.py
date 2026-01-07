"""Data models for the feedback loop system.

Story 20-E2: Implement Self-Improving Feedback Loop
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class FeedbackType(str, Enum):
    """Types of user feedback on retrieval results."""

    RELEVANCE = "relevance"        # Was the result relevant to the query?
    ACCURACY = "accuracy"          # Was the answer factually accurate?
    COMPLETENESS = "completeness"  # Was the answer complete enough?
    PREFERENCE = "preference"      # User preference between options


@dataclass
class UserFeedback:
    """Individual user feedback on a retrieval/response.

    Represents a single piece of feedback from a user about
    the quality of a retrieval result or response.

    Attributes:
        id: Unique feedback identifier
        query_id: ID of the query this feedback relates to
        result_id: ID of the specific result (optional if about query overall)
        feedback_type: Type of feedback being provided
        score: Feedback score from -1.0 (negative) to 1.0 (positive)
        correction: User-provided correction text (optional)
        tenant_id: Tenant identifier for isolation
        user_id: User who provided the feedback
        created_at: When the feedback was recorded
        metadata: Additional context about the feedback
    """

    query_id: str
    feedback_type: FeedbackType
    score: float
    tenant_id: str
    user_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    result_id: Optional[str] = None
    correction: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the feedback after initialization."""
        if not self.query_id:
            raise ValueError("UserFeedback query_id cannot be empty")
        if not self.tenant_id:
            raise ValueError("UserFeedback tenant_id cannot be empty")
        if not self.user_id:
            raise ValueError("UserFeedback user_id cannot be empty")
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(
                f"UserFeedback score must be between -1.0 and 1.0, got {self.score}"
            )
        if isinstance(self.feedback_type, str):
            self.feedback_type = FeedbackType(self.feedback_type)

    @property
    def is_positive(self) -> bool:
        """Check if this is positive feedback (score > 0)."""
        return self.score > 0

    @property
    def is_negative(self) -> bool:
        """Check if this is negative feedback (score < 0)."""
        return self.score < 0

    @property
    def has_correction(self) -> bool:
        """Check if a correction was provided."""
        return bool(self.correction)


@dataclass
class FeedbackStats:
    """Aggregated feedback statistics for a query or result.

    Provides summary statistics for feedback received on
    a particular query or result.

    Attributes:
        query_id: The query these stats relate to
        result_id: The result these stats relate to (optional)
        total_count: Total number of feedback items
        positive_count: Number of positive feedback items
        negative_count: Number of negative feedback items
        average_score: Average score across all feedback
        correction_count: Number of corrections provided
        feedback_types: Count by feedback type
        last_feedback_at: When the most recent feedback was received
    """

    query_id: str
    total_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    average_score: float = 0.0
    correction_count: int = 0
    result_id: Optional[str] = None
    feedback_types: dict[str, int] = field(default_factory=dict)
    last_feedback_at: Optional[datetime] = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def add_feedback(self, feedback: UserFeedback) -> None:
        """Update stats with a new feedback item.

        Args:
            feedback: The new feedback to incorporate
        """
        async with self._lock:
            self.total_count += 1

            if feedback.is_positive:
                self.positive_count += 1
            elif feedback.is_negative:
                self.negative_count += 1

            if feedback.has_correction:
                self.correction_count += 1

            # Update average score
            # new_avg = old_avg + (new_value - old_avg) / count
            self.average_score += (feedback.score - self.average_score) / self.total_count

            # Track by type
            type_key = feedback.feedback_type.value
            self.feedback_types[type_key] = self.feedback_types.get(type_key, 0) + 1

            # Update last feedback time
            if self.last_feedback_at is None or feedback.created_at > self.last_feedback_at:
                self.last_feedback_at = feedback.created_at


@dataclass
class QueryBoost:
    """Boost factors for a query based on feedback.

    Contains the boost factor and metadata about how it was calculated.

    Attributes:
        boost: The boost multiplier (0.5 to 1.5, 1.0 = neutral)
        based_on_queries: Number of similar queries used
        feedback_count: Total feedback items considered
        confidence: Confidence in the boost (0.0 to 1.0)
        decay_applied: Whether time-based decay was applied
    """

    boost: float = 1.0
    based_on_queries: int = 0
    feedback_count: int = 0
    confidence: float = 0.0
    decay_applied: bool = False

    def __post_init__(self) -> None:
        """Validate boost values."""
        if not 0.5 <= self.boost <= 1.5:
            # Clamp to valid range
            self.boost = max(0.5, min(1.5, self.boost))
        if not 0.0 <= self.confidence <= 1.0:
            self.confidence = max(0.0, min(1.0, self.confidence))

    @staticmethod
    def neutral() -> "QueryBoost":
        """Create a neutral boost (no adjustment)."""
        return QueryBoost(boost=1.0, confidence=0.0)


@dataclass
class FeedbackRecordResult:
    """Result of recording feedback.

    Attributes:
        feedback_id: ID of the recorded feedback
        success: Whether recording was successful
        error: Error message if recording failed
        stats_updated: Whether stats were updated
    """

    feedback_id: str
    success: bool = True
    error: Optional[str] = None
    stats_updated: bool = False

    @staticmethod
    def failure(error: str) -> "FeedbackRecordResult":
        """Create a failure result.

        Args:
            error: The error message

        Returns:
            A FeedbackRecordResult indicating failure
        """
        return FeedbackRecordResult(
            feedback_id="",
            success=False,
            error=error,
            stats_updated=False,
        )
