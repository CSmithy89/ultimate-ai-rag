"""Score normalization strategies for retrieval grading.

Story 19-G4: Support Custom Normalization Strategies

This module provides multiple normalization strategies for converting raw
relevance scores into a consistent 0-1 scale for grading decisions.

Available Strategies:
- MIN_MAX: Linear normalization using (score - min) / (max - min)
- Z_SCORE: Standardization using (score - mean) / std
- SOFTMAX: Exponential normalization using exp(score) / sum(exp(scores))
- PERCENTILE: Rank-based normalization using percentile position
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class NormalizationStrategy(str, Enum):
    """Available score normalization strategies.

    Each strategy converts raw scores to a 0-1 scale differently,
    suitable for different score distributions and use cases.
    """

    MIN_MAX = "min_max"
    """Linear normalization: (score - min) / (max - min).

    Use when scores are bounded and evenly distributed.
    Good for: Vector similarity scores, raw relevance scores.
    """

    Z_SCORE = "z_score"
    """Standardization: (score - mean) / std, then scaled to 0-1.

    Use when scores follow a roughly normal distribution.
    Good for: Cross-encoder scores, unbounded relevance signals.
    """

    SOFTMAX = "softmax"
    """Exponential normalization: exp(score) / sum(exp(scores)).

    Use when you want to emphasize score differences.
    Good for: Ranking tasks, when relative differences matter.
    """

    PERCENTILE = "percentile"
    """Rank-based normalization: position in sorted scores.

    Use when you want position-based scoring regardless of score values.
    Good for: Mixed score sources, heterogeneous scoring systems.
    """


def normalize_min_max(
    scores: list[float],
    epsilon: float = 1e-10,
) -> list[float]:
    """Normalize scores using min-max scaling.

    Formula: (score - min) / (max - min)

    Args:
        scores: List of raw scores to normalize
        epsilon: Small value to prevent division by zero

    Returns:
        List of normalized scores in [0, 1] range
    """
    if not scores:
        return []

    if len(scores) == 1:
        return [0.5]  # Single score gets middle value

    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score

    if range_score < epsilon:
        # All scores are the same
        return [0.5] * len(scores)

    return [(s - min_score) / range_score for s in scores]


def normalize_z_score(
    scores: list[float],
    epsilon: float = 1e-10,
) -> list[float]:
    """Normalize scores using z-score standardization.

    Formula: (score - mean) / std, then scaled to approximate 0-1 via sigmoid.

    Uses sigmoid function to map z-scores to [0, 1] range:
    sigmoid(z) = 1 / (1 + exp(-z))

    Args:
        scores: List of raw scores to normalize
        epsilon: Small value to prevent division by zero

    Returns:
        List of normalized scores in [0, 1] range
    """
    if not scores:
        return []

    if len(scores) == 1:
        return [0.5]

    n = len(scores)
    mean_score = sum(scores) / n

    # Calculate standard deviation
    variance = sum((s - mean_score) ** 2 for s in scores) / n
    std_score = math.sqrt(variance)

    if std_score < epsilon:
        # All scores are the same
        return [0.5] * n

    # Calculate z-scores and apply sigmoid
    z_scores = [(s - mean_score) / std_score for s in scores]

    # Apply sigmoid to map to [0, 1]
    return [1 / (1 + math.exp(-z)) for z in z_scores]


def normalize_softmax(
    scores: list[float],
    temperature: float = 1.0,
) -> list[float]:
    """Normalize scores using softmax function.

    Formula: exp(score / T) / sum(exp(scores / T))

    The temperature parameter controls how much the differences
    between scores are emphasized:
    - T < 1: More emphasis on score differences (sharper distribution)
    - T = 1: Standard softmax
    - T > 1: Less emphasis on score differences (smoother distribution)

    Args:
        scores: List of raw scores to normalize
        temperature: Softmax temperature (default: 1.0)

    Returns:
        List of normalized scores that sum to 1.0
    """
    if not scores:
        return []

    if len(scores) == 1:
        return [1.0]  # Single score gets all probability

    # Scale by temperature
    scaled = [s / temperature for s in scores]

    # Subtract max for numerical stability
    max_scaled = max(scaled)
    exp_scores = [math.exp(s - max_scaled) for s in scaled]
    sum_exp = sum(exp_scores)

    return [e / sum_exp for e in exp_scores]


def normalize_percentile(scores: list[float]) -> list[float]:
    """Normalize scores using percentile ranking.

    Each score is converted to its percentile position in the distribution.
    Ties are handled by averaging the percentile positions.

    Args:
        scores: List of raw scores to normalize

    Returns:
        List of normalized scores in [0, 1] range based on percentile rank
    """
    if not scores:
        return []

    n = len(scores)

    if n == 1:
        return [0.5]

    # Create sorted indices with original positions
    indexed_scores = [(score, i) for i, score in enumerate(scores)]
    sorted_scores = sorted(indexed_scores, key=lambda x: x[0])

    # Calculate percentile for each position
    # Handle ties by averaging percentile positions
    normalized = [0.0] * n
    i = 0
    while i < n:
        # Find all scores with the same value (ties)
        j = i
        while j < n and sorted_scores[j][0] == sorted_scores[i][0]:
            j += 1

        # Average percentile for tied scores
        # Percentile ranges from 0 (lowest) to 1 (highest)
        avg_percentile = (i + j - 1) / (2 * (n - 1)) if n > 1 else 0.5

        # Assign to all tied positions
        for k in range(i, j):
            original_idx = sorted_scores[k][1]
            normalized[original_idx] = avg_percentile

        i = j

    return normalized


def normalize_scores(
    scores: list[float],
    strategy: NormalizationStrategy,
    temperature: float = 1.0,
) -> list[float]:
    """Normalize scores using the specified strategy.

    Args:
        scores: List of raw scores to normalize
        strategy: Normalization strategy to use
        temperature: Temperature for softmax (only used with SOFTMAX strategy)

    Returns:
        List of normalized scores
    """
    if not scores:
        return []

    if strategy == NormalizationStrategy.MIN_MAX:
        return normalize_min_max(scores)
    elif strategy == NormalizationStrategy.Z_SCORE:
        return normalize_z_score(scores)
    elif strategy == NormalizationStrategy.SOFTMAX:
        return normalize_softmax(scores, temperature=temperature)
    elif strategy == NormalizationStrategy.PERCENTILE:
        return normalize_percentile(scores)
    else:
        logger.warning(
            "unknown_normalization_strategy",
            strategy=strategy,
            fallback="min_max",
        )
        return normalize_min_max(scores)


def get_normalization_strategy(strategy_name: str) -> NormalizationStrategy:
    """Get normalization strategy enum from string name.

    Args:
        strategy_name: Strategy name (case-insensitive)

    Returns:
        NormalizationStrategy enum value

    Raises:
        ValueError: If strategy name is not valid
    """
    try:
        return NormalizationStrategy(strategy_name.lower())
    except ValueError:
        valid = [s.value for s in NormalizationStrategy]
        raise ValueError(
            f"Invalid normalization strategy: {strategy_name!r}. "
            f"Valid options: {', '.join(valid)}"
        )


def aggregate_normalized_scores(
    normalized_scores: list[float],
    aggregation: str = "mean",
) -> float:
    """Aggregate normalized scores to a single value.

    Args:
        normalized_scores: List of normalized scores
        aggregation: Aggregation method (mean, max, median, weighted_mean)

    Returns:
        Single aggregated score
    """
    if not normalized_scores:
        return 0.0

    if aggregation == "mean":
        return sum(normalized_scores) / len(normalized_scores)
    elif aggregation == "max":
        return max(normalized_scores)
    elif aggregation == "min":
        return min(normalized_scores)
    elif aggregation == "median":
        sorted_scores = sorted(normalized_scores)
        n = len(sorted_scores)
        if n % 2 == 0:
            return (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
        return sorted_scores[n // 2]
    elif aggregation == "weighted_mean":
        # Weight by position (first items weighted more)
        n = len(normalized_scores)
        weights = [1.0 / (i + 1) for i in range(n)]
        sum_weights = sum(weights)
        return sum(s * w for s, w in zip(normalized_scores, weights)) / sum_weights
    else:
        # Default to mean
        return sum(normalized_scores) / len(normalized_scores)
