"""Corrective RAG Grader Agent for evaluating retrieval quality.

This module implements the CRAG (Corrective Retrieval Augmented Generation) pattern,
which evaluates retrieval results and triggers fallback strategies when quality is low.

The grader uses a lightweight approach (cross-encoder or simple heuristics) rather
than full LLM calls to minimize latency and cost.

Story 19-G3: Supports model preloading for reduced first-query latency.
Story 19-G4: Supports configurable score normalization strategies.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

import structlog

from agentic_rag_backend.config import Settings
from agentic_rag_backend.observability.metrics import (
    record_grader_evaluation,
    record_grader_score,
    record_retrieval_latency,
    record_retrieval_fallback,
)
from agentic_rag_backend.retrieval.normalization import (
    NormalizationStrategy,
    normalize_scores,
    aggregate_normalized_scores,
    get_normalization_strategy,
)

logger = structlog.get_logger(__name__)

# Grader configuration constants
MAX_CROSS_ENCODER_HITS = 10  # Maximum hits to evaluate with cross-encoder
# Default heuristic length weight configuration (can be overridden via settings)
DEFAULT_HEURISTIC_LENGTH_WEIGHT = 0.5
DEFAULT_HEURISTIC_MIN_LENGTH = 50
DEFAULT_HEURISTIC_MAX_LENGTH = 2000


class FallbackStrategy(str, Enum):
    """Available fallback strategies when grader score is below threshold."""

    WEB_SEARCH = "web_search"
    EXPANDED_QUERY = "expanded_query"
    ALTERNATE_INDEX = "alternate_index"


@dataclass
class GraderResult:
    """Result of grading retrieval results.

    Attributes:
        score: Relevance score (0.0-1.0)
        passed: Whether the score meets the threshold
        threshold: The threshold used for comparison
        grading_time_ms: Time taken to grade in milliseconds
        fallback_triggered: Whether fallback was triggered
        fallback_strategy: The fallback strategy used (if triggered)
    """

    score: float
    passed: bool
    threshold: float
    grading_time_ms: int
    fallback_triggered: bool = False
    fallback_strategy: Optional[FallbackStrategy] = None


@dataclass
class RetrievalHit:
    """A single retrieval result for grading.

    Attributes:
        content: The retrieved content/chunk text
        score: The retrieval score (if available)
        metadata: Additional metadata about the hit
    """

    content: str
    score: Optional[float] = None
    metadata: Optional[dict] = None


class BaseGrader(ABC):
    """Abstract base class for retrieval graders."""

    @abstractmethod
    async def grade(
        self,
        query: str,
        hits: list[RetrievalHit],
        threshold: float,
        tenant_id: Optional[str] = None,
        strategy: str = "hybrid",
    ) -> GraderResult:
        """Grade the retrieval results for a query.

        Args:
            query: The original query string
            hits: List of retrieval hits to grade
            threshold: Score threshold for passing (0.0-1.0)
            tenant_id: Tenant identifier for metrics
            strategy: Retrieval strategy for metrics labeling

        Returns:
            GraderResult with score and pass/fail status
        """
        pass

    @abstractmethod
    def get_model(self) -> str:
        """Return the model identifier used for grading."""
        pass


class HeuristicGrader(BaseGrader):
    """Simple heuristic-based grader using retrieval scores.

    This grader uses the average retrieval score of top-k hits as the
    relevance signal. It's fast but less accurate than model-based grading.

    When retrieval scores are not available, it uses a configurable content
    length heuristic. Longer content may indicate more comprehensive context,
    though this is domain-dependent. The length weight controls how much
    content length influences the final score vs. a neutral baseline.

    Formula when no retrieval scores available:
        length_factor = min((avg_length - min_length) / (max_length - min_length), 1.0)
        final_score = base_score * (1 - length_weight) + length_factor * length_weight

    Where:
        - length_weight=0: Length has no influence, score is always base_score (0.5)
        - length_weight=0.5 (default): Balanced influence from length
        - length_weight=1.0: Score is purely based on content length
    """

    def __init__(
        self,
        top_k: int = 5,
        length_weight: float = DEFAULT_HEURISTIC_LENGTH_WEIGHT,
        min_length: int = DEFAULT_HEURISTIC_MIN_LENGTH,
        max_length: int = DEFAULT_HEURISTIC_MAX_LENGTH,
    ):
        """Initialize the heuristic grader.

        Args:
            top_k: Number of top hits to consider for scoring
            length_weight: How much content length influences the heuristic score (0.0-1.0).
                          0 = length disabled (use neutral 0.5 score)
                          0.5 = balanced (default)
                          1.0 = score purely based on length
            min_length: Minimum content length for any length contribution (default: 50)
            max_length: Content length at which length bonus maxes out (default: 2000)
        """
        self.top_k = top_k
        self.length_weight = max(0.0, min(1.0, length_weight))
        self.min_length = max(1, min_length)
        self.max_length = max(self.min_length + 1, max_length)

    async def grade(
        self,
        query: str,
        hits: list[RetrievalHit],
        threshold: float,
        tenant_id: Optional[str] = None,
        strategy: str = "hybrid",
    ) -> GraderResult:
        """Grade using average retrieval score of top-k hits."""
        start_time = time.perf_counter()

        if not hits:
            grading_time_ms = int((time.perf_counter() - start_time) * 1000)
            elapsed_seconds = grading_time_ms / 1000
            # Record metrics for empty results
            if tenant_id is not None:
                record_retrieval_latency(
                    strategy=strategy,
                    phase="grade",
                    tenant_id=tenant_id,
                    duration_seconds=elapsed_seconds,
                )
                record_grader_evaluation(result="fallback", tenant_id=tenant_id)
                record_grader_score(
                    model=self.get_model(),
                    tenant_id=tenant_id,
                    score=0.0,
                )
                record_retrieval_fallback(reason="empty_results", tenant_id=tenant_id)
            return GraderResult(
                score=0.0,
                passed=False,
                threshold=threshold,
                grading_time_ms=grading_time_ms,
                fallback_triggered=True,
            )

        # Use top-k hits for scoring
        top_hits = hits[: self.top_k]

        # Track heuristic contribution for logging
        heuristic_applied = False
        length_factor = 0.0
        base_score = 0.5  # Neutral baseline when no retrieval scores available

        # Calculate average score from retrieval scores
        scores = [h.score for h in top_hits if h.score is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            # Normalize to 0-1 range (assuming scores might be higher)
            score = min(1.0, max(0.0, avg_score))
        else:
            # If no retrieval scores available, use configurable content length heuristic
            # The length_weight determines how much content length influences the score
            heuristic_applied = True
            avg_length = sum(len(h.content) for h in top_hits) / len(top_hits)

            # Calculate length factor normalized between min_length and max_length
            # Content below min_length gets 0, content at or above max_length gets 1
            if avg_length <= self.min_length:
                length_factor = 0.0
            else:
                length_factor = min(
                    1.0,
                    (avg_length - self.min_length) / (self.max_length - self.min_length),
                )

            # Apply weighted blend: base_score * (1 - weight) + length_factor * weight
            # weight=0: score = base_score (length disabled)
            # weight=1: score = length_factor (pure length-based)
            score = base_score * (1 - self.length_weight) + length_factor * self.length_weight

        passed = score >= threshold
        grading_time_ms = int((time.perf_counter() - start_time) * 1000)
        elapsed_seconds = grading_time_ms / 1000

        # Record metrics
        if tenant_id is not None:
            record_retrieval_latency(
                strategy=strategy,
                phase="grade",
                tenant_id=tenant_id,
                duration_seconds=elapsed_seconds,
            )
            if passed:
                record_grader_evaluation(result="pass", tenant_id=tenant_id)
            else:
                record_grader_evaluation(result="fail", tenant_id=tenant_id)
                record_retrieval_fallback(reason="low_score", tenant_id=tenant_id)
            record_grader_score(
                model=self.get_model(),
                tenant_id=tenant_id,
                score=score,
            )

        # Build log context with heuristic details when applicable
        log_context = {
            "score": round(score, 4),
            "threshold": threshold,
            "passed": passed,
            "num_hits": len(hits),
            "top_k": self.top_k,
            "grading_time_ms": grading_time_ms,
        }

        if heuristic_applied:
            log_context.update({
                "heuristic_applied": True,
                "length_weight": self.length_weight,
                "length_factor": round(length_factor, 4),
                "base_score": base_score,
                "avg_content_length": round(avg_length, 1),
                "min_length": self.min_length,
                "max_length": self.max_length,
            })
            logger.info(
                "heuristic_grader_result",
                **log_context,
            )
        else:
            logger.debug(
                "heuristic_grader_result",
                **log_context,
            )

        return GraderResult(
            score=score,
            passed=passed,
            threshold=threshold,
            grading_time_ms=grading_time_ms,
            fallback_triggered=not passed,
        )

    def get_model(self) -> str:
        """Return model identifier."""
        return "heuristic"


# Default fallback model for CrossEncoderGrader if configured model fails to load
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Supported cross-encoder models with their characteristics
SUPPORTED_GRADER_MODELS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "description": "Fast, good accuracy (default)",
        "size": "~80MB",
        "speed": "fast",
        "accuracy": "good",
    },
    "cross-encoder/ms-marco-MiniLM-L-12-v2": {
        "description": "Higher accuracy, slower",
        "size": "~120MB",
        "speed": "medium",
        "accuracy": "high",
    },
    "BAAI/bge-reranker-base": {
        "description": "BGE reranker, balanced",
        "size": "~400MB",
        "speed": "medium",
        "accuracy": "high",
    },
    "BAAI/bge-reranker-large": {
        "description": "BGE large, best accuracy",
        "size": "~1.3GB",
        "speed": "slow",
        "accuracy": "highest",
    },
}


class CrossEncoderGrader(BaseGrader):
    """Cross-encoder based grader for more accurate relevance scoring.

    This grader uses a cross-encoder model to score query-document pairs,
    providing more accurate relevance assessment than retrieval scores alone.

    Supported models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good accuracy, default)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (higher accuracy)
    - BAAI/bge-reranker-base (BGE reranker)
    - BAAI/bge-reranker-large (BGE large, best accuracy)

    Story 19-G3: Supports eager preloading for reduced first-query latency.
    Story 19-G4: Supports configurable normalization strategies.

    Note: This requires sentence-transformers to be available.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        fallback_to_default: bool = True,
        preload: bool = False,
        normalization_strategy: NormalizationStrategy = NormalizationStrategy.MIN_MAX,
    ):
        """Initialize the cross-encoder grader.

        Args:
            model_name: Name of the cross-encoder model to use
            fallback_to_default: If True, fall back to default model if configured model fails
            preload: If True, load the model immediately (Story 19-G3)
            normalization_strategy: Strategy for normalizing scores (Story 19-G4)
        """
        self.model_name = model_name
        self.fallback_to_default = fallback_to_default
        self._model: Any | None = None  # Lazy loading
        self._loaded_model_name: str | None = None  # Track which model was actually loaded
        self._preload = preload
        self._normalization_strategy = normalization_strategy

        # Story 19-G3: Preload model if requested
        if preload:
            self._ensure_model()

    def _ensure_model(self) -> None:
        """Lazily load the cross-encoder model.

        Story 19-G3: If preload is True, this is called at initialization.
        Otherwise, called on first use (lazy loading).

        If the configured model fails to load and fallback_to_default is True,
        attempts to load the default model instead.

        Raises:
            ImportError: If sentence-transformers is not installed
            RuntimeError: If model fails to load and fallback is disabled or also fails
        """
        if self._model is not None:
            return

        start_time = time.perf_counter()

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderGrader. "
                "Install with: pip install sentence-transformers"
            )

        # Try loading the configured model
        try:
            logger.info(
                "loading_cross_encoder_model",
                model_name=self.model_name,
                preload=self._preload,
            )
            self._model = CrossEncoder(self.model_name)
            self._loaded_model_name = self.model_name
            load_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "cross_encoder_model_loaded",
                model_name=self.model_name,
                load_time_ms=round(load_time_ms, 2),
                preloaded=self._preload,
            )
        except Exception as e:
            logger.warning(
                "cross_encoder_model_load_failed",
                model_name=self.model_name,
                error=str(e),
            )

            # Try fallback to default model if enabled
            if self.fallback_to_default and self.model_name != DEFAULT_CROSS_ENCODER_MODEL:
                logger.info(
                    "falling_back_to_default_model",
                    failed_model=self.model_name,
                    fallback_model=DEFAULT_CROSS_ENCODER_MODEL,
                )
                try:
                    self._model = CrossEncoder(DEFAULT_CROSS_ENCODER_MODEL)
                    self._loaded_model_name = DEFAULT_CROSS_ENCODER_MODEL
                    load_time_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        "fallback_model_loaded",
                        model_name=DEFAULT_CROSS_ENCODER_MODEL,
                        load_time_ms=round(load_time_ms, 2),
                    )
                except Exception as fallback_error:
                    logger.error(
                        "fallback_model_load_failed",
                        model_name=DEFAULT_CROSS_ENCODER_MODEL,
                        error=str(fallback_error),
                    )
                    raise RuntimeError(
                        f"Failed to load cross-encoder model '{self.model_name}' "
                        f"and fallback model '{DEFAULT_CROSS_ENCODER_MODEL}': {fallback_error}"
                    ) from fallback_error
            else:
                raise RuntimeError(
                    f"Failed to load cross-encoder model '{self.model_name}': {e}"
                ) from e

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded (for health checks).

        Story 19-G3: Health checks can wait for model load completion.
        """
        return self._model is not None

    async def grade(
        self,
        query: str,
        hits: list[RetrievalHit],
        threshold: float,
        tenant_id: Optional[str] = None,
        strategy: str = "hybrid",
    ) -> GraderResult:
        """Grade using cross-encoder relevance scores."""
        start_time = time.perf_counter()

        if not hits:
            grading_time_ms = int((time.perf_counter() - start_time) * 1000)
            elapsed_seconds = grading_time_ms / 1000
            # Record metrics for empty results
            if tenant_id is not None:
                record_retrieval_latency(
                    strategy=strategy,
                    phase="grade",
                    tenant_id=tenant_id,
                    duration_seconds=elapsed_seconds,
                )
                record_grader_evaluation(result="fallback", tenant_id=tenant_id)
                record_grader_score(
                    model=self.get_model(),
                    tenant_id=tenant_id,
                    score=0.0,
                )
                record_retrieval_fallback(reason="empty_results", tenant_id=tenant_id)
            return GraderResult(
                score=0.0,
                passed=False,
                threshold=threshold,
                grading_time_ms=grading_time_ms,
                fallback_triggered=True,
            )

        self._ensure_model()
        model = self._model
        if model is None:
            raise RuntimeError("Cross-encoder model not initialized.")

        # Create query-document pairs (limit to avoid excessive compute)
        pairs = [(query, hit.content) for hit in hits[:MAX_CROSS_ENCODER_HITS]]

        # Score with cross-encoder (run in thread to avoid blocking event loop)
        scores = await asyncio.to_thread(model.predict, pairs)

        # Handle edge case of empty scores (shouldn't happen but be defensive)
        if len(scores) == 0:
            grading_time_ms = int((time.perf_counter() - start_time) * 1000)
            elapsed_seconds = grading_time_ms / 1000
            # Record metrics for empty scores
            if tenant_id is not None:
                record_retrieval_latency(
                    strategy=strategy,
                    phase="grade",
                    tenant_id=tenant_id,
                    duration_seconds=elapsed_seconds,
                )
                record_grader_evaluation(result="fallback", tenant_id=tenant_id)
                record_grader_score(
                    model=self.get_model(),
                    tenant_id=tenant_id,
                    score=0.0,
                )
                record_retrieval_fallback(reason="empty_results", tenant_id=tenant_id)
            return GraderResult(
                score=0.0,
                passed=False,
                threshold=threshold,
                grading_time_ms=grading_time_ms,
                fallback_triggered=True,
            )

        # Story 19-G4: Apply configurable normalization strategy
        raw_scores = list(scores)  # Convert numpy array to list if needed
        normalized_scores = normalize_scores(
            raw_scores,
            strategy=self._normalization_strategy,
        )
        # Aggregate to single score (mean of normalized scores)
        score = aggregate_normalized_scores(normalized_scores, aggregation="mean")

        passed = score >= threshold
        grading_time_ms = int((time.perf_counter() - start_time) * 1000)
        elapsed_seconds = grading_time_ms / 1000

        # Record metrics
        if tenant_id is not None:
            record_retrieval_latency(
                strategy=strategy,
                phase="grade",
                tenant_id=tenant_id,
                duration_seconds=elapsed_seconds,
            )
            if passed:
                record_grader_evaluation(result="pass", tenant_id=tenant_id)
            else:
                record_grader_evaluation(result="fail", tenant_id=tenant_id)
                record_retrieval_fallback(reason="low_score", tenant_id=tenant_id)
            record_grader_score(
                model=self.get_model(),
                tenant_id=tenant_id,
                score=score,
            )

        logger.debug(
            "cross_encoder_grader_result",
            score=score,
            threshold=threshold,
            passed=passed,
            num_hits=len(hits),
            grading_time_ms=grading_time_ms,
            model=self._loaded_model_name or self.model_name,
            normalization_strategy=self._normalization_strategy.value,
        )

        return GraderResult(
            score=score,
            passed=passed,
            threshold=threshold,
            grading_time_ms=grading_time_ms,
            fallback_triggered=not passed,
        )

    def get_model(self) -> str:
        """Return model identifier.

        Returns the actually loaded model name if the model has been loaded
        (which may differ from model_name if fallback was used), otherwise
        returns the configured model_name.
        """
        return self._loaded_model_name or self.model_name


class BaseFallbackHandler(ABC):
    """Abstract base class for fallback handlers."""

    @abstractmethod
    async def execute(self, query: str, tenant_id: Optional[str] = None) -> list[RetrievalHit]:
        """Execute fallback retrieval.

        Args:
            query: The original query string
            tenant_id: Tenant identifier for multi-tenancy isolation

        Returns:
            List of additional retrieval hits from fallback
        """
        pass


class WebSearchFallback(BaseFallbackHandler):
    """Web search fallback using Tavily API.

    This fallback queries the Tavily API for current web data when
    local retrieval quality is insufficient.
    """

    def __init__(self, api_key: str, max_results: int = 5):
        """Initialize the web search fallback.

        Args:
            api_key: Tavily API key
            max_results: Maximum number of results to return
        """
        self.api_key = api_key
        self.max_results = max_results
        self._client: Any | None = None  # Lazy loading

    def _ensure_client(self):
        """Lazily initialize Tavily client."""
        if self._client is None:
            try:
                from tavily import TavilyClient

                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "tavily-python is required for WebSearchFallback. "
                    "Install with: pip install tavily-python"
                )

    async def execute(self, query: str, tenant_id: Optional[str] = None) -> list[RetrievalHit]:
        """Execute web search fallback.

        Args:
            query: The original query string
            tenant_id: Tenant identifier for multi-tenancy isolation (unused for web search)

        Returns:
            List of retrieval hits from web search
        """
        self._ensure_client()
        client = self._client
        if client is None:
            raise RuntimeError("Tavily client not initialized.")

        try:
            # Wrap synchronous Tavily call in asyncio.to_thread to avoid blocking event loop
            response = await asyncio.to_thread(
                client.search,
                query=query,
                max_results=self.max_results,
                search_depth="basic",
            )

            hits = []
            for result in response.get("results", []):
                hits.append(
                    RetrievalHit(
                        content=result.get("content", result.get("snippet", "")),
                        score=result.get("score"),
                        metadata={
                            "source": "tavily_web_search",
                            "url": result.get("url"),
                            "title": result.get("title"),
                        },
                    )
                )

            logger.info(
                "web_search_fallback_executed",
                query=query[:100],
                num_results=len(hits),
            )

            return hits

        except Exception as e:
            logger.error(
                "web_search_fallback_error",
                error=str(e),
                query=query[:100],
            )
            return []


class ExpandedQueryFallback(BaseFallbackHandler):
    """Expanded query fallback that reformulates the query.

    This fallback generates query variations and retries retrieval
    with the expanded queries.
    """

    def __init__(
        self,
        retrieval_func: Optional[Callable[[str], Awaitable[list[RetrievalHit]]]] = None,
    ):
        """Initialize the expanded query fallback.

        Args:
            retrieval_func: Async function to call for retrieval with expanded query
        """
        self.retrieval_func = retrieval_func

    async def execute(self, query: str, tenant_id: Optional[str] = None) -> list[RetrievalHit]:
        """Execute expanded query fallback.

        Args:
            query: The original query string
            tenant_id: Tenant identifier for multi-tenancy isolation

        Returns:
            List of additional retrieval hits from expanded queries

        Note: This is a placeholder implementation. Full implementation
        would use an LLM to generate query variations.
        """
        logger.info(
            "expanded_query_fallback_executed",
            query=query[:100],
            tenant_id=tenant_id,
        )
        # Placeholder - return empty list
        # Full implementation would generate query variations and call retrieval_func with tenant_id
        return []


class RetrievalGrader:
    """Main grader class that orchestrates grading and fallback.

    This class combines a grader with fallback handlers to implement
    the full CRAG pattern.
    """

    def __init__(
        self,
        grader: BaseGrader,
        threshold: float = 0.5,
        fallback_enabled: bool = True,
        fallback_strategy: FallbackStrategy = FallbackStrategy.WEB_SEARCH,
        fallback_handler: Optional[BaseFallbackHandler] = None,
    ):
        """Initialize the retrieval grader.

        Args:
            grader: The base grader to use for scoring
            threshold: Score threshold for passing (0.0-1.0)
            fallback_enabled: Whether to trigger fallback on low scores
            fallback_strategy: The fallback strategy to use
            fallback_handler: Handler for executing fallback
        """
        self.grader = grader
        self.threshold = threshold
        self.fallback_enabled = fallback_enabled
        self.fallback_strategy = fallback_strategy
        self.fallback_handler = fallback_handler

    async def grade_and_fallback(
        self,
        query: str,
        hits: list[RetrievalHit],
        tenant_id: Optional[str] = None,
        strategy: str = "hybrid",
    ) -> tuple[GraderResult, list[RetrievalHit]]:
        """Grade retrieval results and execute fallback if needed.

        Args:
            query: The original query string
            hits: List of retrieval hits to grade
            tenant_id: Tenant identifier for multi-tenancy isolation
            strategy: Retrieval strategy for metrics labeling

        Returns:
            Tuple of (GraderResult, list of additional hits from fallback)
        """
        result = await self.grader.grade(
            query, hits, self.threshold, tenant_id=tenant_id, strategy=strategy
        )

        fallback_hits = []

        if not result.passed and self.fallback_enabled and self.fallback_handler:
            result.fallback_triggered = True
            result.fallback_strategy = self.fallback_strategy

            logger.info(
                "grader_triggering_fallback",
                score=result.score,
                threshold=result.threshold,
                strategy=self.fallback_strategy.value,
                tenant_id=tenant_id,
            )

            fallback_hits = await self.fallback_handler.execute(query, tenant_id)

        return result, fallback_hits

    def get_model(self) -> str:
        """Return the grader model identifier."""
        return self.grader.get_model()


def create_grader(settings: Settings) -> Optional[RetrievalGrader]:
    """Create a RetrievalGrader based on settings.

    The grader model is configurable via settings.grader_model:
    - "heuristic": Lightweight heuristic grader using retrieval scores (default)
    - Any cross-encoder model name: Uses CrossEncoderGrader with that model

    Supported cross-encoder models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good accuracy)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (higher accuracy)
    - BAAI/bge-reranker-base (BGE reranker)
    - BAAI/bge-reranker-large (BGE large, best accuracy)

    Args:
        settings: Application settings

    Returns:
        RetrievalGrader instance if grader is enabled, None otherwise
    """
    if not settings.grader_enabled:
        return None

    # Select grader based on grader_model setting
    grader_model = settings.grader_model.strip().lower()
    grader: BaseGrader

    # Story 19-G4: Get normalization strategy from settings
    try:
        normalization_strategy = get_normalization_strategy(
            settings.grader_normalization_strategy
        )
    except ValueError:
        logger.warning(
            "invalid_normalization_strategy_in_settings",
            strategy=settings.grader_normalization_strategy,
            fallback="min_max",
        )
        normalization_strategy = NormalizationStrategy.MIN_MAX

    if grader_model == "heuristic":
        # Use lightweight heuristic grader with configurable length weight
        grader = HeuristicGrader(
            top_k=5,
            length_weight=settings.grader_heuristic_length_weight,
            min_length=settings.grader_heuristic_min_length,
            max_length=settings.grader_heuristic_max_length,
        )
    else:
        # Use cross-encoder grader with specified model
        # Story 19-G3: Pass preload flag from settings
        # Story 19-G4: Pass normalization strategy
        grader = CrossEncoderGrader(
            model_name=settings.grader_model,
            fallback_to_default=True,
            preload=settings.grader_preload_model,
            normalization_strategy=normalization_strategy,
        )

    # Create fallback handler based on strategy
    fallback_handler: Optional[BaseFallbackHandler] = None
    fallback_strategy = FallbackStrategy(settings.grader_fallback_strategy)

    if settings.grader_fallback_enabled:
        if (
            fallback_strategy == FallbackStrategy.WEB_SEARCH
            and settings.tavily_api_key
        ):
            fallback_handler = WebSearchFallback(
                api_key=settings.tavily_api_key, max_results=5
            )
        elif fallback_strategy == FallbackStrategy.EXPANDED_QUERY:
            fallback_handler = ExpandedQueryFallback()
        # ALTERNATE_INDEX would require additional configuration

    # Build log context
    log_context = {
        "grader_type": type(grader).__name__,
        "grader_model": grader.get_model(),
        "threshold": settings.grader_threshold,
        "fallback_enabled": settings.grader_fallback_enabled,
        "fallback_strategy": fallback_strategy.value,
        "fallback_handler_type": type(fallback_handler).__name__
        if fallback_handler
        else None,
    }

    # Add heuristic settings if using heuristic grader
    if isinstance(grader, HeuristicGrader):
        log_context.update({
            "heuristic_length_weight": settings.grader_heuristic_length_weight,
            "heuristic_min_length": settings.grader_heuristic_min_length,
            "heuristic_max_length": settings.grader_heuristic_max_length,
        })
    # Story 19-G3/G4: Add cross-encoder settings if using cross-encoder grader
    elif isinstance(grader, CrossEncoderGrader):
        log_context.update({
            "preload_model": settings.grader_preload_model,
            "normalization_strategy": normalization_strategy.value,
        })

    logger.info("grader_created", **log_context)

    return RetrievalGrader(
        grader=grader,
        threshold=settings.grader_threshold,
        fallback_enabled=settings.grader_fallback_enabled,
        fallback_strategy=fallback_strategy,
        fallback_handler=fallback_handler,
    )
