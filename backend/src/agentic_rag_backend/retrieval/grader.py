"""Corrective RAG Grader Agent for evaluating retrieval quality.

This module implements the CRAG (Corrective Retrieval Augmented Generation) pattern,
which evaluates retrieval results and triggers fallback strategies when quality is low.

The grader uses a lightweight approach (cross-encoder or simple heuristics) rather
than full LLM calls to minimize latency and cost.
"""

import asyncio
import math
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

logger = structlog.get_logger(__name__)

# Grader configuration constants
MAX_CROSS_ENCODER_HITS = 10  # Maximum hits to evaluate with cross-encoder
CONTENT_LENGTH_NORMALIZATION = 1000  # Characters for content length heuristic


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
    """

    def __init__(self, top_k: int = 5):
        """Initialize the heuristic grader.

        Args:
            top_k: Number of top hits to consider for scoring
        """
        self.top_k = top_k

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

        # Calculate average score from retrieval scores
        scores = [h.score for h in top_hits if h.score is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            # Normalize to 0-1 range (assuming scores might be higher)
            score = min(1.0, max(0.0, avg_score))
        else:
            # If no scores available, use content length heuristic
            # (longer content = potentially more relevant)
            # NOTE: This is a simple heuristic that may not hold for all use cases
            avg_length = sum(len(h.content) for h in top_hits) / len(top_hits)
            score = min(1.0, avg_length / CONTENT_LENGTH_NORMALIZATION)

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
            "heuristic_grader_result",
            score=score,
            threshold=threshold,
            passed=passed,
            num_hits=len(hits),
            top_k=self.top_k,
            grading_time_ms=grading_time_ms,
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

    Note: This requires sentence-transformers to be available.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        fallback_to_default: bool = True,
    ):
        """Initialize the cross-encoder grader.

        Args:
            model_name: Name of the cross-encoder model to use
            fallback_to_default: If True, fall back to default model if configured model fails
        """
        self.model_name = model_name
        self.fallback_to_default = fallback_to_default
        self._model: Any | None = None  # Lazy loading
        self._loaded_model_name: str | None = None  # Track which model was actually loaded

    def _ensure_model(self) -> None:
        """Lazily load the cross-encoder model.

        If the configured model fails to load and fallback_to_default is True,
        attempts to load the default model instead.

        Raises:
            ImportError: If sentence-transformers is not installed
            RuntimeError: If model fails to load and fallback is disabled or also fails
        """
        if self._model is not None:
            return

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
            )
            self._model = CrossEncoder(self.model_name)
            self._loaded_model_name = self.model_name
            logger.info(
                "cross_encoder_model_loaded",
                model_name=self.model_name,
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
                    logger.info(
                        "fallback_model_loaded",
                        model_name=DEFAULT_CROSS_ENCODER_MODEL,
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

        # Average score (cross-encoder scores are typically -inf to +inf, normalize)
        avg_score = sum(scores) / len(scores)
        # Sigmoid normalization to 0-1 range
        score = 1 / (1 + math.exp(-avg_score))

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

    if grader_model == "heuristic":
        # Use lightweight heuristic grader
        grader = HeuristicGrader(top_k=5)
    else:
        # Use cross-encoder grader with specified model
        # The CrossEncoderGrader handles fallback to default if model fails
        grader = CrossEncoderGrader(
            model_name=settings.grader_model,
            fallback_to_default=True,
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

    logger.info(
        "grader_created",
        grader_type=type(grader).__name__,
        grader_model=grader.get_model(),
        threshold=settings.grader_threshold,
        fallback_enabled=settings.grader_fallback_enabled,
        fallback_strategy=fallback_strategy.value,
        fallback_handler_type=type(fallback_handler).__name__
        if fallback_handler
        else None,
    )

    return RetrievalGrader(
        grader=grader,
        threshold=settings.grader_threshold,
        fallback_enabled=settings.grader_fallback_enabled,
        fallback_strategy=fallback_strategy,
        fallback_handler=fallback_handler,
    )
