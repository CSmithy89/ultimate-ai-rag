"""OpenAI-compatible embedding generation with batch processing and retry logic."""

from typing import Optional
from uuid import UUID

import structlog
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agentic_rag_backend.core.errors import EmbeddingError
from agentic_rag_backend.ops import CostTracker

logger = structlog.get_logger(__name__)

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536

# Batch processing limits
MAX_BATCH_SIZE = 100  # OpenAI limit
MAX_TOKENS_PER_REQUEST = 8191  # Model token limit


class EmbeddingGenerator:
    """
    OpenAI-compatible embedding generation with batch processing.

    Features:
    - Batch processing for efficiency (up to 100 texts per API call)
    - Exponential backoff retry for rate limits
    - Automatic text truncation for long inputs
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        """
        Initialize embedding generator.

        Args:
            api_key: OpenAI-compatible API key (optional for local providers)
            model: Embedding model ID (default: text-embedding-ada-002)
            base_url: OpenAI-compatible base URL override
            timeout: Request timeout in seconds (default: 30.0)
        """
        client_kwargs: dict[str, object] = {
            "api_key": api_key or "",
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model
        self._cost_tracker = cost_tracker
        logger.info("embedding_generator_initialized", model=model, timeout=timeout)

    async def _record_usage(
        self,
        tenant_id: Optional[str],
        texts: list[str],
        trajectory_id: Optional[UUID],
    ) -> None:
        if not self._cost_tracker or not tenant_id or not texts:
            return
        try:
            prompt = "\n".join(texts)
            await self._cost_tracker.record_usage(
                tenant_id=tenant_id,
                model_id=self.model,
                prompt=prompt,
                completion="",
                trajectory_id=trajectory_id,
            )
        except Exception as exc:  # pragma: no cover - non-critical telemetry
            logger.warning("embedding_cost_tracking_failed", error=str(exc))

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            "embedding_retry",
            attempt=retry_state.attempt_number,
            error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
        ),
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts with retry logic.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding fails after retries
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(str(e), batch_size=len(texts)) from e

    async def generate_embeddings(
        self,
        texts: list[str],
        batch_size: int = MAX_BATCH_SIZE,
        tenant_id: Optional[str] = None,
        trajectory_id: Optional[UUID] = None,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with batching.

        Processes texts in batches of up to `batch_size` for efficiency.
        Empty texts are handled by returning zero vectors.

        Args:
            texts: List of text strings to embed
            batch_size: Maximum texts per API call (default: 100)

        Returns:
            List of embedding vectors (1536 dimensions each)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        # Pre-process texts: handle empty strings and truncate long texts
        processed_texts = []
        empty_indices = set()

        for i, text in enumerate(texts):
            if not text or not text.strip():
                empty_indices.add(i)
                processed_texts.append(" ")  # Use space as placeholder
            else:
                # Truncate very long texts (rough character limit based on token estimates)
                # Using 4 chars per token as rough estimate
                max_chars = MAX_TOKENS_PER_REQUEST * 4
                if len(text) > max_chars:
                    text = text[:max_chars]
                    logger.warning(
                        "text_truncated",
                        original_length=len(texts[i]),
                        truncated_to=max_chars,
                    )
                processed_texts.append(text)

        embeddings = []
        total_batches = (len(processed_texts) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(processed_texts), batch_size):
            batch = processed_texts[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            logger.debug(
                "embedding_batch",
                batch=batch_num,
                total_batches=total_batches,
                batch_size=len(batch),
            )

            batch_embeddings = await self._embed_batch(batch)
            await self._record_usage(tenant_id, batch, trajectory_id)
            embeddings.extend(batch_embeddings)

        # Replace embeddings for empty texts with zero vectors
        for i in empty_indices:
            embeddings[i] = [0.0] * EMBEDDING_DIMENSION

        logger.info(
            "embeddings_generated",
            total=len(texts),
            batches=total_batches,
        )

        return embeddings

    async def generate_embedding(
        self,
        text: str,
        tenant_id: Optional[str] = None,
        trajectory_id: Optional[UUID] = None,
    ) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector (1536 dimensions)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        embeddings = await self.generate_embeddings(
            [text],
            tenant_id=tenant_id,
            trajectory_id=trajectory_id,
        )
        return embeddings[0]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score (0.0-1.0)
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)
