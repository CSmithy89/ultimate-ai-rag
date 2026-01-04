"""Multi-provider embedding generation with batch processing and retry logic.

Supports:
- OpenAI (text-embedding-ada-002, text-embedding-3-small/large)
- OpenRouter (via OpenAI-compatible API)
- Ollama (via OpenAI-compatible API)
- Google Gemini (text-embedding-004)
- Voyage AI (voyage-3, voyage-3-lite, voyage-code-3)
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, cast
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
from agentic_rag_backend.llm.providers import EmbeddingProviderAdapter, EmbeddingProviderType
from agentic_rag_backend.ops import CostTracker

logger = structlog.get_logger(__name__)

# Default embedding configuration
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_EMBEDDING_DIMENSION = 1536
# Backward compatibility alias
EMBEDDING_DIMENSION = DEFAULT_EMBEDDING_DIMENSION

# Provider-specific dimensions
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-004": 768,  # Gemini
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
}

# Batch processing limits
MAX_BATCH_SIZE = 100  # OpenAI limit
MAX_TOKENS_PER_REQUEST = 8191  # Model token limit


class EmbeddingClient(ABC):
    """Abstract base class for embedding clients."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension for this model."""
        pass


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI-compatible embedding client (works with OpenAI, OpenRouter, Ollama)."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        if base_url:
            self.client = AsyncOpenAI(
                api_key=api_key or "",
                timeout=timeout,
                base_url=base_url,
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key or "",
                timeout=timeout,
            )
        self.model = model
        logger.info("openai_embedding_client_initialized", model=model)

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
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch using OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(str(e), batch_size=len(texts)) from e

    def get_dimension(self) -> int:
        return EMBEDDING_DIMENSIONS.get(self.model, DEFAULT_EMBEDDING_DIMENSION)


class GeminiEmbeddingClient(EmbeddingClient):
    """Google Gemini embedding client."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "text-embedding-004",
    ) -> None:
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            raise EmbeddingError(
                "google-genai package required for Gemini embeddings. "
                "Install with: uv add google-genai"
            )
        self.model = model
        logger.info("gemini_embedding_client_initialized", model=model)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            "gemini_embedding_retry",
            attempt=retry_state.attempt_number,
            error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
        ),
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch using Gemini API.

        Uses asyncio.to_thread() to avoid blocking the event loop since
        the Google Genai SDK uses synchronous HTTP calls.
        """
        try:
            # Run synchronous Gemini API calls in thread pool to avoid blocking
            embeddings = []
            for text in texts:
                result = await asyncio.to_thread(
                    self._client.models.embed_content,
                    model=self.model,
                    contents=text,
                )
                embedding_payload = getattr(result, "embedding", None)
                if embedding_payload is None:
                    embedding_payload = getattr(result, "embeddings", None)
                if embedding_payload is None:
                    raise EmbeddingError("Gemini embedding response missing embedding data.", batch_size=1)
                vector = embedding_payload.values if hasattr(embedding_payload, "values") else embedding_payload
                embeddings.append([float(x) for x in vector])
            return embeddings
        except Exception as e:
            raise EmbeddingError(str(e), batch_size=len(texts)) from e

    def get_dimension(self) -> int:
        return EMBEDDING_DIMENSIONS.get(self.model, 768)


class VoyageEmbeddingClient(EmbeddingClient):
    """Voyage AI embedding client."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "voyage-3",
    ) -> None:
        try:
            import voyageai
            self._client = voyageai.AsyncClient(api_key=api_key)
        except ImportError:
            raise EmbeddingError(
                "voyageai package required for Voyage embeddings. "
                "Install with: uv add voyageai"
            )
        self.model = model
        logger.info("voyage_embedding_client_initialized", model=model)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            "voyage_embedding_retry",
            attempt=retry_state.attempt_number,
            error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
        ),
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch using Voyage AI API."""
        try:
            result = await self._client.embed(
                texts=texts,
                model=self.model,
            )
            return [[float(x) for x in embedding] for embedding in result.embeddings]
        except Exception as e:
            raise EmbeddingError(str(e), batch_size=len(texts)) from e

    def get_dimension(self) -> int:
        return EMBEDDING_DIMENSIONS.get(self.model, 1024)


def create_embedding_client(
    adapter: EmbeddingProviderAdapter,
    timeout: float = 30.0,
) -> EmbeddingClient:
    """Factory function to create the appropriate embedding client.

    Args:
        adapter: Embedding provider adapter with config
        timeout: Request timeout in seconds

    Returns:
        Configured EmbeddingClient instance
    """
    if adapter.provider in (
        EmbeddingProviderType.OPENAI,
        EmbeddingProviderType.OPENROUTER,
        EmbeddingProviderType.OLLAMA,
    ):
        return OpenAIEmbeddingClient(
            api_key=adapter.api_key,
            model=adapter.model,
            base_url=adapter.base_url,
            timeout=timeout,
        )
    elif adapter.provider == EmbeddingProviderType.GEMINI:
        return GeminiEmbeddingClient(
            api_key=adapter.api_key,
            model=adapter.model,
        )
    elif adapter.provider == EmbeddingProviderType.VOYAGE:
        return VoyageEmbeddingClient(
            api_key=adapter.api_key,
            model=adapter.model,
        )
    else:
        raise EmbeddingError(f"Unsupported embedding provider: {adapter.provider}")


class EmbeddingGenerator:
    """
    Multi-provider embedding generation with batch processing.

    Features:
    - Batch processing for efficiency (up to 100 texts per API call)
    - Exponential backoff retry for rate limits
    - Automatic text truncation for long inputs
    - Support for OpenAI, Gemini, Voyage, and OpenAI-compatible providers
    """

    def __init__(
        self,
        client: EmbeddingClient,
        cost_tracker: Optional[CostTracker] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialize embedding generator with a provider client.

        Args:
            client: Embedding client for the provider
            cost_tracker: Optional cost tracker for usage monitoring
            model: Model name for cost tracking (optional, inferred from client)
        """
        self._client = client
        self._cost_tracker = cost_tracker
        inferred_model = getattr(client, "model", DEFAULT_EMBEDDING_MODEL)
        self._model: str = model or cast(str, inferred_model)
        self._dimension = client.get_dimension()
        logger.info(
            "embedding_generator_initialized",
            model=self._model,
            dimension=self._dimension,
        )

    @classmethod
    def from_adapter(
        cls,
        adapter: EmbeddingProviderAdapter,
        cost_tracker: Optional[CostTracker] = None,
        timeout: float = 30.0,
    ) -> "EmbeddingGenerator":
        """Create EmbeddingGenerator from a provider adapter.

        Args:
            adapter: Embedding provider adapter with config
            cost_tracker: Optional cost tracker
            timeout: Request timeout in seconds

        Returns:
            Configured EmbeddingGenerator instance
        """
        client = create_embedding_client(adapter, timeout=timeout)
        return cls(client=client, cost_tracker=cost_tracker, model=adapter.model)

    @classmethod
    def from_openai_config(
        cls,
        api_key: Optional[str],
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        cost_tracker: Optional[CostTracker] = None,
    ) -> "EmbeddingGenerator":
        """Create EmbeddingGenerator with OpenAI-compatible config (backward compatible).

        Args:
            api_key: OpenAI-compatible API key
            model: Embedding model ID
            base_url: OpenAI-compatible base URL override
            timeout: Request timeout in seconds
            cost_tracker: Optional cost tracker

        Returns:
            Configured EmbeddingGenerator instance
        """
        client = OpenAIEmbeddingClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
        return cls(client=client, cost_tracker=cost_tracker, model=model)

    @property
    def model(self) -> str:
        """Get the embedding model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

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
                model_id=self._model,
                prompt=prompt,
                completion="",
                trajectory_id=trajectory_id,
            )
        except Exception as exc:  # pragma: no cover - non-critical telemetry
            logger.warning("embedding_cost_tracking_failed", error=str(exc))

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
            tenant_id: Optional tenant ID for cost tracking
            trajectory_id: Optional trajectory ID for cost tracking

        Returns:
            List of embedding vectors

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
                original_length = len(text)
                if original_length > max_chars:
                    text = text[:max_chars]
                    logger.warning(
                        "text_truncated",
                        original_length=original_length,
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

            batch_embeddings = await self._client.embed_batch(batch)
            await self._record_usage(tenant_id, batch, trajectory_id)
            embeddings.extend(batch_embeddings)

        # Replace embeddings for empty texts with zero vectors
        for i in empty_indices:
            embeddings[i] = [0.0] * self._dimension

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
            tenant_id: Optional tenant ID for cost tracking
            trajectory_id: Optional trajectory ID for cost tracking

        Returns:
            Embedding vector

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
