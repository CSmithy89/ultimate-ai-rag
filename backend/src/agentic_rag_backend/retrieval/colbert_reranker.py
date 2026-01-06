"""ColBERT (Contextualized Late Interaction) reranking.

Story 20-H5: Implement ColBERT Reranking

This module provides ColBERT-style reranking using late interaction (MaxSim)
between query and document token embeddings.

ColBERT offers a middle ground between:
- Fast bi-encoder retrieval (single embedding per doc)
- Slow cross-encoder reranking (joint query-doc encoding)

By pre-computing document token embeddings and computing MaxSim at query time,
ColBERT achieves high accuracy with better efficiency than cross-encoders.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import structlog

logger = structlog.get_logger(__name__)

# Default configuration values
DEFAULT_COLBERT_ENABLED = False
DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"
DEFAULT_COLBERT_MAX_LENGTH = 512


class TokenEmbeddingsProtocol(Protocol):
    """Protocol for token embeddings."""

    def encode_tokens(self, text: str) -> list[list[float]]:
        """Encode text to token embeddings."""
        ...


@dataclass
class ColBERTResult:
    """Result of ColBERT reranking.

    Attributes:
        doc_id: Document identifier
        content: Document content
        score: MaxSim score
        original_score: Original retrieval score
        original_rank: Original rank before reranking
        metadata: Additional metadata
    """

    doc_id: str
    content: str
    score: float
    original_score: float = 0.0
    original_rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenEmbeddings:
    """Token embeddings for a document or query.

    Attributes:
        tokens: List of tokens
        embeddings: 2D array of token embeddings [num_tokens, embedding_dim]
    """

    tokens: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        """Return number of tokens."""
        return len(self.embeddings)


class ColBERTEncoder:
    """Encoder for ColBERT token embeddings.

    Encodes text into token-level embeddings using a transformer model.
    Unlike sentence embeddings, this preserves individual token representations
    for late interaction scoring.

    Example:
        encoder = ColBERTEncoder(model_name="colbert-ir/colbertv2.0")
        query_emb = await encoder.encode_query("What is machine learning?")
        doc_emb = await encoder.encode_document("Machine learning is...")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_COLBERT_MODEL,
        max_length: int = DEFAULT_COLBERT_MAX_LENGTH,
    ) -> None:
        """Initialize ColBERT encoder.

        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
        """
        self._model_name = model_name
        self._max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None

        self._logger = logger.bind(
            component="ColBERTEncoder",
            model=model_name,
        )

    def _ensure_model(self) -> None:
        """Lazily load the model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)

            self._logger.info(
                "colbert_model_loaded",
                model=self._model_name,
            )
        except ImportError as e:
            # Fall back to sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
                self._tokenizer = None  # Use model's tokenizer

                self._logger.info(
                    "colbert_fallback_to_sentence_transformers",
                    model=self._model_name,
                )
            except ImportError as e2:
                raise ImportError(
                    "transformers or sentence-transformers is required for ColBERT. "
                    "Install with: pip install transformers or pip install sentence-transformers"
                ) from e2

    async def encode_query(self, query: str) -> TokenEmbeddings:
        """Encode query to token embeddings.

        Args:
            query: Query text

        Returns:
            TokenEmbeddings with query token representations
        """
        self._ensure_model()

        loop = asyncio.get_event_loop()

        def _encode() -> TokenEmbeddings:
            return self._encode_text(query, is_query=True)

        return await loop.run_in_executor(None, _encode)

    async def encode_document(self, document: str) -> TokenEmbeddings:
        """Encode document to token embeddings.

        Args:
            document: Document text

        Returns:
            TokenEmbeddings with document token representations
        """
        self._ensure_model()

        loop = asyncio.get_event_loop()

        def _encode() -> TokenEmbeddings:
            return self._encode_text(document, is_query=False)

        return await loop.run_in_executor(None, _encode)

    async def encode_documents_batch(
        self,
        documents: list[str],
    ) -> list[TokenEmbeddings]:
        """Encode multiple documents.

        Args:
            documents: List of document texts

        Returns:
            List of TokenEmbeddings
        """
        if not documents:
            return []

        self._ensure_model()

        loop = asyncio.get_event_loop()

        def _encode_batch() -> list[TokenEmbeddings]:
            return [self._encode_text(doc, is_query=False) for doc in documents]

        return await loop.run_in_executor(None, _encode_batch)

    def _encode_text(self, text: str, is_query: bool = False) -> TokenEmbeddings:
        """Encode text to token embeddings.

        Args:
            text: Text to encode
            is_query: Whether this is a query (for special tokens)

        Returns:
            TokenEmbeddings
        """
        import torch

        try:
            if self._tokenizer is not None:
                # Using transformers directly
                encoded = self._tokenizer(
                    text,
                    max_length=self._max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )

                with torch.no_grad():
                    outputs = self._model(**encoded)

                # Get token embeddings (last hidden state)
                embeddings = outputs.last_hidden_state[0].cpu().numpy()

                tokens = self._tokenizer.convert_ids_to_tokens(
                    encoded["input_ids"][0].tolist()
                )

                return TokenEmbeddings(
                    tokens=tokens,
                    embeddings=embeddings.tolist(),
                )
            else:
                # Using sentence-transformers
                # Encode with output_value="token_embeddings"
                embeddings = self._model.encode(
                    text,
                    convert_to_numpy=True,
                    output_value="token_embeddings",
                )

                # If we get a single embedding, wrap it
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.reshape(1, -1)

                return TokenEmbeddings(
                    tokens=[],  # sentence-transformers doesn't expose tokens easily
                    embeddings=embeddings.tolist(),
                )

        except Exception as e:
            self._logger.error(
                "colbert_encode_failed",
                error=str(e),
            )
            raise


class MaxSimScorer:
    """MaxSim scorer for ColBERT late interaction.

    Computes the MaxSim score between query and document token embeddings:
    score = sum(max(cos_sim(q_i, d_j)) for all query tokens i over all doc tokens j)

    This captures the best-matching document token for each query token,
    enabling fine-grained relevance matching.
    """

    @staticmethod
    def compute_score(
        query_embeddings: TokenEmbeddings,
        doc_embeddings: TokenEmbeddings,
    ) -> float:
        """Compute MaxSim score between query and document.

        Args:
            query_embeddings: Query token embeddings
            doc_embeddings: Document token embeddings

        Returns:
            MaxSim score (higher is more relevant)
        """
        import numpy as np

        if not query_embeddings.embeddings or not doc_embeddings.embeddings:
            return 0.0

        q_emb = np.array(query_embeddings.embeddings)
        d_emb = np.array(doc_embeddings.embeddings)

        # Normalize for cosine similarity
        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
        d_norm = d_emb / (np.linalg.norm(d_emb, axis=1, keepdims=True) + 1e-8)

        # Compute similarity matrix [num_query_tokens, num_doc_tokens]
        sim_matrix = np.dot(q_norm, d_norm.T)

        # MaxSim: for each query token, take max similarity over all doc tokens
        max_sims = np.max(sim_matrix, axis=1)

        # Sum (or could use mean) of max similarities
        score = float(np.sum(max_sims))

        return score

    @staticmethod
    def compute_scores_batch(
        query_embeddings: TokenEmbeddings,
        doc_embeddings_list: list[TokenEmbeddings],
    ) -> list[float]:
        """Compute MaxSim scores for multiple documents.

        Args:
            query_embeddings: Query token embeddings
            doc_embeddings_list: List of document token embeddings

        Returns:
            List of MaxSim scores
        """
        return [
            MaxSimScorer.compute_score(query_embeddings, doc_emb)
            for doc_emb in doc_embeddings_list
        ]


class ColBERTReranker:
    """ColBERT reranker with feature flag support.

    Reranks retrieval results using ColBERT late interaction scoring.
    When disabled, passes through results unchanged.

    Example:
        reranker = ColBERTReranker(enabled=True)
        results = await reranker.rerank(
            query="What is AI?",
            documents=[{"id": "1", "content": "AI is..."}],
        )
    """

    def __init__(
        self,
        enabled: bool = DEFAULT_COLBERT_ENABLED,
        model_name: str = DEFAULT_COLBERT_MODEL,
        max_length: int = DEFAULT_COLBERT_MAX_LENGTH,
        top_k: int = 10,
    ) -> None:
        """Initialize ColBERT reranker.

        Args:
            enabled: Whether ColBERT reranking is enabled
            model_name: ColBERT model name
            max_length: Maximum sequence length
            top_k: Number of top results to return
        """
        self._enabled = enabled
        self._model_name = model_name
        self._max_length = max_length
        self._top_k = top_k
        self._encoder: Optional[ColBERTEncoder] = None
        self._scorer = MaxSimScorer()

        self._logger = logger.bind(component="ColBERTReranker")

        if enabled:
            self._logger.info(
                "colbert_reranker_enabled",
                model=model_name,
                max_length=max_length,
            )
        else:
            self._logger.info("colbert_reranker_disabled")

    @property
    def enabled(self) -> bool:
        """Check if ColBERT reranking is enabled."""
        return self._enabled

    def _ensure_encoder(self) -> ColBERTEncoder:
        """Get or create encoder."""
        if self._encoder is None:
            self._encoder = ColBERTEncoder(
                model_name=self._model_name,
                max_length=self._max_length,
            )
        return self._encoder

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> list[ColBERTResult]:
        """Rerank documents using ColBERT MaxSim.

        Args:
            query: Query text
            documents: List of documents with 'id' and 'content' keys
            top_k: Number of top results to return

        Returns:
            List of ColBERTResult sorted by score
        """
        if not self._enabled:
            # Pass through without reranking
            return [
                ColBERTResult(
                    doc_id=doc.get("id", str(i)),
                    content=doc.get("content", ""),
                    score=doc.get("score", 1.0),
                    original_score=doc.get("score", 1.0),
                    original_rank=i,
                    metadata=doc.get("metadata", {}),
                )
                for i, doc in enumerate(documents)
            ]

        if not documents:
            return []

        top_k = top_k or self._top_k

        try:
            encoder = self._ensure_encoder()

            # Encode query
            query_embeddings = await encoder.encode_query(query)

            # Encode all documents
            doc_contents = [doc.get("content", "") for doc in documents]
            doc_embeddings_list = await encoder.encode_documents_batch(doc_contents)

            # Compute MaxSim scores
            scores = self._scorer.compute_scores_batch(
                query_embeddings, doc_embeddings_list
            )

            # Create results with scores
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                results.append(
                    ColBERTResult(
                        doc_id=doc.get("id", str(i)),
                        content=doc.get("content", ""),
                        score=score,
                        original_score=doc.get("score", 0.0),
                        original_rank=i,
                        metadata=doc.get("metadata", {}),
                    )
                )

            # Sort by score descending
            results.sort(key=lambda r: r.score, reverse=True)

            # Return top_k
            return results[:top_k]

        except Exception as e:
            self._logger.error(
                "colbert_rerank_failed",
                query=query[:50],
                num_docs=len(documents),
                error=str(e),
            )
            # Fall back to original order
            return [
                ColBERTResult(
                    doc_id=doc.get("id", str(i)),
                    content=doc.get("content", ""),
                    score=doc.get("score", 0.0),
                    original_score=doc.get("score", 0.0),
                    original_rank=i,
                    metadata=doc.get("metadata", {}),
                )
                for i, doc in enumerate(documents)
            ]


def create_colbert_reranker(
    enabled: bool = DEFAULT_COLBERT_ENABLED,
    model_name: str = DEFAULT_COLBERT_MODEL,
    max_length: int = DEFAULT_COLBERT_MAX_LENGTH,
    top_k: int = 10,
) -> ColBERTReranker:
    """Factory function to create a ColBERT reranker.

    Args:
        enabled: Whether to enable ColBERT
        model_name: Model name
        max_length: Max sequence length
        top_k: Number of results to return

    Returns:
        Configured ColBERTReranker
    """
    return ColBERTReranker(
        enabled=enabled,
        model_name=model_name,
        max_length=max_length,
        top_k=top_k,
    )
