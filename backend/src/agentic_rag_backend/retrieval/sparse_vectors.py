"""Sparse vector search with BM42 encoding.

Story 20-H1: Implement Sparse Vector Search (BM42)

This module provides sparse vector search capabilities using BM42 encoding
for improved lexical matching alongside dense vectors.

Components:
- SparseVector: Sparse vector representation (indices + values)
- BM42Encoder: BM42 sparse vector encoder using fastembed
- HybridVectorSearch: Combined dense and sparse search with RRF fusion
- SparseVectorAdapter: Feature flag wrapper for sparse vector features
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import structlog

logger = structlog.get_logger(__name__)

# Default configuration values
DEFAULT_SPARSE_VECTORS_ENABLED = False
DEFAULT_SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
DEFAULT_HYBRID_DENSE_WEIGHT = 0.7
DEFAULT_HYBRID_SPARSE_WEIGHT = 0.3
DEFAULT_RRF_K = 60  # Standard RRF constant


@dataclass
class SparseVector:
    """A sparse vector representation.

    Sparse vectors store only non-zero values with their indices,
    making them efficient for high-dimensional but sparse data
    like term-weighted vectors.

    Attributes:
        indices: List of non-zero positions in the vector
        values: List of corresponding weights for each position

    Example:
        vector = SparseVector(
            indices=[10, 42, 100],
            values=[0.8, 0.3, 0.5],
        )
        # Represents a vector with non-zero values at positions 10, 42, 100
    """

    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate that indices and values have matching lengths."""
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"Indices and values must have same length: "
                f"{len(self.indices)} != {len(self.values)}"
            )

    def to_dict(self) -> dict[str, list]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with 'indices' and 'values' keys
        """
        return {"indices": self.indices, "values": self.values}

    @classmethod
    def from_dict(cls, data: dict[str, list]) -> "SparseVector":
        """Create from dictionary representation.

        Args:
            data: Dictionary with 'indices' and 'values' keys

        Returns:
            SparseVector instance
        """
        return cls(
            indices=data.get("indices", []),
            values=data.get("values", []),
        )

    def dot_product(self, other: "SparseVector") -> float:
        """Calculate dot product with another sparse vector.

        Uses index intersection for efficient computation.

        Args:
            other: Another sparse vector

        Returns:
            Dot product (similarity score)
        """
        # Build index -> value map for the smaller vector
        if len(self.indices) <= len(other.indices):
            smaller, larger = self, other
        else:
            smaller, larger = other, self

        smaller_map = dict(zip(smaller.indices, smaller.values))
        larger_map = dict(zip(larger.indices, larger.values))

        # Calculate dot product for overlapping indices
        result = 0.0
        for idx, val in smaller_map.items():
            if idx in larger_map:
                result += val * larger_map[idx]

        return result

    @property
    def is_empty(self) -> bool:
        """Check if the vector is empty."""
        return len(self.indices) == 0


class DenseSearchProtocol(Protocol):
    """Protocol for dense vector search implementations."""

    async def search(
        self,
        query: str,
        tenant_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform dense vector search."""
        ...


class BM42Encoder:
    """BM42 sparse vector encoder using fastembed.

    BM42 uses attention-based term weighting, providing better
    sparse representations than traditional BM25.

    Example:
        encoder = BM42Encoder()
        sparse = encoder.encode_query("what is machine learning?")
        print(f"Non-zero terms: {len(sparse.indices)}")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SPARSE_MODEL,
    ) -> None:
        """Initialize the BM42 encoder.

        Args:
            model_name: Name of the sparse embedding model to use.
                Default is the Qdrant BM42 model.
        """
        self._model_name = model_name
        self._model: Any = None  # Lazy initialization

    def _ensure_model(self) -> None:
        """Lazily initialize the model on first use."""
        if self._model is not None:
            return

        try:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(model_name=self._model_name)
            logger.info(
                "bm42_encoder_initialized",
                model_name=self._model_name,
            )
        except ImportError as e:
            raise ImportError(
                "fastembed is required for BM42 encoding. "
                "Install with: pip install fastembed"
            ) from e

    def encode(self, texts: list[str]) -> list[SparseVector]:
        """Encode texts to sparse vectors.

        Args:
            texts: List of texts to encode

        Returns:
            List of SparseVector instances
        """
        if not texts:
            return []

        self._ensure_model()

        try:
            embeddings = list(self._model.embed(texts))

            return [
                SparseVector(
                    indices=list(emb.indices),
                    values=list(emb.values),
                )
                for emb in embeddings
            ]
        except Exception as e:
            logger.error(
                "bm42_encode_failed",
                num_texts=len(texts),
                error=str(e),
            )
            raise

    def encode_query(self, query: str) -> SparseVector:
        """Encode a single query to sparse vector.

        Args:
            query: Query text to encode

        Returns:
            SparseVector instance
        """
        results = self.encode([query])
        return results[0] if results else SparseVector()

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name


@dataclass
class HybridSearchResult:
    """Result from hybrid search with scores from both sources.

    Attributes:
        id: Document identifier
        data: Full result data from the source
        dense_score: Score from dense search (0 if not found)
        sparse_score: Score from sparse search (0 if not found)
        combined_score: RRF combined score
    """

    id: str
    data: dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0


class HybridVectorSearch:
    """Combine dense and sparse vector search using RRF fusion.

    Reciprocal Rank Fusion (RRF) combines rankings from multiple
    search systems without needing score normalization.

    Formula: score(d) = Σ 1/(k + rank_i(d)) for each system i

    Example:
        hybrid = HybridVectorSearch(
            dense_search=dense_search,
            sparse_encoder=BM42Encoder(),
            sparse_index=sparse_index,
            dense_weight=0.7,
            sparse_weight=0.3,
        )
        results = await hybrid.search("machine learning", "tenant-1")
    """

    def __init__(
        self,
        dense_search: DenseSearchProtocol,
        sparse_encoder: BM42Encoder,
        sparse_index: Optional[Any] = None,
        dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_HYBRID_SPARSE_WEIGHT,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        """Initialize hybrid search.

        Args:
            dense_search: Dense vector search implementation
            sparse_encoder: BM42 encoder for sparse vectors
            sparse_index: Sparse vector index (or None for in-memory)
            dense_weight: Weight for dense search in RRF (default 0.7)
            sparse_weight: Weight for sparse search in RRF (default 0.3)
            rrf_k: RRF constant k (default 60, per original paper)
        """
        self._dense = dense_search
        self._sparse_encoder = sparse_encoder
        self._sparse_index = sparse_index
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self._rrf_k = rrf_k

        # In-memory sparse vector storage (for development/testing)
        self._sparse_vectors: dict[str, SparseVector] = {}

    async def search(
        self,
        query: str,
        tenant_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search combining dense and sparse results.

        Args:
            query: Search query
            tenant_id: Tenant identifier for isolation
            limit: Maximum number of results to return

        Returns:
            List of result dictionaries with combined scores
        """
        # Fetch more results from each source for better fusion
        fetch_limit = limit * 2

        # Dense search
        try:
            dense_results = await self._dense.search(
                query, tenant_id, limit=fetch_limit
            )
        except Exception as e:
            logger.warning(
                "dense_search_failed",
                query=query[:50],
                error=str(e),
            )
            dense_results = []

        # Sparse search
        try:
            query_vector = self._sparse_encoder.encode_query(query)
            sparse_results = await self._sparse_search(
                query_vector, tenant_id, limit=fetch_limit
            )
        except Exception as e:
            logger.warning(
                "sparse_search_failed",
                query=query[:50],
                error=str(e),
            )
            sparse_results = []

        # Combine with RRF
        combined = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
        )

        logger.debug(
            "hybrid_search_complete",
            query=query[:50],
            tenant_id=tenant_id,
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            combined_count=len(combined),
        )

        return combined[:limit]

    async def _sparse_search(
        self,
        query_vector: SparseVector,
        tenant_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform sparse vector similarity search.

        Args:
            query_vector: Query sparse vector
            tenant_id: Tenant identifier for isolation
            limit: Maximum results to return

        Returns:
            List of results with similarity scores
        """
        if query_vector.is_empty:
            return []

        # Use external index if available
        if self._sparse_index is not None:
            return await self._sparse_index.search(
                query_vector, tenant_id, limit=limit
            )

        # Fall back to in-memory search
        results: list[tuple[str, float, dict]] = []

        for doc_id, doc_vector in self._sparse_vectors.items():
            # Check tenant (doc_id format: tenant_id:doc_id)
            if not doc_id.startswith(f"{tenant_id}:"):
                continue

            score = query_vector.dot_product(doc_vector)
            if score > 0:
                results.append((doc_id, score, {"id": doc_id.split(":", 1)[1]}))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return [
            {"id": r[2]["id"], "score": r[1], **r[2]}
            for r in results[:limit]
        ]

    def _reciprocal_rank_fusion(
        self,
        dense: list[dict[str, Any]],
        sparse: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Combine results using Reciprocal Rank Fusion.

        RRF is a rank-based fusion method that doesn't require
        score normalization. Each result gets a score based on
        its rank position, weighted by the source weight.

        Args:
            dense: Dense search results (ordered by relevance)
            sparse: Sparse search results (ordered by relevance)

        Returns:
            Combined results sorted by RRF score
        """
        scores: dict[str, float] = {}
        data: dict[str, dict[str, Any]] = {}

        # Score dense results
        for i, result in enumerate(dense):
            doc_id = str(result.get("id", result.get("doc_id", "")))
            if not doc_id:
                continue

            rrf_score = self.dense_weight / (self._rrf_k + i + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in data:
                data[doc_id] = result

        # Score sparse results
        for i, result in enumerate(sparse):
            doc_id = str(result.get("id", result.get("doc_id", "")))
            if not doc_id:
                continue

            rrf_score = self.sparse_weight / (self._rrf_k + i + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in data:
                data[doc_id] = result

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build result list with combined scores
        results = []
        for doc_id in sorted_ids:
            result = data[doc_id].copy()
            result["rrf_score"] = scores[doc_id]
            results.append(result)

        return results

    def index_document(
        self,
        doc_id: str,
        text: str,
        tenant_id: str,
    ) -> SparseVector:
        """Index a document for sparse search.

        Args:
            doc_id: Document identifier
            text: Document text to encode
            tenant_id: Tenant identifier

        Returns:
            The generated sparse vector
        """
        sparse_vector = self._sparse_encoder.encode_query(text)

        # Store with tenant prefix for isolation
        storage_id = f"{tenant_id}:{doc_id}"
        self._sparse_vectors[storage_id] = sparse_vector

        logger.debug(
            "document_indexed_sparse",
            doc_id=doc_id,
            tenant_id=tenant_id,
            non_zero_terms=len(sparse_vector.indices),
        )

        return sparse_vector

    def remove_document(self, doc_id: str, tenant_id: str) -> bool:
        """Remove a document from sparse index.

        Args:
            doc_id: Document identifier
            tenant_id: Tenant identifier

        Returns:
            True if document was removed
        """
        storage_id = f"{tenant_id}:{doc_id}"
        if storage_id in self._sparse_vectors:
            del self._sparse_vectors[storage_id]
            return True
        return False

    def clear_tenant(self, tenant_id: str) -> int:
        """Clear all documents for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Number of documents removed
        """
        prefix = f"{tenant_id}:"
        to_remove = [k for k in self._sparse_vectors if k.startswith(prefix)]
        for key in to_remove:
            del self._sparse_vectors[key]
        return len(to_remove)


class SparseVectorAdapter:
    """Feature flag wrapper for sparse vector functionality.

    When disabled, all operations return neutral results without
    performing actual sparse vector operations.

    Example:
        adapter = SparseVectorAdapter(
            enabled=settings.sparse_vectors_enabled,
            dense_search=dense_search,
        )

        # Safe to call even when disabled
        results = await adapter.search("query", "tenant-1")
    """

    def __init__(
        self,
        enabled: bool = DEFAULT_SPARSE_VECTORS_ENABLED,
        dense_search: Optional[DenseSearchProtocol] = None,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
        dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_HYBRID_SPARSE_WEIGHT,
    ) -> None:
        """Initialize the adapter.

        Args:
            enabled: Whether sparse vector features are enabled
            dense_search: Dense vector search implementation
            sparse_model: BM42 model name for sparse encoding
            dense_weight: Weight for dense search in RRF
            sparse_weight: Weight for sparse search in RRF
        """
        self._enabled = enabled
        self._dense_search = dense_search
        self._hybrid: Optional[HybridVectorSearch] = None

        if enabled and dense_search:
            self._hybrid = HybridVectorSearch(
                dense_search=dense_search,
                sparse_encoder=BM42Encoder(model_name=sparse_model),
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
            logger.info(
                "sparse_vector_adapter_enabled",
                model=sparse_model,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        else:
            logger.info("sparse_vector_adapter_disabled")

    @property
    def enabled(self) -> bool:
        """Check if sparse vectors are enabled."""
        return self._enabled

    async def search(
        self,
        query: str,
        tenant_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform search (hybrid if enabled, dense-only if disabled).

        Args:
            query: Search query
            tenant_id: Tenant identifier
            limit: Maximum results

        Returns:
            Search results
        """
        if not self._enabled or self._hybrid is None:
            # Fall back to dense-only search
            if self._dense_search:
                return await self._dense_search.search(query, tenant_id, limit)
            return []

        return await self._hybrid.search(query, tenant_id, limit)

    def index_document(
        self,
        doc_id: str,
        text: str,
        tenant_id: str,
    ) -> Optional[SparseVector]:
        """Index document for sparse search.

        Args:
            doc_id: Document identifier
            text: Document text
            tenant_id: Tenant identifier

        Returns:
            Sparse vector if enabled, None otherwise
        """
        if not self._enabled or self._hybrid is None:
            return None

        return self._hybrid.index_document(doc_id, text, tenant_id)

    def remove_document(self, doc_id: str, tenant_id: str) -> bool:
        """Remove document from sparse index.

        Args:
            doc_id: Document identifier
            tenant_id: Tenant identifier

        Returns:
            True if removed (always False when disabled)
        """
        if not self._enabled or self._hybrid is None:
            return False

        return self._hybrid.remove_document(doc_id, tenant_id)

    def encode_query(self, query: str) -> Optional[SparseVector]:
        """Encode query to sparse vector.

        Args:
            query: Query text

        Returns:
            Sparse vector if enabled, None otherwise
        """
        if not self._enabled or self._hybrid is None:
            return None

        return self._hybrid._sparse_encoder.encode_query(query)
