# Advanced Retrieval Deep Dive

**Date:** 2026-01-13
**Version:** 1.0
**Related Story:** Story 18-20 - Create Advanced Retrieval Deep Dive Guide
**Related Epics:** Epic 12 (Advanced Retrieval), Epic 19 (Quality Foundation), Epic 20 (Advanced Retrieval Intelligence)

---

## Overview

This technical deep dive explores the advanced retrieval internals of the Agentic RAG platform. It covers reranking technologies, CRAG (Corrective Retrieval Augmented Generation), sparse vectors, cross-language retrieval, and performance optimization strategies.

For basic configuration, see [Advanced Retrieval Configuration Guide](./advanced-retrieval-configuration.md).

---

## Table of Contents

1. [Retrieval Pipeline Architecture](#retrieval-pipeline-architecture)
2. [Reranking Technologies](#reranking-technologies)
   - [Cohere Rerank](#cohere-rerank)
   - [FlashRank](#flashrank)
   - [ColBERT Late Interaction](#colbert-late-interaction)
   - [Graph-Based Rerankers](#graph-based-rerankers)
3. [CRAG: Corrective Retrieval Augmented Generation](#crag-corrective-retrieval-augmented-generation)
4. [Sparse Vectors and BM42](#sparse-vectors-and-bm42)
5. [Cross-Language Retrieval](#cross-language-retrieval)
6. [Score Normalization Strategies](#score-normalization-strategies)
7. [Caching Architecture](#caching-architecture)
8. [Performance Optimization](#performance-optimization)
9. [Benchmarking and Evaluation](#benchmarking-and-evaluation)

---

## Retrieval Pipeline Architecture

The retrieval pipeline is designed as a multi-stage system where each stage progressively refines results.

### Pipeline Flow

```
                                    RETRIEVAL PIPELINE
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Query ───────────────────────────────────────────────────────────────────────►  |
|    │                                                                              |
|    ▼                                                                              |
|  ┌─────────────────────────────────────────────────────────────────────────────┐ |
|  │                        STAGE 1: INITIAL RETRIEVAL                            │ |
|  │                                                                               │ |
|  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐           │ |
|  │  │  Dense Vector    │  │  Sparse Vector   │  │   Graph Search   │           │ |
|  │  │  (pgvector)      │  │  (BM42/BM25)     │  │   (Graphiti)     │           │ |
|  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘           │ |
|  │           │                     │                     │                      │ |
|  │           └─────────────────────┴─────────────────────┘                      │ |
|  │                                 │                                             │ |
|  │                                 ▼                                             │ |
|  │                   ┌─────────────────────────────┐                            │ |
|  │                   │   RRF Fusion (Hybrid)       │  ~50 candidates            │ |
|  │                   └─────────────────────────────┘                            │ |
|  └─────────────────────────────────────────────────────────────────────────────┘ |
|                                    │                                              |
|                                    ▼                                              |
|  ┌─────────────────────────────────────────────────────────────────────────────┐ |
|  │                        STAGE 2: RERANKING                                    │ |
|  │                                                                               │ |
|  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐           │ |
|  │  │  Cross-Encoder   │  │  ColBERT         │  │  Graph Reranker  │           │ |
|  │  │  (Cohere/Flash)  │  │  (MaxSim)        │  │  (Episode/Dist)  │           │ |
|  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘           │ |
|  │           └─────────────────────┴─────────────────────┘                      │ |
|  │                                 │                                             │ |
|  │                                 ▼                                             │ |
|  │                   ┌─────────────────────────────┐                            │ |
|  │                   │   Combined Reranking        │  top_k results             │ |
|  │                   └─────────────────────────────┘                            │ |
|  └─────────────────────────────────────────────────────────────────────────────┘ |
|                                    │                                              |
|                                    ▼                                              |
|  ┌─────────────────────────────────────────────────────────────────────────────┐ |
|  │                        STAGE 3: QUALITY GRADING (CRAG)                       │ |
|  │                                                                               │ |
|  │       ┌──────────────────┐       ┌──────────────────────────────────┐       │ |
|  │       │  Grader Agent    │ ──►   │  Score >= threshold? ─► Results  │       │ |
|  │       │  (Heuristic/CE)  │       │  Score < threshold?  ─► Fallback │       │ |
|  │       └──────────────────┘       └──────────────────────────────────┘       │ |
|  │                                                                               │ |
|  │                         ┌──────────────────────────────┐                     │ |
|  │                         │  Fallback Strategies:        │                     │ |
|  │                         │  - Web Search (Tavily)       │                     │ |
|  │                         │  - Expanded Query            │                     │ |
|  │                         │  - Alternate Index           │                     │ |
|  │                         └──────────────────────────────┘                     │ |
|  └─────────────────────────────────────────────────────────────────────────────┘ |
|                                    │                                              |
|                                    ▼                                              |
|                              Final Results ──────────────────────────────────►   |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### Core Classes

The pipeline is implemented in `backend/src/agentic_rag_backend/retrieval/pipeline.py`:

```python
@dataclass(frozen=True)
class VectorSearchResult:
    """Vector search result with optional reranking details."""
    hits: list[VectorHit]
    original_hits: list[VectorHit]
    reranked: list[RerankedHit] | None
    reranking_applied: bool
    reranker_model: str | None

class RetrievalPipeline:
    """Unified retrieval operations shared across entrypoints."""

    def __init__(
        self,
        vector_search: Optional[VectorSearchService],
        graph_traversal: Optional[GraphTraversalService],
        graphiti_client: Optional[GraphitiClient] = None,
        reranker: Optional[RerankerClient] = None,
        reranker_top_k: int = 10,
        small_to_big: Optional[SmallToBigAdapter] = None,
        graph_reranker: Optional[GraphReranker] = None,
    ) -> None:
        # Pipeline components are injected for flexibility
```

---

## Reranking Technologies

Reranking improves retrieval precision by scoring query-document pairs together rather than independently.

### Cohere Rerank

Cohere's cross-encoder model provides high-accuracy reranking with multilingual support.

**Implementation:** `backend/src/agentic_rag_backend/retrieval/reranking.py`

```python
class CohereRerankerClient(RerankerClient):
    """Cohere Rerank API client.

    Uses Cohere's cross-encoder models for high-accuracy reranking.
    Supports 100+ languages and 32K context window.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v3.5",
    ) -> None:
        self._client = cohere.AsyncClient(api_key=api_key)
        self._model = model
```

**Key Features:**
- **100+ Language Support:** Native multilingual reranking without translation
- **32K Context Window:** Can process long documents without truncation
- **Automatic Retry:** Tenacity-based retry with exponential backoff
- **Metrics Integration:** Records latency and improvement ratio

**Model Options:**
| Model | Description | Best For |
|-------|-------------|----------|
| `rerank-v3.5` | Latest, highest accuracy | Production deployments |
| `rerank-multilingual-v3.0` | Optimized for non-English | International content |
| `rerank-english-v3.0` | English-only, faster | English-only corpora |

### FlashRank

FlashRank provides CPU-optimized local reranking without API costs.

**Implementation:** `backend/src/agentic_rag_backend/retrieval/reranking.py`

```python
class FlashRankRerankerClient(RerankerClient):
    """FlashRank local reranker client.

    Uses CPU-optimized models for cost-effective local reranking.
    No API costs, good for cost-sensitive deployments.

    Story 19-G3: Supports lazy loading or eager preloading of the model.
    """

    def __init__(
        self,
        model: str = "ms-marco-MiniLM-L-12-v2",
        preload: bool = False,
    ) -> None:
        self._model = model
        self._ranker = None
        self._preload = preload
        self._model_loaded = False
```

**Key Features:**
- **Zero API Cost:** Runs entirely locally
- **CPU Optimized:** No GPU required
- **Lazy Loading:** Model loads on first use (or at startup with preload=True)
- **Thread-Safe:** Uses asyncio.to_thread for non-blocking inference

**Model Options:**
| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `ms-marco-MiniLM-L-12-v2` | ~120MB | Fast | High |
| `ms-marco-MiniLM-L-6-v2` | ~80MB | Fastest | Good |
| `ms-marco-TinyBERT-L-2-v2` | ~25MB | Ultra-fast | Lower |

### ColBERT Late Interaction

ColBERT uses a late interaction approach that computes token-level similarity, offering a middle ground between bi-encoders and cross-encoders.

**Implementation:** `backend/src/agentic_rag_backend/retrieval/colbert_reranker.py`

```python
class ColBERTEncoder:
    """Encoder for ColBERT token embeddings.

    Encodes text into token-level embeddings using a transformer model.
    Unlike sentence embeddings, this preserves individual token representations
    for late interaction scoring.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length

class MaxSimScorer:
    """MaxSim scorer for ColBERT late interaction.

    Computes the MaxSim score between query and document token embeddings:
    score = sum(max(cos_sim(q_i, d_j)) for all query tokens i over all doc tokens j)
    """

    @staticmethod
    def compute_score(
        query_embeddings: TokenEmbeddings,
        doc_embeddings: TokenEmbeddings,
    ) -> float:
        # Normalize for cosine similarity
        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
        d_norm = d_emb / (np.linalg.norm(d_emb, axis=1, keepdims=True) + 1e-8)

        # Compute similarity matrix [num_query_tokens, num_doc_tokens]
        sim_matrix = np.dot(q_norm, d_norm.T)

        # MaxSim: for each query token, take max similarity over all doc tokens
        max_sims = np.max(sim_matrix, axis=1)
        score = float(np.sum(max_sims))
        return score
```

**MaxSim Algorithm Explained:**

```
Query:  "What is machine learning?"
        [Q1] [Q2] [Q3] [Q4] [Q5]

Document: "Machine learning is a subset of AI..."
          [D1] [D2] [D3] [D4] [D5] [D6] ...

Similarity Matrix (cosine):
              D1    D2    D3    D4    D5    D6
        Q1  [0.9]  0.2   0.1   0.3   0.1   0.2    max(Q1) = 0.9
        Q2   0.2  [0.8]  0.1   0.1   0.2   0.1    max(Q2) = 0.8
        Q3   0.1   0.2  [0.9]  0.3   0.1   0.2    max(Q3) = 0.9
        Q4   0.3   0.2   0.1  [0.7]  0.2   0.1    max(Q4) = 0.7
        Q5   0.1   0.1   0.2   0.1  [0.6]  0.2    max(Q5) = 0.6

MaxSim Score = 0.9 + 0.8 + 0.9 + 0.7 + 0.6 = 3.9
```

**Trade-offs:**
| Approach | Speed | Accuracy | Index Size |
|----------|-------|----------|------------|
| Bi-Encoder | Fastest | Good | 1x |
| ColBERT | Medium | High | ~10x |
| Cross-Encoder | Slowest | Highest | N/A |

### Graph-Based Rerankers

Graph rerankers use knowledge graph signals to boost contextually relevant results.

**Implementation:** `backend/src/agentic_rag_backend/retrieval/graph_rerankers.py`

#### Episode Mentions Reranker

Boosts entities mentioned in recent episodes (Zep-style temporal scoring):

```python
class EpisodeMentionsReranker(GraphReranker):
    """Reranker based on episode mention frequency.

    Entities mentioned in more recent episodes are considered more
    contextually relevant. This implements Zep-style temporal scoring.
    """

    async def _get_total_mentions(
        self,
        entity_ids: list[str],
        tenant_id: str,
    ) -> int:
        # Neo4j query for episode mentions within time window
        query = """
        MATCH (e:Entity {tenant_id: $tenant_id})<-[:MENTIONS]-(ep:Episode)
        WHERE (e.id IN $entity_ids OR e.uuid IN $entity_ids)
          AND ep.created_at >= datetime($cutoff)
        RETURN count(DISTINCT ep) as mention_count
        """
```

#### Node Distance Reranker

Boosts results by proximity to query entities in the knowledge graph:

```python
class NodeDistanceReranker(GraphReranker):
    """Reranker based on graph distance from query entities.

    Entities closer to query concepts in the knowledge graph receive
    higher scores. Uses Neo4j's shortest path algorithm.
    """

    def _distance_to_score(self, distance: Optional[int]) -> float:
        """Convert distance to 0-1 score.
        Closer = higher score. No path or distance > max = 0.
        """
        if distance is None or distance > self._max_distance:
            return 0.0
        # distance=0 -> 1.0, distance=max_distance -> 0.0
        return max(0.0, 1.0 - (distance / self._max_distance))
```

#### Hybrid Graph Reranker

Combines episode and distance signals with configurable weights:

```python
class HybridGraphReranker(GraphReranker):
    """Hybrid reranker combining episode and distance signals.

    Runs both sub-rerankers in parallel and combines their scores
    with configurable weights.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        graphiti_client: Optional[GraphitiClient] = None,
        episode_weight: float = 0.3,
        distance_weight: float = 0.3,
        original_weight: float = 0.4,
        episode_window_days: int = 30,
        max_distance: int = 3,
    ) -> None:
        # Weights must sum to 1.0
        # Combined score = original*0.4 + episode*0.3 + distance*0.3
```

**Configuration:**
```bash
GRAPH_RERANKER_ENABLED=true
GRAPH_RERANKER_TYPE=hybrid  # episode | distance | hybrid
GRAPH_RERANKER_EPISODE_WEIGHT=0.3
GRAPH_RERANKER_DISTANCE_WEIGHT=0.3
GRAPH_RERANKER_ORIGINAL_WEIGHT=0.4
GRAPH_RERANKER_EPISODE_WINDOW_DAYS=30
GRAPH_RERANKER_MAX_DISTANCE=3
```

---

## CRAG: Corrective Retrieval Augmented Generation

CRAG evaluates retrieval quality and triggers fallback strategies when results are insufficient.

**Implementation:** `backend/src/agentic_rag_backend/retrieval/grader.py`

### Grader Types

#### Heuristic Grader

Uses retrieval scores or content length as a proxy for quality:

```python
class HeuristicGrader(BaseGrader):
    """Simple heuristic-based grader using retrieval scores.

    When retrieval scores are not available, uses content length heuristic.

    Formula when no retrieval scores available:
        length_factor = min((avg_length - min_length) / (max_length - min_length), 1.0)
        final_score = base_score * (1 - length_weight) + length_factor * length_weight
    """

    def __init__(
        self,
        top_k: int = 5,
        length_weight: float = 0.5,  # 0 = disabled, 1 = pure length-based
        min_length: int = 50,
        max_length: int = 2000,
    ):
```

#### Cross-Encoder Grader

Uses a cross-encoder model for higher-accuracy scoring:

```python
class CrossEncoderGrader(BaseGrader):
    """Cross-encoder based grader for more accurate relevance scoring.

    Supported models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good accuracy, default)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (higher accuracy)
    - BAAI/bge-reranker-base (BGE reranker)
    - BAAI/bge-reranker-large (BGE large, best accuracy)

    Story 19-G3: Supports eager preloading for reduced first-query latency.
    Story 19-G4: Supports configurable normalization strategies.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fallback_to_default: bool = True,
        preload: bool = False,
        normalization_strategy: NormalizationStrategy = NormalizationStrategy.MIN_MAX,
    ):
```

### Fallback Strategies

#### Web Search Fallback (Tavily)

```python
class WebSearchFallback(BaseFallbackHandler):
    """Web search fallback using Tavily API.

    Queries the Tavily API for current web data when
    local retrieval quality is insufficient.
    """

    async def execute(self, query: str, tenant_id: Optional[str] = None) -> list[RetrievalHit]:
        response = await asyncio.to_thread(
            self._client.search,
            query=query,
            max_results=self.max_results,
            search_depth="basic",
        )
```

#### Expanded Query Fallback

Reformulates the query and retries retrieval (placeholder for LLM-based expansion):

```python
class ExpandedQueryFallback(BaseFallbackHandler):
    """Expanded query fallback that reformulates the query."""

    async def execute(self, query: str, tenant_id: Optional[str] = None) -> list[RetrievalHit]:
        # Full implementation would use an LLM to generate query variations
        # Then call retrieval with tenant_id isolation
```

### CRAG Flow

```
                        CRAG GRADING FLOW
+-------------------------------------------------------------------+
|                                                                   |
|  Retrieval Results (N documents)                                  |
|         │                                                         |
|         ▼                                                         |
|  ┌──────────────────────────────────────────────────────────────┐|
|  │                    SCORE EXTRACTION                          │|
|  │                                                              │|
|  │  For each hit in top_k:                                      │|
|  │    - Use retrieval score if available                        │|
|  │    - Otherwise: cross-encoder(query, document)               │|
|  │    - Or fallback to heuristic (content length)               │|
|  └──────────────────────────────────────────────────────────────┘|
|         │                                                         |
|         ▼                                                         |
|  ┌──────────────────────────────────────────────────────────────┐|
|  │                    SCORE NORMALIZATION                       │|
|  │                                                              │|
|  │  Apply normalization strategy:                               │|
|  │    - MIN_MAX:     (score - min) / (max - min)               │|
|  │    - Z_SCORE:     sigmoid((score - mean) / std)              │|
|  │    - SOFTMAX:     exp(score) / sum(exp(scores))              │|
|  │    - PERCENTILE:  rank-based position                        │|
|  └──────────────────────────────────────────────────────────────┘|
|         │                                                         |
|         ▼                                                         |
|  ┌──────────────────────────────────────────────────────────────┐|
|  │                    AGGREGATION                               │|
|  │                                                              │|
|  │  final_score = aggregate(normalized_scores)                  │|
|  │    - mean (default)                                          │|
|  │    - max                                                     │|
|  │    - weighted_mean (position-weighted)                       │|
|  └──────────────────────────────────────────────────────────────┘|
|         │                                                         |
|         ▼                                                         |
|  ┌────────────────┐      ┌────────────────────────────────────┐  |
|  │ final_score >= │  Yes │                                    │  |
|  │   threshold?   │─────►│  Return results (PASS)             │  |
|  └────────┬───────┘      └────────────────────────────────────┘  |
|           │ No                                                    |
|           ▼                                                       |
|  ┌──────────────────────────────────────────────────────────────┐|
|  │                    FALLBACK EXECUTION                        │|
|  │                                                              │|
|  │  Strategy = settings.grader_fallback_strategy:               │|
|  │    - web_search:     Query Tavily for current data          │|
|  │    - expanded_query: Reformulate + retry retrieval          │|
|  │    - alternate_index: Search different knowledge base        │|
|  │                                                              │|
|  │  Merge fallback_hits with original results                   │|
|  └──────────────────────────────────────────────────────────────┘|
|         │                                                         |
|         ▼                                                         |
|  Return merged results (FALLBACK)                                 |
|                                                                   |
+-------------------------------------------------------------------+
```

---

## Sparse Vectors and BM42

Sparse vectors provide lexical matching that complements dense semantic search.

**Implementation:** `backend/src/agentic_rag_backend/retrieval/sparse_vectors.py`

### SparseVector Data Structure

```python
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

    indices: list[int]
    values: list[float]

    def dot_product(self, other: "SparseVector") -> float:
        """Calculate dot product with another sparse vector.
        Uses index intersection for efficient computation.
        """
        # Build index -> value map for smaller vector
        smaller_map = dict(zip(smaller.indices, smaller.values))
        larger_map = dict(zip(larger.indices, larger.values))

        # Calculate dot product for overlapping indices
        result = 0.0
        for idx, val in smaller_map.items():
            if idx in larger_map:
                result += val * larger_map[idx]
        return result
```

### BM42 Encoder

BM42 uses attention-based term weighting (better than traditional BM25):

```python
class BM42Encoder:
    """BM42 sparse vector encoder using fastembed.

    BM42 uses attention-based term weighting, providing better
    sparse representations than traditional BM25.
    """

    def __init__(
        self,
        model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
    ) -> None:
        self._model_name = model_name
        self._model = None  # Lazy initialization
        self._model_lock = threading.Lock()  # Thread-safe loading

    def encode(self, texts: list[str]) -> list[SparseVector]:
        """Encode texts to sparse vectors."""
        self._ensure_model()
        embeddings = list(self._model.embed(texts))
        return [
            SparseVector(
                indices=list(emb.indices),
                values=list(emb.values),
            )
            for emb in embeddings
        ]
```

### Reciprocal Rank Fusion (RRF)

RRF combines dense and sparse results without score normalization:

```python
class HybridVectorSearch:
    """Combine dense and sparse vector search using RRF fusion.

    Reciprocal Rank Fusion (RRF) combines rankings from multiple
    search systems without needing score normalization.

    Formula: score(d) = sum(1/(k + rank_i(d))) for each system i
    """

    def __init__(
        self,
        dense_search: DenseSearchProtocol,
        sparse_encoder: BM42Encoder,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,  # Standard RRF constant
    ) -> None:

    def _reciprocal_rank_fusion(
        self,
        dense: list[dict],
        sparse: list[dict],
    ) -> list[dict]:
        """Combine results using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}

        # Score dense results
        for i, result in enumerate(dense):
            doc_id = str(result.get("id", ""))
            rrf_score = self.dense_weight / (self._rrf_k + i + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        # Score sparse results
        for i, result in enumerate(sparse):
            doc_id = str(result.get("id", ""))
            rrf_score = self.sparse_weight / (self._rrf_k + i + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [{"id": doc_id, "rrf_score": scores[doc_id]} for doc_id in sorted_ids]
```

**RRF Example:**

```
Dense Results:        Sparse Results:
1. doc_A (score 0.9)  1. doc_C (score 0.8)
2. doc_B (score 0.8)  2. doc_A (score 0.7)
3. doc_C (score 0.7)  3. doc_D (score 0.6)

RRF Calculation (k=60, dense_weight=0.7, sparse_weight=0.3):

doc_A: 0.7/(60+1) + 0.3/(60+2) = 0.0115 + 0.0048 = 0.0163
doc_B: 0.7/(60+2) + 0           = 0.0113           = 0.0113
doc_C: 0.7/(60+3) + 0.3/(60+1) = 0.0111 + 0.0049 = 0.0160
doc_D: 0           + 0.3/(60+3) =           0.0048 = 0.0048

Final Ranking: doc_A, doc_C, doc_B, doc_D
```

**Configuration:**
```bash
SPARSE_VECTORS_ENABLED=true
SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions
HYBRID_DENSE_WEIGHT=0.7
HYBRID_SPARSE_WEIGHT=0.3
RRF_K=60
```

---

## Cross-Language Retrieval

Cross-language retrieval enables querying content in any language.

**Implementation:** `backend/src/agentic_rag_backend/retrieval/cross_language.py`

### Language Detection

```python
class LanguageDetector:
    """Detect language from text using Unicode patterns and word markers.

    This is a lightweight detector that doesn't require external libraries.
    For production use with higher accuracy, consider langdetect or fasttext.
    """

    # Unicode patterns for script detection
    LANGUAGE_PATTERNS = {
        "zh": re.compile(r"[\u4e00-\u9fff]"),      # Chinese
        "ja": re.compile(r"[\u3040-\u309f\u30a0-\u30ff]"),  # Japanese
        "ko": re.compile(r"[\uac00-\ud7af]"),      # Korean
        "ar": re.compile(r"[\u0600-\u06ff]"),      # Arabic
        "ru": re.compile(r"[\u0400-\u04ff]"),      # Cyrillic
        # ... more patterns
    }

    # Word markers for Latin-script languages
    LATIN_LANGUAGE_MARKERS = {
        "en": {"the", "is", "are", "was", "have", "and", "or", "but"},
        "es": {"el", "la", "los", "las", "es", "son", "y", "de", "que"},
        "fr": {"le", "la", "les", "est", "sont", "et", "de", "que"},
        # ... more markers
    }
```

### Multilingual Embeddings

```python
@dataclass
class CrossLanguageEmbedding:
    """Multilingual embedding using sentence-transformers.

    Uses models like multilingual-e5 that map text from different
    languages to the same vector space.
    """

    model_name: str = "intfloat/multilingual-e5-base"

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text (any language)."""
        self._ensure_model()

        # Add query prefix for e5 models
        if "e5" in self.model_name.lower():
            text = f"query: {text}"

        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
```

### Query Translation

```python
@dataclass
class QueryTranslator:
    """Translate queries using LLM for cross-language search.

    Uses the existing LLM provider to translate queries to a target
    language (typically English for English-indexed content).
    """

    llm_provider: Optional[Any] = None
    target_language: str = "en"
    _cache: LRUCache = field(default_factory=LRUCache)  # Max 1000 entries

    async def translate(
        self,
        text: str,
        target_language: Optional[str] = None,
    ) -> str:
        """Translate text to target language."""
        # Check cache first
        cache_key = f"{text}:{target}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Use LLM for translation
        prompt = (
            f"Translate the following text to {target}. "
            f"Return only the translation, nothing else:\n\n{text}"
        )
        result = await self.llm_provider.generate(prompt)
        translated = result.strip()
        self._cache.set(cache_key, translated)
        return translated
```

### Cross-Language Adapter

```python
class CrossLanguageAdapter:
    """Feature flag wrapper for cross-language query support.

    When enabled, uses multilingual embeddings and optional translation.
    """

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if not self._enabled:
            return await self._base_embedding.embed(text)

        # Optionally translate non-English queries first
        if self._translator and self._translation_enabled:
            detected = self._language_detector.detect(text)
            if detected.language != "en" and detected.confidence > 0.5:
                text = await self._translator.translate(text)

        # Use multilingual embedding
        return await self._cross_language_embedding.embed(text)
```

**Configuration:**
```bash
CROSS_LANGUAGE_ENABLED=true
CROSS_LANGUAGE_EMBEDDING=intfloat/multilingual-e5-base
CROSS_LANGUAGE_TRANSLATION=true
CROSS_LANGUAGE_TARGET=en
```

---

## Score Normalization Strategies

Different normalization strategies convert raw scores to a consistent 0-1 scale.

**Implementation:** `backend/src/agentic_rag_backend/retrieval/normalization.py`

### Available Strategies

```python
class NormalizationStrategy(str, Enum):
    MIN_MAX = "min_max"       # Linear: (score - min) / (max - min)
    Z_SCORE = "z_score"       # Standardization with sigmoid
    SOFTMAX = "softmax"       # Exponential: exp(score) / sum(exp)
    PERCENTILE = "percentile" # Rank-based position
```

### Implementation Details

#### Min-Max Normalization

```python
def normalize_min_max(scores: list[float], epsilon: float = 1e-10) -> list[float]:
    """Normalize scores using min-max scaling.
    Formula: (score - min) / (max - min)
    """
    if len(scores) == 1:
        return [0.5]  # Single score gets middle value

    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score

    if range_score < epsilon:
        return [0.5] * len(scores)  # All scores equal

    return [(s - min_score) / range_score for s in scores]
```

#### Z-Score Normalization

```python
def normalize_z_score(scores: list[float], epsilon: float = 1e-10) -> list[float]:
    """Normalize using z-score standardization with sigmoid mapping.
    Formula: sigmoid((score - mean) / std)
    """
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_score = math.sqrt(variance)

    if std_score < epsilon:
        return [0.5] * len(scores)

    z_scores = [(s - mean_score) / std_score for s in scores]
    return [1 / (1 + math.exp(-z)) for z in z_scores]  # Sigmoid
```

#### Softmax Normalization

```python
def normalize_softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Normalize using softmax function.
    Formula: exp(score / T) / sum(exp(scores / T))

    Temperature controls emphasis on score differences:
    - T < 1: Sharper (more difference emphasis)
    - T = 1: Standard softmax
    - T > 1: Smoother (less difference emphasis)
    """
    scaled = [s / temperature for s in scores]
    max_scaled = max(scaled)  # Numerical stability
    exp_scores = [math.exp(s - max_scaled) for s in scaled]
    sum_exp = sum(exp_scores)
    return [e / sum_exp for e in exp_scores]
```

#### Percentile Normalization

```python
def normalize_percentile(scores: list[float]) -> list[float]:
    """Normalize using percentile ranking.
    Each score converted to its percentile position.
    """
    n = len(scores)
    indexed_scores = [(score, i) for i, score in enumerate(scores)]
    sorted_scores = sorted(indexed_scores, key=lambda x: x[0])

    normalized = [0.0] * n
    i = 0
    while i < n:
        # Find all tied scores
        j = i
        while j < n and sorted_scores[j][0] == sorted_scores[i][0]:
            j += 1

        # Average percentile for tied scores
        avg_percentile = (i + j - 1) / (2 * (n - 1))
        for k in range(i, j):
            normalized[sorted_scores[k][1]] = avg_percentile
        i = j

    return normalized
```

### Choosing a Strategy

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| `min_max` | Bounded scores | Simple, interpretable | Sensitive to outliers |
| `z_score` | Normal distributions | Handles outliers | Requires multiple scores |
| `softmax` | Ranking tasks | Emphasizes differences | Sensitive to gaps |
| `percentile` | Mixed sources | Robust to outliers | Loses magnitude info |

---

## Caching Architecture

Caching reduces redundant computations and API calls.

**Implementation:** `backend/src/agentic_rag_backend/retrieval/cache.py`

### TTL Cache

```python
@dataclass
class CacheEntry(Generic[T]):
    value: T
    expires_at: float

class TTLCache(Generic[T]):
    """Simple size-bounded TTL cache."""

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[Hashable, CacheEntry[T]] = OrderedDict()
        self._stats = CacheStats()

    def get(self, key: Hashable) -> T | None:
        now = monotonic()
        entry = self._store.get(key)

        if not entry:
            self._stats.misses += 1
            return None

        if entry.expires_at <= now:
            self._store.pop(key, None)
            self._stats.misses += 1
            return None

        self._store.move_to_end(key)  # LRU update
        self._stats.hits += 1
        return entry.value

    def _prune(self, now: float) -> None:
        """Remove expired entries and enforce max_size."""
        expired = [k for k, v in self._store.items() if v.expires_at <= now]
        for key in expired:
            self._store.pop(key, None)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)  # Remove oldest
```

### Reranker Cache

```python
def generate_reranker_cache_key(
    query_text: str,
    document_ids: list[str],
    chunk_ids: list[str],
    reranker_model: str,
    tenant_id: str,
    top_k: int,
) -> str:
    """Generate a cache key for reranking results.

    The cache key is a SHA-256 hash ensuring:
    - Tenant isolation is preserved
    - Different queries/documents get different entries
    - Model changes invalidate cached results
    """
    sorted_doc_ids = sorted(document_ids)
    sorted_chunk_ids = sorted(chunk_ids)
    key_components = [
        query_text,
        "|".join(sorted_doc_ids),
        "|".join(sorted_chunk_ids),
        reranker_model,
        tenant_id,
        str(top_k),
    ]
    combined = "\n".join(key_components)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

class RerankerCache:
    """Cache for reranking results with tenant isolation.

    Helps avoid redundant reranking calls for identical query-document combinations.
    """

    def __init__(
        self,
        enabled: bool = False,
        ttl_seconds: int = 300,   # 5 minutes default
        max_size: int = 1000,
    ) -> None:
```

**Cache Configuration:**
```bash
RERANKER_CACHE_ENABLED=true
RERANKER_CACHE_TTL_SECONDS=300
RERANKER_CACHE_MAX_SIZE=1000
```

---

## Performance Optimization

### Latency Optimization Strategies

#### 1. Model Preloading

Load models at startup instead of first query:

```bash
GRADER_PRELOAD_MODEL=true
RERANKER_PRELOAD_MODEL=true  # FlashRank only
```

**Trade-offs:**
| Metric | Without Preload | With Preload |
|--------|-----------------|--------------|
| Startup time | Fast (~1-2s) | Slower (+3-10s) |
| First query | Slow (+3-10s) | Fast |
| Memory at startup | Lower | Higher |

#### 2. Parallel Execution

The pipeline runs independent operations in parallel:

```python
# From pipeline.py - vector + graph search in parallel
vector_task = self.vector_search(query, tenant_id, ...)
graph_task = graphiti_search(graphiti_client, query, tenant_id, ...)
vector_result, graph_result = await asyncio.gather(vector_task, graph_task)
```

```python
# From graph_rerankers.py - episode + distance rerankers in parallel
episode_task = self._episode_reranker.rerank(query, results, tenant_id)
distance_task = self._distance_reranker.rerank(query, results, tenant_id)
episode_results, distance_results = await asyncio.gather(episode_task, distance_task)
```

#### 3. Lazy Loading

Models load on first use to reduce startup time:

```python
def _ensure_model(self) -> None:
    """Lazily load the model on first use."""
    if self._model is not None:
        return

    with self._model_lock:  # Thread-safe double-check
        if self._model is not None:
            return
        self._model = load_model(...)
```

#### 4. Result Limiting

Limit candidates at each stage:

```python
# Stage 1: Limit initial retrieval
VECTOR_LIMIT=50

# Stage 2: Limit reranking output
RERANKER_TOP_K=10

# Stage 3: Limit cross-encoder grading
MAX_CROSS_ENCODER_HITS=10
```

### Memory Optimization

#### Sparse Vector Efficiency

Sparse vectors store only non-zero values:

```python
# Dense vector (1024 dims): 4KB per document
vector = [0.0, 0.0, 0.123, 0.0, ..., 0.456, 0.0]  # 1024 floats

# Sparse vector: ~50-200 bytes per document
sparse = SparseVector(
    indices=[2, 1023],   # Only non-zero positions
    values=[0.123, 0.456]
)
```

#### Cache Size Limits

All caches have configurable size limits:

```python
class TTLCache:
    def _prune(self, now: float) -> None:
        # Enforce max_size with LRU eviction
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)  # Remove oldest
```

### Throughput Optimization

#### Batch Processing

Batch operations where possible:

```python
# ColBERT batch encoding
async def encode_documents_batch(
    self,
    documents: list[str],
) -> list[TokenEmbeddings]:
    def _encode_batch() -> list[TokenEmbeddings]:
        return [self._encode_text(doc) for doc in documents]
    return await loop.run_in_executor(None, _encode_batch)
```

#### Connection Pooling

Database clients use connection pooling:

```python
# Neo4j driver with connection pool
self._neo4j.driver.session()  # Sessions from pool

# PostgreSQL with asyncpg pool
self.postgres.search_similar_chunks(...)  # Uses connection pool
```

---

## Benchmarking and Evaluation

### Retrieval Quality Metrics

The system supports standard IR evaluation metrics:

| Metric | Description | Formula |
|--------|-------------|---------|
| **MRR@K** | Mean Reciprocal Rank | `1/rank_of_first_relevant` |
| **NDCG@K** | Normalized Discounted Cumulative Gain | `DCG/IDCG` |
| **Precision@K** | Fraction of top K that are relevant | `relevant_in_K / K` |
| **Recall@K** | Fraction of relevant docs in top K | `relevant_in_K / total_relevant` |

### Observability

All retrieval operations emit structured logs and metrics:

```python
# Latency tracking
record_retrieval_latency(
    strategy="hybrid",
    phase="rerank",  # search, rerank, grade
    tenant_id=tenant_id,
    duration_seconds=elapsed_seconds,
)

# Reranking improvement
record_reranking_improvement(
    tenant_id=tenant_id,
    pre_score=hits[0].similarity,
    post_score=reranked[0].rerank_score,
)

# Grader evaluation
record_grader_evaluation(result="pass", tenant_id=tenant_id)
record_grader_score(model=self.get_model(), tenant_id=tenant_id, score=score)

# Fallback tracking
record_retrieval_fallback(reason="low_score", tenant_id=tenant_id)
```

### Prometheus Metrics

Available metrics for monitoring:

```
# Reranker cache
reranker_cache_hits_total{tenant_id}
reranker_cache_misses_total{tenant_id}
reranker_cache_size

# Retrieval latency
retrieval_latency_seconds{strategy, phase, tenant_id}

# Grader
grader_evaluations_total{result, tenant_id}
grader_scores{model, tenant_id}

# Fallbacks
retrieval_fallbacks_total{reason, tenant_id}
```

### Example Benchmark Script

```bash
# Run retrieval benchmark
uv run python -m agentic_rag_backend.scripts.benchmark_retrieval \
  --dataset evaluation/queries.json \
  --output results/baseline.json \
  --tenant-id your-tenant \
  --iterations 3

# Compare configurations
uv run python -m agentic_rag_backend.scripts.compare_benchmarks \
  --baseline results/baseline.json \
  --experiment results/with_reranker.json
```

---

## References

### Academic Papers
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832)
- [CRAG: Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
- [BM42: Attention-based Sparse Representations](https://qdrant.tech/articles/bm42/)
- [Multilingual-E5: Multilingual Text Embeddings](https://arxiv.org/abs/2402.05672)
- [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

### Documentation
- [Cohere Rerank Documentation](https://docs.cohere.com/docs/rerank)
- [FlashRank GitHub](https://github.com/AnswerDotAI/rerankers)
- [FastEmbed Documentation](https://qdrant.github.io/fastembed/)
- [Sentence Transformers](https://www.sbert.net/)

### Internal References
- `backend/src/agentic_rag_backend/retrieval/` - All retrieval implementations
- `_bmad-output/epics/epic-12-tech-spec.md` - Advanced Retrieval epic
- `_bmad-output/epics/epic-19-tech-spec.md` - Quality Foundation epic
- `_bmad-output/epics/epic-20-tech-spec.md` - Advanced Retrieval Intelligence epic
- [Advanced Retrieval Configuration Guide](./advanced-retrieval-configuration.md) - Configuration reference

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-13 | Initial release - Story 18-20 |
