"""Benchmark runner for retrieval quality evaluation.

This module orchestrates benchmark runs across multiple retrieval strategies
and k values, producing comprehensive quality reports.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable, Protocol

from .datasets import EvaluationDataset, Query
from .metrics import RetrievalMetrics, compute_all_metrics


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies for benchmarking."""

    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """Result of a single retrieval operation.

    Attributes:
        query_id: The query identifier
        retrieved_ids: Ordered list of retrieved document IDs
        latency_ms: Retrieval latency in milliseconds
        metadata: Additional metadata about the retrieval
    """

    query_id: str
    retrieved_ids: list[str]
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalFunction(Protocol):
    """Protocol for retrieval function implementations."""

    async def __call__(
        self, query: str, k: int, tenant_id: str
    ) -> list[str]:
        """Execute retrieval and return ordered document IDs.

        Args:
            query: The query text
            k: Number of results to retrieve
            tenant_id: Tenant identifier

        Returns:
            Ordered list of retrieved document IDs
        """
        ...


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes:
        k_values: List of K values to evaluate (e.g., [1, 5, 10])
        strategies: List of strategies to evaluate
        tenant_id: Tenant ID for multi-tenant isolation
        output_dir: Directory to save results
        run_id: Optional run identifier (auto-generated if not provided)
    """

    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])
    strategies: list[RetrievalStrategy] = field(
        default_factory=lambda: [RetrievalStrategy.VECTOR]
    )
    tenant_id: str = "benchmark-tenant"
    output_dir: Path = field(default_factory=lambda: Path("tests/benchmarks/results"))
    run_id: str | None = None

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class StrategyResult:
    """Results for a single strategy across all K values.

    Attributes:
        strategy: The retrieval strategy
        metrics_by_k: Mapping of K value -> metrics
        total_queries: Total number of queries evaluated
        avg_latency_ms: Average retrieval latency
    """

    strategy: RetrievalStrategy
    metrics_by_k: dict[int, RetrievalMetrics]
    total_queries: int
    avg_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy": self.strategy.value,
            "total_queries": self.total_queries,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "metrics_by_k": {
                str(k): metrics.to_dict() for k, metrics in self.metrics_by_k.items()
            },
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report.

    Attributes:
        run_id: Unique run identifier
        timestamp: When the benchmark was run
        dataset_summary: Summary of the evaluation dataset
        config: Benchmark configuration
        results: Results for each strategy
        baseline_comparison: Optional comparison to baseline
    """

    run_id: str
    timestamp: str
    dataset_summary: dict[str, Any]
    config: dict[str, Any]
    results: list[StrategyResult]
    baseline_comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "dataset": self.dataset_summary,
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
            "baseline_comparison": self.baseline_comparison,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save report to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Benchmark Report: {self.run_id}",
            f"Timestamp: {self.timestamp}",
            f"Dataset: {self.dataset_summary.get('name', 'unknown')}",
            f"Queries: {self.dataset_summary.get('num_queries', 0)}",
            "",
            "Results:",
        ]

        for result in self.results:
            lines.append(f"\n  Strategy: {result.strategy.value}")
            lines.append(f"  Avg Latency: {result.avg_latency_ms:.2f}ms")
            for k, metrics in sorted(result.metrics_by_k.items()):
                lines.append(
                    f"    @{k}: MRR={metrics.mrr:.4f}, NDCG={metrics.ndcg:.4f}, "
                    f"P={metrics.precision:.4f}, R={metrics.recall:.4f}"
                )

        return "\n".join(lines)


class MockRetriever:
    """Mock retriever for testing benchmarks without real infrastructure.

    Returns documents based on simple text matching heuristics.
    """

    def __init__(self, documents: dict[str, str]):
        """Initialize with document corpus.

        Args:
            documents: Mapping of doc_id -> text
        """
        self.documents = documents
        self._doc_ids = list(documents.keys())

    async def retrieve(self, query: str, k: int, tenant_id: str) -> list[str]:
        """Mock retrieval based on word overlap.

        Args:
            query: Query text
            k: Number of results
            tenant_id: Tenant ID (unused in mock)

        Returns:
            List of document IDs ordered by word overlap score
        """
        query_words = set(query.lower().split())

        scores = []
        for doc_id, text in self.documents.items():
            doc_words = set(text.lower().split())
            overlap = len(query_words & doc_words)
            scores.append((doc_id, overlap))

        # Sort by overlap descending, then by doc_id for stability
        scores.sort(key=lambda x: (-x[1], x[0]))

        return [doc_id for doc_id, _ in scores[:k]]


class BenchmarkRunner:
    """Orchestrates benchmark runs across strategies and K values."""

    def __init__(
        self,
        dataset: EvaluationDataset,
        config: BenchmarkConfig,
        retrievers: dict[RetrievalStrategy, Callable[[str, int, str], Awaitable[list[str]]]] | None = None,
    ):
        """Initialize the benchmark runner.

        Args:
            dataset: Evaluation dataset
            config: Benchmark configuration
            retrievers: Optional mapping of strategy -> retrieval function.
                       If not provided, uses MockRetriever for all strategies.
        """
        self.dataset = dataset
        self.config = config

        if retrievers is None:
            # Use mock retriever for all strategies
            mock = MockRetriever(dataset.get_document_text_map())
            self.retrievers = {
                strategy: mock.retrieve for strategy in RetrievalStrategy
            }
        else:
            self.retrievers = retrievers

    async def run_single_query(
        self,
        query: Query,
        strategy: RetrievalStrategy,
        k: int,
    ) -> RetrievalResult:
        """Run retrieval for a single query.

        Args:
            query: The benchmark query
            strategy: Retrieval strategy to use
            k: Number of results to retrieve

        Returns:
            RetrievalResult with retrieved IDs and latency
        """
        import time

        retriever = self.retrievers.get(strategy)
        if retriever is None:
            return RetrievalResult(
                query_id=query.query_id,
                retrieved_ids=[],
                latency_ms=0.0,
                metadata={"error": f"No retriever for {strategy}"},
            )

        start = time.perf_counter()
        retrieved_ids = await retriever(query.text, k, self.config.tenant_id)
        latency_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query_id=query.query_id,
            retrieved_ids=retrieved_ids,
            latency_ms=latency_ms,
        )

    async def run_strategy(
        self,
        strategy: RetrievalStrategy,
    ) -> StrategyResult:
        """Run benchmark for a single strategy across all K values.

        Args:
            strategy: Retrieval strategy to evaluate

        Returns:
            StrategyResult with metrics for all K values
        """
        max_k = max(self.config.k_values)

        # Run all queries with max K (we'll slice for smaller K values)
        results = await asyncio.gather(
            *[
                self.run_single_query(query, strategy, max_k)
                for query in self.dataset.queries
            ]
        )

        # Calculate average latency
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0.0

        # Build relevance data structures
        all_retrieved_ids = [r.retrieved_ids for r in results]
        all_relevance_grades = [q.relevant_docs for q in self.dataset.queries]

        # Compute metrics for each K
        metrics_by_k = {}
        for k in self.config.k_values:
            # Slice retrieved IDs to K
            retrieved_at_k = [ids[:k] for ids in all_retrieved_ids]
            metrics = compute_all_metrics(retrieved_at_k, all_relevance_grades, k)
            metrics_by_k[k] = metrics

        return StrategyResult(
            strategy=strategy,
            metrics_by_k=metrics_by_k,
            total_queries=len(self.dataset.queries),
            avg_latency_ms=avg_latency,
        )

    async def run(self) -> BenchmarkReport:
        """Run the complete benchmark.

        Returns:
            BenchmarkReport with all results
        """
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Run all strategies
        results = await asyncio.gather(
            *[self.run_strategy(strategy) for strategy in self.config.strategies]
        )

        report = BenchmarkReport(
            run_id=self.config.run_id or timestamp,
            timestamp=timestamp,
            dataset_summary=self.dataset.summary(),
            config={
                "k_values": self.config.k_values,
                "strategies": [s.value for s in self.config.strategies],
                "tenant_id": self.config.tenant_id,
            },
            results=list(results),
        )

        # Save results
        output_path = self.config.output_dir / f"benchmark_{self.config.run_id}.json"
        report.save(output_path)

        return report


def load_baseline(path: Path) -> dict[str, Any] | None:
    """Load baseline results for comparison.

    Args:
        path: Path to baseline JSON file

    Returns:
        Parsed baseline data or None if not found
    """
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compare_to_baseline(
    current: BenchmarkReport,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare current results to baseline.

    Args:
        current: Current benchmark report
        baseline: Baseline results

    Returns:
        Comparison summary with deltas
    """
    comparison = {
        "baseline_run_id": baseline.get("run_id"),
        "baseline_timestamp": baseline.get("timestamp"),
        "deltas": [],
    }

    baseline_results = {r["strategy"]: r for r in baseline.get("results", [])}

    for result in current.results:
        strategy = result.strategy.value
        baseline_strategy = baseline_results.get(strategy)

        if not baseline_strategy:
            continue

        for k, metrics in result.metrics_by_k.items():
            baseline_metrics = baseline_strategy.get("metrics_by_k", {}).get(str(k), {})

            delta = {
                "strategy": strategy,
                "k": k,
                "mrr_delta": metrics.mrr - baseline_metrics.get(f"mrr@{k}", 0),
                "ndcg_delta": metrics.ndcg - baseline_metrics.get(f"ndcg@{k}", 0),
                "precision_delta": metrics.precision - baseline_metrics.get(f"precision@{k}", 0),
                "recall_delta": metrics.recall - baseline_metrics.get(f"recall@{k}", 0),
            }
            comparison["deltas"].append(delta)

    return comparison
