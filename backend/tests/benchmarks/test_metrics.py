"""Unit tests for IR metrics module.

Tests verify correctness of MRR, NDCG, Precision, and Recall calculations
using known examples with hand-calculated expected values.
"""

from __future__ import annotations

import pytest

from agentic_rag_backend.benchmarks.metrics import (
    RetrievalMetrics,
    compute_all_metrics,
    dcg_at_k,
    idcg_at_k,
    mrr_at_k,
    ndcg_at_k,
    ndcg_at_k_single,
    precision_at_k,
    precision_at_k_single,
    recall_at_k,
    recall_at_k_single,
    reciprocal_rank,
)


class TestReciprocalRank:
    """Tests for reciprocal_rank function."""

    def test_first_result_relevant(self):
        """First result is relevant -> RR = 1.0"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        assert reciprocal_rank(retrieved, relevant, k=3) == 1.0

    def test_second_result_relevant(self):
        """Second result is relevant -> RR = 0.5"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2"}
        assert reciprocal_rank(retrieved, relevant, k=3) == 0.5

    def test_third_result_relevant(self):
        """Third result is relevant -> RR = 1/3"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc3"}
        assert reciprocal_rank(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    def test_no_relevant_in_top_k(self):
        """No relevant in top-k -> RR = 0"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4"}
        assert reciprocal_rank(retrieved, relevant, k=3) == 0.0

    def test_multiple_relevant_returns_first(self):
        """Multiple relevant -> returns RR for first one"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2", "doc3"}
        assert reciprocal_rank(retrieved, relevant, k=3) == 0.5

    def test_k_limits_search(self):
        """K limits the search depth"""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc3"}
        assert reciprocal_rank(retrieved, relevant, k=2) == 0.0
        assert reciprocal_rank(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    def test_empty_retrieved(self):
        """Empty retrieved list -> RR = 0"""
        assert reciprocal_rank([], {"doc1"}, k=3) == 0.0

    def test_empty_relevant(self):
        """Empty relevant set -> RR = 0"""
        assert reciprocal_rank(["doc1", "doc2"], set(), k=3) == 0.0


class TestMRRAtK:
    """Tests for mrr_at_k function."""

    def test_single_query(self):
        """Single query MRR equals its RR"""
        all_retrieved = [["doc1", "doc2", "doc3"]]
        all_relevant = [{"doc2"}]
        assert mrr_at_k(all_retrieved, all_relevant, k=3) == 0.5

    def test_multiple_queries(self):
        """Average RR across multiple queries"""
        all_retrieved = [
            ["doc1", "doc2", "doc3"],  # RR = 1.0 (doc1 is first)
            ["doc4", "doc5", "doc6"],  # RR = 0.5 (doc5 is second)
        ]
        all_relevant = [{"doc1"}, {"doc5"}]
        # MRR = (1.0 + 0.5) / 2 = 0.75
        assert mrr_at_k(all_retrieved, all_relevant, k=3) == 0.75

    def test_mismatched_lengths_raises(self):
        """Mismatched input lengths raise ValueError"""
        with pytest.raises(ValueError, match="Mismatched lengths"):
            mrr_at_k([["doc1"]], [{"doc1"}, {"doc2"}], k=3)

    def test_empty_input_raises(self):
        """Empty input raises ValueError"""
        with pytest.raises(ValueError, match="No queries provided"):
            mrr_at_k([], [], k=3)


class TestDCGAtK:
    """Tests for DCG calculation."""

    def test_perfect_ranking(self):
        """DCG for perfect ranking with graded relevance"""
        retrieved = ["doc1", "doc2", "doc3"]
        grades = {"doc1": 1.0, "doc2": 0.5, "doc3": 0.25}
        # DCG = 1.0/log2(2) + 0.5/log2(3) + 0.25/log2(4)
        # DCG = 1.0/1 + 0.5/1.585 + 0.25/2 = 1.0 + 0.315 + 0.125 = 1.440
        dcg = dcg_at_k(retrieved, grades, k=3)
        assert dcg == pytest.approx(1.440, rel=0.01)

    def test_non_relevant_docs(self):
        """Non-relevant docs (not in grades) contribute 0"""
        retrieved = ["doc1", "doc2", "doc3"]
        grades = {"doc1": 1.0}  # Only doc1 is relevant
        dcg = dcg_at_k(retrieved, grades, k=3)
        assert dcg == pytest.approx(1.0, rel=0.01)  # 1.0/log2(2) = 1.0

    def test_empty_retrieved(self):
        """Empty retrieved list -> DCG = 0"""
        assert dcg_at_k([], {"doc1": 1.0}, k=3) == 0.0


class TestIDCGAtK:
    """Tests for Ideal DCG calculation."""

    def test_ideal_ranking(self):
        """IDCG sorts by relevance grade"""
        grades = {"doc1": 1.0, "doc2": 0.5, "doc3": 0.25}
        # Ideal order: doc1(1.0), doc2(0.5), doc3(0.25)
        # IDCG = 1.0/log2(2) + 0.5/log2(3) + 0.25/log2(4)
        idcg = idcg_at_k(grades, k=3)
        assert idcg == pytest.approx(1.440, rel=0.01)

    def test_k_limits(self):
        """IDCG respects K limit"""
        grades = {"doc1": 1.0, "doc2": 0.5, "doc3": 0.25}
        idcg = idcg_at_k(grades, k=1)
        assert idcg == pytest.approx(1.0, rel=0.01)  # Only top-1

    def test_empty_grades(self):
        """Empty grades -> IDCG = 0"""
        assert idcg_at_k({}, k=3) == 0.0


class TestNDCGAtKSingle:
    """Tests for single-query NDCG calculation."""

    def test_perfect_ranking(self):
        """Perfect ranking -> NDCG = 1.0"""
        # Retrieved in perfect order
        retrieved = ["doc1", "doc2", "doc3"]
        grades = {"doc1": 1.0, "doc2": 0.5, "doc3": 0.25}
        ndcg = ndcg_at_k_single(retrieved, grades, k=3)
        assert ndcg == pytest.approx(1.0, rel=0.01)

    def test_inverted_ranking(self):
        """Inverted ranking -> NDCG < 1.0"""
        # Retrieved in reverse order of relevance
        retrieved = ["doc3", "doc2", "doc1"]
        grades = {"doc1": 1.0, "doc2": 0.5, "doc3": 0.25}
        ndcg = ndcg_at_k_single(retrieved, grades, k=3)
        assert ndcg < 1.0
        assert ndcg > 0.0

    def test_no_relevant_docs(self):
        """No relevant docs -> NDCG = 0.0"""
        retrieved = ["doc1", "doc2", "doc3"]
        grades = {}  # No relevant docs
        ndcg = ndcg_at_k_single(retrieved, grades, k=3)
        assert ndcg == 0.0


class TestNDCGAtK:
    """Tests for average NDCG across queries."""

    def test_multiple_queries(self):
        """Average NDCG across queries"""
        all_retrieved = [
            ["doc1", "doc2"],  # Perfect order
            ["doc4", "doc3"],  # Imperfect order
        ]
        all_grades = [
            {"doc1": 1.0, "doc2": 0.5},
            {"doc3": 1.0, "doc4": 0.5},  # doc3 should be first
        ]
        ndcg = ndcg_at_k(all_retrieved, all_grades, k=2)
        # First query: perfect NDCG = 1.0
        # Second query: imperfect NDCG < 1.0
        # Average should be between 0.5 and 1.0
        assert 0.5 < ndcg < 1.0

    def test_mismatched_lengths_raises(self):
        """Mismatched input lengths raise ValueError"""
        with pytest.raises(ValueError):
            ndcg_at_k([["doc1"]], [{"doc1": 1.0}, {"doc2": 1.0}], k=3)


class TestPrecisionAtK:
    """Tests for Precision@K calculation."""

    def test_all_relevant(self):
        """All retrieved docs are relevant -> P@K = 1.0"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        assert precision_at_k_single(retrieved, relevant, k=3) == 1.0

    def test_half_relevant(self):
        """Half are relevant -> P@K = 0.5"""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc3"}
        assert precision_at_k_single(retrieved, relevant, k=4) == 0.5

    def test_none_relevant(self):
        """None are relevant -> P@K = 0.0"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4"}
        assert precision_at_k_single(retrieved, relevant, k=3) == 0.0

    def test_k_limits(self):
        """Precision respects K limit"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc3"}
        assert precision_at_k_single(retrieved, relevant, k=1) == 1.0  # 1/1
        assert precision_at_k_single(retrieved, relevant, k=2) == 0.5  # 1/2
        assert precision_at_k_single(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    def test_zero_k(self):
        """K=0 returns 0 (avoid division by zero)"""
        assert precision_at_k_single(["doc1"], {"doc1"}, k=0) == 0.0

    def test_average_precision(self):
        """Average precision across queries"""
        all_retrieved = [
            ["doc1", "doc2"],  # 1 relevant at k=2 -> P = 0.5
            ["doc3", "doc4"],  # 1 relevant at k=2 -> P = 0.5
        ]
        all_relevant = [{"doc1"}, {"doc4"}]
        assert precision_at_k(all_retrieved, all_relevant, k=2) == 0.5


class TestRecallAtK:
    """Tests for Recall@K calculation."""

    def test_all_retrieved(self):
        """All relevant docs are retrieved -> R@K = 1.0"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2"}
        assert recall_at_k_single(retrieved, relevant, k=3) == 1.0

    def test_half_retrieved(self):
        """Half of relevant retrieved -> R@K = 0.5"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc4"}  # Only doc1 in top-3
        assert recall_at_k_single(retrieved, relevant, k=3) == 0.5

    def test_none_retrieved(self):
        """No relevant retrieved -> R@K = 0.0"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        assert recall_at_k_single(retrieved, relevant, k=3) == 0.0

    def test_no_relevant_docs(self):
        """No relevant docs exist -> R@K = 0.0"""
        assert recall_at_k_single(["doc1", "doc2"], set(), k=2) == 0.0

    def test_k_limits(self):
        """Recall respects K limit"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc3"}
        assert recall_at_k_single(retrieved, relevant, k=1) == 0.5  # 1/2
        assert recall_at_k_single(retrieved, relevant, k=2) == 0.5  # 1/2
        assert recall_at_k_single(retrieved, relevant, k=3) == 1.0  # 2/2

    def test_average_recall(self):
        """Average recall across queries"""
        all_retrieved = [
            ["doc1", "doc2"],  # 1/2 relevant -> R = 0.5
            ["doc3", "doc4"],  # 1/1 relevant -> R = 1.0
        ]
        all_relevant = [{"doc1", "doc5"}, {"doc4"}]
        assert recall_at_k(all_retrieved, all_relevant, k=2) == 0.75


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_perfect_retrieval(self):
        """Perfect retrieval returns all metrics = 1.0"""
        all_retrieved = [["doc1", "doc2"]]
        all_grades = [{"doc1": 1.0, "doc2": 1.0}]
        metrics = compute_all_metrics(all_retrieved, all_grades, k=2)

        assert metrics.mrr == 1.0
        assert metrics.ndcg == pytest.approx(1.0, rel=0.01)
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.k == 2
        assert metrics.num_queries == 1

    def test_no_relevant_retrieval(self):
        """No relevant docs retrieved returns all metrics = 0.0"""
        all_retrieved = [["doc1", "doc2"]]
        all_grades = [{}]  # No relevant docs
        metrics = compute_all_metrics(all_retrieved, all_grades, k=2)

        assert metrics.mrr == 0.0
        assert metrics.ndcg == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0

    def test_to_dict(self):
        """to_dict returns properly formatted dict"""
        all_retrieved = [["doc1", "doc2"]]
        all_grades = [{"doc1": 1.0}]
        metrics = compute_all_metrics(all_retrieved, all_grades, k=2)
        result = metrics.to_dict()

        assert "mrr@2" in result
        assert "ndcg@2" in result
        assert "precision@2" in result
        assert "recall@2" in result
        assert result["k"] == 2
        assert result["num_queries"] == 1

    def test_graded_relevance_for_binary_metrics(self):
        """Graded relevance > 0 is treated as relevant for binary metrics"""
        all_retrieved = [["doc1", "doc2", "doc3"]]
        # doc1 has grade 0.1 (relevant), doc2 has grade 0.0 (not in dict = not relevant)
        all_grades = [{"doc1": 0.1, "doc3": 0.8}]
        metrics = compute_all_metrics(all_retrieved, all_grades, k=3)

        # Binary: doc1 and doc3 are relevant (grade > 0)
        # Precision: 2 relevant in top-3 = 2/3
        assert metrics.precision == pytest.approx(2 / 3, rel=0.01)

    def test_mismatched_lengths_raises(self):
        """Mismatched input lengths raise ValueError"""
        with pytest.raises(ValueError):
            compute_all_metrics([["doc1"]], [{"doc1": 1.0}, {"doc2": 1.0}], k=2)

    def test_empty_input_raises(self):
        """Empty input raises ValueError"""
        with pytest.raises(ValueError):
            compute_all_metrics([], [], k=2)


class TestRetrievalMetricsDataclass:
    """Tests for RetrievalMetrics dataclass."""

    def test_to_dict_rounds_values(self):
        """to_dict rounds values to 4 decimal places"""
        metrics = RetrievalMetrics(
            mrr=0.123456789,
            ndcg=0.987654321,
            precision=0.555555555,
            recall=0.111111111,
            k=5,
            num_queries=10,
        )
        result = metrics.to_dict()

        assert result["mrr@5"] == 0.1235
        assert result["ndcg@5"] == 0.9877
        assert result["precision@5"] == 0.5556
        assert result["recall@5"] == 0.1111
