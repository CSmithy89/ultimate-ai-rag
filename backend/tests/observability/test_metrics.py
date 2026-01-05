"""Unit tests for Prometheus retrieval metrics.

Story 19-C5: Prometheus Retrieval Metrics
"""

import pytest
from prometheus_client import CollectorRegistry

from agentic_rag_backend.observability.metrics import (
    RETRIEVAL_REQUESTS_TOTAL,
    RETRIEVAL_FALLBACK_TRIGGERED_TOTAL,
    GRADER_EVALUATIONS_TOTAL,
    RETRIEVAL_LATENCY_SECONDS,
    RERANKING_IMPROVEMENT_RATIO,
    GRADER_SCORE,
    RETRIEVAL_PRECISION,
    RETRIEVAL_RECALL,
    ACTIVE_RETRIEVAL_OPERATIONS,
    record_retrieval_request,
    record_retrieval_fallback,
    record_grader_evaluation,
    record_retrieval_latency,
    record_reranking_improvement,
    record_grader_score,
    set_retrieval_precision,
    set_retrieval_recall,
    track_active_retrieval,
    get_metrics_registry,
)


class TestMetricDefinitions:
    """Tests for metric definitions."""

    def test_retrieval_requests_total_exists(self) -> None:
        """Test that RETRIEVAL_REQUESTS_TOTAL counter is defined."""
        assert RETRIEVAL_REQUESTS_TOTAL is not None
        # Verify labels
        assert "strategy" in RETRIEVAL_REQUESTS_TOTAL._labelnames
        assert "tenant_id" in RETRIEVAL_REQUESTS_TOTAL._labelnames

    def test_retrieval_fallback_triggered_total_exists(self) -> None:
        """Test that RETRIEVAL_FALLBACK_TRIGGERED_TOTAL counter is defined."""
        assert RETRIEVAL_FALLBACK_TRIGGERED_TOTAL is not None
        assert "reason" in RETRIEVAL_FALLBACK_TRIGGERED_TOTAL._labelnames
        assert "tenant_id" in RETRIEVAL_FALLBACK_TRIGGERED_TOTAL._labelnames

    def test_grader_evaluations_total_exists(self) -> None:
        """Test that GRADER_EVALUATIONS_TOTAL counter is defined."""
        assert GRADER_EVALUATIONS_TOTAL is not None
        assert "result" in GRADER_EVALUATIONS_TOTAL._labelnames
        assert "tenant_id" in GRADER_EVALUATIONS_TOTAL._labelnames

    def test_retrieval_latency_seconds_exists(self) -> None:
        """Test that RETRIEVAL_LATENCY_SECONDS histogram is defined."""
        assert RETRIEVAL_LATENCY_SECONDS is not None
        assert "strategy" in RETRIEVAL_LATENCY_SECONDS._labelnames
        assert "phase" in RETRIEVAL_LATENCY_SECONDS._labelnames
        assert "tenant_id" in RETRIEVAL_LATENCY_SECONDS._labelnames

    def test_reranking_improvement_ratio_exists(self) -> None:
        """Test that RERANKING_IMPROVEMENT_RATIO histogram is defined."""
        assert RERANKING_IMPROVEMENT_RATIO is not None
        assert "tenant_id" in RERANKING_IMPROVEMENT_RATIO._labelnames

    def test_grader_score_exists(self) -> None:
        """Test that GRADER_SCORE histogram is defined."""
        assert GRADER_SCORE is not None
        assert "model" in GRADER_SCORE._labelnames
        assert "tenant_id" in GRADER_SCORE._labelnames

    def test_retrieval_precision_exists(self) -> None:
        """Test that RETRIEVAL_PRECISION gauge is defined."""
        assert RETRIEVAL_PRECISION is not None
        assert "strategy" in RETRIEVAL_PRECISION._labelnames
        assert "k" in RETRIEVAL_PRECISION._labelnames
        assert "tenant_id" in RETRIEVAL_PRECISION._labelnames

    def test_retrieval_recall_exists(self) -> None:
        """Test that RETRIEVAL_RECALL gauge is defined."""
        assert RETRIEVAL_RECALL is not None
        assert "strategy" in RETRIEVAL_RECALL._labelnames
        assert "k" in RETRIEVAL_RECALL._labelnames
        assert "tenant_id" in RETRIEVAL_RECALL._labelnames

    def test_active_retrieval_operations_exists(self) -> None:
        """Test that ACTIVE_RETRIEVAL_OPERATIONS gauge is defined."""
        assert ACTIVE_RETRIEVAL_OPERATIONS is not None
        assert "tenant_id" in ACTIVE_RETRIEVAL_OPERATIONS._labelnames


class TestHelperFunctions:
    """Tests for metric helper functions."""

    def test_record_retrieval_request(self) -> None:
        """Test recording a retrieval request."""
        # Should not raise
        record_retrieval_request(strategy="vector", tenant_id="test-tenant")
        record_retrieval_request(strategy="graph", tenant_id="test-tenant")
        record_retrieval_request(strategy="hybrid", tenant_id="test-tenant")

    def test_record_retrieval_fallback(self) -> None:
        """Test recording a retrieval fallback."""
        record_retrieval_fallback(reason="low_score", tenant_id="test-tenant")
        record_retrieval_fallback(reason="empty_results", tenant_id="test-tenant")
        record_retrieval_fallback(reason="timeout", tenant_id="test-tenant")

    def test_record_grader_evaluation(self) -> None:
        """Test recording a grader evaluation."""
        record_grader_evaluation(result="pass", tenant_id="test-tenant")
        record_grader_evaluation(result="fail", tenant_id="test-tenant")
        record_grader_evaluation(result="fallback", tenant_id="test-tenant")

    def test_record_retrieval_latency(self) -> None:
        """Test recording retrieval latency."""
        record_retrieval_latency(
            strategy="vector",
            phase="embed",
            tenant_id="test-tenant",
            duration_seconds=0.1,
        )
        record_retrieval_latency(
            strategy="hybrid",
            phase="search",
            tenant_id="test-tenant",
            duration_seconds=0.5,
        )
        record_retrieval_latency(
            strategy="hybrid",
            phase="rerank",
            tenant_id="test-tenant",
            duration_seconds=0.3,
        )
        record_retrieval_latency(
            strategy="hybrid",
            phase="grade",
            tenant_id="test-tenant",
            duration_seconds=0.2,
        )

    def test_record_reranking_improvement(self) -> None:
        """Test recording reranking improvement ratio."""
        # Normal case: improvement
        record_reranking_improvement(
            tenant_id="test-tenant",
            pre_score=0.5,
            post_score=0.8,
        )
        # Edge case: no improvement
        record_reranking_improvement(
            tenant_id="test-tenant",
            pre_score=0.5,
            post_score=0.5,
        )
        # Edge case: degradation
        record_reranking_improvement(
            tenant_id="test-tenant",
            pre_score=0.8,
            post_score=0.6,
        )

    def test_record_reranking_improvement_zero_pre_score(self) -> None:
        """Test that zero pre_score does not cause division by zero."""
        # Should not raise
        record_reranking_improvement(
            tenant_id="test-tenant",
            pre_score=0.0,
            post_score=0.5,
        )

    def test_record_grader_score(self) -> None:
        """Test recording grader scores."""
        record_grader_score(model="heuristic", tenant_id="test-tenant", score=0.7)
        record_grader_score(model="cross-encoder", tenant_id="test-tenant", score=0.85)

    def test_set_retrieval_precision(self) -> None:
        """Test setting retrieval precision."""
        set_retrieval_precision(
            strategy="vector",
            k=10,
            tenant_id="test-tenant",
            precision=0.8,
        )
        set_retrieval_precision(
            strategy="hybrid",
            k=5,
            tenant_id="test-tenant",
            precision=0.9,
        )

    def test_set_retrieval_recall(self) -> None:
        """Test setting retrieval recall."""
        set_retrieval_recall(
            strategy="vector",
            k=10,
            tenant_id="test-tenant",
            recall=0.75,
        )
        set_retrieval_recall(
            strategy="hybrid",
            k=5,
            tenant_id="test-tenant",
            recall=0.85,
        )


class TestTrackActiveRetrieval:
    """Tests for the track_active_retrieval context manager."""

    def test_track_active_retrieval_increments_and_decrements(self) -> None:
        """Test that context manager properly tracks active operations."""
        tenant_id = "test-tenant-active"

        # Get initial value
        initial = ACTIVE_RETRIEVAL_OPERATIONS.labels(tenant_id=tenant_id)._value.get()

        with track_active_retrieval(tenant_id):
            # Value should be incremented inside context
            inside = ACTIVE_RETRIEVAL_OPERATIONS.labels(tenant_id=tenant_id)._value.get()
            assert inside == initial + 1

        # Value should be decremented after context
        after = ACTIVE_RETRIEVAL_OPERATIONS.labels(tenant_id=tenant_id)._value.get()
        assert after == initial

    def test_track_active_retrieval_decrements_on_exception(self) -> None:
        """Test that context manager decrements even on exception."""
        tenant_id = "test-tenant-exception"

        initial = ACTIVE_RETRIEVAL_OPERATIONS.labels(tenant_id=tenant_id)._value.get()

        with pytest.raises(ValueError):
            with track_active_retrieval(tenant_id):
                # Verify incremented
                inside = ACTIVE_RETRIEVAL_OPERATIONS.labels(tenant_id=tenant_id)._value.get()
                assert inside == initial + 1
                raise ValueError("Test exception")

        # Should still be decremented
        after = ACTIVE_RETRIEVAL_OPERATIONS.labels(tenant_id=tenant_id)._value.get()
        assert after == initial


class TestGetMetricsRegistry:
    """Tests for the registry getter."""

    def test_get_metrics_registry_returns_registry(self) -> None:
        """Test that get_metrics_registry returns a valid registry."""
        registry = get_metrics_registry()
        assert registry is not None
        assert isinstance(registry, CollectorRegistry)


class TestMetricLabels:
    """Tests for metric label validation."""

    def test_strategy_labels_accept_valid_values(self) -> None:
        """Test that strategy labels accept vector, graph, and hybrid."""
        for strategy in ["vector", "graph", "hybrid"]:
            # Should not raise
            record_retrieval_request(strategy=strategy, tenant_id="test")

    def test_phase_labels_accept_valid_values(self) -> None:
        """Test that phase labels accept embed, search, rerank, and grade."""
        for phase in ["embed", "search", "rerank", "grade"]:
            # Should not raise
            record_retrieval_latency(
                strategy="hybrid",
                phase=phase,
                tenant_id="test",
                duration_seconds=0.1,
            )

    def test_result_labels_accept_valid_values(self) -> None:
        """Test that result labels accept pass, fail, and fallback."""
        for result in ["pass", "fail", "fallback"]:
            # Should not raise
            record_grader_evaluation(result=result, tenant_id="test")

    def test_reason_labels_accept_valid_values(self) -> None:
        """Test that reason labels accept low_score, empty_results, and timeout."""
        for reason in ["low_score", "empty_results", "timeout"]:
            # Should not raise
            record_retrieval_fallback(reason=reason, tenant_id="test")
