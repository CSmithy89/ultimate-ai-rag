"""Unit tests for model routing heuristics."""

from agentic_rag_backend.ops.model_router import ModelRouter


def test_routing_score_thresholds() -> None:
    router = ModelRouter(
        simple_model="simple",
        medium_model="medium",
        complex_model="complex",
        baseline_model="baseline",
        simple_max_score=0,
        complex_min_score=3,
    )

    decision_simple = router.route("hello")
    assert decision_simple.complexity == "simple"
    assert decision_simple.model_id == "simple"

    decision_medium = router.route("compare")
    assert decision_medium.complexity == "medium"
    assert decision_medium.model_id == "medium"

    decision_complex = router.route("compare architecture performance")
    assert decision_complex.complexity == "complex"
    assert decision_complex.model_id == "complex"
    assert isinstance(decision_complex.reason, tuple)
