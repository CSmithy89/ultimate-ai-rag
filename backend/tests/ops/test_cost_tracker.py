"""Unit tests for cost tracking logic."""

from decimal import Decimal

import tiktoken

from agentic_rag_backend.ops.cost_tracker import CostTracker, DEFAULT_PRICING, _load_pricing


class DummyPool:
    """Minimal pool stub for CostTracker construction."""


def test_cost_calculation_precision() -> None:
    tracker = CostTracker(pool=DummyPool())
    input_cost, output_cost, total_cost = tracker._calculate_costs(
        "gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=2000,
    )

    assert input_cost == Decimal("0.00015")
    assert output_cost == Decimal("0.00120")
    assert total_cost == Decimal("0.00135")


def test_pricing_json_parsing() -> None:
    pricing = _load_pricing('{"custom": {"input_per_1k": "1.5", "output_per_1k": "2.5"}}')

    assert pricing["custom"].input_per_1k == Decimal("1.5")
    assert pricing["custom"].output_per_1k == Decimal("2.5")


def test_pricing_invalid_json_falls_back() -> None:
    pricing = _load_pricing("not-json")
    assert pricing["gpt-4o-mini"] == DEFAULT_PRICING["gpt-4o-mini"]


def test_token_estimation_matches_tiktoken() -> None:
    tracker = CostTracker(pool=DummyPool())
    text = "Quick brown fox jumps."
    expected = len(tiktoken.get_encoding("cl100k_base").encode(text))

    assert tracker._estimate_tokens(text, model_id="gpt-4o-mini") == expected


def test_token_estimation_fallback_for_unknown_model() -> None:
    tracker = CostTracker(pool=DummyPool())
    text = "one two three four"

    assert tracker._estimate_tokens(text, model_id="claude-3") == 6
