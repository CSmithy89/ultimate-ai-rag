from __future__ import annotations

from dataclasses import dataclass
import re

import tiktoken


COMPLEXITY_KEYWORDS = {
    "compare",
    "versus",
    "tradeoff",
    "trade-off",
    "pros",
    "cons",
    "architecture",
    "design",
    "scalability",
    "performance",
    "optimize",
    "migration",
    "security",
    "compliance",
    "explain",
    "reason",
    "strategy",
    "plan",
    "step-by-step",
}


@dataclass(frozen=True)
class RoutingDecision:
    model_id: str
    baseline_model_id: str
    complexity: str
    score: int
    reason: tuple[str, ...]


class ModelRouter:
    def __init__(
        self,
        simple_model: str,
        medium_model: str,
        complex_model: str,
        baseline_model: str,
        simple_max_score: int = 2,
        complex_min_score: int = 5,
    ) -> None:
        self._simple_model = simple_model
        self._medium_model = medium_model
        self._complex_model = complex_model
        self._baseline_model = baseline_model
        self._simple_max_score = simple_max_score
        self._complex_min_score = complex_min_score
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def route(self, query: str) -> RoutingDecision:
        tokens = len(self._encoding.encode(query)) if query else 0
        reason: list[str] = []
        score = 0

        if tokens > 60:
            score += 2
            reason.append("long_query")
        elif tokens > 25:
            score += 1
            reason.append("medium_query")

        keyword_hits = self._count_keyword_hits(query)
        if keyword_hits:
            score += min(3, keyword_hits)
            reason.append(f"keyword_hits:{keyword_hits}")

        if score <= self._simple_max_score:
            complexity = "simple"
            model_id = self._simple_model
        elif score >= self._complex_min_score:
            complexity = "complex"
            model_id = self._complex_model
        else:
            complexity = "medium"
            model_id = self._medium_model

        return RoutingDecision(
            model_id=model_id,
            baseline_model_id=self._baseline_model,
            complexity=complexity,
            score=score,
            reason=tuple(reason),
        )

    def _count_keyword_hits(self, query: str) -> int:
        if not query:
            return 0
        text = query.lower()
        hits = 0
        for keyword in COMPLEXITY_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                hits += 1
        return hits
