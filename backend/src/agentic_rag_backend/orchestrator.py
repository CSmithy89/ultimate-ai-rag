from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .retrieval_router import RetrievalStrategy, select_retrieval_strategy

try:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
except ImportError:  # pragma: no cover - optional dependency at runtime
    Agent = None
    OpenAIChat = None


@dataclass
class PlanStep:
    step: str
    status: str


@dataclass
class OrchestratorResult:
    answer: str
    plan: List[PlanStep]
    thoughts: List[str]
    retrieval_strategy: RetrievalStrategy


class OrchestratorAgent:
    def __init__(self, api_key: str, model_id: str = "gpt-4o-mini") -> None:
        self._agent = None
        if Agent and OpenAIChat:
            self._agent = Agent(model=OpenAIChat(api_key=api_key, id=model_id))

    def run(self, query: str) -> OrchestratorResult:
        plan = self._build_plan(query)
        thoughts = self._execute_plan(plan)
        strategy = select_retrieval_strategy(query)
        thoughts.append(f"Selected retrieval strategy: {strategy.value}")

        if self._agent:
            response = self._agent.run(query)
            content = getattr(response, "content", response)
            answer = str(content)
        else:
            answer = f"Received query: {query}"

        return OrchestratorResult(
            answer=answer,
            plan=plan,
            thoughts=thoughts,
            retrieval_strategy=strategy,
        )

    def _build_plan(self, query: str) -> List[PlanStep]:
        base_steps = [
            "Understand the question intent",
            "Select retrieval strategy",
            "Gather evidence",
            "Synthesize response",
        ]
        if "compare" in query.lower() or " and " in query.lower():
            base_steps.insert(2, "Break down into sub-questions")

        plan = [PlanStep(step=step, status="pending") for step in base_steps]

        if "if" in query.lower() or "depending" in query.lower():
            plan.insert(3, PlanStep(step="Refine plan based on intermediate signals", status="pending"))

        return plan

    def _execute_plan(self, plan: List[PlanStep]) -> List[str]:
        thoughts: List[str] = []
        for step in plan:
            thoughts.append(f"Plan step: {step.step}")
            step.status = "completed"
        return thoughts
