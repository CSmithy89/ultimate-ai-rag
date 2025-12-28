from __future__ import annotations

from dataclasses import dataclass
from typing import List
from uuid import UUID

from .retrieval_router import RetrievalStrategy, select_retrieval_strategy
from .trajectory import TrajectoryLogger

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
    trajectory_id: UUID | None


class OrchestratorAgent:
    def __init__(
        self,
        api_key: str,
        model_id: str = "gpt-4o-mini",
        logger: TrajectoryLogger | None = None,
    ) -> None:
        self._agent = None
        self._logger = logger
        if Agent and OpenAIChat:
            self._agent = Agent(model=OpenAIChat(api_key=api_key, id=model_id))

    def run(self, query: str, session_id: str | None = None) -> OrchestratorResult:
        trajectory_id = (
            self._logger.start_trajectory(session_id) if self._logger else None
        )
        plan = self._build_plan(query)
        thoughts = self._execute_plan(plan, trajectory_id)
        strategy = select_retrieval_strategy(query)
        strategy_note = f"Selected retrieval strategy: {strategy.value}"
        thoughts.append(strategy_note)
        self._log_action(trajectory_id, strategy_note)

        if self._agent:
            response = self._agent.run(query)
            content = getattr(response, "content", response)
            answer = str(content)
        else:
            answer = f"Received query: {query}"

        self._log_observation(trajectory_id, f"Generated response ({len(answer)} chars)")

        return OrchestratorResult(
            answer=answer,
            plan=plan,
            thoughts=thoughts,
            retrieval_strategy=strategy,
            trajectory_id=trajectory_id,
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

    def _execute_plan(self, plan: List[PlanStep], trajectory_id: UUID | None) -> List[str]:
        thoughts: List[str] = []
        for step in plan:
            thought = f"Plan step: {step.step}"
            thoughts.append(thought)
            self._log_thought(trajectory_id, thought)
            step.status = "completed"
        return thoughts

    def _log_thought(self, trajectory_id: UUID | None, content: str) -> None:
        if self._logger and trajectory_id:
            self._logger.log_thought(trajectory_id, content)

    def _log_action(self, trajectory_id: UUID | None, content: str) -> None:
        if self._logger and trajectory_id:
            self._logger.log_action(trajectory_id, content)

    def _log_observation(self, trajectory_id: UUID | None, content: str) -> None:
        if self._logger and trajectory_id:
            self._logger.log_observation(trajectory_id, content)
