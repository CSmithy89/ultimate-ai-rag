from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import List
from uuid import UUID

from .retrieval_router import RetrievalStrategy, select_retrieval_strategy
from .schemas import PlanStep
from .trajectory import TrajectoryLogger

try:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
except ImportError:  # pragma: no cover - optional dependency at runtime
    Agent = None
    OpenAIChat = None

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    answer: str
    plan: List[PlanStep]
    thoughts: List[str]
    retrieval_strategy: RetrievalStrategy
    trajectory_id: UUID | None


class OrchestratorAgent:
    """Run orchestration flow for user queries."""

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

    def run(
        self, query: str, tenant_id: str, session_id: str | None = None
    ) -> OrchestratorResult:
        """Run the orchestrator for a query and return the response payload."""
        trajectory_id = (
            self._logger.start_trajectory(tenant_id, session_id)
            if self._logger
            else None
        )
        plan = self._build_plan(query)
        logger.debug("Generated plan with %s steps", len(plan))
        thoughts = self._execute_plan(tenant_id, plan, trajectory_id)
        strategy = select_retrieval_strategy(query)
        strategy_note = f"Selected retrieval strategy: {strategy.value}"
        thoughts.append(strategy_note)
        self._log_action(tenant_id, trajectory_id, strategy_note)
        logger.debug("Retrieval strategy selected: %s", strategy.value)

        if self._agent:
            response = self._agent.run(query)
            content = getattr(response, "content", response)
            answer = str(content)
        else:
            answer = f"Received query: {query}"

        self._log_observation(
            tenant_id, trajectory_id, f"Generated response ({len(answer)} chars)"
        )

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
        if self._has_token(query, "compare") or self._has_token(query, "versus"):
            base_steps.insert(2, "Break down into sub-questions")

        plan = [PlanStep(step=step, status="pending") for step in base_steps]

        if self._has_token(query, "if") or self._has_token(query, "depending"):
            plan.insert(
                3,
                PlanStep(
                    step="Refine plan based on intermediate signals", status="pending"
                ),
            )

        return plan

    def _has_token(self, query: str, token: str) -> bool:
        return re.search(rf"\\b{re.escape(token)}\\b", query.lower()) is not None

    def _execute_plan(
        self, tenant_id: str, plan: List[PlanStep], trajectory_id: UUID | None
    ) -> List[str]:
        thoughts: List[str] = []
        for step in plan:
            thought = f"Plan step: {step.step}"
            thoughts.append(thought)
            self._log_thought(tenant_id, trajectory_id, thought)
            step.status = "completed"
        return thoughts

    def _log_thought(self, tenant_id: str, trajectory_id: UUID | None, content: str) -> None:
        if self._logger and trajectory_id:
            self._logger.log_thought(tenant_id, trajectory_id, content)

    def _log_action(self, tenant_id: str, trajectory_id: UUID | None, content: str) -> None:
        if self._logger and trajectory_id:
            self._logger.log_action(tenant_id, trajectory_id, content)

    def _log_observation(
        self, tenant_id: str, trajectory_id: UUID | None, content: str
    ) -> None:
        if self._logger and trajectory_id:
            self._logger.log_observation(tenant_id, trajectory_id, content)
