from __future__ import annotations

from dataclasses import dataclass
import asyncio
import logging
import re
from uuid import UUID

from ..retrieval_router import RetrievalStrategy, select_retrieval_strategy
from ..schemas import PlanStep
from ..trajectory import EventType, TrajectoryLogger

try:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
except ImportError:  # pragma: no cover - optional dependency at runtime
    Agent = None
    OpenAIChat = None

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Result payload returned by the orchestrator."""
    answer: str
    plan: list[PlanStep]
    thoughts: list[str]
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

    async def run(
        self, query: str, tenant_id: str, session_id: str | None = None
    ) -> OrchestratorResult:
        """Run the orchestrator for a query and return the response payload."""
        trajectory_id = (
            await self._logger.start_trajectory(tenant_id, session_id)
            if self._logger
            else None
        )
        plan = self._build_plan(query)
        logger.debug("Generated plan with %s steps", len(plan))
        completed_plan, thoughts, events = self._execute_plan(plan)
        strategy = select_retrieval_strategy(query)
        strategy_note = f"Selected retrieval strategy: {strategy.value}"
        thoughts.append(strategy_note)
        events.append((EventType.ACTION, strategy_note))
        logger.debug("Retrieval strategy selected: %s", strategy.value)

        if self._agent:
            response = await asyncio.to_thread(self._agent.run, query)
            content = getattr(response, "content", response)
            answer = str(content)
        else:
            logger.warning("Agno agent unavailable; returning placeholder response.")
            answer = f"Received query: {query}"

        events.append(
            (EventType.OBSERVATION, f"Generated response ({len(answer)} chars)")
        )

        if self._logger and trajectory_id:
            await self._logger.log_events(tenant_id, trajectory_id, events)

        return OrchestratorResult(
            answer=answer,
            plan=completed_plan,
            thoughts=thoughts,
            retrieval_strategy=strategy,
            trajectory_id=trajectory_id,
        )

    def _build_plan(self, query: str) -> list[PlanStep]:
        base_steps = [
            "Understand the question intent",
            "Select retrieval strategy",
            "Gather evidence",
            "Synthesize response",
        ]
        if self._has_token(query, "compare") or self._has_token(query, "versus"):
            base_steps = self._insert_after(
                base_steps,
                "Understand the question intent",
                "Break down into sub-questions",
            )

        if self._has_token(query, "if") or self._has_token(query, "depending"):
            base_steps = self._insert_after(
                base_steps,
                "Gather evidence",
                "Refine plan based on intermediate signals",
            )

        return [PlanStep(step=step, status="pending") for step in base_steps]

    def _has_token(self, query: str, token: str) -> bool:
        """Return True when a token appears as a whole word in the query."""
        return re.search(rf"\b{re.escape(token)}\b", query.lower()) is not None

    def _insert_after(self, steps: list[str], anchor: str, new_step: str) -> list[str]:
        """Insert a step after the anchor label when present."""
        if anchor not in steps:
            return steps + [new_step]
        index = steps.index(anchor)
        return steps[: index + 1] + [new_step] + steps[index + 1 :]

    def _execute_plan(
        self, plan: list[PlanStep]
    ) -> tuple[list[PlanStep], list[str], list[tuple[EventType, str]]]:
        """Execute a plan and return updated steps, thoughts, and log events."""
        thoughts: list[str] = []
        events: list[tuple[EventType, str]] = []
        completed_plan: list[PlanStep] = []
        for step in plan:
            thought = f"Plan step: {step.step}"
            thoughts.append(thought)
            events.append((EventType.THOUGHT, thought))
            completed_plan.append(PlanStep(step=step.step, status="completed"))
        return completed_plan, thoughts, events
