from __future__ import annotations

from dataclasses import dataclass

try:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
except ImportError:  # pragma: no cover - optional dependency at runtime
    Agent = None
    OpenAIChat = None


@dataclass
class OrchestratorResult:
    answer: str


class OrchestratorAgent:
    def __init__(self, api_key: str, model_id: str = "gpt-4o-mini") -> None:
        self._agent = None
        if Agent and OpenAIChat:
            self._agent = Agent(model=OpenAIChat(api_key=api_key, id=model_id))

    def run(self, query: str) -> OrchestratorResult:
        if self._agent:
            response = self._agent.run(query)
            content = getattr(response, "content", response)
            return OrchestratorResult(answer=str(content))

        return OrchestratorResult(answer=f"Received query: {query}")
