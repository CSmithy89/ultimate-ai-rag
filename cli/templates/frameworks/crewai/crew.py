"""CrewAI crew that delegates to RAG via MCP protocol."""

import os
import httpx
from crewai import Agent, Crew, Task
from crewai.tools import tool

from tasks import create_research_task

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")


@tool("Search RAG Knowledge Base")
def search_knowledge(query: str, tenant_id: str = "default") -> str:
    """Search the knowledge graph and vector store for relevant information."""
    payload = {"tool": "knowledge.query", "arguments": {"query": query, "tenant_id": tenant_id}}
    response = httpx.post(f"{RAG_BASE_URL}/api/v1/mcp/call", json=payload, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    return data.get("result", "")


researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert at finding information",
    tools=[search_knowledge],
)

summarizer = Agent(
    role="Summarizer",
    goal="Summarize research findings",
    backstory="Clear and concise summarizer",
)


def main() -> None:
    task = create_research_task(researcher)
    summary_task = Task(
        description="Summarize the findings",
        expected_output="Concise summary",
        agent=summarizer,
        context=[task],
    )
    crew = Crew(agents=[researcher, summarizer], tasks=[task, summary_task])
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()
