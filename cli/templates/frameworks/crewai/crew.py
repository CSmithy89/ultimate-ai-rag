"""CrewAI crew that delegates to RAG via A2A protocol."""

import os
from crewai import Agent, Crew, Task

from tasks import create_research_task

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")

researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert at finding information",
    a2a_agents=[
        {
            "url": f"{RAG_BASE_URL}/api/v1/a2a",
            "name": "rag_knowledge_base",
            "description": "Search the knowledge graph and vector store",
        }
    ],
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
