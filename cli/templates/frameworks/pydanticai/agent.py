"""PydanticAI agent that connects to RAG via A2A."""

import os
import asyncio
from pydantic_ai import Agent
from pydantic import BaseModel

from mcp_client import query_knowledge

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")


class SearchResult(BaseModel):
    content: str


rag_agent = Agent(
    "openai:gpt-4o",
    result_type=SearchResult,
)


@rag_agent.tool
async def search_knowledge(query: str) -> str:
    return await query_knowledge(RAG_BASE_URL, query)


async def main() -> None:
    result = await rag_agent.run("What is GraphRAG?")
    print(result.data.content)


if __name__ == "__main__":
    asyncio.run(main())
