"""LangGraph workflow with MCP tool calls."""

from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from nodes import query_rag

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")


class AgentState(TypedDict):
    query: str
    context: str


def retrieve(state: AgentState) -> AgentState:
    context = query_rag(RAG_BASE_URL, state["query"])
    return {"query": state["query"], "context": context}


def respond(state: AgentState) -> AgentState:
    return state


def main() -> None:
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("respond", respond)
    graph.add_edge("retrieve", "respond")
    graph.set_entry_point("retrieve")
    graph.set_finish_point("respond")

    app = graph.compile()
    result = app.invoke({"query": "What is GraphRAG?", "context": ""})
    print(result["context"])


if __name__ == "__main__":
    main()
