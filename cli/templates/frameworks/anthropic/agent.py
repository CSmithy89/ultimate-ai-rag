"""Anthropic SDK agent with MCP tool access."""

import os
import httpx
from anthropic import Anthropic

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    if "tenant_id" not in arguments:
        arguments["tenant_id"] = "default"
    payload = {"tool": tool_name, "arguments": arguments}
    response = httpx.post(f"{RAG_BASE_URL}/api/v1/mcp/call", json=payload, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    return data.get("result", "")


def main() -> None:
    tools = [
        {
            "name": "knowledge.query",
            "description": "Search the RAG knowledge base",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        tools=tools,
        messages=[{"role": "user", "content": "Search for GraphRAG"}],
    )

    if response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")
        tool_result = call_mcp_tool(tool_use.name, tool_use.input)
        print(tool_result)
    else:
        print(response.content[0].text)


if __name__ == "__main__":
    main()
