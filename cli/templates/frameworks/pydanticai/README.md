# PydanticAI Starter

This starter connects to the Agentic RAG platform via A2A and MCP.

## Setup

```bash
pip install pydantic-ai httpx
export RAG_BASE_URL=http://localhost:8000
```

## Run

```bash
python agent.py
```

## Files

- `agent.py`: PydanticAI agent that delegates to A2A.
- `mcp_client.py`: Example MCP tool call helper.

## Troubleshooting

- **Backend not reachable:** Confirm `RAG_BASE_URL` points to a running backend (`/health` should return 200).
- **Auth errors:** Ensure your API key environment variables are set before running the agent.
- **MCP tool failures:** Verify the backend is running with MCP enabled and check the server logs.
