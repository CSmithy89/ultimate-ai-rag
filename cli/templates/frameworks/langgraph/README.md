# LangGraph Starter

This starter shows a LangGraph flow that queries the RAG MCP server.

## Setup

```bash
pip install langgraph langchain-openai httpx
export RAG_BASE_URL=http://localhost:8000
```

## Run

```bash
python graph.py
```

## Troubleshooting

- **Backend not responding:** Check that `RAG_BASE_URL` is correct and the backend `/health` endpoint is up.
- **Auth failures:** Ensure any required API keys are exported in your shell before running.
- **Graph errors:** Confirm `langgraph` and `langchain-openai` versions are installed as listed.
