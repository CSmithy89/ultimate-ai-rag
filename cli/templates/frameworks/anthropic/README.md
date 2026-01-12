# Anthropic SDK Starter

This starter demonstrates using Claude with MCP tools.

## Setup

```bash
pip install anthropic httpx
export RAG_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
python agent.py
```

## Troubleshooting

- **401/403 errors:** Double-check `ANTHROPIC_API_KEY` is set and valid.
- **Backend not reachable:** Confirm `RAG_BASE_URL` points to a running backend service.
- **Tool call failures:** Review backend logs to ensure MCP endpoints are enabled and healthy.
