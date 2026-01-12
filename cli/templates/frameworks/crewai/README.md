# CrewAI Starter

This starter demonstrates A2A delegation and MCP tool usage.

## Setup

```bash
pip install crewai httpx
export RAG_BASE_URL=http://localhost:8000
```

## Run

```bash
python crew.py
```

## Troubleshooting

- **Connection refused:** Start the backend and confirm `RAG_BASE_URL` points to it.
- **Tool calls fail:** Verify the MCP endpoints are reachable and the backend logs show requests.
- **Unexpected output:** Re-check `crewai` and `httpx` versions match the setup instructions.
