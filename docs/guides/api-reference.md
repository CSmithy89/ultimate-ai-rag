# API Reference Guide

The Agentic RAG Backend provides a comprehensive REST API for knowledge retrieval, document ingestion, and AI agent orchestration. This guide explains how to access and use the API documentation.

## Interactive API Documentation

FastAPI automatically generates interactive API documentation from the OpenAPI specification. These are available when the backend server is running:

### Swagger UI (`/docs`)

**URL:** `http://localhost:8000/docs`

Swagger UI provides an interactive interface to:
- Browse all available endpoints grouped by tags
- View request/response schemas
- Test endpoints directly in the browser
- Authenticate and test protected endpoints

### ReDoc (`/redoc`)

**URL:** `http://localhost:8000/redoc`

ReDoc provides a clean, readable API reference with:
- Nested schema documentation
- Code samples for request bodies
- Search functionality
- Responsive design for easy reading

### OpenAPI JSON Schema (`/openapi.json`)

**URL:** `http://localhost:8000/openapi.json`

The raw OpenAPI 3.x specification in JSON format. Use this for:
- Generating client SDKs
- Importing into API tools (Postman, Insomnia)
- CI/CD validation
- Custom documentation generation

## Exporting OpenAPI Schema

For offline documentation or CI/CD pipelines, export the schema without starting the server:

```bash
# Export to stdout (JSON)
cd backend && uv run python scripts/export_openapi.py

# Export to file with pretty-printing
cd backend && uv run python scripts/export_openapi.py --pretty --output ../docs/openapi.json

# Export as YAML (requires pyyaml)
cd backend && uv run python scripts/export_openapi.py --format yaml --output ../docs/openapi.yaml
```

## API Overview

The API is organized into the following endpoint groups:

### Core Query Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (no auth required) |
| `/query` | POST | Execute agentic RAG query with reasoning |

### Ingestion (`/api/v1/ingest`)

Document and URL ingestion pipeline:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ingest/url` | POST | Start URL crawl job |
| `/api/v1/ingest/document` | POST | Upload PDF document |
| `/api/v1/ingest/jobs/{job_id}` | GET | Get job status |
| `/api/v1/ingest/jobs` | GET | List jobs for tenant |

### Knowledge Graph (`/api/v1/knowledge`)

Graph visualization and temporal queries:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/knowledge/graph` | GET | Get graph nodes and edges |
| `/api/v1/knowledge/stats` | GET | Get graph statistics |
| `/api/v1/knowledge/orphans` | GET | Get orphan nodes |
| `/api/v1/knowledge/temporal/search` | POST | Temporal search (point-in-time) |
| `/api/v1/knowledge/temporal/changes` | GET | Get knowledge changes over time |

### CopilotKit Integration (`/api/v1/copilot`)

AG-UI protocol endpoints for CopilotKit:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/copilot` | POST | AG-UI streaming handler |
| `/api/v1/copilot/validation/{validation_id}` | POST | HITL validation response |
| `/api/v1/copilot/transcribe` | POST | Speech-to-text (Voice I/O) |
| `/api/v1/copilot/synthesize` | POST | Text-to-speech (Voice I/O) |

### MCP Tools (`/api/v1/mcp`)

Model Context Protocol tool discovery and invocation:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/mcp/tools` | GET | List available tools |
| `/api/v1/mcp/call` | POST | Invoke a tool |
| `/api/v1/mcp/ui/config` | GET | Get MCP-UI configuration |

### A2A Protocol (`/api/v1/a2a`)

Agent-to-Agent collaboration:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/a2a/sessions` | POST | Create A2A session |
| `/api/v1/a2a/sessions/{session_id}` | GET | Get session details |
| `/api/v1/a2a/sessions/{session_id}/message` | POST | Send message |
| `/api/v1/a2a/agents` | POST | Register agent |
| `/api/v1/a2a/agents` | GET | List registered agents |
| `/api/v1/a2a/agents/{agent_id}` | GET | Get agent details |
| `/api/v1/a2a/tasks` | POST | Delegate task |
| `/api/v1/a2a/tasks/{task_id}` | GET | Get task status |

### Memory Platform (`/api/v1/memories`)

Scoped memory management:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memories` | GET | List memories |
| `/api/v1/memories` | POST | Store memory |
| `/api/v1/memories/{memory_id}` | GET | Get memory |
| `/api/v1/memories/{memory_id}` | DELETE | Delete memory |
| `/api/v1/memories/search` | POST | Search memories |

### Graph Intelligence

Community detection and advanced retrieval:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/communities/detect` | POST | Detect communities |
| `/api/v1/communities/{community_id}` | GET | Get community details |
| `/api/v1/lazy-rag/query` | POST | LazyRAG query |
| `/api/v1/query-router/route` | POST | Route query type |
| `/api/v1/dual-level/retrieve` | POST | Dual-level retrieval |

### Operations (`/api/v1/ops`)

Cost tracking and observability:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ops/costs` | GET | Get cost metrics |
| `/api/v1/ops/trajectories` | GET | List agent trajectories |
| `/api/v1/ops/trajectories/{trajectory_id}` | GET | Get trajectory details |

### Codebase Intelligence (`/api/v1/codebase`)

Code-aware validation and indexing:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/codebase/index` | POST | Index repository |
| `/api/v1/codebase/validate` | POST | Validate code references |
| `/api/v1/codebase/symbols` | GET | Search symbols |

### Telemetry (`/api/v1/telemetry`)

Client-side telemetry and metrics:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/telemetry/events` | POST | Record telemetry events |

### Prometheus Metrics (`/metrics`)

Prometheus-compatible metrics endpoint (when enabled):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Prometheus metrics |

## Authentication

The API uses tenant-based multi-tenancy. Most endpoints require a `tenant_id` parameter:

- **Query Parameter:** `?tenant_id=your-tenant-id`
- **Request Body:** `{"tenant_id": "your-tenant-id", ...}`
- **Header (A2A):** `X-A2A-API-Key: your-api-key`

### Rate Limiting

Endpoints are rate-limited per tenant. Default limits:
- Query endpoints: 60 requests/minute
- Ingestion endpoints: 10 requests/minute (URL), 5 requests/minute (document)
- Knowledge endpoints: 30-60 requests/minute

Rate limit headers are returned in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining in window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Response Format

### Success Response

All successful responses follow a standard envelope:

```json
{
  "data": {
    // Response payload
  },
  "meta": {
    "requestId": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-01-12T10:30:00Z"
  }
}
```

### Error Response (RFC 7807)

Errors follow the RFC 7807 Problem Details format:

```json
{
  "type": "https://api.example.com/errors/validation-error",
  "title": "Validation Error",
  "status": 422,
  "detail": "The 'query' field is required",
  "instance": "/api/v1/query"
}
```

## Example Usage

### Query with curl

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main features of GraphRAG?",
    "tenant_id": "demo-tenant",
    "session_id": "session-123"
  }'
```

### Query with Python

```python
import httpx

async def query_rag(query: str, tenant_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/query",
            json={
                "query": query,
                "tenant_id": tenant_id,
            },
        )
        response.raise_for_status()
        return response.json()

# Usage
result = await query_rag("What is GraphRAG?", "demo-tenant")
print(result["data"]["answer"])
```

### Ingest URL with Python

```python
import httpx

async def ingest_url(url: str, tenant_id: str, max_depth: int = 2) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/ingest/url",
            json={
                "url": url,
                "tenant_id": tenant_id,
                "max_depth": max_depth,
                "options": {
                    "extract_links": True,
                    "follow_external_links": False,
                }
            },
        )
        response.raise_for_status()
        return response.json()

# Start crawl job
result = await ingest_url("https://docs.example.com", "demo-tenant")
job_id = result["data"]["job_id"]
```

### MCP Tool Invocation

```python
import httpx

async def call_mcp_tool(tool_name: str, arguments: dict, tenant_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        # List available tools
        tools_response = await client.get(
            "http://localhost:8000/api/v1/mcp/tools",
            params={"tenant_id": tenant_id},
        )
        tools = tools_response.json()["tools"]

        # Call specific tool
        response = await client.post(
            "http://localhost:8000/api/v1/mcp/call",
            json={
                "tool": tool_name,
                "arguments": arguments,
            },
            params={"tenant_id": tenant_id},
        )
        response.raise_for_status()
        return response.json()

# Example: vector search
result = await call_mcp_tool(
    "vector_search",
    {"query": "machine learning", "top_k": 5},
    "demo-tenant"
)
```

## SDK Generation

Generate client SDKs from the OpenAPI schema:

### TypeScript/JavaScript (openapi-typescript)

```bash
# Install generator
npm install -g openapi-typescript

# Generate types
openapi-typescript http://localhost:8000/openapi.json -o src/api-types.ts
```

### Python (openapi-python-client)

```bash
# Install generator
pip install openapi-python-client

# Generate client
openapi-python-client generate --url http://localhost:8000/openapi.json
```

### Go (oapi-codegen)

```bash
# Install generator
go install github.com/deepmap/oapi-codegen/cmd/oapi-codegen@latest

# Generate client
oapi-codegen -package api http://localhost:8000/openapi.json > api/client.go
```

## Related Documentation

- [Provider Configuration Guide](./provider-configuration.md) - LLM and embedding provider setup
- [Advanced Retrieval Configuration](./advanced-retrieval-configuration.md) - Reranking and grading options
- [Protocol Integration Guide](./protocol-integration/overview.md) - MCP, A2A, AG-UI protocols
- [Memory Platform Guide](./memory-platform.md) - Memory scopes and consolidation
- [Observability Guide](./observability.md) - Metrics and monitoring
