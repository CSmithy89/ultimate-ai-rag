# MCP Wrapper Architecture Guide

**Date:** 2026-01-03
**Version:** 1.0
**Related Epic:** Epic 14 - Connectivity

---

## Overview

This guide documents the MCP (Model Context Protocol) wrapper architecture decision made during the 2026-01-03 roadmap analysis. The key decision is to **wrap Graphiti's built-in MCP server** rather than build a duplicate, and **extend it with RAG-specific tools**.

### Architecture Decision

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED MCP SERVER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐  │
│  │   GRAPHITI MCP (Wrapped) │  │   RAG EXTENSIONS MCP         │  │
│  ├──────────────────────────┤  ├──────────────────────────────┤  │
│  │ • add_memory/add_episode │  │ • vector_search              │  │
│  │ • search_nodes           │  │ • hybrid_retrieve            │  │
│  │ • search_facts           │  │ • ingest_url                 │  │
│  │ • delete_episode         │  │ • ingest_pdf                 │  │
│  │ • clear_graph            │  │ • ingest_youtube             │  │
│  │                          │  │ • query_with_reranking       │  │
│  │ (Graph operations)       │  │ • explain_answer             │  │
│  └──────────────────────────┘  └──────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      EXTERNAL CLIENTS          │
              ├───────────────────────────────┤
              │ • Claude Desktop              │
              │ • Cursor IDE                  │
              │ • VS Code + Continue          │
              │ • Custom Agents               │
              └───────────────────────────────┘
```

---

## Decision Rationale

### What Graphiti MCP Already Provides

From [Graphiti's MCP server documentation](https://deepwiki.com/getzep/graphiti):

| Tool | Description |
|------|-------------|
| `add_memory` / `add_episode` | Ingest knowledge into temporal graph |
| `search_nodes` | Find entities in the knowledge graph |
| `search_facts` | Find relationships/facts between entities |
| `delete_episode` | Remove specific knowledge |
| `clear_graph` | Reset the entire graph |

### What Our RAG System Adds

| Tool | Description | Why Not in Graphiti |
|------|-------------|---------------------|
| `vector_search` | pgvector semantic search | Different database |
| `hybrid_retrieve` | Combined vector + graph | Our orchestration logic |
| `ingest_url` | Crawl4AI/Apify web ingestion | Our ingestion pipeline |
| `ingest_pdf` | Docling PDF processing | Our ingestion pipeline |
| `ingest_youtube` | YouTube transcript extraction | Our ingestion pipeline |
| `query_with_reranking` | Cross-encoder reranked results | Epic 12 feature |
| `explain_answer` | Trajectory/explainability | Our observability layer |

### Why Wrap, Not Duplicate

1. **DRY Principle:** Graphiti already has a tested, maintained MCP server
2. **Future-Proof:** Graphiti updates automatically flow through
3. **Reduced Maintenance:** One less codebase to maintain
4. **Feature Addition:** Easy to add RAG-specific tools alongside

---

## Implementation Architecture

### Server Structure

```python
# backend/src/agentic_rag_backend/mcp/server.py

from mcp.server import Server
from graphiti_core.mcp import GraphitiMCPServer

class UnifiedMCPServer:
    """
    Wraps Graphiti MCP and extends with RAG-specific tools.
    """

    def __init__(
        self,
        graphiti_client: Graphiti,
        vector_store: VectorStore,
        ingestion_service: IngestionService,
        retrieval_service: RetrievalService
    ):
        self.server = Server("agentic-rag-mcp")
        self.graphiti_mcp = GraphitiMCPServer(graphiti_client)

        # Register Graphiti tools (wrapped)
        self._register_graphiti_tools()

        # Register RAG extension tools
        self._register_rag_tools()

    def _register_graphiti_tools(self):
        """Proxy Graphiti MCP tools through our server."""
        for tool in self.graphiti_mcp.tools:
            self.server.register_tool(
                name=tool.name,
                description=tool.description,
                handler=tool.handler
            )

    def _register_rag_tools(self):
        """Register RAG-specific extension tools."""

        @self.server.tool(
            name="vector_search",
            description="Search for documents using semantic similarity"
        )
        async def vector_search(query: str, top_k: int = 10) -> list[Document]:
            return await self.vector_store.search(query, top_k=top_k)

        @self.server.tool(
            name="hybrid_retrieve",
            description="Combined vector + graph retrieval with optional reranking"
        )
        async def hybrid_retrieve(
            query: str,
            use_reranking: bool = True,
            top_k: int = 10
        ) -> RetrievalResult:
            return await self.retrieval_service.hybrid_retrieve(
                query,
                use_reranking=use_reranking,
                top_k=top_k
            )

        @self.server.tool(
            name="ingest_url",
            description="Crawl and ingest a URL into the knowledge base"
        )
        async def ingest_url(
            url: str,
            depth: int = 1,
            provider: str = "crawl4ai"
        ) -> IngestionResult:
            return await self.ingestion_service.ingest_url(
                url,
                depth=depth,
                provider=provider
            )

        @self.server.tool(
            name="ingest_pdf",
            description="Parse and ingest a PDF document"
        )
        async def ingest_pdf(file_path: str) -> IngestionResult:
            return await self.ingestion_service.ingest_pdf(file_path)

        @self.server.tool(
            name="ingest_youtube",
            description="Extract transcript and ingest a YouTube video"
        )
        async def ingest_youtube(video_url: str) -> IngestionResult:
            return await self.ingestion_service.ingest_youtube(video_url)

        @self.server.tool(
            name="query_with_reranking",
            description="Query with explicit reranking control"
        )
        async def query_with_reranking(
            query: str,
            reranker: str = "flashrank",
            top_k: int = 10
        ) -> RetrievalResult:
            return await self.retrieval_service.query_with_reranking(
                query,
                reranker=reranker,
                top_k=top_k
            )

        @self.server.tool(
            name="explain_answer",
            description="Get explanation of how an answer was derived"
        )
        async def explain_answer(query_id: str) -> ExplanationResult:
            return await self.retrieval_service.get_explanation(query_id)
```

---

## Tool Reference

### Graphiti Tools (Wrapped)

These tools are passed through from Graphiti's MCP server:

#### `add_memory` / `add_episode`

Ingest knowledge into the temporal graph.

```json
{
  "name": "add_memory",
  "parameters": {
    "content": "The quarterly revenue was $5.2M",
    "source": "financial_report_q4_2024.pdf",
    "timestamp": "2024-12-31T00:00:00Z"
  }
}
```

#### `search_nodes`

Find entities in the knowledge graph.

```json
{
  "name": "search_nodes",
  "parameters": {
    "query": "revenue",
    "limit": 10
  }
}
```

#### `search_facts`

Find relationships between entities.

```json
{
  "name": "search_facts",
  "parameters": {
    "query": "revenue growth 2024"
  }
}
```

### RAG Extension Tools

These tools are added by our wrapper:

#### `vector_search`

Semantic similarity search using pgvector.

```json
{
  "name": "vector_search",
  "parameters": {
    "query": "How to configure authentication?",
    "top_k": 10
  }
}
```

**Returns:**
```json
{
  "results": [
    {
      "content": "Authentication is configured via...",
      "score": 0.89,
      "source": "docs/auth.md",
      "chunk_id": "chunk-123"
    }
  ]
}
```

#### `hybrid_retrieve`

Combined vector + graph retrieval with optional reranking.

```json
{
  "name": "hybrid_retrieve",
  "parameters": {
    "query": "What is the relationship between User and Project?",
    "use_reranking": true,
    "top_k": 10
  }
}
```

**Returns:**
```json
{
  "vector_results": [...],
  "graph_results": [...],
  "reranked_results": [...],
  "synthesis": "The User entity is connected to Project via..."
}
```

#### `ingest_url`

Crawl and ingest a URL.

```json
{
  "name": "ingest_url",
  "parameters": {
    "url": "https://docs.example.com/",
    "depth": 2,
    "provider": "crawl4ai"
  }
}
```

**Provider Options:**
- `crawl4ai` (default): Open-source crawler
- `apify`: Enterprise fallback
- `brightdata`: Enterprise fallback

#### `ingest_youtube`

Extract and ingest YouTube transcript.

```json
{
  "name": "ingest_youtube",
  "parameters": {
    "video_url": "https://www.youtube.com/watch?v=abc123"
  }
}
```

---

## Configuration

### Environment Variables

```bash
# MCP Server Configuration
MCP_SERVER_PORT=8080
MCP_AUTH_ENABLED=true
MCP_API_KEY=your-mcp-api-key

# Rate Limiting
MCP_RATE_LIMIT_RPM=60
MCP_RATE_LIMIT_BURST=10

# Logging
MCP_LOG_REQUESTS=true
MCP_LOG_RESPONSES=false  # Avoid logging sensitive data
```

### Client Configuration

#### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "agentic-rag": {
      "command": "uvx",
      "args": ["agentic-rag-mcp"],
      "env": {
        "MCP_API_KEY": "your-api-key"
      }
    }
  }
}
```

#### Cursor IDE (`.cursor/mcp.json`)

```json
{
  "servers": {
    "agentic-rag": {
      "url": "http://localhost:8080/mcp",
      "apiKey": "your-api-key"
    }
  }
}
```

---

## Security Considerations

### Authentication

All MCP endpoints require API key authentication:

```python
@self.server.middleware
async def auth_middleware(request, next):
    api_key = request.headers.get("X-API-Key")
    if not verify_api_key(api_key):
        raise UnauthorizedError("Invalid API key")
    return await next(request)
```

### Rate Limiting

Protect against abuse with configurable rate limits:

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_api_key_from_request)

@limiter.limit("60/minute")
async def ingest_url(url: str) -> IngestionResult:
    ...
```

### Tenant Isolation

All tools enforce tenant isolation:

```python
@self.server.tool(name="vector_search")
async def vector_search(query: str, top_k: int = 10) -> list[Document]:
    tenant_id = get_current_tenant()
    return await self.vector_store.search(
        query,
        top_k=top_k,
        filters={"tenant_id": tenant_id}
    )
```

---

## Testing

### Integration Test Example

```python
import pytest
from agentic_rag_backend.mcp.server import UnifiedMCPServer

@pytest.mark.integration
async def test_hybrid_retrieve_tool():
    server = UnifiedMCPServer(...)

    result = await server.call_tool(
        "hybrid_retrieve",
        {"query": "test query", "use_reranking": True}
    )

    assert "vector_results" in result
    assert "graph_results" in result
    assert len(result["reranked_results"]) <= 10
```

---

## References

- [Graphiti MCP Server](https://deepwiki.com/getzep/graphiti#8)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- `_bmad-output/implementation-artifacts/epic-14-tech-spec.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/guides/advanced-retrieval-configuration.md`
