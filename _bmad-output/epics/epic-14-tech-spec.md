# Epic 14 Tech Spec: Connectivity (MCP Wrapper Architecture)

**Date:** 2026-01-04
**Status:** Complete
**Epic Owner:** Platform Engineering
**Related Documents:**
- `docs/guides/mcp-wrapper-architecture.md`
- `_bmad-output/architecture.md`
- `_bmad-output/project-context.md`

---

## Executive Summary

Epic 14 delivers external connectivity for the Agentic RAG platform, enabling third-party AI tools and agents to interact with the knowledge base through standardized protocols. The epic focuses on two key capabilities:

1. **MCP Server with RAG Extensions (Story 14-1):** Wrap Graphiti's MCP server and extend it with RAG-specific tools including vector search, content ingestion (URL, PDF, YouTube), and query with reranking.

2. **Robust A2A Protocol Implementation (Story 14-2):** Enhance the existing A2A protocol implementation with agent capabilities discovery, standardized task delegation, and bidirectional communication patterns.

### Strategic Context

**Key Decision (2026-01-03):** WRAP Graphiti MCP, don't duplicate - extend with RAG-specific tools.

**Rationale:**
- Graphiti already provides a tested, maintained MCP server with graph operations
- DRY principle: avoid duplicating existing functionality
- Future-proof: Graphiti updates automatically flow through the wrapper
- Easy extension: add RAG-specific tools alongside existing Graphiti tools

**Decision Document:** `docs/guides/mcp-wrapper-architecture.md`

### Target Clients

| Client | Integration Method | Primary Use Cases |
|--------|-------------------|-------------------|
| Claude Desktop | MCP via stdio/SSE | Knowledge Q&A, document ingestion |
| Cursor IDE | MCP via HTTP | Code-aware RAG queries, documentation lookup |
| VS Code + Continue | MCP via HTTP | Development assistance, API discovery |
| Custom Agents | A2A Protocol | Multi-agent orchestration, task delegation |

---

## Technical Architecture

### High-Level Architecture

```
+--------------------------------------------------------------------------------+
|                          CONNECTIVITY LAYER (Epic 14)                            |
+--------------------------------------------------------------------------------+
|                                                                                  |
|  +-------------------------------+    +------------------------------------+     |
|  |     UNIFIED MCP SERVER        |    |         A2A PROTOCOL LAYER         |     |
|  +-------------------------------+    +------------------------------------+     |
|  |                               |    |                                    |     |
|  |  +-------------------------+  |    |  +------------------------------+  |     |
|  |  | Graphiti MCP (Wrapped)  |  |    |  | Agent Discovery Service      |  |     |
|  |  | - add_memory/episode    |  |    |  | - Capabilities registry      |  |     |
|  |  | - search_nodes/facts    |  |    |  | - Health monitoring          |  |     |
|  |  | - delete_episode        |  |    |  +------------------------------+  |     |
|  |  | - clear_graph           |  |    |                                    |     |
|  |  +-------------------------+  |    |  +------------------------------+  |     |
|  |                               |    |  | Task Delegation Manager      |  |     |
|  |  +-------------------------+  |    |  | - Request routing            |  |     |
|  |  | RAG Extensions MCP      |  |    |  | - Response aggregation       |  |     |
|  |  | - vector_search         |  |    |  | - Error handling             |  |     |
|  |  | - hybrid_retrieve       |  |    |  +------------------------------+  |     |
|  |  | - ingest_url            |  |    |                                    |     |
|  |  | - ingest_pdf            |  |    |  +------------------------------+  |     |
|  |  | - ingest_youtube        |  |    |  | Session Manager (Enhanced)   |  |     |
|  |  | - query_with_reranking  |  |    |  | - Redis persistence          |  |     |
|  |  | - explain_answer        |  |    |  | - Bi-directional comms       |  |     |
|  |  +-------------------------+  |    |  +------------------------------+  |     |
|  |                               |    |                                    |     |
|  +-------------------------------+    +------------------------------------+     |
|               |                                        |                         |
|               v                                        v                         |
|  +-------------------------------+    +------------------------------------+     |
|  |  Transport Layer              |    |  Message Layer                     |     |
|  |  - HTTP/SSE (primary)         |    |  - JSON-RPC 2.0                    |     |
|  |  - stdio (Claude Desktop)     |    |  - Structured request/response     |     |
|  |  - WebSocket (optional)       |    |  - Error codes (RFC 7807)          |     |
|  +-------------------------------+    +------------------------------------+     |
|                                                                                  |
+--------------------------------------------------------------------------------+
                                        |
                                        v
+--------------------------------------------------------------------------------+
|                             EXTERNAL CLIENTS                                     |
+--------------------------------------------------------------------------------+
|  Claude Desktop | Cursor IDE | VS Code + Continue | Custom Agents              |
+--------------------------------------------------------------------------------+
```

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| MCP SDK | `mcp` Python package | Latest | MCP server implementation |
| Graphiti MCP | `graphiti-core` | 0.x | Wrapped MCP server for graph ops |
| Transport | FastAPI + SSE | 0.115+ | HTTP transport with streaming |
| A2A Protocol | Custom implementation | 1.0 | Agent-to-agent communication |
| Session Store | Redis | 7.x | Persistent A2A sessions |
| Auth | API Key + tenant isolation | N/A | Security layer |

### New Dependencies

```toml
# backend/pyproject.toml additions for Epic 14
dependencies = [
  # ... existing deps ...
  # Epic 14 - Connectivity
  "mcp>=1.0.0",                    # MCP SDK
  # graphiti-core already included (provides MCP tools)
]
```

---

## Story 14-1: Expose RAG Engine via MCP Server

### Objective

Create a unified MCP server that wraps Graphiti's MCP tools and extends with RAG-specific capabilities. This enables external AI tools (Claude Desktop, Cursor, VS Code) to interact with the knowledge base.

### Why This Matters

- **Developer Reach:** Claude Desktop has 1M+ users who could access our RAG capabilities
- **IDE Integration:** Cursor and VS Code users can query documentation directly
- **Extensibility:** MCP is becoming the standard for AI tool integration
- **Composability:** External agents can use our RAG as a knowledge source

### Tool Inventory

#### Graphiti MCP Tools (Wrapped Pass-Through)

| Tool | Description | Pass-Through Strategy |
|------|-------------|----------------------|
| `add_memory` | Add episodic memory to graph | Direct pass-through |
| `add_episode` | Ingest knowledge as episode | Direct pass-through |
| `search_nodes` | Find entities in graph | Direct pass-through |
| `search_facts` | Find relationships/facts | Direct pass-through |
| `delete_episode` | Remove specific knowledge | Direct pass-through |
| `clear_graph` | Reset entire graph | Add tenant isolation guard |

#### RAG Extension Tools (New)

| Tool | Description | Implementation |
|------|-------------|---------------|
| `vector_search` | pgvector semantic search | Uses `VectorSearchService` |
| `hybrid_retrieve` | Combined vector + graph retrieval | Uses hybrid synthesis |
| `ingest_url` | Web URL ingestion | Uses `Crawler` + Crawl4AI |
| `ingest_pdf` | PDF document ingestion | Uses `Parser` (Docling) |
| `ingest_youtube` | YouTube transcript ingestion | Uses `YouTubeIngestionService` |
| `query_with_reranking` | Query with explicit reranking | Uses `RerankerClient` |
| `explain_answer` | Get answer explanation/trajectory | Uses trajectory logging |

### Technical Design

#### Module Structure

```
backend/src/agentic_rag_backend/
+-- mcp/                              # NEW: MCP server module
|   +-- __init__.py
|   +-- server.py                     # UnifiedMCPServer class
|   +-- tools/
|   |   +-- __init__.py
|   |   +-- graphiti_wrapper.py       # Graphiti tool wrapping
|   |   +-- vector_tools.py           # vector_search, hybrid_retrieve
|   |   +-- ingestion_tools.py        # ingest_url, ingest_pdf, ingest_youtube
|   |   +-- query_tools.py            # query_with_reranking, explain_answer
|   +-- transport/
|   |   +-- __init__.py
|   |   +-- http.py                   # HTTP/SSE transport
|   |   +-- stdio.py                  # stdio transport for Claude Desktop
|   +-- auth.py                       # API key authentication
|   +-- schemas.py                    # Pydantic models for MCP I/O
```

#### Core Classes

```python
# backend/src/agentic_rag_backend/mcp/server.py
from __future__ import annotations

import asyncio
from typing import Any, Optional
import structlog

from mcp.server import Server
from mcp.server.models import Tool, TextContent
from mcp.types import ToolResult

from ..db.graphiti import GraphitiClient
from ..db.postgres import PostgresClient
from ..retrieval.vector_search import VectorSearchService
from ..retrieval.reranking import RerankerClient, create_reranker_client
from ..indexing.crawler import Crawler
from ..indexing.parser import Parser
from ..indexing.youtube_ingestion import ingest_youtube_video

logger = structlog.get_logger(__name__)


class UnifiedMCPServer:
    """
    Unified MCP server wrapping Graphiti and extending with RAG tools.

    This server follows the decision to WRAP Graphiti MCP rather than
    duplicate it, adding RAG-specific extensions for ingestion, retrieval,
    and query operations.
    """

    def __init__(
        self,
        graphiti_client: GraphitiClient,
        postgres_client: PostgresClient,
        vector_search: VectorSearchService,
        reranker: Optional[RerankerClient] = None,
        crawler: Optional[Crawler] = None,
        parser: Optional[Parser] = None,
    ):
        self.server = Server("agentic-rag-mcp")
        self._graphiti = graphiti_client
        self._postgres = postgres_client
        self._vector_search = vector_search
        self._reranker = reranker
        self._crawler = crawler
        self._parser = parser

        self._tenant_id: Optional[str] = None

        # Register all tools
        self._register_graphiti_tools()
        self._register_rag_tools()

        logger.info("mcp_server_initialized", tool_count=len(self.server.list_tools()))

    def set_tenant(self, tenant_id: str) -> None:
        """Set the current tenant context for multi-tenancy."""
        self._tenant_id = tenant_id

    def _register_graphiti_tools(self) -> None:
        """Register wrapped Graphiti MCP tools."""

        @self.server.list_tools()
        async def list_graphiti_tools() -> list[Tool]:
            # Graphiti tools are registered alongside RAG tools
            return []

        @self.server.call_tool()
        async def handle_graphiti_call(name: str, arguments: dict) -> list[TextContent]:
            if name.startswith("graphiti_"):
                # Delegate to Graphiti client
                return await self._delegate_graphiti(name, arguments)
            raise ValueError(f"Unknown tool: {name}")

    def _register_rag_tools(self) -> None:
        """Register RAG-specific extension tools."""

        # Tool: vector_search
        @self.server.tool(
            name="vector_search",
            description="Search for documents using semantic similarity via pgvector"
        )
        async def vector_search(
            query: str,
            top_k: int = 10,
            similarity_threshold: float = 0.7,
        ) -> ToolResult:
            if not self._tenant_id:
                return ToolResult(error="Tenant context not set")

            try:
                hits = await self._vector_search.search(query, self._tenant_id)
                return ToolResult(content=[
                    TextContent(type="text", text=self._format_vector_results(hits[:top_k]))
                ])
            except Exception as e:
                logger.error("vector_search_failed", error=str(e))
                return ToolResult(error=str(e))

        # Tool: hybrid_retrieve
        @self.server.tool(
            name="hybrid_retrieve",
            description="Combined vector + graph retrieval with optional reranking"
        )
        async def hybrid_retrieve(
            query: str,
            use_reranking: bool = True,
            top_k: int = 10,
        ) -> ToolResult:
            if not self._tenant_id:
                return ToolResult(error="Tenant context not set")

            try:
                # Vector search
                vector_hits = await self._vector_search.search(query, self._tenant_id)

                # Graph search via Graphiti
                graph_results = await self._graphiti.client.search(query)

                # Optional reranking
                if use_reranking and self._reranker and vector_hits:
                    reranked = await self._reranker.rerank(query, vector_hits, top_k)
                    vector_hits = [r.hit for r in reranked]

                return ToolResult(content=[
                    TextContent(type="text", text=self._format_hybrid_results(
                        vector_hits[:top_k],
                        graph_results,
                    ))
                ])
            except Exception as e:
                logger.error("hybrid_retrieve_failed", error=str(e))
                return ToolResult(error=str(e))

        # Tool: ingest_url
        @self.server.tool(
            name="ingest_url",
            description="Crawl and ingest a URL into the knowledge base"
        )
        async def ingest_url(
            url: str,
            depth: int = 1,
            profile: str = "auto",
        ) -> ToolResult:
            if not self._tenant_id:
                return ToolResult(error="Tenant context not set")
            if not self._crawler:
                return ToolResult(error="Crawler not configured")

            try:
                result = await self._crawler.crawl_url(url, max_depth=depth)
                # Index the crawled content
                # ... indexing logic using existing pipeline
                return ToolResult(content=[
                    TextContent(type="text", text=f"Ingested {len(result.pages)} pages from {url}")
                ])
            except Exception as e:
                logger.error("ingest_url_failed", url=url, error=str(e))
                return ToolResult(error=str(e))

        # Tool: ingest_pdf
        @self.server.tool(
            name="ingest_pdf",
            description="Parse and ingest a PDF document"
        )
        async def ingest_pdf(
            file_path: str,
            chunk_size: int = 1000,
        ) -> ToolResult:
            if not self._tenant_id:
                return ToolResult(error="Tenant context not set")
            if not self._parser:
                return ToolResult(error="Parser not configured")

            try:
                result = await self._parser.parse_document(file_path)
                # Index the parsed content
                return ToolResult(content=[
                    TextContent(type="text", text=f"Ingested PDF with {result.page_count} pages")
                ])
            except Exception as e:
                logger.error("ingest_pdf_failed", file_path=file_path, error=str(e))
                return ToolResult(error=str(e))

        # Tool: ingest_youtube
        @self.server.tool(
            name="ingest_youtube",
            description="Extract transcript and ingest a YouTube video"
        )
        async def ingest_youtube(
            video_url: str,
            languages: list[str] = None,
        ) -> ToolResult:
            if not self._tenant_id:
                return ToolResult(error="Tenant context not set")

            try:
                result = await ingest_youtube_video(video_url, languages=languages)
                return ToolResult(content=[
                    TextContent(type="text", text=(
                        f"Ingested YouTube video {result.video_id}\n"
                        f"Language: {result.language}\n"
                        f"Duration: {result.duration_seconds:.0f}s\n"
                        f"Chunks: {len(result.chunks)}"
                    ))
                ])
            except Exception as e:
                logger.error("ingest_youtube_failed", url=video_url, error=str(e))
                return ToolResult(error=str(e))

        # Tool: query_with_reranking
        @self.server.tool(
            name="query_with_reranking",
            description="Query with explicit reranking control"
        )
        async def query_with_reranking(
            query: str,
            reranker: str = "flashrank",
            top_k: int = 10,
        ) -> ToolResult:
            if not self._tenant_id:
                return ToolResult(error="Tenant context not set")

            try:
                # Get vector results
                vector_hits = await self._vector_search.search(query, self._tenant_id)

                # Apply reranking
                if self._reranker:
                    reranked = await self._reranker.rerank(query, vector_hits, top_k)
                    results = [r.hit for r in reranked]
                else:
                    results = vector_hits[:top_k]

                return ToolResult(content=[
                    TextContent(type="text", text=self._format_vector_results(results))
                ])
            except Exception as e:
                logger.error("query_with_reranking_failed", error=str(e))
                return ToolResult(error=str(e))

        # Tool: explain_answer
        @self.server.tool(
            name="explain_answer",
            description="Get explanation of how an answer was derived"
        )
        async def explain_answer(
            trajectory_id: str,
        ) -> ToolResult:
            if not self._tenant_id:
                return ToolResult(error="Tenant context not set")

            try:
                # Fetch trajectory from PostgreSQL
                trajectory = await self._postgres.get_trajectory(
                    trajectory_id,
                    tenant_id=self._tenant_id,
                )
                if not trajectory:
                    return ToolResult(error="Trajectory not found")

                return ToolResult(content=[
                    TextContent(type="text", text=self._format_trajectory(trajectory))
                ])
            except Exception as e:
                logger.error("explain_answer_failed", error=str(e))
                return ToolResult(error=str(e))

    def _format_vector_results(self, hits: list) -> str:
        """Format vector search results for MCP response."""
        lines = ["Vector Search Results:"]
        for i, hit in enumerate(hits, 1):
            lines.append(f"\n{i}. [Score: {hit.similarity:.3f}]")
            lines.append(f"   Content: {hit.content[:200]}...")
            lines.append(f"   Chunk ID: {hit.chunk_id}")
        return "\n".join(lines)

    def _format_hybrid_results(self, vector_hits: list, graph_results: Any) -> str:
        """Format hybrid retrieval results."""
        lines = ["=== Hybrid Retrieval Results ==="]

        lines.append("\n--- Vector Results ---")
        for i, hit in enumerate(vector_hits[:5], 1):
            lines.append(f"{i}. {hit.content[:150]}... (score: {hit.similarity:.3f})")

        lines.append("\n--- Graph Results ---")
        if hasattr(graph_results, 'nodes'):
            for node in graph_results.nodes[:5]:
                lines.append(f"- {node.name} ({node.type})")

        return "\n".join(lines)

    def _format_trajectory(self, trajectory: dict) -> str:
        """Format trajectory for explanation."""
        lines = ["=== Answer Explanation ==="]

        if "plan" in trajectory:
            lines.append("\nPlan:")
            for step in trajectory["plan"]:
                lines.append(f"  - {step}")

        if "thoughts" in trajectory:
            lines.append("\nReasoning:")
            for thought in trajectory["thoughts"]:
                lines.append(f"  * {thought}")

        if "sources" in trajectory:
            lines.append("\nSources:")
            for source in trajectory["sources"]:
                lines.append(f"  [{source['type']}] {source['id']}")

        return "\n".join(lines)

    async def _delegate_graphiti(self, name: str, arguments: dict) -> list[TextContent]:
        """Delegate a call to the underlying Graphiti MCP tools."""
        # Strip the graphiti_ prefix
        graphiti_tool_name = name.replace("graphiti_", "")

        # Add tenant isolation
        if self._tenant_id:
            arguments["group_id"] = self._tenant_id

        # Call Graphiti client methods
        if graphiti_tool_name == "add_episode":
            result = await self._graphiti.client.add_episode(**arguments)
            return [TextContent(type="text", text=f"Episode added: {result.uuid}")]

        elif graphiti_tool_name == "search_nodes":
            results = await self._graphiti.client.search(arguments.get("query", ""))
            return [TextContent(type="text", text=self._format_graph_nodes(results))]

        # ... other Graphiti tool delegations

        raise ValueError(f"Unknown Graphiti tool: {graphiti_tool_name}")

    def _format_graph_nodes(self, results: Any) -> str:
        """Format Graphiti search results."""
        if not results:
            return "No results found."
        lines = ["Graph Search Results:"]
        for node in results[:10]:
            lines.append(f"- {node.name} ({node.type})")
        return "\n".join(lines)
```

#### Transport Layer

```python
# backend/src/agentic_rag_backend/mcp/transport/http.py
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from ..server import UnifiedMCPServer
from ..auth import verify_api_key, get_tenant_from_key

router = APIRouter(prefix="/mcp", tags=["mcp"])


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict = {}
    id: str | int | None = None


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: dict | None = None
    error: dict | None = None
    id: str | int | None = None


@router.post("/")
async def mcp_call(
    request: MCPRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Handle MCP JSON-RPC calls over HTTP."""
    tenant_id = get_tenant_from_key(api_key)
    server: UnifiedMCPServer = raw_request.app.state.mcp_server
    server.set_tenant(tenant_id)

    try:
        if request.method == "tools/list":
            tools = await server.server.list_tools()
            return MCPResponse(
                result={"tools": [t.model_dump() for t in tools]},
                id=request.id,
            )

        elif request.method == "tools/call":
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            result = await server.server.call_tool(tool_name, arguments)
            return MCPResponse(
                result={"content": [c.model_dump() for c in result.content]},
                id=request.id,
            )

        else:
            return MCPResponse(
                error={"code": -32601, "message": f"Method not found: {request.method}"},
                id=request.id,
            )

    except Exception as e:
        return MCPResponse(
            error={"code": -32000, "message": str(e)},
            id=request.id,
        )


@router.get("/sse")
async def mcp_sse(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Server-Sent Events endpoint for streaming MCP."""
    tenant_id = get_tenant_from_key(api_key)
    server: UnifiedMCPServer = request.app.state.mcp_server
    server.set_tenant(tenant_id)

    async def event_generator():
        # Send initial tools list
        tools = await server.server.list_tools()
        yield f"event: tools\ndata: {json.dumps([t.model_dump() for t in tools])}\n\n"

        # Keep connection alive
        while True:
            yield f"event: ping\ndata: {{}}\n\n"
            await asyncio.sleep(30)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp/` | POST | JSON-RPC endpoint for MCP calls |
| `/mcp/sse` | GET | SSE endpoint for streaming MCP |
| `/mcp/tools` | GET | List available tools (convenience) |

### Configuration

```bash
# Epic 14 - MCP Server Configuration
MCP_SERVER_ENABLED=true|false        # Default: false
MCP_SERVER_PORT=8080                 # Separate from main API if needed
MCP_AUTH_ENABLED=true                # Always require API key
MCP_RATE_LIMIT_RPM=60                # Requests per minute per API key
MCP_RATE_LIMIT_BURST=10              # Burst allowance
MCP_LOG_REQUESTS=true                # Log incoming requests
MCP_LOG_RESPONSES=false              # Don't log responses (may contain sensitive data)

# Graphiti MCP Wrapper
MCP_WRAP_GRAPHITI=true               # Enable Graphiti tool pass-through
MCP_GRAPHITI_TOOLS=add_episode,search_nodes,search_facts,delete_episode,clear_graph
```

### Client Configuration Examples

#### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "agentic-rag": {
      "command": "uvx",
      "args": ["agentic-rag-mcp"],
      "env": {
        "MCP_API_KEY": "your-api-key",
        "MCP_SERVER_URL": "http://localhost:8000"
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
      "url": "http://localhost:8000/mcp",
      "apiKey": "${MCP_API_KEY}"
    }
  }
}
```

### Acceptance Criteria

- [ ] Given the MCP server is running, when a client calls `tools/list`, then all Graphiti + RAG tools are returned
- [ ] Given a valid API key, when calling `vector_search`, then results are scoped to the tenant
- [ ] Given a URL, when calling `ingest_url`, then the content is crawled and indexed
- [ ] Given a YouTube URL, when calling `ingest_youtube`, then the transcript is extracted and indexed
- [ ] Given a PDF path, when calling `ingest_pdf`, then the document is parsed and indexed
- [ ] Given a query, when calling `hybrid_retrieve`, then both vector and graph results are returned
- [ ] Given a trajectory ID, when calling `explain_answer`, then the reasoning chain is returned
- [ ] All MCP tools enforce tenant isolation via `tenant_id` filtering
- [ ] Rate limiting is enforced per API key
- [ ] Invalid API keys receive 401 Unauthorized

---

## Story 14-2: Implement Robust A2A Protocol

### Objective

Enhance the existing A2A (Agent-to-Agent) protocol implementation with agent capabilities discovery, standardized task delegation, and bidirectional communication patterns. This enables external agents to delegate tasks to our RAG system and vice versa.

### Current State Analysis

The existing A2A implementation (`backend/src/agentic_rag_backend/protocols/a2a.py`) provides:
- Basic session management with `A2ASessionManager`
- Message storage with `A2AMessage` dataclass
- Redis persistence for session recovery
- TTL-based session expiration

**Gaps to Address:**
1. No agent capabilities discovery (agents can't know what others can do)
2. No structured task delegation (just free-form messages)
3. No response/result tracking (no way to correlate requests with responses)
4. No health monitoring of peer agents
5. Limited error handling for delegation failures

### Technical Design

#### Enhanced A2A Architecture

```
+-------------------------------------------------------------------+
|                    ENHANCED A2A LAYER                               |
+-------------------------------------------------------------------+
|                                                                     |
|  +-----------------------+    +---------------------------------+   |
|  | Agent Registry        |    | Task Delegation Manager         |   |
|  +-----------------------+    +---------------------------------+   |
|  | - agent_id            |    | - task_id                       |   |
|  | - capabilities[]      |    | - source_agent                  |   |
|  | - endpoint_url        |    | - target_agent                  |   |
|  | - health_status       |    | - task_type                     |   |
|  | - last_heartbeat      |    | - parameters                    |   |
|  +-----------------------+    | - status (pending/running/done) |   |
|                               | - result                        |   |
|                               +---------------------------------+   |
|                                                                     |
|  +-----------------------+    +---------------------------------+   |
|  | Session Manager       |    | Message Types                   |   |
|  | (Enhanced)            |    +---------------------------------+   |
|  +-----------------------+    | - CAPABILITY_QUERY              |   |
|  | - session_id          |    | - CAPABILITY_RESPONSE           |   |
|  | - participants[]      |    | - TASK_REQUEST                  |   |
|  | - messages[]          |    | - TASK_PROGRESS                 |   |
|  | - task_references[]   |    | - TASK_RESULT                   |   |
|  +-----------------------+    | - HEARTBEAT                     |   |
|                               | - ERROR                         |   |
|                               +---------------------------------+   |
|                                                                     |
+-------------------------------------------------------------------+
```

#### Module Structure

```
backend/src/agentic_rag_backend/protocols/
+-- __init__.py
+-- a2a.py                          # ENHANCED: Existing session manager
+-- a2a_registry.py                 # NEW: Agent capabilities registry
+-- a2a_delegation.py               # NEW: Task delegation manager
+-- a2a_messages.py                 # NEW: Structured message types
+-- a2a_health.py                   # NEW: Health monitoring
```

#### Core Classes

```python
# backend/src/agentic_rag_backend/protocols/a2a_messages.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class A2AMessageType(str, Enum):
    """Standardized A2A message types."""
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    TASK_REQUEST = "task_request"
    TASK_PROGRESS = "task_progress"
    TASK_RESULT = "task_result"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCapability:
    """A capability offered by an agent."""
    name: str
    description: str
    parameters_schema: dict[str, Any]
    returns_schema: dict[str, Any]
    estimated_duration_ms: Optional[int] = None


@dataclass
class AgentRegistration:
    """Registration record for an agent in the A2A network."""
    agent_id: str
    agent_type: str
    endpoint_url: str
    capabilities: list[AgentCapability]
    tenant_id: str
    registered_at: datetime
    last_heartbeat: datetime
    health_status: str = "healthy"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "endpoint_url": self.endpoint_url,
            "capabilities": [
                {
                    "name": c.name,
                    "description": c.description,
                    "parameters_schema": c.parameters_schema,
                    "returns_schema": c.returns_schema,
                }
                for c in self.capabilities
            ],
            "tenant_id": self.tenant_id,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "health_status": self.health_status,
            "metadata": self.metadata,
        }


@dataclass
class TaskRequest:
    """A task delegated from one agent to another."""
    task_id: str = field(default_factory=lambda: str(uuid4()))
    source_agent: str = ""
    target_agent: str = ""
    capability_name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, higher = more urgent
    timeout_seconds: int = 300
    correlation_id: Optional[str] = None  # For request/response correlation
    created_at: datetime = field(default_factory=lambda: datetime.now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "capability_name": self.capability_name,
            "parameters": self.parameters,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    status: TaskStatus
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    completed_at: datetime = field(default_factory=lambda: datetime.now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "completed_at": self.completed_at.isoformat(),
        }
```

```python
# backend/src/agentic_rag_backend/protocols/a2a_registry.py
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import structlog

from ..db.redis import RedisClient
from .a2a_messages import AgentCapability, AgentRegistration

logger = structlog.get_logger(__name__)


class A2AAgentRegistry:
    """Registry for agent capabilities discovery.

    Enables agents to:
    - Register their capabilities
    - Discover other agents and their capabilities
    - Monitor agent health via heartbeats
    """

    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        redis_prefix: str = "a2a:agents",
        heartbeat_timeout_seconds: int = 60,
        cleanup_interval_seconds: int = 30,
    ):
        self._agents: dict[str, AgentRegistration] = {}
        self._redis_client = redis_client
        self._redis_prefix = redis_prefix
        self._heartbeat_timeout = heartbeat_timeout_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        endpoint_url: str,
        capabilities: list[AgentCapability],
        tenant_id: str,
        metadata: dict = None,
    ) -> AgentRegistration:
        """Register an agent with its capabilities."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            registration = AgentRegistration(
                agent_id=agent_id,
                agent_type=agent_type,
                endpoint_url=endpoint_url,
                capabilities=capabilities,
                tenant_id=tenant_id,
                registered_at=now,
                last_heartbeat=now,
                health_status="healthy",
                metadata=metadata or {},
            )
            self._agents[agent_id] = registration
            await self._persist_registration(registration)

            logger.info(
                "a2a_agent_registered",
                agent_id=agent_id,
                agent_type=agent_type,
                capability_count=len(capabilities),
            )

            return registration

    async def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the registry."""
        async with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                await self._delete_registration(agent_id)
                logger.info("a2a_agent_unregistered", agent_id=agent_id)
                return True
            return False

    async def heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat timestamp."""
        async with self._lock:
            if agent_id not in self._agents:
                return False

            self._agents[agent_id].last_heartbeat = datetime.now(timezone.utc)
            self._agents[agent_id].health_status = "healthy"
            await self._persist_registration(self._agents[agent_id])
            return True

    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get a specific agent registration."""
        return self._agents.get(agent_id)

    async def find_agents_by_capability(
        self,
        capability_name: str,
        tenant_id: str,
        healthy_only: bool = True,
    ) -> list[AgentRegistration]:
        """Find agents that offer a specific capability."""
        results = []
        for agent in self._agents.values():
            if agent.tenant_id != tenant_id:
                continue
            if healthy_only and agent.health_status != "healthy":
                continue
            for cap in agent.capabilities:
                if cap.name == capability_name:
                    results.append(agent)
                    break
        return results

    async def list_all_capabilities(
        self,
        tenant_id: str,
    ) -> dict[str, list[str]]:
        """List all capabilities available in the tenant's A2A network."""
        capabilities: dict[str, list[str]] = {}
        for agent in self._agents.values():
            if agent.tenant_id != tenant_id:
                continue
            for cap in agent.capabilities:
                if cap.name not in capabilities:
                    capabilities[cap.name] = []
                capabilities[cap.name].append(agent.agent_id)
        return capabilities

    async def _persist_registration(self, registration: AgentRegistration) -> None:
        """Persist registration to Redis."""
        if not self._redis_client:
            return
        try:
            redis = self._redis_client.client
            key = f"{self._redis_prefix}:{registration.agent_id}"
            import json
            await redis.set(key, json.dumps(registration.to_dict()))
        except Exception as e:
            logger.warning("a2a_persist_failed", error=str(e))

    async def _delete_registration(self, agent_id: str) -> None:
        """Delete registration from Redis."""
        if not self._redis_client:
            return
        try:
            redis = self._redis_client.client
            await redis.delete(f"{self._redis_prefix}:{agent_id}")
        except Exception as e:
            logger.warning("a2a_delete_failed", error=str(e))

    async def start_health_monitor(self) -> None:
        """Start background health monitoring."""
        if self._cleanup_task and not self._cleanup_task.done():
            return
        self._cleanup_task = asyncio.create_task(self._health_check_loop())

    async def stop_health_monitor(self) -> None:
        """Stop background health monitoring."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _health_check_loop(self) -> None:
        """Periodic health check for registered agents."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                async with self._lock:
                    now = datetime.now(timezone.utc)
                    for agent in self._agents.values():
                        elapsed = (now - agent.last_heartbeat).total_seconds()
                        if elapsed > self._heartbeat_timeout:
                            agent.health_status = "unhealthy"
                            logger.warning(
                                "a2a_agent_unhealthy",
                                agent_id=agent.agent_id,
                                seconds_since_heartbeat=elapsed,
                            )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("a2a_health_check_error", error=str(e))
```

```python
# backend/src/agentic_rag_backend/protocols/a2a_delegation.py
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

import httpx
import structlog

from ..db.redis import RedisClient
from .a2a_messages import TaskRequest, TaskResult, TaskStatus
from .a2a_registry import A2AAgentRegistry

logger = structlog.get_logger(__name__)


class TaskDelegationManager:
    """Manages task delegation between agents.

    Handles:
    - Sending task requests to target agents
    - Tracking task progress and results
    - Timeout and retry handling
    - Result correlation
    """

    def __init__(
        self,
        agent_id: str,
        registry: A2AAgentRegistry,
        redis_client: Optional[RedisClient] = None,
        redis_prefix: str = "a2a:tasks",
        default_timeout_seconds: int = 300,
        max_retries: int = 3,
    ):
        self._agent_id = agent_id
        self._registry = registry
        self._redis_client = redis_client
        self._redis_prefix = redis_prefix
        self._default_timeout = default_timeout_seconds
        self._max_retries = max_retries
        self._pending_tasks: dict[str, TaskRequest] = {}
        self._results: dict[str, TaskResult] = {}
        self._callbacks: dict[str, Callable] = {}
        self._lock = asyncio.Lock()

    async def delegate_task(
        self,
        target_agent: str,
        capability_name: str,
        parameters: dict[str, Any],
        priority: int = 5,
        timeout_seconds: Optional[int] = None,
        callback: Optional[Callable[[TaskResult], None]] = None,
    ) -> TaskRequest:
        """Delegate a task to another agent.

        Args:
            target_agent: Agent ID to delegate to
            capability_name: The capability to invoke
            parameters: Parameters for the capability
            priority: Task priority (1-10)
            timeout_seconds: Optional timeout override
            callback: Optional callback when result arrives

        Returns:
            The created TaskRequest
        """
        # Verify target agent exists and is healthy
        agent = await self._registry.get_agent(target_agent)
        if not agent:
            raise ValueError(f"Target agent not found: {target_agent}")
        if agent.health_status != "healthy":
            raise ValueError(f"Target agent unhealthy: {target_agent}")

        # Verify capability exists
        has_capability = any(c.name == capability_name for c in agent.capabilities)
        if not has_capability:
            raise ValueError(f"Agent {target_agent} lacks capability: {capability_name}")

        # Create task request
        task = TaskRequest(
            source_agent=self._agent_id,
            target_agent=target_agent,
            capability_name=capability_name,
            parameters=parameters,
            priority=priority,
            timeout_seconds=timeout_seconds or self._default_timeout,
            correlation_id=str(uuid4()),
        )

        async with self._lock:
            self._pending_tasks[task.task_id] = task
            if callback:
                self._callbacks[task.task_id] = callback

        # Send task to target agent
        await self._send_task(agent.endpoint_url, task)

        logger.info(
            "a2a_task_delegated",
            task_id=task.task_id,
            target_agent=target_agent,
            capability=capability_name,
        )

        return task

    async def receive_result(self, result: TaskResult) -> None:
        """Handle an incoming task result."""
        async with self._lock:
            task = self._pending_tasks.pop(result.task_id, None)
            if not task:
                logger.warning("a2a_result_for_unknown_task", task_id=result.task_id)
                return

            self._results[result.task_id] = result

            # Invoke callback if registered
            callback = self._callbacks.pop(result.task_id, None)
            if callback:
                try:
                    callback(result)
                except Exception as e:
                    logger.error("a2a_callback_error", task_id=result.task_id, error=str(e))

        logger.info(
            "a2a_task_result_received",
            task_id=result.task_id,
            status=result.status.value,
            execution_time_ms=result.execution_time_ms,
        )

    async def get_result(
        self,
        task_id: str,
        timeout_seconds: Optional[int] = None,
    ) -> Optional[TaskResult]:
        """Wait for and return a task result.

        Args:
            task_id: The task to wait for
            timeout_seconds: How long to wait

        Returns:
            TaskResult if received, None on timeout
        """
        timeout = timeout_seconds or self._default_timeout
        start = time.monotonic()

        while (time.monotonic() - start) < timeout:
            async with self._lock:
                if task_id in self._results:
                    return self._results.pop(task_id)
            await asyncio.sleep(0.1)

        # Timeout - mark task as failed
        async with self._lock:
            task = self._pending_tasks.pop(task_id, None)
            if task:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error="Timeout waiting for result",
                )

        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        async with self._lock:
            task = self._pending_tasks.pop(task_id, None)
            if not task:
                return False

            self._callbacks.pop(task_id, None)

        # Notify target agent of cancellation
        agent = await self._registry.get_agent(task.target_agent)
        if agent:
            await self._send_cancellation(agent.endpoint_url, task_id)

        logger.info("a2a_task_cancelled", task_id=task_id)
        return True

    async def _send_task(self, endpoint_url: str, task: TaskRequest) -> None:
        """Send task request to target agent."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    f"{endpoint_url}/a2a/tasks",
                    json=task.to_dict(),
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error(
                    "a2a_send_task_failed",
                    task_id=task.task_id,
                    target=endpoint_url,
                    error=str(e),
                )
                raise

    async def _send_cancellation(self, endpoint_url: str, task_id: str) -> None:
        """Send task cancellation to target agent."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                await client.delete(f"{endpoint_url}/a2a/tasks/{task_id}")
            except httpx.HTTPError as e:
                logger.warning("a2a_send_cancel_failed", task_id=task_id, error=str(e))
```

### API Endpoints (Enhanced)

```python
# backend/src/agentic_rag_backend/api/routes/a2a.py (enhanced)
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional

from ...protocols.a2a_messages import (
    AgentCapability,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from ...protocols.a2a_registry import A2AAgentRegistry
from ...protocols.a2a_delegation import TaskDelegationManager

router = APIRouter(prefix="/a2a", tags=["a2a"])


class RegisterAgentRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent (e.g., 'rag', 'search')")
    endpoint_url: str = Field(..., description="HTTP endpoint for receiving tasks")
    capabilities: list[dict] = Field(..., description="List of capability definitions")


class RegisterAgentResponse(BaseModel):
    agent_id: str
    registered_at: str


@router.post("/agents/register", response_model=RegisterAgentResponse)
async def register_agent(
    request: RegisterAgentRequest,
    tenant_id: str = Depends(get_tenant_id),
    registry: A2AAgentRegistry = Depends(get_registry),
):
    """Register a new agent with the A2A network."""
    capabilities = [
        AgentCapability(
            name=c["name"],
            description=c.get("description", ""),
            parameters_schema=c.get("parameters_schema", {}),
            returns_schema=c.get("returns_schema", {}),
        )
        for c in request.capabilities
    ]

    agent_id = f"{tenant_id}-{request.agent_type}-{uuid4().hex[:8]}"
    registration = await registry.register_agent(
        agent_id=agent_id,
        agent_type=request.agent_type,
        endpoint_url=request.endpoint_url,
        capabilities=capabilities,
        tenant_id=tenant_id,
    )

    return RegisterAgentResponse(
        agent_id=registration.agent_id,
        registered_at=registration.registered_at.isoformat(),
    )


@router.post("/agents/{agent_id}/heartbeat")
async def agent_heartbeat(
    agent_id: str,
    registry: A2AAgentRegistry = Depends(get_registry),
):
    """Update agent heartbeat."""
    success = await registry.heartbeat(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "ok"}


@router.get("/capabilities")
async def list_capabilities(
    tenant_id: str = Depends(get_tenant_id),
    registry: A2AAgentRegistry = Depends(get_registry),
):
    """List all available capabilities in the A2A network."""
    capabilities = await registry.list_all_capabilities(tenant_id)
    return {"capabilities": capabilities}


@router.get("/agents")
async def list_agents(
    tenant_id: str = Depends(get_tenant_id),
    capability: Optional[str] = None,
    healthy_only: bool = True,
    registry: A2AAgentRegistry = Depends(get_registry),
):
    """List registered agents, optionally filtered by capability."""
    if capability:
        agents = await registry.find_agents_by_capability(
            capability, tenant_id, healthy_only
        )
    else:
        agents = [
            a for a in registry._agents.values()
            if a.tenant_id == tenant_id and (not healthy_only or a.health_status == "healthy")
        ]

    return {"agents": [a.to_dict() for a in agents]}


class DelegateTaskRequest(BaseModel):
    target_agent: str
    capability_name: str
    parameters: dict[str, Any] = {}
    priority: int = Field(5, ge=1, le=10)
    timeout_seconds: int = Field(300, ge=1, le=3600)


@router.post("/tasks/delegate")
async def delegate_task(
    request: DelegateTaskRequest,
    delegation_manager: TaskDelegationManager = Depends(get_delegation_manager),
):
    """Delegate a task to another agent."""
    try:
        task = await delegation_manager.delegate_task(
            target_agent=request.target_agent,
            capability_name=request.capability_name,
            parameters=request.parameters,
            priority=request.priority,
            timeout_seconds=request.timeout_seconds,
        )
        return {"task_id": task.task_id, "status": "pending"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tasks/{task_id}/result")
async def get_task_result(
    task_id: str,
    wait_seconds: int = 30,
    delegation_manager: TaskDelegationManager = Depends(get_delegation_manager),
):
    """Wait for and retrieve a task result."""
    result = await delegation_manager.get_result(task_id, timeout_seconds=wait_seconds)
    if not result:
        raise HTTPException(status_code=408, detail="Timeout waiting for result")
    return result.to_dict()


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    delegation_manager: TaskDelegationManager = Depends(get_delegation_manager),
):
    """Cancel a pending task."""
    success = await delegation_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "cancelled"}


# Endpoint for receiving delegated tasks (this agent as target)
@router.post("/tasks")
async def receive_task(
    task: dict[str, Any],
    task_handler = Depends(get_task_handler),
):
    """Receive a delegated task from another agent."""
    task_request = TaskRequest(**task)

    # Execute the task asynchronously
    asyncio.create_task(task_handler.execute_task(task_request))

    return {"status": "accepted", "task_id": task_request.task_id}
```

### Configuration

```bash
# Epic 14 - A2A Protocol Configuration
A2A_ENABLED=true|false                    # Default: true
A2A_AGENT_ID=agentic-rag-001              # This agent's ID
A2A_ENDPOINT_URL=http://localhost:8000    # This agent's endpoint
A2A_HEARTBEAT_INTERVAL_SECONDS=30         # Heartbeat frequency
A2A_HEARTBEAT_TIMEOUT_SECONDS=60          # Unhealthy threshold
A2A_TASK_DEFAULT_TIMEOUT_SECONDS=300      # Default task timeout
A2A_TASK_MAX_RETRIES=3                    # Max retry attempts
A2A_SESSION_TTL_SECONDS=21600             # Session TTL (6 hours)
A2A_MAX_SESSIONS_PER_TENANT=100           # Session limit
```

### Acceptance Criteria

- [ ] Given an agent, when it registers with the A2A network, then its capabilities are discoverable by other agents
- [ ] Given a registered agent, when it sends heartbeats, then its health status remains "healthy"
- [ ] Given a missing heartbeat, when timeout is exceeded, then the agent is marked "unhealthy"
- [ ] Given a capability query, when calling `/a2a/capabilities`, then all available capabilities are listed
- [ ] Given a task delegation, when calling `/a2a/tasks/delegate`, then the task is sent to the target agent
- [ ] Given a delegated task, when the target agent completes it, then the result is returned to the source agent
- [ ] Given a task timeout, when waiting for result, then the task is marked as failed
- [ ] Given a task cancellation, when calling DELETE `/a2a/tasks/{task_id}`, then the task is cancelled
- [ ] All A2A operations enforce tenant isolation
- [ ] Redis persistence ensures session and registration recovery after restart

---

## Integration with Existing Systems

### MCP Integration Points

| Existing Component | Integration Approach |
|-------------------|---------------------|
| `VectorSearchService` | Direct injection into MCP server |
| `GraphitiClient` | Wrapped with tenant isolation |
| `RerankerClient` | Optional injection for reranking tools |
| `Crawler` (Crawl4AI) | Used by `ingest_url` tool |
| `Parser` (Docling) | Used by `ingest_pdf` tool |
| `YouTubeIngestionService` | Used by `ingest_youtube` tool |

### A2A Integration Points

| Existing Component | Integration Approach |
|-------------------|---------------------|
| `A2ASessionManager` | Enhanced with task references |
| `OrchestratorAgent` | Can delegate to external agents |
| `RedisClient` | Used for persistence |
| Rate Limiter | Applied to A2A endpoints |

### Startup Initialization

```python
# backend/src/agentic_rag_backend/main.py (additions)
from agentic_rag_backend.mcp.server import UnifiedMCPServer
from agentic_rag_backend.mcp.transport.http import router as mcp_router
from agentic_rag_backend.protocols.a2a_registry import A2AAgentRegistry
from agentic_rag_backend.protocols.a2a_delegation import TaskDelegationManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing initialization ...

    # Initialize MCP server
    if settings.mcp_server_enabled:
        app.state.mcp_server = UnifiedMCPServer(
            graphiti_client=app.state.graphiti_client,
            postgres_client=app.state.postgres_client,
            vector_search=app.state.vector_search,
            reranker=app.state.reranker if settings.reranker_enabled else None,
            crawler=app.state.crawler,
            parser=app.state.parser,
        )
        app.include_router(mcp_router)
        logger.info("mcp_server_initialized")

    # Initialize A2A registry
    if settings.a2a_enabled:
        app.state.a2a_registry = A2AAgentRegistry(
            redis_client=app.state.redis_client,
            heartbeat_timeout_seconds=settings.a2a_heartbeat_timeout_seconds,
        )
        await app.state.a2a_registry.start_health_monitor()

        app.state.a2a_delegation = TaskDelegationManager(
            agent_id=settings.a2a_agent_id,
            registry=app.state.a2a_registry,
            redis_client=app.state.redis_client,
        )

        # Register this agent's capabilities
        await app.state.a2a_registry.register_agent(
            agent_id=settings.a2a_agent_id,
            agent_type="agentic-rag",
            endpoint_url=settings.a2a_endpoint_url,
            capabilities=_get_rag_capabilities(),
            tenant_id="system",
        )
        logger.info("a2a_protocol_initialized")

    yield

    # Cleanup
    if settings.a2a_enabled:
        await app.state.a2a_registry.stop_health_monitor()


def _get_rag_capabilities() -> list[AgentCapability]:
    """Define this agent's A2A capabilities."""
    return [
        AgentCapability(
            name="hybrid_retrieve",
            description="Combined vector + graph retrieval",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            returns_schema={"type": "object"},
        ),
        AgentCapability(
            name="ingest_url",
            description="Crawl and ingest a URL",
            parameters_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "depth": {"type": "integer", "default": 1},
                },
                "required": ["url"],
            },
            returns_schema={"type": "object"},
        ),
        # ... additional capabilities
    ]
```

---

## Testing Strategy

### Unit Tests

```python
# backend/tests/mcp/test_server.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_rag_backend.mcp.server import UnifiedMCPServer


@pytest.fixture
def mock_dependencies():
    return {
        "graphiti_client": MagicMock(),
        "postgres_client": MagicMock(),
        "vector_search": AsyncMock(),
        "reranker": AsyncMock(),
    }


@pytest.mark.asyncio
async def test_vector_search_requires_tenant(mock_dependencies):
    server = UnifiedMCPServer(**mock_dependencies)

    # Without tenant set
    result = await server.server.call_tool("vector_search", {"query": "test"})
    assert result.error == "Tenant context not set"


@pytest.mark.asyncio
async def test_vector_search_with_tenant(mock_dependencies):
    mock_dependencies["vector_search"].search.return_value = []
    server = UnifiedMCPServer(**mock_dependencies)
    server.set_tenant("tenant-123")

    result = await server.server.call_tool("vector_search", {"query": "test"})
    assert result.error is None
    mock_dependencies["vector_search"].search.assert_called_once()


# backend/tests/protocols/test_a2a_registry.py
@pytest.mark.asyncio
async def test_agent_registration():
    registry = A2AAgentRegistry()

    registration = await registry.register_agent(
        agent_id="test-agent",
        agent_type="test",
        endpoint_url="http://localhost:8001",
        capabilities=[AgentCapability(
            name="test_cap",
            description="Test capability",
            parameters_schema={},
            returns_schema={},
        )],
        tenant_id="tenant-123",
    )

    assert registration.agent_id == "test-agent"
    assert registration.health_status == "healthy"


@pytest.mark.asyncio
async def test_find_agents_by_capability():
    registry = A2AAgentRegistry()

    await registry.register_agent(
        agent_id="agent-1",
        agent_type="rag",
        endpoint_url="http://localhost:8001",
        capabilities=[AgentCapability(
            name="hybrid_retrieve",
            description="",
            parameters_schema={},
            returns_schema={},
        )],
        tenant_id="tenant-123",
    )

    agents = await registry.find_agents_by_capability("hybrid_retrieve", "tenant-123")
    assert len(agents) == 1
    assert agents[0].agent_id == "agent-1"


@pytest.mark.asyncio
async def test_heartbeat_timeout_marks_unhealthy():
    registry = A2AAgentRegistry(heartbeat_timeout_seconds=1)

    await registry.register_agent(
        agent_id="test-agent",
        agent_type="test",
        endpoint_url="http://localhost:8001",
        capabilities=[],
        tenant_id="tenant-123",
    )

    # Simulate time passing
    registry._agents["test-agent"].last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=5)

    await registry._health_check_loop()  # Run once

    agent = await registry.get_agent("test-agent")
    assert agent.health_status == "unhealthy"
```

### Integration Tests

```python
# backend/tests/integration/test_mcp_integration.py
import pytest
from httpx import AsyncClient


@pytest.mark.integration
async def test_mcp_list_tools(async_client: AsyncClient, api_key: str):
    response = await async_client.post(
        "/mcp/",
        json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "tools" in data["result"]

    tool_names = [t["name"] for t in data["result"]["tools"]]
    assert "vector_search" in tool_names
    assert "hybrid_retrieve" in tool_names
    assert "ingest_url" in tool_names


@pytest.mark.integration
async def test_mcp_vector_search(async_client: AsyncClient, api_key: str, indexed_documents):
    response = await async_client.post(
        "/mcp/",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "vector_search",
                "arguments": {"query": "test query", "top_k": 5},
            },
            "id": 1,
        },
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "content" in data["result"]


# backend/tests/integration/test_a2a_integration.py
@pytest.mark.integration
async def test_a2a_agent_lifecycle(async_client: AsyncClient, api_key: str):
    # Register agent
    response = await async_client.post(
        "/a2a/agents/register",
        json={
            "agent_type": "test",
            "endpoint_url": "http://localhost:8001",
            "capabilities": [
                {"name": "test_cap", "description": "Test"}
            ],
        },
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 200
    agent_id = response.json()["agent_id"]

    # Send heartbeat
    response = await async_client.post(
        f"/a2a/agents/{agent_id}/heartbeat",
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 200

    # List capabilities
    response = await async_client.get(
        "/a2a/capabilities",
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 200
    assert "test_cap" in response.json()["capabilities"]


@pytest.mark.integration
async def test_a2a_task_delegation(async_client: AsyncClient, api_key: str, registered_agent):
    response = await async_client.post(
        "/a2a/tasks/delegate",
        json={
            "target_agent": registered_agent["agent_id"],
            "capability_name": "hybrid_retrieve",
            "parameters": {"query": "test query"},
        },
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 200
    assert "task_id" in response.json()
```

---

## Migration and Deployment

### Phase 1: MCP Server Foundation

1. Add MCP SDK dependency to pyproject.toml
2. Create `mcp/` module structure
3. Implement `UnifiedMCPServer` with Graphiti wrapper
4. Add HTTP transport endpoints
5. Add authentication middleware

### Phase 2: RAG Tool Extensions

1. Implement `vector_search` tool
2. Implement `hybrid_retrieve` tool
3. Implement ingestion tools (URL, PDF, YouTube)
4. Implement `query_with_reranking` tool
5. Implement `explain_answer` tool

### Phase 3: A2A Protocol Enhancement

1. Create `a2a_messages.py` with structured types
2. Implement `A2AAgentRegistry`
3. Implement `TaskDelegationManager`
4. Enhance existing A2A API endpoints
5. Add health monitoring

### Phase 4: Testing and Documentation

1. Unit tests for all new components
2. Integration tests for MCP and A2A
3. Update API documentation
4. Create client configuration examples

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| MCP tool response time | < 500ms | P95 for vector_search, hybrid_retrieve |
| MCP server availability | 99.9% | Uptime monitoring |
| A2A task success rate | >= 95% | Completed vs. failed tasks |
| A2A agent health detection | < 60s | Time to detect unhealthy agent |
| Client integration success | >= 3 clients | Claude Desktop, Cursor, VS Code |

---

## Dependencies

- **Epic 3:** Hybrid retrieval infrastructure (VectorSearchService)
- **Epic 5:** Graphiti temporal knowledge graph (GraphitiClient)
- **Epic 12:** Reranking and grading (RerankerClient)
- **Epic 13:** Enterprise ingestion (Crawler, YouTubeIngestionService)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Graphiti MCP API changes | High | Pin graphiti-core version, monitor releases |
| MCP SDK breaking changes | Medium | Pin mcp package version, test thoroughly |
| A2A network partitions | Medium | Health monitoring, graceful degradation |
| Rate limiting for external clients | Medium | Configurable limits, burst allowance |
| Security vulnerabilities in MCP transport | High | API key auth, tenant isolation, input validation |

---

## References

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Graphiti MCP Server Documentation](https://deepwiki.com/getzep/graphiti)
- [A2A Protocol Design](https://google.github.io/A2A/)
- `docs/guides/mcp-wrapper-architecture.md`
- `_bmad-output/architecture.md`
- `_bmad-output/project-context.md`
