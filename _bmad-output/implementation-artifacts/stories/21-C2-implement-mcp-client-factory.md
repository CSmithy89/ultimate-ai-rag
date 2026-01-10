# Story 21-C2: Implement MCP Client Factory

## Status: Review

## Story

As a backend developer, I need an MCP client factory that creates and manages connections to external MCP servers, so that I can integrate tools from the broader MCP ecosystem (GitHub, Notion, etc.).

## Implementation Summary

### New Files Created

1. **`backend/src/agentic_rag_backend/mcp_client/client.py`** - Core client implementation
   - `MCPClient`: HTTP client for MCP server communication
     - `list_tools()`: Discover available tools from server
     - `call_tool(name, arguments)`: Execute a tool
     - Retry logic with exponential backoff and jitter
     - Timeout handling
     - JSON-RPC 2.0 protocol support
   - `MCPClientFactory`: Factory for managing client instances
     - `get_client(name)`: Get or create client by server name
     - `discover_all_tools()`: Discover tools from all configured servers
     - `call_tool(server, tool, args)`: Execute tool on specific server
     - `close_all()`: Graceful shutdown of all clients
     - `lifespan()`: Context manager for lifecycle management

2. **`backend/src/agentic_rag_backend/mcp_client/errors.py`** - Error classes
   - `MCPClientError`: Base exception
   - `MCPClientTimeoutError`: Request timeout after retries
   - `MCPClientConnectionError`: Connection failure
   - `MCPServerNotFoundError`: Unknown server name
   - `MCPProtocolError`: JSON-RPC error response
   - `MCPToolNotFoundError`: Unknown tool on server
   - `MCPClientNotEnabledError`: Feature disabled

3. **`backend/src/agentic_rag_backend/mcp_client/dependencies.py`** - FastAPI DI
   - `create_mcp_client_settings()`: Convert app Settings to MCPClientSettings
   - `create_mcp_client_factory()`: Create factory from app Settings
   - `get_mcp_factory()`: Dependency injection for routes

4. **`backend/src/agentic_rag_backend/mcp_client/__init__.py`** - Module exports

### Files Modified

1. **`backend/src/agentic_rag_backend/mcp_client/config.py`**
   - Fixed Pydantic deprecation warning (ConfigDict instead of class Config)

2. **`backend/src/agentic_rag_backend/main.py`**
   - Added MCP client factory initialization on startup
   - Added graceful shutdown in lifespan

### Tests Added

1. **`backend/tests/mcp_client/test_client.py`** (29 tests)
   - MCPClient tests: headers, list_tools, call_tool, retries, errors
   - MCPClientFactory tests: lifecycle, discovery, tool calls
   - Error class tests

2. **`backend/tests/mcp_client/test_config.py`** (11 tests)
   - MCPServerConfig validation tests
   - MCPClientSettings validation tests

3. **`backend/tests/mcp_client/test_dependencies.py`** (5 tests)
   - Dependency function tests

## Key Features

### Retry Logic
- Exponential backoff: `base_delay * 2^attempt`
- 10% jitter to prevent thundering herd
- Max backoff capped at 30 seconds
- No retry on 4xx client errors (they won't succeed with retry)
- Retry on 5xx server errors and timeouts

### Connection Pooling
- Clients are reused via factory (lazy creation)
- Thread-safe with asyncio.Lock
- Graceful shutdown closes all clients

### Protocol Support
- JSON-RPC 2.0 over HTTP
- Bearer token authentication
- SSE/HTTP transport (HTTP implementation)

## Configuration

Uses existing environment variables from Story 21-C1:
```bash
MCP_CLIENTS_ENABLED=true
MCP_CLIENT_TIMEOUT=30000
MCP_CLIENT_RETRY_COUNT=3
MCP_CLIENT_RETRY_DELAY=1000
MCP_CLIENT_SERVERS='[{"name":"github","url":"https://mcp.github.com/sse","apiKey":"..."}]'
```

## Usage Example

```python
from fastapi import Depends

from agentic_rag_backend.mcp_client import MCPClientFactory, get_mcp_factory

@router.get("/tools")
async def list_external_tools(
    factory: MCPClientFactory = Depends(get_mcp_factory),
):
    if factory and factory.is_enabled:
        return await factory.discover_all_tools()
    return {}
```

## Acceptance Criteria

- [x] `MCPClient` class handles HTTP communication
- [x] `MCPClientFactory` manages client lifecycle
- [x] Tool discovery from all configured servers
- [x] Retry logic with exponential backoff
- [x] Timeout handling
- [x] Connection pooling (client reuse)
- [x] Graceful shutdown on application exit
- [x] 45 unit tests passing

## Files Changed

- `backend/src/agentic_rag_backend/mcp_client/client.py` (new)
- `backend/src/agentic_rag_backend/mcp_client/errors.py` (new)
- `backend/src/agentic_rag_backend/mcp_client/dependencies.py` (new)
- `backend/src/agentic_rag_backend/mcp_client/__init__.py` (updated)
- `backend/src/agentic_rag_backend/mcp_client/config.py` (updated)
- `backend/src/agentic_rag_backend/main.py` (updated)
- `backend/tests/mcp_client/` (new directory with tests)
