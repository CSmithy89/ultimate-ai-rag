# Story 21-C3: Wire MCP Client to CopilotRuntime

## Status: Review

## Story

As a backend developer, I need to integrate MCP client with CopilotKit runtime for unified tool access, enabling external MCP tools to be discoverable and callable through the copilot endpoints.

## Implementation Summary

### New Files Created

1. **`backend/src/agentic_rag_backend/mcp_client/registry.py`** - Tool registry utilities
   - `ToolInfo`: Class representing tool information from any source
   - `merge_tool_registries()`: Merge internal and external tools with namespacing
   - `discover_all_tools()`: Discover tools from factory and merge with internal
   - `parse_namespaced_tool()`: Parse "server:tool" format into components

2. **`backend/tests/mcp_client/test_registry.py`** - Registry tests (13 tests)
   - Tests for merging registries
   - Tests for parsing namespaced tools
   - Tests for tool discovery with various factory states

3. **`backend/tests/api/routes/test_copilot_tools.py`** - Endpoint tests (9 tests)
   - Tests for GET /copilot/tools endpoint
   - Tests for POST /copilot/tools/call endpoint

### Files Modified

1. **`backend/src/agentic_rag_backend/mcp_client/__init__.py`**
   - Added exports for registry functions

2. **`backend/src/agentic_rag_backend/api/routes/copilot.py`**
   - Added imports for MCP client
   - Added `ToolDefinition`, `ToolsResponse`, `ToolCallRequest`, `ToolCallResponse` models
   - Added `GET /copilot/tools` endpoint for tool discovery
   - Added `POST /copilot/tools/call` endpoint for external tool execution

3. **`frontend/app/api/copilotkit/route.ts`**
   - Added MCP server configuration parsing from environment
   - Added `remoteEndpoints` configuration for backend integration
   - Added GET endpoint for debugging MCP status (dev only)

## API Endpoints

### GET /api/v1/copilot/tools

List all available tools (internal + external MCP).

**Response:**
```json
{
  "tools": [
    {
      "name": "github:create_issue",
      "description": "Create a GitHub issue",
      "inputSchema": {...},
      "source": "external",
      "serverName": "github"
    }
  ],
  "mcpEnabled": true,
  "serverCount": 2
}
```

### POST /api/v1/copilot/tools/call

Execute an external MCP tool.

**Request:**
```json
{
  "toolName": "github:create_issue",
  "arguments": {"title": "Bug report", "body": "..."}
}
```

**Response:**
```json
{
  "result": {"issue_url": "...", "issue_number": 1},
  "serverName": "github"
}
```

## Tool Namespacing

External tools are namespaced by server name to prevent conflicts:
- Internal tool: `search`
- External tool: `github:create_issue`, `notion:create_page`

## Frontend Configuration

Environment variables for MCP servers:
```bash
MCP_SERVERS='[{"name":"github","url":"https://mcp.github.com/sse","apiKey":"..."}]'
BACKEND_URL=http://localhost:8000
```

## Acceptance Criteria

- [x] External MCP tools discoverable via /copilot/tools endpoint
- [x] Tools namespaced by source server (server:tool format)
- [x] Unified tool registry merges internal and external tools
- [x] Proper error handling for unavailable servers
- [x] Tool metadata includes source for UI display
- [x] Tests verify tool discovery and merging (22 tests)

## Files Changed

- `backend/src/agentic_rag_backend/mcp_client/registry.py` (new)
- `backend/src/agentic_rag_backend/mcp_client/__init__.py` (updated)
- `backend/src/agentic_rag_backend/api/routes/copilot.py` (updated)
- `backend/tests/mcp_client/test_registry.py` (new)
- `backend/tests/api/routes/test_copilot_tools.py` (new)
- `frontend/app/api/copilotkit/route.ts` (updated)
