# Story 21-C1: Implement MCP Client Configuration

## Status: Review

## Story

As a developer, I need to configure connections to external MCP servers, so that I can integrate tools from the broader MCP ecosystem (GitHub, Notion, etc.).

## Implementation Summary

### Backend Changes

1. **Added MCP Client Settings to Settings dataclass** (`config.py`)
   - `mcp_clients_enabled`: bool - Enable/disable MCP client feature
   - `mcp_client_timeout_ms`: int - Request timeout in milliseconds (default: 30000)
   - `mcp_client_retry_count`: int - Number of retries (default: 3, max: 10)
   - `mcp_client_retry_delay_ms`: int - Base delay between retries (default: 1000ms)
   - `mcp_client_servers`: list[dict] - JSON array of server configurations

2. **Added Environment Variable Loading** (`config.py`)
   - `MCP_CLIENTS_ENABLED` - Boolean to enable MCP client feature
   - `MCP_CLIENT_TIMEOUT` - Timeout in milliseconds
   - `MCP_CLIENT_RETRY_COUNT` - Retry count (capped at 10)
   - `MCP_CLIENT_RETRY_DELAY` - Retry delay in milliseconds
   - `MCP_CLIENT_SERVERS` - JSON array of server configs

3. **Updated .env.example**
   - Added all MCP client environment variables with documentation
   - Example server config format: `[{"name": "github", "url": "https://...", "apiKey": "..."}]`

### Configuration Schema

```json
MCP_CLIENT_SERVERS='[
  {"name": "github", "url": "https://mcp.github.com/sse", "apiKey": "${GITHUB_MCP_KEY}"},
  {"name": "notion", "url": "https://mcp.notion.so/sse", "apiKey": "${NOTION_MCP_KEY}"}
]'
```

## Dev Notes

- MCP Client configuration is separate from MCP Server (hosting) configuration
- Server configs support `name`, `url`, `apiKey`, `transport`, and `timeout` fields
- Retry count is capped at 10 to prevent infinite retry loops
- JSON parsing fails gracefully to empty array with warning log

## Acceptance Criteria

- [x] MCP client configuration schema defined in Settings
- [x] Environment variables parsed at startup
- [x] Multiple MCP server endpoints supported via JSON array
- [x] Configuration validation on startup (min values enforced)
- [x] Settings accessible via dependency injection (get_settings)
- [x] .env.example updated with documentation

## Files Changed

- `backend/src/agentic_rag_backend/config.py` - Added MCP client settings and env loading
- `.env.example` - Added MCP client environment variables
