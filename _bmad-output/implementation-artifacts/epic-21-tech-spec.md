# Epic 21 Tech Spec: CopilotKit Full Integration

**Date:** 2026-01-06
**Status:** Backlog
**Epic Owner:** Product and Engineering
**Origin:** Party Mode Deep Dive Analysis (2026-01-06)

---

## Overview

Epic 21 delivers full CopilotKit feature integration, unlocking capabilities we're not currently using despite having the CopilotKit dependency. This epic addresses the gap between what CopilotKit provides and what our implementation utilizes.

### Strategic Context

This epic was identified through comprehensive analysis using DeepWiki and Context7 MCP tools against the CopilotKit repository. The analysis revealed:

1. **Native A2UI Support**: CopilotKit is a launch partner for Google's A2UI spec - we're not using it
2. **MCP Client Integration**: CopilotKit supports outbound MCP connections - we only have an MCP server
3. **Modern Hook Patterns**: Our code uses deprecated `useCopilotAction` - should migrate to `useFrontendTool`, `useHumanInTheLoop`, `useRenderToolCall`
4. **Observability Hooks**: CopilotKit provides comprehensive observability - we have none configured
5. **Missing AG-UI Events**: We're missing `STATE_DELTA`, `RUN_ERROR`, and other event types

### Research Sources

| Source | Usage |
|--------|-------|
| DeepWiki: CopilotKit/CopilotKit | AG-UI protocol architecture, event types, hook documentation |
| Context7: /copilotkit/copilotkit | Code examples, API patterns, configuration schemas |
| Our Codebase Audit | Gap analysis against current implementation |

### Goals

- Achieve full utilization of CopilotKit capabilities
- Migrate from deprecated to modern hook patterns
- Enable MCP client integration for external tool ecosystem
- Implement A2UI declarative widget rendering
- Add comprehensive observability for debugging and analytics
- Support all AG-UI event types for rich UI experiences

### Related Epics

| Epic | Relationship |
|------|-------------|
| Epic 6: Interactive Copilot Experience | Original CopilotKit integration (completed) |
| Epic 7: Protocol Integration | MCP server, A2A protocol (completed) |
| Epic 22: Advanced Protocol Integration | A2A middleware, advanced protocols (future) |

---

## Current Implementation Audit

### What We Have

| Feature | Location | Status |
|---------|----------|--------|
| CopilotKit Provider | `frontend/components/copilot/CopilotProvider.tsx` | Minimal - runtimeUrl only |
| Chat Sidebar | `frontend/components/copilot/ChatSidebar.tsx` | Uses `@copilotkit/react-ui` |
| Generative UI | `frontend/hooks/use-generative-ui.tsx` | Custom actions, not A2UI |
| Source Validation | `frontend/hooks/use-source-validation.ts` | Uses deprecated pattern |
| Frontend Actions | `frontend/hooks/use-copilot-actions.ts` | 5x deprecated `useCopilotAction` |
| AG-UI Bridge | `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` | Basic events only |
| MCP Server | `backend/src/agentic_rag_backend/mcp_server/` | Fully implemented |
| AG-UI Events | `backend/src/agentic_rag_backend/models/copilot.py` | Missing STATE_DELTA, RUN_ERROR |

### What We're Missing

| Feature | CopilotKit Capability | Gap |
|---------|----------------------|-----|
| Tool Call Visualization | `useRenderToolCall` | Not used anywhere |
| Observability | `observabilityHooks` + `publicApiKey` | Not configured |
| MCP Client | `mcpServers` + `createMCPClient` | No outbound MCP |
| A2UI Widgets | Native A2UI rendering | Using custom actions instead |
| STATE_DELTA | Incremental state updates | Only STATE_SNAPSHOT |
| RUN_ERROR | Error event emission | Errors embedded in text |
| useFrontendTool | Modern tool pattern | Using deprecated useCopilotAction |
| useHumanInTheLoop | Modern HITL pattern | Using deprecated render pattern |
| useCoAgent | Agent state sync | Not used |
| useCopilotReadable | Context provision | Not used |

---

## Story Groups

### Group A: Modern Hook Migration (Tech Debt)

*Focus: Migrate from deprecated patterns to modern CopilotKit hooks*
*Priority: P0 - Future-proofs codebase*

#### Story 21-A1: Migrate to useFrontendTool Pattern

**Priority:** P0 - HIGH
**Story Points:** 5
**Owner:** Frontend

**Objective:** Replace deprecated `useCopilotAction` with handler to modern `useFrontendTool`.

**Current Pattern (Deprecated):**
```tsx
// frontend/hooks/use-copilot-actions.ts (current)
useCopilotAction({
  name: "save_to_workspace",
  description: "Save content to workspace",
  parameters: [...],
  handler: async ({ content_id, content_text }) => {
    // handler logic
  },
});
```

**Target Pattern (Modern):**
```tsx
// frontend/hooks/use-frontend-tools.ts (new)
import { useFrontendTool } from "@copilotkit/react-core";

useFrontendTool({
  name: "save_to_workspace",
  description: "Save content to workspace",
  parameters: z.object({
    content_id: z.string().describe("Unique ID of the content"),
    content_text: z.string().describe("Content to save"),
    title: z.string().optional().describe("Optional title"),
  }),
  handler: async ({ content_id, content_text, title }) => {
    // handler logic - now with Zod validation
    return { saved: true, id: content_id };
  },
});
```

**Files to Migrate:**
| File | Actions to Migrate | Effort |
|------|-------------------|--------|
| `use-copilot-actions.ts` | save_to_workspace, export_content, share_content, bookmark_content, suggest_follow_up | High |
| `use-generative-ui.tsx` | show_sources, show_answer, show_knowledge_graph | Medium |

**Configuration:**
```bash
# No new configuration - using existing CopilotKit setup
```

**Acceptance Criteria:**
1. All `useCopilotAction` with `handler` migrated to `useFrontendTool`
2. Zod schemas replace inline parameter definitions
3. Type safety improved with inferred types from Zod
4. All existing tests updated and passing
5. No runtime behavior changes for users
6. Documentation updated in code comments

**Technical Approach:**
1. Create new `frontend/hooks/use-frontend-tools.ts`
2. Define Zod schemas for each action's parameters
3. Migrate handlers one by one with tests
4. Update imports in `GenerativeUIRenderer.tsx`
5. Remove deprecated action registrations
6. Run full test suite

---

#### Story 21-A2: Migrate to useHumanInTheLoop Pattern

**Priority:** P0 - HIGH
**Story Points:** 5
**Owner:** Frontend

**Objective:** Replace deprecated `useCopilotAction` with `render` to modern `useHumanInTheLoop`.

**Current Pattern (Deprecated):**
```tsx
// frontend/hooks/use-source-validation.ts (current)
useCopilotAction({
  name: "validate_sources",
  description: "Request human approval for sources",
  parameters: [...],
  render: ({ status, args }) => {
    if (status === "executing" && args.sources) {
      setTimeout(() => startValidation(args.sources), 0);
    }
    return <></>;
  },
});
```

**Target Pattern (Modern):**
```tsx
// frontend/hooks/use-hitl-validation.ts (new)
import { useHumanInTheLoop } from "@copilotkit/react-core";

useHumanInTheLoop({
  name: "validate_sources",
  description: "Request human approval for sources",
  parameters: z.object({
    sources: z.array(SourceSchema).describe("Sources requiring validation"),
    query: z.string().optional().describe("Original query for context"),
  }),
  render: ({ args, respond, status }) => {
    if (status === "executing" && respond) {
      return (
        <SourceValidationDialog
          sources={args.sources}
          onApprove={(ids) => respond({ approved: ids })}
          onReject={() => respond({ approved: [] })}
        />
      );
    }
    return null;
  },
});
```

**Files to Migrate:**
| File | Pattern | New Hook |
|------|---------|----------|
| `use-source-validation.ts` | render + setTimeout hack | `useHumanInTheLoop` |
| `use-generative-ui.tsx` | render-only actions | `useRenderToolCall` (see 21-A3) |

**Acceptance Criteria:**
1. `validate_sources` migrated to `useHumanInTheLoop`
2. Remove setTimeout workaround (proper respond callback usage)
3. HITL dialog integrates with respond callback
4. Tests verify approval/rejection flows
5. No changes to backend AG-UI protocol

---

#### Story 21-A3: Implement Tool Call Visualization

**Priority:** P0 - HIGH
**Story Points:** 5
**Owner:** Frontend

**Objective:** Add `useRenderToolCall` to visualize all tool calls in the chat interface.

**What `useRenderToolCall` Provides:**
- Status tracking: InProgress → Executing → Complete
- Automatic renderer selection by tool name
- Wildcard renderer for unregistered tools
- Clean separation of visualization from execution

**Implementation:**

```tsx
// frontend/components/copilot/ToolCallRenderer.tsx (new)
import { useRenderToolCall } from "@copilotkit/react-core";

export function ToolCallRenderer() {
  // Render MCP tool calls
  useRenderToolCall({
    name: "*", // Wildcard - catches all unregistered tools
    render: ({ name, args, status, result }) => (
      <MCPToolCallCard
        name={name}
        args={args}
        status={status}
        result={result}
      />
    ),
  });

  // Specific renderer for RAG tools
  useRenderToolCall({
    name: "vector_search",
    render: ({ args, status, result }) => (
      <VectorSearchCard
        query={args.query}
        isSearching={status === "executing"}
        results={status === "complete" ? result : null}
      />
    ),
  });

  return null; // Renderers are registered, no visible output
}
```

**MCPToolCallCard Component:**
```tsx
// frontend/components/copilot/MCPToolCallCard.tsx (new)
interface MCPToolCallCardProps {
  name: string;
  args: Record<string, unknown>;
  status: "inProgress" | "executing" | "complete";
  result?: unknown;
}

export function MCPToolCallCard({ name, args, status, result }: MCPToolCallCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <Card className="my-2">
      <CardHeader className="p-3 cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="flex items-center gap-2">
          <StatusBadge status={status} />
          <span className="font-mono text-sm">{name}</span>
          <ChevronIcon expanded={isExpanded} />
        </div>
      </CardHeader>
      {isExpanded && (
        <CardContent className="p-3 pt-0">
          <div className="space-y-2">
            <div>
              <Label>Arguments</Label>
              <pre className="text-xs bg-muted p-2 rounded overflow-auto max-h-40">
                {JSON.stringify(args, null, 2)}
              </pre>
            </div>
            {status === "complete" && result && (
              <div>
                <Label>Result</Label>
                <pre className="text-xs bg-muted p-2 rounded overflow-auto max-h-40">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </CardContent>
      )}
    </Card>
  );
}
```

**Files to Create:**
| File | Purpose |
|------|---------|
| `frontend/components/copilot/ToolCallRenderer.tsx` | Register tool call renderers |
| `frontend/components/copilot/MCPToolCallCard.tsx` | Generic tool call UI |
| `frontend/components/copilot/StatusBadge.tsx` | Status indicator (InProgress/Executing/Complete) |
| `frontend/components/copilot/VectorSearchCard.tsx` | RAG-specific tool card |

**Acceptance Criteria:**
1. All MCP tool calls visible in chat interface
2. Status badges show InProgress → Executing → Complete transitions
3. Collapsible JSON viewer for arguments and results
4. Specific renderers for common tools (vector_search, ingest_url)
5. Wildcard renderer catches unregistered tools
6. Tests verify rendering for each status state

---

### Group B: Observability & Debugging

*Focus: Enable production debugging and analytics*
*Priority: P0 - Critical for operations*

#### Story 21-B1: Configure Observability Hooks

**Priority:** P0 - HIGH
**Story Points:** 3
**Owner:** Frontend + Backend

**Objective:** Wire CopilotKit observability hooks to our telemetry pipeline.

**Available Observability Hooks:**
| Hook | Trigger | Data |
|------|---------|------|
| `onMessageSent` | User sends message | Message content |
| `onChatExpanded` | Chat sidebar opens | - |
| `onChatMinimized` | Chat sidebar closes | - |
| `onMessageRegenerated` | Regenerate clicked | Message ID |
| `onMessageCopied` | Copy button clicked | Content |
| `onFeedbackGiven` | Thumbs up/down | Message ID, type |
| `onChatStarted` | AI starts responding | - |
| `onChatStopped` | AI stops responding | - |
| `onError` | Error occurs | Error event |

**Implementation:**

```tsx
// frontend/components/copilot/CopilotProvider.tsx (updated)
import { CopilotKit } from "@copilotkit/react-core";
import { CopilotSidebar } from "@copilotkit/react-ui";
import { useAnalytics } from "@/hooks/use-analytics";

export function CopilotProvider({ children }: CopilotProviderProps) {
  const analytics = useAnalytics();

  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      publicApiKey={process.env.NEXT_PUBLIC_COPILOTKIT_API_KEY}
    >
      <CopilotSidebar
        observabilityHooks={{
          onMessageSent: (message) => {
            analytics.track("copilot_message_sent", {
              messageLength: message.length,
              timestamp: new Date().toISOString(),
            });
          },
          onChatExpanded: () => {
            analytics.track("copilot_chat_expanded");
          },
          onChatMinimized: () => {
            analytics.track("copilot_chat_minimized");
          },
          onMessageRegenerated: (messageId) => {
            analytics.track("copilot_message_regenerated", { messageId });
          },
          onMessageCopied: (content) => {
            analytics.track("copilot_message_copied", {
              contentLength: content.length
            });
          },
          onFeedbackGiven: (messageId, type) => {
            analytics.track("copilot_feedback", { messageId, type });
          },
          onChatStarted: () => {
            analytics.track("copilot_generation_started");
          },
          onChatStopped: () => {
            analytics.track("copilot_generation_stopped");
          },
          onError: (error) => {
            analytics.track("copilot_error", {
              error: error.message,
              code: error.code,
            });
            console.error("CopilotKit error:", error);
          },
        }}
      />
      {children}
    </CopilotKit>
  );
}
```

**Configuration:**
```bash
# .env.local
NEXT_PUBLIC_COPILOTKIT_API_KEY=ck_pub_xxx  # For Copilot Cloud (optional)
# OR
NEXT_PUBLIC_COPILOTKIT_LICENSE_KEY=ck_lic_xxx  # For self-hosted with Inspector
```

**Analytics Hook:**
```tsx
// frontend/hooks/use-analytics.ts (new or extend existing)
export function useAnalytics() {
  const track = useCallback((event: string, properties?: Record<string, unknown>) => {
    // Send to backend telemetry endpoint
    fetch('/api/telemetry', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event, properties, timestamp: new Date().toISOString() }),
    }).catch(console.error);

    // Also log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.log('[Analytics]', event, properties);
    }
  }, []);

  return { track };
}
```

**Acceptance Criteria:**
1. All observability hooks wired to analytics
2. Events emitted to backend telemetry endpoint
3. Error events logged with full context
4. Development mode console logging
5. Optional Copilot Cloud integration (with API key)
6. Metrics visible in application logs

---

#### Story 21-B2: Add RUN_ERROR Event Support

**Priority:** P1 - MEDIUM
**Story Points:** 3
**Owner:** Backend

**Objective:** Emit proper RUN_ERROR events instead of embedding errors in text.

**Current Behavior:**
```python
# backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py (current)
except Exception as e:
    logger.exception("copilot_request_failed", error=str(e))
    yield TextMessageStartEvent()
    yield TextDeltaEvent(content=GENERIC_ERROR_MESSAGE)  # Error in text!
    yield TextMessageEndEvent()
```

**Target Behavior:**
```python
# backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py (new)
except Exception as e:
    logger.exception("copilot_request_failed", error=str(e))
    yield RunErrorEvent(
        code="AGENT_EXECUTION_ERROR",
        message=GENERIC_ERROR_MESSAGE,
        details={"error_type": type(e).__name__} if settings.debug else None,
    )
```

**New Event Model:**
```python
# backend/src/agentic_rag_backend/models/copilot.py (add)
class RunErrorEvent(AGUIEvent):
    """Event emitted when agent run fails with error."""

    event: AGUIEventType = AGUIEventType.RUN_ERROR

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            data={
                "code": code,
                "message": message,
                "details": details or {},
            },
            **kwargs
        )
```

**Error Codes:**
| Code | When | User Message |
|------|------|--------------|
| `AGENT_EXECUTION_ERROR` | Agent throws exception | "An error occurred processing your request" |
| `TENANT_REQUIRED` | Missing tenant_id | "Authentication required" |
| `RATE_LIMITED` | Too many requests | "Please wait before trying again" |
| `TIMEOUT` | Request timeout | "Request took too long" |
| `INVALID_REQUEST` | Bad request format | "Invalid request format" |

**Frontend Handling:**
```tsx
// frontend/components/copilot/ErrorHandler.tsx (new)
import { useEffect } from "react";
import { useCopilotChat } from "@copilotkit/react-core";
import { toast } from "@/hooks/use-toast";

export function CopilotErrorHandler() {
  // Listen for RUN_ERROR events
  // Display toast notification with error details
  // Provide retry option where appropriate

  return null;
}
```

**Acceptance Criteria:**
1. `RUN_ERROR` event type added to `AGUIEventType` enum
2. `RunErrorEvent` class implemented with code, message, details
3. AGUIBridge emits `RUN_ERROR` on exceptions
4. Frontend handles `RUN_ERROR` events gracefully
5. Toast notification for user-facing errors
6. Error details hidden in production (visible in debug mode)
7. Tests verify error event emission

---

#### Story 21-B3: Implement STATE_DELTA Support

**Priority:** P1 - MEDIUM
**Story Points:** 5
**Owner:** Backend + Frontend

**Objective:** Add incremental state updates using JSON Patch operations.

**Why STATE_DELTA:**
- `STATE_SNAPSHOT` replaces entire state (inefficient for large states)
- `STATE_DELTA` applies incremental JSON Patch operations
- Enables real-time progress updates without full state transfer
- Matches CopilotKit's native state synchronization

**Implementation:**

```python
# backend/src/agentic_rag_backend/models/copilot.py (add)
class StateDeltaEvent(AGUIEvent):
    """Event for incremental state updates using JSON Patch."""

    event: AGUIEventType = AGUIEventType.STATE_DELTA

    def __init__(self, operations: list[dict[str, Any]], **kwargs: Any) -> None:
        """
        Args:
            operations: JSON Patch operations (RFC 6902)
                [{"op": "add", "path": "/steps/0", "value": {...}}]
                [{"op": "replace", "path": "/currentStep", "value": "processing"}]
        """
        super().__init__(data={"delta": operations}, **kwargs)
```

**Usage in AGUIBridge:**
```python
# Emit step progress incrementally
yield StateDeltaEvent(operations=[
    {"op": "add", "path": "/steps/-", "value": {"step": "Searching...", "status": "in_progress"}},
])

# Later, update step status
yield StateDeltaEvent(operations=[
    {"op": "replace", "path": "/steps/0/status", "value": "completed"},
])
```

**Frontend Handling:**
```tsx
// Frontend automatically handles STATE_DELTA via useCoAgent/useCoAgentStateRender
// The CopilotKit runtime applies JSON Patch operations to shared state
```

**Acceptance Criteria:**
1. `STATE_DELTA` event type added to enum
2. `StateDeltaEvent` class with JSON Patch operations
3. AGUIBridge uses STATE_DELTA for incremental updates
4. Frontend receives and applies deltas correctly
5. Tests verify delta application
6. Performance improvement for large state updates

---

### Group C: MCP Client Integration

*Focus: Connect to external MCP tool ecosystem*
*Priority: P1 - Ecosystem expansion*

#### Story 21-C1: Implement MCP Client Configuration

**Priority:** P1 - HIGH
**Story Points:** 5
**Owner:** Backend

**Objective:** Add configuration for connecting to external MCP servers.

**Configuration Schema:**
```bash
# .env
# MCP Client Configuration
MCP_CLIENTS_ENABLED=true|false  # Default: false
MCP_CLIENT_TIMEOUT=30000  # Timeout in ms
MCP_CLIENT_RETRY_COUNT=3  # Number of retries
MCP_CLIENT_RETRY_DELAY=1000  # Delay between retries in ms

# MCP Server Endpoints (JSON array)
MCP_CLIENT_SERVERS='[
  {"name": "github", "url": "https://mcp.github.com/sse", "apiKey": "${GITHUB_MCP_KEY}"},
  {"name": "notion", "url": "https://mcp.notion.so/sse", "apiKey": "${NOTION_MCP_KEY}"}
]'
```

**Configuration Model:**
```python
# backend/src/agentic_rag_backend/mcp_client/config.py (new)
from pydantic import BaseModel, HttpUrl
from typing import Optional

class MCPServerConfig(BaseModel):
    """Configuration for an external MCP server."""
    name: str  # Unique identifier
    url: HttpUrl  # MCP server endpoint
    apiKey: Optional[str] = None  # Optional API key
    transport: str = "sse"  # "sse" or "http"
    timeout: int = 30000  # Timeout in ms

class MCPClientSettings(BaseModel):
    """MCP Client configuration."""
    enabled: bool = False
    servers: list[MCPServerConfig] = []
    default_timeout: int = 30000
    retry_count: int = 3
    retry_delay: int = 1000
```

**Acceptance Criteria:**
1. MCP client configuration schema defined
2. Environment variables parsed at startup
3. Multiple MCP server endpoints supported
4. API key injection from environment
5. Configuration validation on startup
6. Settings accessible via dependency injection

---

#### Story 21-C2: Implement MCP Client Factory

**Priority:** P1 - HIGH
**Story Points:** 8
**Owner:** Backend

**Objective:** Create MCP client instances for connecting to external servers.

**Implementation:**
```python
# backend/src/agentic_rag_backend/mcp_client/client.py (new)
import httpx
from typing import AsyncIterator, Any
import structlog

logger = structlog.get_logger(__name__)

class MCPClient:
    """Client for connecting to external MCP servers."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._http_client = httpx.AsyncClient(
            timeout=config.timeout / 1000,  # Convert to seconds
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.apiKey:
            headers["Authorization"] = f"Bearer {self.config.apiKey}"
        return headers

    async def list_tools(self) -> list[dict[str, Any]]:
        """Discover available tools from the MCP server."""
        response = await self._request("tools/list", {})
        return response.get("tools", [])

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool on the MCP server."""
        return await self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send JSON-RPC request to MCP server."""
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        for attempt in range(self.config.retry_count + 1):
            try:
                response = await self._http_client.post(
                    str(self.config.url),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    raise MCPClientError(data["error"])

                return data.get("result", {})

            except httpx.TimeoutException:
                if attempt == self.config.retry_count:
                    raise MCPClientTimeoutError(f"Timeout after {attempt + 1} attempts")
                await asyncio.sleep(self.config.retry_delay / 1000)

            except httpx.HTTPStatusError as e:
                if attempt == self.config.retry_count:
                    raise MCPClientError(f"HTTP error: {e.response.status_code}")
                await asyncio.sleep(self.config.retry_delay / 1000)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http_client.aclose()


class MCPClientFactory:
    """Factory for creating and managing MCP clients."""

    def __init__(self, settings: MCPClientSettings) -> None:
        self.settings = settings
        self._clients: dict[str, MCPClient] = {}

    async def get_client(self, name: str) -> MCPClient:
        """Get or create an MCP client by server name."""
        if name not in self._clients:
            config = self._find_server_config(name)
            if not config:
                raise MCPClientError(f"Unknown MCP server: {name}")
            self._clients[name] = MCPClient(config)
        return self._clients[name]

    def _find_server_config(self, name: str) -> MCPServerConfig | None:
        for server in self.settings.servers:
            if server.name == name:
                return server
        return None

    async def discover_all_tools(self) -> dict[str, list[dict]]:
        """Discover tools from all configured MCP servers."""
        tools = {}
        for server in self.settings.servers:
            try:
                client = await self.get_client(server.name)
                tools[server.name] = await client.list_tools()
            except Exception as e:
                logger.warning("mcp_tool_discovery_failed", server=server.name, error=str(e))
        return tools

    async def close_all(self) -> None:
        """Close all MCP clients."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
```

**Acceptance Criteria:**
1. `MCPClient` class handles HTTP/SSE communication
2. `MCPClientFactory` manages client lifecycle
3. Tool discovery from all configured servers
4. Retry logic with exponential backoff
5. Timeout handling with circuit breaker
6. Connection pooling for efficiency
7. Graceful shutdown on application exit

---

#### Story 21-C3: Wire MCP Client to CopilotRuntime

**Priority:** P1 - MEDIUM
**Story Points:** 5
**Owner:** Backend + Frontend

**Objective:** Integrate MCP client with CopilotKit runtime for unified tool access.

**Backend Integration:**
```python
# backend/src/agentic_rag_backend/api/routes/copilot.py (update)
from agentic_rag_backend.mcp_client import MCPClientFactory

@router.post("/copilotkit")
async def copilotkit_endpoint(
    request: CopilotRequest,
    mcp_factory: MCPClientFactory = Depends(get_mcp_factory),
):
    # Discover external MCP tools
    external_tools = await mcp_factory.discover_all_tools()

    # Merge with internal tools
    all_tools = merge_tool_registries(internal_tools, external_tools)

    # Process request with unified tool access
    ...
```

**Frontend Integration:**
```tsx
// frontend/app/api/copilotkit/route.ts (update)
import { CopilotRuntime } from "@copilotkit/runtime";
import { experimental_createMCPClient } from "ai";

const runtime = new CopilotRuntime({
  // Existing configuration...

  // Add MCP client configuration
  mcpServers: [
    { endpoint: process.env.GITHUB_MCP_URL },
    { endpoint: process.env.NOTION_MCP_URL },
  ],

  createMCPClient: async (config) => {
    return await experimental_createMCPClient({
      transport: {
        type: "sse",
        url: config.endpoint,
        headers: config.apiKey
          ? { Authorization: `Bearer ${config.apiKey}` }
          : undefined,
      },
    });
  },
});
```

**Tool Registry Merging:**
```python
def merge_tool_registries(
    internal: dict[str, list],
    external: dict[str, list],
) -> dict[str, list]:
    """Merge internal and external tool registries with namespacing."""
    merged = {}

    # Add internal tools (no namespace)
    for tool in internal.get("internal", []):
        merged[tool["name"]] = tool

    # Add external tools (namespaced by server)
    for server_name, tools in external.items():
        for tool in tools:
            namespaced_name = f"{server_name}:{tool['name']}"
            merged[namespaced_name] = {
                **tool,
                "name": namespaced_name,
                "source": server_name,
            }

    return merged
```

**Acceptance Criteria:**
1. External MCP tools discoverable at startup
2. Tools namespaced by source server
3. Unified tool registry for orchestrator
4. Proper error handling for unavailable servers
5. Tool metadata includes source for UI display
6. Tests verify tool discovery and merging

---

### Group D: A2UI Widget Rendering

*Focus: Enable Google's A2UI declarative UI specification*
*Priority: P1 - Richer agent responses*

#### Story 21-D1: Enable A2UI Support

**Priority:** P1 - MEDIUM
**Story Points:** 3
**Owner:** Backend

**Objective:** Allow agents to emit A2UI widget payloads in responses.

**A2UI Widget Types:**
| Type | Purpose | Example |
|------|---------|---------|
| `card` | Display content card | Weather, status, summary |
| `table` | Tabular data | Search results, comparisons |
| `form` | Input collection | Settings, preferences |
| `chart` | Data visualization | Metrics, trends |
| `image` | Image display | Diagrams, photos |
| `list` | Item listing | Steps, options |

**Backend Emission:**
```python
# backend/src/agentic_rag_backend/protocols/a2ui.py (new)
from pydantic import BaseModel
from typing import Any, Literal

class A2UIWidget(BaseModel):
    """A2UI widget payload."""
    type: Literal["card", "table", "form", "chart", "image", "list"]
    properties: dict[str, Any]

def create_a2ui_card(
    title: str,
    content: str,
    actions: list[dict] | None = None,
) -> A2UIWidget:
    """Create an A2UI card widget."""
    return A2UIWidget(
        type="card",
        properties={
            "title": title,
            "content": content,
            "actions": actions or [],
        },
    )

def create_a2ui_table(
    headers: list[str],
    rows: list[list[str]],
    caption: str | None = None,
) -> A2UIWidget:
    """Create an A2UI table widget."""
    return A2UIWidget(
        type="table",
        properties={
            "headers": headers,
            "rows": rows,
            "caption": caption,
        },
    )
```

**AG-UI Event Emission:**
```python
# In AGUIBridge
yield StateSnapshotEvent(state={
    "a2ui_widgets": [
        create_a2ui_card(
            title="Search Results",
            content=f"Found {len(sources)} relevant documents",
        ).model_dump(),
    ],
})
```

**Acceptance Criteria:**
1. A2UI widget models defined with validation
2. Helper functions for common widget types
3. Widgets emitted via STATE_SNAPSHOT
4. Schema validation for widget properties
5. Tests verify widget serialization

---

#### Story 21-D2: Implement A2UI Widget Renderer

**Priority:** P1 - MEDIUM
**Story Points:** 5
**Owner:** Frontend

**Objective:** Map A2UI widget types to React components.

**Implementation:**
```tsx
// frontend/components/copilot/A2UIRenderer.tsx (new)
import { useCoAgentStateRender } from "@copilotkit/react-core";
import { Card, Table, Chart } from "@/components/ui";

interface A2UIWidget {
  type: "card" | "table" | "form" | "chart" | "image" | "list";
  properties: Record<string, unknown>;
}

interface A2UIState {
  a2ui_widgets?: A2UIWidget[];
}

export function A2UIRenderer() {
  useCoAgentStateRender<A2UIState>({
    name: "orchestrator",
    render: ({ state }) => {
      if (!state?.a2ui_widgets?.length) return null;

      return (
        <div className="space-y-2 my-2">
          {state.a2ui_widgets.map((widget, idx) => (
            <A2UIWidget key={idx} widget={widget} />
          ))}
        </div>
      );
    },
  });

  return null;
}

function A2UIWidget({ widget }: { widget: A2UIWidget }) {
  switch (widget.type) {
    case "card":
      return <A2UICard {...widget.properties} />;
    case "table":
      return <A2UITable {...widget.properties} />;
    case "chart":
      return <A2UIChart {...widget.properties} />;
    case "form":
      return <A2UIForm {...widget.properties} />;
    case "list":
      return <A2UIList {...widget.properties} />;
    case "image":
      return <A2UIImage {...widget.properties} />;
    default:
      return <A2UIFallback widget={widget} />;
  }
}

function A2UICard({ title, content, actions }: CardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>{content}</CardContent>
      {actions?.length > 0 && (
        <CardFooter>
          {actions.map((action, idx) => (
            <Button key={idx} onClick={() => handleAction(action)}>
              {action.label}
            </Button>
          ))}
        </CardFooter>
      )}
    </Card>
  );
}

function A2UIFallback({ widget }: { widget: A2UIWidget }) {
  // Fallback for unsupported widget types
  return (
    <Card className="bg-muted">
      <CardContent className="text-sm text-muted-foreground">
        Unsupported widget type: {widget.type}
        <pre className="mt-2 text-xs">
          {JSON.stringify(widget.properties, null, 2)}
        </pre>
      </CardContent>
    </Card>
  );
}
```

**Component Mapping:**
| A2UI Type | Our Component | Notes |
|-----------|---------------|-------|
| `card` | `A2UICard` → shadcn `Card` | Maps to our design system |
| `table` | `A2UITable` → shadcn `Table` | Column headers + rows |
| `chart` | `A2UIChart` → Recharts | Line, bar, pie support |
| `form` | `A2UIForm` → React Hook Form | Dynamic field generation |
| `list` | `A2UIList` → Ordered/unordered list | Bullet/numbered |
| `image` | `A2UIImage` → Next.js Image | With lazy loading |

**Acceptance Criteria:**
1. All A2UI widget types mapped to components
2. Graceful fallback for unsupported types
3. Components use our shadcn/ui design system
4. Charts rendered via Recharts
5. Forms handle submit via callback
6. Tests verify each widget type rendering
7. Visual regression tests for widget styles

---

## Testing Requirements

### Unit Tests

| Story | Test Focus | Coverage Target |
|-------|------------|-----------------|
| 21-A1 | useFrontendTool migration | 90% |
| 21-A2 | useHumanInTheLoop migration | 90% |
| 21-A3 | useRenderToolCall renderers | 85% |
| 21-B1 | Observability hook firing | 80% |
| 21-B2 | RUN_ERROR event emission | 90% |
| 21-B3 | STATE_DELTA JSON Patch | 90% |
| 21-C1 | MCP client configuration | 85% |
| 21-C2 | MCP client factory | 85% |
| 21-C3 | Tool registry merging | 90% |
| 21-D1 | A2UI widget models | 90% |
| 21-D2 | A2UI widget rendering | 85% |

### Integration Tests

| Test Scenario | Stories |
|---------------|---------|
| Full tool call lifecycle (start → execute → complete → render) | 21-A3 |
| HITL dialog approval flow | 21-A2 |
| MCP client discovery → tool call → result | 21-C1, 21-C2, 21-C3 |
| A2UI widget emission → render | 21-D1, 21-D2 |
| Error event emission → toast display | 21-B2 |
| STATE_DELTA application → UI update | 21-B3 |

### E2E Tests (Playwright)

| Test | Purpose |
|------|---------|
| Tool call card visibility | Verify MCP tools show in chat |
| HITL dialog interaction | Click approve/reject |
| A2UI card rendering | Verify card displays correctly |
| Error toast appearance | Verify error notification |

---

## Configuration Summary

### New Environment Variables

```bash
# Epic 21 - CopilotKit Full Integration

# --- Observability (21-B1) ---
NEXT_PUBLIC_COPILOTKIT_API_KEY=ck_pub_xxx  # Optional: Copilot Cloud
NEXT_PUBLIC_COPILOTKIT_LICENSE_KEY=ck_lic_xxx  # Optional: Self-hosted Inspector

# --- MCP Client (21-C1, 21-C2) ---
MCP_CLIENTS_ENABLED=true|false  # Default: false
MCP_CLIENT_TIMEOUT=30000  # Timeout in ms
MCP_CLIENT_RETRY_COUNT=3
MCP_CLIENT_RETRY_DELAY=1000
MCP_CLIENT_SERVERS='[{"name":"github","url":"...","apiKey":"${GITHUB_MCP_KEY}"}]'

# --- A2UI (21-D1) ---
A2UI_ENABLED=true|false  # Default: true (if using CopilotKit)
A2UI_FALLBACK_RENDER=true  # Show fallback for unsupported widgets
```

---

## Dependencies

### External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `@copilotkit/react-core` | ^1.x | Modern hooks |
| `@copilotkit/react-ui` | ^1.x | UI components |
| `@copilotkit/runtime` | ^1.x | Runtime configuration |
| `zod` | ^3.x | Schema validation |
| `jsonpointer` | ^5.x | JSON Patch operations |

### Internal Dependencies

| Module | Dependency |
|--------|-----------|
| 21-A2 | 21-A1 (shared hook patterns) |
| 21-A3 | 21-A1, 21-A2 (tool renderers) |
| 21-C3 | 21-C1, 21-C2 (MCP client) |
| 21-D2 | 21-D1 (A2UI models) |

---

## Sprint Allocation

### Sprint 1 (Stories: 21-A1, 21-A2, 21-A3, 21-B1)
- Focus: Hook migration + Tool visualization + Observability
- Points: 18
- Goal: Modern patterns, debugging capabilities

### Sprint 2 (Stories: 21-B2, 21-B3, 21-C1, 21-C2)
- Focus: Error handling + STATE_DELTA + MCP client foundation
- Points: 16
- Goal: Robust error handling, MCP client ready

### Sprint 3 (Stories: 21-C3, 21-D1, 21-D2)
- Focus: MCP integration + A2UI rendering
- Points: 13
- Goal: External tools + Rich UI widgets

**Total: 47 story points across 3 sprints**

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Deprecated hook usage | 12 instances | 0 | Code search |
| Tool call visibility | 0% | 100% | All MCP tools rendered |
| Observability coverage | 0 hooks | 9 hooks | All hooks wired |
| MCP ecosystem tools | 0 | 10+ | External server tools |
| A2UI widget types | 0 | 6 | Supported widget types |
| Error transparency | Text-only | Event-based | RUN_ERROR emissions |

---

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| CopilotKit API changes | High | Low | Pin versions, monitor releases |
| External MCP server downtime | Medium | Medium | Circuit breakers, fallbacks |
| A2UI spec changes | Low | Low | Pin A2UI version, fallback rendering |
| Hook migration regressions | High | Medium | Comprehensive tests, gradual rollout |

---

## References

- [CopilotKit Documentation](https://docs.copilotkit.ai/)
- [A2UI Specification](https://github.com/anthropics/a2ui)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [AG-UI Protocol](https://github.com/copilotkit/ag-ui)
- DeepWiki: CopilotKit/CopilotKit (research source)
- Context7: /copilotkit/copilotkit (code examples)
