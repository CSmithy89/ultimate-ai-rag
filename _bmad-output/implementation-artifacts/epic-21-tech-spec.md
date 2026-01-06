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
| useCopilotReadable | Context provision | Not used → 21-A4 |
| useCopilotChatSuggestions | Contextual suggestions | Not used |
| useCopilotChat | Headless chat control | Not used |
| useCopilotAdditionalInstructions | Dynamic prompts | Not used |
| showDevConsole | Development debugging | Not configured |
| MESSAGES_SNAPSHOT | Message history sync | Not in enum |
| Voice features | Speech-to-text, TTS | Not configured |
| CopilotPopup | Alternative floating UI | Not used |
| CopilotChat | Embedded chat panel | Not used |
| CopilotTextarea | AI-powered textarea | Not used |

---

## Design Decisions & Clarifications

### Open Questions Resolved

| Question | Decision | Rationale |
|----------|----------|-----------|
| CopilotKit analytics vs internal telemetry? | **Both**: CopilotKit Cloud for official features, internal `/api/telemetry` for custom metrics | CopilotKit Cloud provides Inspector; internal telemetry integrates with existing Prometheus/Grafana |
| A2UI behind feature flag per tenant? | **Yes**: `A2UI_ENABLED` global + tenant override via config table | Allows gradual rollout and per-tenant control |
| Tool args/results redaction in UI logs? | **Redact by default**: Mask keys matching `password|secret|token|key|auth` | See Security Policies below |
| useCoAgent usage? | **Deferred**: Not in scope for Epic 21 | useCoAgent is for CoAgent patterns; we use standard AG-UI state sync |

### Tool Status Enum Verification

CopilotKit's actual `useRenderToolCall` status enum (verified via DeepWiki):
```typescript
// IMPORTANT: Status values are PascalCase, NOT lowercase!
type ToolCallStatus = "InProgress" | "Executing" | "Complete";
```

**Warning**: Many examples show lowercase comparisons, but the actual enum values are PascalCase. Code must use exact casing:
```typescript
// CORRECT
if (status === "Executing") { ... }

// WRONG - will never match!
if (status === "executing") { ... }
```

### CopilotSidebar Tool Call Rendering

**Important**: CopilotSidebar (and CopilotChat) **automatically render tool calls** via the internal `CopilotChatToolCallsView` component. You do NOT need `useRenderToolCall` separately when using these components.

| Scenario | Approach |
|----------|----------|
| Using CopilotSidebar/CopilotChat | Tool calls auto-rendered. Use `renderToolCalls` prop on `CopilotKitProvider` for custom renderers |
| Building custom chat UI | Use `useRenderToolCall` hook directly |
| Catch-all rendering | Pass `wildcardRenderer` to `renderToolCalls` prop |

**Story 21-A3 Scope Clarification**: Since we use CopilotSidebar, 21-A3 should configure custom renderers via `CopilotKitProvider.renderToolCalls` prop rather than calling `useRenderToolCall` directly.

### useCopilotAction Deprecation Status

`useCopilotAction` is deprecated but **still functional** for backwards compatibility. Migration is recommended for:
- Better type safety
- Clearer intent (separate hooks for different use cases)
- Future-proofing against eventual removal

Migration does NOT need to be urgent - existing code will continue to work.

### A2UI + STATE_DELTA Merge Strategy

When both A2UI widgets and STATE_DELTA are active:
1. **A2UI widgets** are delivered via `STATE_SNAPSHOT.a2ui_widgets` array
2. **STATE_DELTA** applies JSON Patch operations to the same state object
3. **Merge rule**: STATE_DELTA can add/remove widgets via path `/a2ui_widgets/-` or `/a2ui_widgets/{index}`
4. **Conflict resolution**: Last-write-wins; STATE_DELTA operations are applied atomically
5. **Frontend**: CopilotKit's `useCoAgentStateRender` handles merging automatically

### Zod Schema DRY Pattern

To avoid schema duplication across hooks:

```typescript
// frontend/lib/schemas/tools.ts (shared schemas)
export const SaveToWorkspaceSchema = z.object({
  content_id: z.string().describe("Unique ID of the content"),
  content_text: z.string().describe("Content to save"),
  title: z.string().optional().describe("Optional title"),
});

// Use in hooks
import { SaveToWorkspaceSchema } from "@/lib/schemas/tools";
useFrontendTool({ name: "save_to_workspace", parameters: SaveToWorkspaceSchema, ... });
```

### Observability Endpoint (21-B1 Scope)

Story 21-B1 **includes** creation of `/api/telemetry` endpoint:
- **Backend**: `backend/src/agentic_rag_backend/api/routes/telemetry.py`
- **Auth**: Requires valid session (no API key needed - same-origin)
- **PII**: Message content is SHA-256 hashed, never stored raw
- **Retention**: 7-day default, configurable via `TELEMETRY_RETENTION_DAYS`

---

## Security Policies

### Observability Data Handling

| Data Type | Policy | Implementation |
|-----------|--------|----------------|
| Message content | Hash only, never store raw | SHA-256 hash of first 100 chars |
| Tool arguments | Redact sensitive keys | Mask `password\|secret\|token\|key\|auth\|credential` |
| Tool results | Redact + truncate | Same masking + 500 char limit |
| User identity | Pseudonymize | Emit tenant_id, not user_id |
| Timestamps | Keep | ISO 8601 format |

### Tool Call Masking Rules

```typescript
// frontend/lib/utils/redact.ts
const SENSITIVE_PATTERNS = /password|secret|token|key|auth|credential|apikey|api_key/i;

export function redactSensitiveKeys(obj: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(obj).map(([key, value]) => [
      key,
      SENSITIVE_PATTERNS.test(key) ? "[REDACTED]" : value,
    ])
  );
}
```

### MCP Client Security

| Concern | Mitigation |
|---------|------------|
| Tool namespace collisions | Prefix external tools with server name: `github:create_issue` |
| Circuit breaker | Wrap calls with `tenacity` retry + circuit breaker pattern |
| Credential exposure | API keys injected at runtime, never logged |

### MCP Client Experimental Status

**Warning**: CopilotKit's `mcpServers` and `createMCPClient` are marked `@experimental` in the CopilotKit source code.

| Concern | Mitigation |
|---------|------------|
| API instability | Pin CopilotKit version: `@copilotkit/runtime@1.x.y` (exact version) |
| Feature flag | Gate with `MCP_CLIENTS_ENABLED=false` by default |
| Breaking changes | Monitor CopilotKit changelog before upgrades |
| Fallback | If MCP client fails, log error and continue without external tools |

```typescript
// Import uses experimental_ prefix in some versions
import { experimental_createMCPClient } from "@ai-sdk/mcp";
// OR in newer versions:
import { createMCPClient } from "@copilotkit/runtime";
```

### Generative UI Scope (Epic 21 vs Epic 22)

Epic 21 implements **A2UI only**. Other UI specs are in Epic 22:

| UI Spec | Epic | Description |
|---------|------|-------------|
| A2UI | **Epic 21** | Native CopilotKit widget rendering |
| MCP-UI | Epic 22 | Iframe-based external tool embedding |
| Open-JSON-UI | Epic 22 | OpenAI-style declarative UI components |

This is intentional - A2UI is the foundation; MCP-UI and Open-JSON-UI extend it for interoperability.

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
    if (status === "Executing" && args.sources) {
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
    if (status === "Executing" && respond) {
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
        isSearching={status === "Executing"}
        results={status === "Complete" ? result : null}
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
  status: "InProgress" | "Executing" | "Complete";
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
            {status === "Complete" && result && (
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

#### Story 21-A4: Implement useCopilotReadable for App Context

**Priority:** P0 - HIGH
**Story Points:** 5
**Owner:** Frontend

**Objective:** Expose application state and context to the AI using `useCopilotReadable`.

**Why This Matters:**
- AI needs context about the current workspace, user preferences, and query history
- Without readable state, AI makes decisions without understanding app context
- Critical for RAG systems where context improves retrieval quality

**What `useCopilotReadable` Provides:**
- Exposes any React state or derived data to the AI
- AI can reference this context when generating responses
- Enables personalized, context-aware interactions

**Implementation:**

```tsx
// frontend/hooks/use-copilot-context.ts (new)
import { useCopilotReadable } from "@copilotkit/react-core";
import { useWorkspace } from "@/hooks/use-workspace";
import { useQueryHistory } from "@/hooks/use-query-history";
import { useUserPreferences } from "@/hooks/use-user-preferences";

export function useCopilotContext() {
  const { workspace, documents, activeDocument } = useWorkspace();
  const { recentQueries, frequentTopics } = useQueryHistory();
  const { preferences } = useUserPreferences();

  // Expose current workspace context
  useCopilotReadable({
    description: "Current workspace information including saved documents and notes",
    value: {
      workspaceId: workspace?.id,
      workspaceName: workspace?.name,
      documentCount: documents?.length ?? 0,
      activeDocumentTitle: activeDocument?.title,
      activeDocumentType: activeDocument?.type,
    },
  });

  // Expose query history for context
  useCopilotReadable({
    description: "Recent search queries and frequently discussed topics",
    value: {
      recentQueries: recentQueries?.slice(0, 10) ?? [],
      frequentTopics: frequentTopics?.slice(0, 5) ?? [],
    },
  });

  // Expose user preferences
  useCopilotReadable({
    description: "User preferences for response formatting and behavior",
    value: {
      preferredResponseLength: preferences?.responseLength ?? "medium",
      includeSourceCitations: preferences?.includeCitations ?? true,
      preferredLanguage: preferences?.language ?? "en",
      expertiseLevel: preferences?.expertiseLevel ?? "intermediate",
    },
  });

  // Expose current session state
  useCopilotReadable({
    description: "Current session state including selected sources and filters",
    value: {
      selectedSources: workspace?.selectedSources ?? [],
      activeFilters: workspace?.filters ?? {},
      currentTenant: workspace?.tenantId,
    },
  });
}
```

**Integration in CopilotProvider:**
```tsx
// frontend/components/copilot/CopilotProvider.tsx (update)
import { useCopilotContext } from "@/hooks/use-copilot-context";

export function CopilotProvider({ children }: CopilotProviderProps) {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <CopilotContextProvider />
      {children}
    </CopilotKit>
  );
}

function CopilotContextProvider() {
  useCopilotContext();
  return null;
}
```

**Context Categories to Expose:**

| Category | Data | AI Use Case |
|----------|------|-------------|
| Workspace | Current docs, active selection | "Summarize my current document" |
| Query History | Recent searches, topics | "Continue from my last question about X" |
| User Preferences | Response style, language | Personalized response formatting |
| Session State | Filters, selected sources | Scoped search within constraints |
| Tenant Context | Tenant ID, permissions | Multi-tenant aware responses |

**Acceptance Criteria:**
1. `useCopilotReadable` registered for workspace context
2. Query history exposed (last 10 queries, top 5 topics)
3. User preferences available to AI
4. Session state (filters, sources) accessible
5. Context updates reactively when state changes
6. No sensitive data exposed (passwords, tokens)
7. Tests verify context availability in AI responses
8. Documentation for extending readable context

---

#### Story 21-A5: Implement useCopilotChatSuggestions for Smart Follow-ups

**Priority:** P1 - MEDIUM
**Story Points:** 3
**Owner:** Frontend

**Objective:** Generate contextual chat suggestions based on application state.

**What `useCopilotChatSuggestions` Provides:**
- Dynamic suggestion chips based on current context
- Reduces friction for common follow-up actions
- AI-generated suggestions based on conversation flow

**Implementation:**

```tsx
// frontend/hooks/use-chat-suggestions.ts (new)
import { useCopilotChatSuggestions } from "@copilotkit/react-core";
import { useWorkspace } from "@/hooks/use-workspace";

export function useChatSuggestions() {
  const { activeDocument, documents } = useWorkspace();

  useCopilotChatSuggestions({
    instructions: `Based on the current context, suggest helpful follow-up actions.

Current context:
- Active document: ${activeDocument?.title ?? "None"}
- Document type: ${activeDocument?.type ?? "N/A"}
- Total documents in workspace: ${documents?.length ?? 0}

Generate 2-4 concise suggestions that would help the user:
1. Explore related topics
2. Perform common actions (summarize, compare, explain)
3. Ask clarifying questions
4. Navigate to related content`,

    minSuggestions: 2,
    maxSuggestions: 4,
  });
}
```

**Contextual Suggestion Examples:**

| Context | Generated Suggestions |
|---------|----------------------|
| After search results | "Show more sources", "Explain the first result", "Compare top 3" |
| Viewing document | "Summarize this document", "Find related topics", "Extract key points" |
| Empty workspace | "Import a document", "Search for a topic", "View recent queries" |
| After error | "Try a different query", "Broaden search terms", "Contact support" |

**Integration:**
```tsx
// frontend/components/copilot/CopilotProvider.tsx (add)
import { useChatSuggestions } from "@/hooks/use-chat-suggestions";

function CopilotContextProvider() {
  useCopilotContext();
  useChatSuggestions();
  return null;
}
```

**Acceptance Criteria:**
1. `useCopilotChatSuggestions` configured with context-aware instructions
2. 2-4 suggestions generated based on current state
3. Suggestions appear as clickable chips in chat UI
4. Suggestions update when context changes
5. Tests verify suggestion generation
6. Performance: suggestions don't block chat rendering

---

#### Story 21-A6: Implement useCopilotChat for Headless Control

**Priority:** P2 - LOW
**Story Points:** 3
**Owner:** Frontend

**Objective:** Enable programmatic chat control for testing, automation, and advanced use cases.

**What `useCopilotChat` Provides:**
- Send messages programmatically
- Reload/regenerate responses
- Stop generation mid-stream
- Access chat history
- Headless chat for custom UIs

**Implementation:**

```tsx
// frontend/hooks/use-programmatic-chat.ts (new)
import { useCopilotChat } from "@copilotkit/react-core";

export function useProgrammaticChat() {
  const {
    visibleMessages,
    appendMessage,
    reloadMessages,
    stopGeneration,
    isLoading,
    setMessages,
  } = useCopilotChat();

  // Send a message programmatically
  const sendMessage = async (content: string) => {
    await appendMessage({
      role: "user",
      content,
    });
  };

  // Regenerate the last response
  const regenerateLastResponse = async () => {
    await reloadMessages();
  };

  // Clear chat history
  const clearHistory = () => {
    setMessages([]);
  };

  // Get message count
  const messageCount = visibleMessages.length;

  return {
    messages: visibleMessages,
    sendMessage,
    regenerateLastResponse,
    stopGeneration,
    clearHistory,
    isLoading,
    messageCount,
  };
}
```

**Use Cases:**

| Use Case | Method | Example |
|----------|--------|---------|
| Quick action buttons | `sendMessage` | "Summarize" button sends preset message |
| Keyboard shortcuts | `regenerateLastResponse` | Cmd+R regenerates response |
| Test automation | `sendMessage` + assertions | E2E tests send messages and verify |
| Custom chat UI | All methods | Build entirely custom chat interface |
| Onboarding flows | `sendMessage` | Guided tutorials with scripted messages |

**Example: Quick Action Buttons:**
```tsx
// frontend/components/QuickActions.tsx
import { useProgrammaticChat } from "@/hooks/use-programmatic-chat";

export function QuickActions() {
  const { sendMessage, isLoading } = useProgrammaticChat();

  return (
    <div className="flex gap-2">
      <Button
        disabled={isLoading}
        onClick={() => sendMessage("Summarize the current document")}
      >
        Summarize
      </Button>
      <Button
        disabled={isLoading}
        onClick={() => sendMessage("Extract key insights")}
      >
        Key Insights
      </Button>
      <Button
        disabled={isLoading}
        onClick={() => sendMessage("Find related topics")}
      >
        Related Topics
      </Button>
    </div>
  );
}
```

**Acceptance Criteria:**
1. `useCopilotChat` hook wrapped with convenient methods
2. `sendMessage` function for programmatic message sending
3. `regenerateLastResponse` for response regeneration
4. `stopGeneration` for interrupting streaming
5. `clearHistory` for resetting conversation
6. Quick action buttons component implemented
7. Tests verify programmatic control
8. Keyboard shortcuts documented

---

#### Story 21-A7: Implement useCopilotAdditionalInstructions for Dynamic Prompts

**Priority:** P2 - LOW
**Story Points:** 2
**Owner:** Frontend

**Objective:** Enable dynamic system prompt modifications based on context.

**What `useCopilotAdditionalInstructions` Provides:**
- Add context-specific instructions to the system prompt
- Instructions can change based on component state
- Enables feature-specific AI behavior

**Implementation:**

```tsx
// frontend/hooks/use-dynamic-instructions.ts (new)
import { useCopilotAdditionalInstructions } from "@copilotkit/react-core";
import { useWorkspace } from "@/hooks/use-workspace";

export function useDynamicInstructions() {
  const { activeDocument, expertMode } = useWorkspace();

  // Add document-specific instructions
  useCopilotAdditionalInstructions(
    activeDocument?.type === "code"
      ? "When discussing code, include syntax highlighting hints and best practices."
      : activeDocument?.type === "pdf"
      ? "When referencing the PDF, cite page numbers when available."
      : ""
  );

  // Add expert mode instructions
  useCopilotAdditionalInstructions(
    expertMode
      ? "Provide detailed technical explanations with citations. Assume advanced knowledge."
      : "Provide clear, accessible explanations. Define technical terms when used."
  );

  // Add tenant-specific instructions (if any)
  useCopilotAdditionalInstructions(
    `Always scope searches to the current tenant context.
     Never reveal information from other tenants.`
  );
}
```

**Instruction Categories:**

| Category | Condition | Instruction Added |
|----------|-----------|-------------------|
| Document Type | PDF active | "Cite page numbers" |
| Document Type | Code active | "Include syntax hints" |
| User Mode | Expert mode on | "Detailed technical explanations" |
| User Mode | Beginner mode | "Define technical terms" |
| Security | Always | "Scope to current tenant" |

**Acceptance Criteria:**
1. `useCopilotAdditionalInstructions` used for dynamic prompts
2. Document-type-specific instructions implemented
3. Expert/beginner mode switching works
4. Security instructions always applied
5. Instructions update reactively
6. Tests verify instruction injection

---

### Group B: Observability & Debugging

*Focus: Enable production debugging and analytics*
*Priority: P0 - Critical for operations*

#### Story 21-B1: Configure Observability Hooks and Dev Console

**Priority:** P0 - HIGH
**Story Points:** 5
**Owner:** Frontend + Backend

**Objective:** Wire CopilotKit observability hooks to our telemetry pipeline and enable development debugging tools.

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

# Dev Console (development only)
NEXT_PUBLIC_SHOW_DEV_CONSOLE=true  # Show visual debugging console
```

**Updated CopilotProvider with showDevConsole:**
```tsx
// frontend/components/copilot/CopilotProvider.tsx (updated)
export function CopilotProvider({ children }: CopilotProviderProps) {
  const analytics = useAnalytics();
  const isDev = process.env.NODE_ENV === "development";

  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      publicApiKey={process.env.NEXT_PUBLIC_COPILOTKIT_API_KEY}
      showDevConsole={isDev && process.env.NEXT_PUBLIC_SHOW_DEV_CONSOLE === "true"}
      onError={(errorEvent) => {
        analytics.track("copilot_error", {
          type: errorEvent.type,
          error: errorEvent.error?.message,
          context: errorEvent.context,
          timestamp: errorEvent.timestamp,
        });
        if (isDev) {
          console.error("CopilotKit Error:", errorEvent);
        }
      }}
    >
      <CopilotSidebar
        observabilityHooks={{...}}
      />
      {children}
    </CopilotKit>
  );
}
```

**What showDevConsole Provides:**
- Visual error overlay in development
- Request/response inspector
- Event timeline visualization
- State debugging tools
- Real-time message stream viewer

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
7. `showDevConsole` enabled in development mode
8. `onError` handler configured with structured logging
9. Dev console provides visual debugging in development
10. Production builds hide dev console

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

#### Story 21-B3: Implement STATE_DELTA and MESSAGES_SNAPSHOT Support

**Priority:** P1 - MEDIUM
**Story Points:** 8
**Owner:** Backend + Frontend

**Objective:** Add incremental state updates using JSON Patch operations and message history synchronization.

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

**MESSAGES_SNAPSHOT Implementation:**

The `MESSAGES_SNAPSHOT` event syncs the entire message history, useful for:
- Session restoration after reconnection
- Multi-tab synchronization
- Chat history persistence across page reloads

```python
# backend/src/agentic_rag_backend/models/copilot.py (add to enum)
class AGUIEventType(str, Enum):
    # ... existing events ...
    STATE_DELTA = "STATE_DELTA"
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"  # NEW

class MessagesSnapshotEvent(AGUIEvent):
    """Event for syncing full message history."""

    event: AGUIEventType = AGUIEventType.MESSAGES_SNAPSHOT

    def __init__(self, messages: list[dict[str, Any]], **kwargs: Any) -> None:
        """
        Args:
            messages: Full message history
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
        """
        super().__init__(data={"messages": messages}, **kwargs)


class CustomEvent(AGUIEvent):
    """Event for application-specific custom events.

    CUSTOM events enable application-specific functionality that isn't covered
    by standard AG-UI events. Use for:
    - A2UI widget updates
    - Application-specific notifications
    - Custom state synchronization
    - Third-party integration events
    """

    event: AGUIEventType = AGUIEventType.CUSTOM_EVENT

    def __init__(
        self,
        event_name: str,
        payload: dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Args:
            event_name: Application-specific event type (e.g., "a2ui_widget_update")
            payload: Event-specific data
        """
        super().__init__(
            data={"name": event_name, "payload": payload},
            **kwargs
        )
```

**CUSTOM_EVENT Use Cases:**

| Event Name | Payload | Use Case |
|------------|---------|----------|
| `a2ui_widget_update` | Widget ID + props | Dynamic widget updates |
| `progress_update` | Percent + message | Long-running task progress |
| `notification` | Type + message | User notifications |
| `analytics_event` | Event name + data | Frontend analytics triggers |

**Usage for Session Restoration:**
```python
# backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py (update)
async def restore_session(self, session_id: str) -> AsyncIterator[AGUIEvent]:
    """Restore a previous chat session."""
    messages = await self._load_session_messages(session_id)
    if messages:
        yield MessagesSnapshotEvent(messages=messages)
```

**Acceptance Criteria:**
1. `STATE_DELTA` event type added to enum
2. `StateDeltaEvent` class with JSON Patch operations
3. AGUIBridge uses STATE_DELTA for incremental updates
4. Frontend receives and applies deltas correctly
5. Tests verify delta application
6. Performance improvement for large state updates
7. `MESSAGES_SNAPSHOT` event type added to enum
8. `MessagesSnapshotEvent` class for full history sync
9. Session restoration uses MESSAGES_SNAPSHOT
10. Tests verify message history synchronization
11. `CUSTOM_EVENT` event type added to enum
12. `CustomEvent` class for application-specific events
13. Frontend handles CUSTOM events appropriately

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
                # Exponential backoff: base_delay * 2^attempt (with jitter)
                delay = (self.config.retry_delay / 1000) * (2 ** attempt)
                delay += random.uniform(0, delay * 0.1)  # 10% jitter
                await asyncio.sleep(min(delay, 30))  # Cap at 30 seconds

            except httpx.HTTPStatusError as e:
                # Don't retry 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    raise MCPClientError(f"HTTP error: {e.response.status_code}")
                if attempt == self.config.retry_count:
                    raise MCPClientError(f"HTTP error: {e.response.status_code}")
                delay = (self.config.retry_delay / 1000) * (2 ** attempt)
                await asyncio.sleep(min(delay, 30))

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

### Group E: Voice & Audio Features

*Focus: Enable speech-to-text and text-to-speech capabilities*
*Priority: P2 - Accessibility enhancement*

#### Story 21-E1: Implement Voice Input (Speech-to-Text)

**Priority:** P2 - MEDIUM
**Story Points:** 5
**Owner:** Frontend + Backend

**Objective:** Enable voice input for chat messages using CopilotKit's transcription service.

**What Voice Input Provides:**
- Accessibility for users who prefer voice
- Hands-free operation
- Natural language input without typing
- Mobile-friendly interaction

**Configuration:**
```bash
# .env
# Voice Input Configuration
NEXT_PUBLIC_TRANSCRIBE_AUDIO_URL=/api/copilotkit/transcribe  # Transcription endpoint
COPILOTKIT_TRANSCRIPTION_ENABLED=true|false  # Default: false
COPILOTKIT_TRANSCRIPTION_LANGUAGE=en  # ISO language code
```

**Backend Implementation:**
```python
# backend/src/agentic_rag_backend/api/routes/copilot.py (add)
from fastapi import UploadFile, File

@router.post("/copilotkit/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Query(default="en"),
) -> dict[str, str]:
    """Transcribe audio to text using configured transcription service."""
    if not settings.copilotkit_transcription_enabled:
        raise HTTPException(status_code=403, detail="Transcription disabled")

    # Use configured transcription service (Whisper, AssemblyAI, etc.)
    transcription = await transcription_service.transcribe(
        audio_data=await audio.read(),
        language=language,
        format=audio.content_type,
    )

    return {
        "text": transcription.text,
        "language": transcription.detected_language,
        "confidence": transcription.confidence,
    }
```

**Frontend Implementation:**
```tsx
// frontend/components/copilot/CopilotProvider.tsx (update)
<CopilotKit
  runtimeUrl="/api/copilotkit"
  transcribeAudioUrl={process.env.NEXT_PUBLIC_TRANSCRIBE_AUDIO_URL}
>
```

**Audio Recording Component:**
```tsx
// frontend/components/copilot/VoiceInput.tsx (new)
import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Mic, MicOff, Loader2 } from "lucide-react";

interface VoiceInputProps {
  onTranscription: (text: string) => void;
  disabled?: boolean;
}

export function VoiceInput({ onTranscription, disabled }: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder.current = new MediaRecorder(stream);
    chunks.current = [];

    mediaRecorder.current.ondataavailable = (e) => {
      chunks.current.push(e.data);
    };

    mediaRecorder.current.onstop = async () => {
      const blob = new Blob(chunks.current, { type: "audio/webm" });
      await transcribeAudio(blob);
    };

    mediaRecorder.current.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    mediaRecorder.current?.stop();
    setIsRecording(false);
  };

  const transcribeAudio = async (blob: Blob) => {
    setIsTranscribing(true);
    try {
      const formData = new FormData();
      formData.append("audio", blob);

      const response = await fetch("/api/copilotkit/transcribe", {
        method: "POST",
        body: formData,
      });

      const { text } = await response.json();
      onTranscription(text);
    } finally {
      setIsTranscribing(false);
    }
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      disabled={disabled || isTranscribing}
      onClick={isRecording ? stopRecording : startRecording}
    >
      {isTranscribing ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : isRecording ? (
        <MicOff className="h-4 w-4 text-red-500" />
      ) : (
        <Mic className="h-4 w-4" />
      )}
    </Button>
  );
}
```

**Acceptance Criteria:**
1. `transcribeAudioUrl` configured in CopilotKit
2. Backend transcription endpoint implemented
3. Audio recording component functional
4. Transcribed text populates chat input
5. Visual feedback during recording/transcription
6. Error handling for microphone permissions
7. Tests verify transcription flow
8. Accessibility labels for screen readers

---

#### Story 21-E2: Implement Voice Output (Text-to-Speech)

**Priority:** P2 - LOW
**Story Points:** 5
**Owner:** Frontend + Backend

**Objective:** Enable text-to-speech for AI responses.

**What Voice Output Provides:**
- Accessibility for visually impaired users
- Hands-free consumption of AI responses
- Enhanced user experience for audio learners
- Mobile-friendly output

**Configuration:**
```bash
# .env
# Voice Output Configuration
NEXT_PUBLIC_TEXT_TO_SPEECH_URL=/api/copilotkit/tts  # TTS endpoint
COPILOTKIT_TTS_ENABLED=true|false  # Default: false
COPILOTKIT_TTS_VOICE=alloy  # Voice selection (alloy, echo, fable, onyx, nova, shimmer)
COPILOTKIT_TTS_SPEED=1.0  # Speed multiplier (0.25 to 4.0)
```

**Backend Implementation:**
```python
# backend/src/agentic_rag_backend/api/routes/copilot.py (add)
from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"
    speed: float = 1.0

@router.post("/copilotkit/tts")
async def text_to_speech(request: TTSRequest) -> StreamingResponse:
    """Convert text to speech audio stream."""
    if not settings.copilotkit_tts_enabled:
        raise HTTPException(status_code=403, detail="TTS disabled")

    # Use configured TTS service (OpenAI, ElevenLabs, etc.)
    audio_stream = await tts_service.synthesize(
        text=request.text,
        voice=request.voice,
        speed=request.speed,
    )

    return StreamingResponse(
        audio_stream,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=response.mp3"},
    )
```

**Frontend Implementation:**
```tsx
// frontend/components/copilot/CopilotProvider.tsx (update)
<CopilotKit
  runtimeUrl="/api/copilotkit"
  textToSpeechUrl={process.env.NEXT_PUBLIC_TEXT_TO_SPEECH_URL}
>
```

**Speech Playback Component:**
```tsx
// frontend/components/copilot/SpeakButton.tsx (new)
import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Volume2, VolumeX, Loader2 } from "lucide-react";

interface SpeakButtonProps {
  text: string;
  disabled?: boolean;
}

export function SpeakButton({ text, disabled }: SpeakButtonProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const speak = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("/api/copilotkit/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      audioRef.current = new Audio(url);
      audioRef.current.onended = () => setIsPlaying(false);
      audioRef.current.play();
      setIsPlaying(true);
    } finally {
      setIsLoading(false);
    }
  };

  const stop = () => {
    audioRef.current?.pause();
    setIsPlaying(false);
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      disabled={disabled || isLoading}
      onClick={isPlaying ? stop : speak}
    >
      {isLoading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : isPlaying ? (
        <VolumeX className="h-4 w-4" />
      ) : (
        <Volume2 className="h-4 w-4" />
      )}
    </Button>
  );
}
```

**Acceptance Criteria:**
1. `textToSpeechUrl` configured in CopilotKit
2. Backend TTS endpoint implemented
3. Speak button on assistant messages
4. Audio playback with stop control
5. Voice/speed configuration options
6. Error handling for synthesis failures
7. Tests verify TTS flow
8. Accessibility: screen reader compatibility

---

### Group F: Alternative UI Components

*Focus: Provide flexible UI options for different use cases*
*Priority: P2 - UX flexibility*

#### Story 21-F1: Implement CopilotPopup Component

**Priority:** P2 - LOW
**Story Points:** 3
**Owner:** Frontend

**Objective:** Add floating popup chat option as alternative to sidebar.

**What CopilotPopup Provides:**
- Floating button with expandable chat window
- Less intrusive than full sidebar
- Ideal for secondary/contextual assistance
- Customizable position and styling

**Implementation:**
```tsx
// frontend/components/copilot/PopupChat.tsx (new)
"use client";

import { CopilotPopup } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";

interface PopupChatProps {
  position?: "bottom-right" | "bottom-left" | "top-right" | "top-left";
  buttonLabel?: string;
}

export function PopupChat({
  position = "bottom-right",
  buttonLabel = "AI Assistant",
}: PopupChatProps) {
  return (
    <CopilotPopup
      labels={{
        title: "RAG Assistant",
        initial: "How can I help you today?",
      }}
      defaultOpen={false}
      clickOutsideToClose={true}
      className={getPositionClass(position)}
    />
  );
}

function getPositionClass(position: string): string {
  switch (position) {
    case "bottom-left":
      return "!left-4 !right-auto";
    case "top-right":
      return "!bottom-auto !top-4";
    case "top-left":
      return "!left-4 !right-auto !bottom-auto !top-4";
    default:
      return "";
  }
}
```

**Configuration Option:**
```bash
# .env
NEXT_PUBLIC_COPILOT_UI_MODE=sidebar|popup  # Default: sidebar
```

**Conditional Rendering:**
```tsx
// frontend/components/copilot/ChatInterface.tsx (new)
import { ChatSidebar } from "./ChatSidebar";
import { PopupChat } from "./PopupChat";

export function ChatInterface() {
  const uiMode = process.env.NEXT_PUBLIC_COPILOT_UI_MODE || "sidebar";

  if (uiMode === "popup") {
    return <PopupChat />;
  }

  return <ChatSidebar />;
}
```

**Acceptance Criteria:**
1. CopilotPopup component implemented
2. Configurable position (4 corners)
3. Click-outside-to-close behavior
4. Environment variable for UI mode selection
5. Consistent styling with design system
6. Tests verify popup behavior
7. Responsive on mobile devices

---

#### Story 21-F2: Implement CopilotChat Embedded Component

**Priority:** P2 - LOW
**Story Points:** 3
**Owner:** Frontend

**Objective:** Add inline embedded chat option for page integration.

**What CopilotChat Provides:**
- Embedded chat panel within page content
- No sidebar or popup - inline experience
- Ideal for dedicated chat pages or sections
- Full-width or contained layouts

**Implementation:**
```tsx
// frontend/components/copilot/EmbeddedChat.tsx (new)
"use client";

import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";

interface EmbeddedChatProps {
  className?: string;
  welcomeMessage?: string;
}

export function EmbeddedChat({
  className,
  welcomeMessage = "Welcome! Ask me anything about your documents.",
}: EmbeddedChatProps) {
  return (
    <div className={className}>
      <CopilotChat
        labels={{
          initial: welcomeMessage,
        }}
        className="h-full"
      />
    </div>
  );
}
```

**Page Integration Example:**
```tsx
// frontend/app/chat/page.tsx (new)
import { EmbeddedChat } from "@/components/copilot/EmbeddedChat";

export default function ChatPage() {
  return (
    <div className="container mx-auto h-screen py-4">
      <h1 className="text-2xl font-bold mb-4">AI Assistant</h1>
      <EmbeddedChat className="h-[calc(100vh-8rem)] border rounded-lg" />
    </div>
  );
}
```

**Acceptance Criteria:**
1. CopilotChat embedded component implemented
2. Configurable welcome message
3. Responsive height/width
4. Works in container layouts
5. Dedicated chat page example
6. Tests verify embedded rendering

---

#### Story 21-F3: Implement CopilotTextarea Component

**Priority:** P3 - LOW
**Story Points:** 5
**Owner:** Frontend

**Objective:** Add AI-powered textarea for inline text assistance.

**What CopilotTextarea Provides:**
- AI autocompletion in any textarea
- Inline suggestions while typing
- Tab to accept suggestions
- Ideal for document editing, notes, forms

**Implementation:**
```tsx
// frontend/components/copilot/AITextarea.tsx (new)
"use client";

import { CopilotTextarea } from "@copilotkit/react-textarea";
import "@copilotkit/react-textarea/styles.css";

interface AITextareaProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  autosuggestionsConfig?: {
    textareaPurpose: string;
    chatApiConfigs: Record<string, unknown>;
  };
}

export function AITextarea({
  value,
  onChange,
  placeholder = "Start typing...",
  className,
  autosuggestionsConfig,
}: AITextareaProps) {
  return (
    <CopilotTextarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={className}
      autosuggestionsConfig={
        autosuggestionsConfig ?? {
          textareaPurpose: "General purpose text editor",
          chatApiConfigs: {},
        }
      }
    />
  );
}
```

**Usage Example:**
```tsx
// frontend/components/notes/NoteEditor.tsx
import { AITextarea } from "@/components/copilot/AITextarea";
import { useState } from "react";

export function NoteEditor() {
  const [content, setContent] = useState("");

  return (
    <AITextarea
      value={content}
      onChange={setContent}
      placeholder="Write your notes here... AI will help!"
      className="min-h-[200px] w-full"
      autosuggestionsConfig={{
        textareaPurpose: "Note-taking and documentation",
        chatApiConfigs: {},
      }}
    />
  );
}
```

**Acceptance Criteria:**
1. CopilotTextarea component implemented
2. AI autocompletion working
3. Tab-to-accept suggestions
4. Configurable purpose/context
5. Works with React Hook Form
6. Tests verify suggestion flow
7. Performance: no typing lag

---

## Testing Requirements

### Unit Tests

| Story | Test Focus | Coverage Target |
|-------|------------|-----------------|
| 21-A1 | useFrontendTool migration | 90% |
| 21-A2 | useHumanInTheLoop migration | 90% |
| 21-A3 | useRenderToolCall renderers | 85% |
| 21-A4 | useCopilotReadable context provision | 85% |
| 21-A5 | useCopilotChatSuggestions generation | 80% |
| 21-A6 | useCopilotChat programmatic control | 85% |
| 21-A7 | useCopilotAdditionalInstructions dynamic prompts | 80% |
| 21-B1 | Observability hook firing + showDevConsole | 80% |
| 21-B2 | RUN_ERROR event emission | 90% |
| 21-B3 | STATE_DELTA + MESSAGES_SNAPSHOT events | 90% |
| 21-C1 | MCP client configuration | 85% |
| 21-C2 | MCP client factory | 85% |
| 21-C3 | Tool registry merging | 90% |
| 21-D1 | A2UI widget models | 90% |
| 21-D2 | A2UI widget rendering | 85% |
| 21-E1 | Voice input transcription flow | 80% |
| 21-E2 | Text-to-speech synthesis flow | 80% |
| 21-F1 | CopilotPopup positioning + behavior | 80% |
| 21-F2 | CopilotChat embedded rendering | 80% |
| 21-F3 | CopilotTextarea autocompletion | 75% |

### Integration Tests

| Test Scenario | Stories |
|---------------|---------|
| Full tool call lifecycle (start → execute → complete → render) | 21-A3 |
| HITL dialog approval flow | 21-A2 |
| Context provision → AI response uses context | 21-A4 |
| Chat suggestions generation based on state | 21-A5 |
| Programmatic chat control (send/reload/stop) | 21-A6 |
| Dynamic instructions based on document type | 21-A7 |
| MCP client discovery → tool call → result | 21-C1, 21-C2, 21-C3 |
| A2UI widget emission → render | 21-D1, 21-D2 |
| Error event emission → toast display | 21-B2 |
| STATE_DELTA application → UI update | 21-B3 |
| MESSAGES_SNAPSHOT → session restoration | 21-B3 |
| Voice input recording → transcription → chat input | 21-E1 |
| TTS synthesis → audio playback | 21-E2 |

### E2E Tests (Playwright)

| Test | Purpose |
|------|---------|
| Tool call card visibility | Verify MCP tools show in chat |
| HITL dialog interaction | Click approve/reject |
| A2UI card rendering | Verify card displays correctly |
| Error toast appearance | Verify error notification |
| Chat suggestions appear and are clickable | Verify suggestion chips work |
| Quick action buttons send messages | Verify programmatic chat |
| Dev console appears in development mode | Verify showDevConsole |
| Microphone button records and transcribes | Verify voice input (21-E1) |
| Speak button plays audio | Verify TTS (21-E2) |
| CopilotPopup opens and closes | Verify popup behavior (21-F1) |
| CopilotChat renders inline | Verify embedded chat (21-F2) |
| CopilotTextarea shows suggestions | Verify autocompletion (21-F3) |

---

## Configuration Summary

### New Environment Variables

```bash
# Epic 21 - CopilotKit Full Integration

# --- Observability (21-B1) ---
NEXT_PUBLIC_COPILOTKIT_API_KEY=ck_pub_xxx  # Optional: Copilot Cloud
NEXT_PUBLIC_COPILOTKIT_LICENSE_KEY=ck_lic_xxx  # Optional: Self-hosted Inspector
NEXT_PUBLIC_SHOW_DEV_CONSOLE=true|false  # Default: false (true in dev only)

# --- MCP Client (21-C1, 21-C2) ---
MCP_CLIENTS_ENABLED=true|false  # Default: false
MCP_CLIENT_TIMEOUT=30000  # Timeout in ms
MCP_CLIENT_RETRY_COUNT=3
MCP_CLIENT_RETRY_DELAY=1000
MCP_CLIENT_SERVERS='[{"name":"github","url":"...","apiKey":"${GITHUB_MCP_KEY}"}]'

# --- A2UI (21-D1) ---
A2UI_ENABLED=true|false  # Default: true (if using CopilotKit)
A2UI_FALLBACK_RENDER=true  # Show fallback for unsupported widgets

# --- Voice Input (21-E1) ---
NEXT_PUBLIC_TRANSCRIBE_AUDIO_URL=/api/copilotkit/transcribe  # Transcription endpoint
COPILOTKIT_TRANSCRIPTION_ENABLED=true|false  # Default: false
COPILOTKIT_TRANSCRIPTION_LANGUAGE=en  # ISO language code (en, es, fr, de, etc.)

# --- Voice Output (21-E2) ---
NEXT_PUBLIC_TEXT_TO_SPEECH_URL=/api/copilotkit/tts  # TTS endpoint
COPILOTKIT_TTS_ENABLED=true|false  # Default: false
COPILOTKIT_TTS_VOICE=alloy  # Voice: alloy, echo, fable, onyx, nova, shimmer
COPILOTKIT_TTS_SPEED=1.0  # Speed multiplier: 0.25 to 4.0

# --- Alternative UI Components (21-F1, 21-F2, 21-F3) ---
NEXT_PUBLIC_COPILOT_UI_MODE=sidebar|popup|embedded  # Default: sidebar
```

---

## Dependencies

### External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `@copilotkit/react-core` | ^1.x | Modern hooks |
| `@copilotkit/react-ui` | ^1.x | UI components (Sidebar, Popup, Chat) |
| `@copilotkit/react-textarea` | ^1.x | AI-powered textarea (21-F3) |
| `@copilotkit/runtime` | ^1.x | Runtime configuration |
| `zod` | ^3.x | Schema validation |
| `jsonpatch` | ^1.x | JSON Patch operations (RFC 6902) |

### Internal Dependencies

| Module | Dependency |
|--------|-----------|
| 21-A2 | 21-A1 (shared hook patterns) |
| 21-A3 | 21-A1, 21-A2 (tool renderers) |
| 21-A4 | 21-A1 (provider pattern established) |
| 21-A5 | 21-A4 (context from useCopilotReadable) |
| 21-A6 | 21-A1 (hook patterns) |
| 21-A7 | 21-A4 (context-aware instructions) |
| 21-B3 | 21-B2 (error events before state sync) |
| 21-C3 | 21-C1, 21-C2 (MCP client) |
| 21-D2 | 21-D1 (A2UI models) |
| 21-E2 | 21-E1 (voice infrastructure) |
| 21-F1 | 21-B1 (provider config with observability) |
| 21-F2 | 21-B1 (provider config with observability) |
| 21-F3 | 21-A4 (context for suggestions) |

---

## Sprint Allocation

### Sprint 1 (Stories: 21-A1, 21-A2, 21-A3, 21-A4, 21-B1)
- Focus: Hook migration + Tool visualization + Context provision + Observability
- Points: 25
- Goal: Modern patterns, context awareness, debugging capabilities

### Sprint 2 (Stories: 21-A5, 21-A6, 21-A7, 21-B2, 21-B3)
- Focus: Chat enhancements + Dynamic instructions + Error handling + State sync
- Points: 19
- Goal: Enhanced chat UX, robust error handling

### Sprint 3 (Stories: 21-C1, 21-C2, 21-C3)
- Focus: MCP client integration
- Points: 18
- Goal: External MCP tool ecosystem access

### Sprint 4 (Stories: 21-D1, 21-D2, 21-E1, 21-E2)
- Focus: A2UI rendering + Voice features
- Points: 18
- Goal: Rich UI widgets + Accessibility via voice

### Sprint 5 (Stories: 21-F1, 21-F2, 21-F3)
- Focus: Alternative UI components
- Points: 11
- Goal: Flexible UI options (popup, embedded, textarea)

**Total: 91 story points across 5 sprints**

### Story Points Summary by Group

| Group | Focus | Stories | Points |
|-------|-------|---------|--------|
| A | Modern Hook Migration | 21-A1 to 21-A7 | 28 |
| B | Observability & Debugging | 21-B1 to 21-B3 | 16 |
| C | MCP Client Integration | 21-C1 to 21-C3 | 18 |
| D | A2UI Widget Rendering | 21-D1 to 21-D2 | 8 |
| E | Voice & Audio Features | 21-E1 to 21-E2 | 10 |
| F | Alternative UI Components | 21-F1 to 21-F3 | 11 |
| **Total** | | **18 stories** | **91** |

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Deprecated hook usage | 12 instances | 0 | Code search |
| Tool call visibility | 0% | 100% | All MCP tools rendered |
| Observability coverage | 0 hooks | 9 hooks | All hooks wired |
| Context provision hooks | 0 | 4+ | useCopilotReadable registrations |
| Chat suggestion accuracy | N/A | 70%+ | User click-through rate |
| MCP ecosystem tools | 0 | 10+ | External server tools |
| A2UI widget types | 0 | 6 | Supported widget types |
| Error transparency | Text-only | Event-based | RUN_ERROR emissions |
| State sync efficiency | Snapshot-only | Delta + Snapshot | STATE_DELTA usage |
| Voice input availability | 0% | 100% | Transcription endpoint uptime |
| Voice output availability | 0% | 100% | TTS endpoint uptime |
| UI component options | 1 (sidebar) | 3 | sidebar, popup, embedded |
| Dev console adoption | 0% | 80%+ | Developers using showDevConsole |

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
