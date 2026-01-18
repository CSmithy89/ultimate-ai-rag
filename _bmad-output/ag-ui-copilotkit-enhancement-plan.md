# Comprehensive AG-UI & CopilotKit Enhancement Plan (Self-Hosted)

## Executive Summary

This plan consolidates all discoveries from our analysis of CopilotKit Premium, AG-UI protocol, and A2UI Composer to maximize the project's capabilities while remaining fully self-hosted (free unlimited users).

**Current State:** ~95% of free features + full AG-UI protocol compliance
**Target State:** ~95% of free features + full AG-UI protocol compliance

---

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: AG-UI Protocol (Backend) | ✅ COMPLETE | 19 event types, STATE_DELTA, TOOL_CALL_RESULT |
| Phase 2: Frontend Hooks | ✅ COMPLETE | useOrchestratorState, HITL v2, context |
| Phase 3: Widget System | ✅ COMPLETE | Registry + 8 widgets (Step, Approval, Table, Activity, Status, Charts) |
| Phase 4: THINKING Events | ✅ COMPLETE | THINKING_START, _CONTENT, _END events |
| Phase 5: ACTIVITY Events | ✅ COMPLETE | ACTIVITY_SNAPSHOT, _DELTA + ActivityTrackerWidget + useActivityTracker |
| Phase 6.1: Cancel/Resume | ✅ COMPLETE | RunManager + useRunControl hook |
| Phase 6.2: Agent Steering | ✅ COMPLETE | /steer API + useAgentSteering hook |
| Phase 7.1: Sub-Agent Composition | ✅ COMPLETE | SubAgentManager + context passing |
| Phase 7.2: Multimodal Input | ✅ COMPLETE | TextInputContent, BinaryInputContent, MultimodalMessage + useMultimodalInput |
| Phase 7.3: RAW Events | ✅ COMPLETE | RawEvent for MCP/A2A protocol wrapping |
| Phase 8: Code Review Fixes | ✅ COMPLETE | Security, type safety, tests, documentation |

**Tests:** 48 passing (backend), 1043 passing (frontend)
**Last Updated:** 2026-01-18

---

## Completed Features Summary

### Backend (Python/FastAPI)
- **19 AG-UI Event Types**: Full protocol compliance
- **Run Manager**: Cancel, resume, checkpoint agent runs
- **Agent Steering**: Inject guidance mid-execution
- **Sub-Agent Composition**: Delegation with context isolation
- **Multimodal Support**: Text, images, audio, documents
- **RAW Events**: Wrap MCP/A2A external protocol events

### Frontend (Next.js/React)
- **8 Widget Components**: StepProgress, ApprovalDialog, DataTable, ActivityTracker, StatusIndicator, BarChart, LineChart, PieChart
- **6 New Hooks**: useOrchestratorState, useRunControl, useAgentSteering, useMultimodalInput, useActivityTracker, useSourceValidationV2
- **Widget Registry**: Dynamic widget rendering from backend events

---

## Part 1: AG-UI Protocol Enhancements (Backend)

### 1.1 Missing Event Types to Implement

**Currently Implemented (11 events):**
- RUN_STARTED, RUN_FINISHED, RUN_ERROR
- TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, TEXT_MESSAGE_END
- TOOL_CALL_START, TOOL_CALL_ARGS, TOOL_CALL_END
- STATE_SNAPSHOT
- ACTION_REQUEST

**Events to Add (8 events):**

| Event | Priority | Purpose |
|-------|----------|---------|
| `STATE_DELTA` | High | Incremental state updates (RFC 6902 JSON Patch) |
| `TOOL_CALL_RESULT` | High | Tool execution results (currently only END emitted) |
| `MESSAGES_SNAPSHOT` | Medium | Conversation history sync |
| `ACTIVITY_SNAPSHOT` | Medium | Long-running operation progress |
| `ACTIVITY_DELTA` | Medium | Incremental activity updates |
| `THINKING_START` | Low | Agent reasoning phase start |
| `THINKING_TEXT_MESSAGE_CONTENT` | Low | Reasoning content |
| `THINKING_END` | Low | Agent reasoning phase end |
| `CUSTOM` | Low | Application-specific events |

### 1.2 Implementation: STATE_DELTA Events

**File:** `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`

```python
from ag_ui.events import StateDeltaEvent

# Replace full snapshots with deltas for step updates
async def emit_step_update(self, step_index: int, status: str):
    yield StateDeltaEvent(delta=[
        {"op": "replace", "path": f"/steps/{step_index}/status", "value": status}
    ])

# Use snapshots only for initial state or major changes
yield StateSnapshotEvent(state={"steps": steps, ...})  # Initial
yield StateDeltaEvent(delta=[...])  # Subsequent updates
```

### 1.3 Implementation: TOOL_CALL_RESULT Events

**File:** `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`

```python
# After tool execution completes, emit result before END
yield ToolCallResultEvent(
    tool_call_id=tool_call_id,
    result=json.dumps(tool_result)
)
yield ToolCallEndEvent(tool_call_id=tool_call_id)
```

### 1.4 Implementation: THINKING Events (Agent Reasoning)

**File:** `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`

```python
# Wrap orchestrator reasoning in thinking events
yield ThinkingStartEvent()
for thought in result.thoughts:
    yield ThinkingTextMessageContentEvent(content=thought.content)
yield ThinkingEndEvent()
```

### 1.5 Implementation: ACTIVITY Events (Long Operations)

**File:** `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`

```python
# For multi-step operations like document ingestion
yield ActivitySnapshotEvent(activity={
    "id": activity_id,
    "type": "indexing",
    "progress": 0.0,
    "message": "Starting document processing..."
})

# Update progress incrementally
yield ActivityDeltaEvent(delta=[
    {"op": "replace", "path": "/progress", "value": 0.45},
    {"op": "replace", "path": "/message", "value": "Processing page 3/7..."}
])
```

### 1.6 Event Models to Add

**File:** `backend/src/agentic_rag_backend/models/copilot.py`

```python
class StateDeltaEvent(BaseAGUIEvent):
    type: Literal[AGUIEventType.STATE_DELTA] = AGUIEventType.STATE_DELTA
    delta: List[Dict[str, Any]]  # RFC 6902 JSON Patch operations

class ToolCallResultEvent(BaseAGUIEvent):
    type: Literal[AGUIEventType.TOOL_CALL_RESULT] = AGUIEventType.TOOL_CALL_RESULT
    tool_call_id: str
    result: str

class ActivitySnapshotEvent(BaseAGUIEvent):
    type: Literal[AGUIEventType.ACTIVITY_SNAPSHOT] = AGUIEventType.ACTIVITY_SNAPSHOT
    activity: Dict[str, Any]

class ActivityDeltaEvent(BaseAGUIEvent):
    type: Literal[AGUIEventType.ACTIVITY_DELTA] = AGUIEventType.ACTIVITY_DELTA
    delta: List[Dict[str, Any]]

class ThinkingStartEvent(BaseAGUIEvent):
    type: Literal[AGUIEventType.THINKING_START] = AGUIEventType.THINKING_START

class ThinkingTextMessageContentEvent(BaseAGUIEvent):
    type: Literal[AGUIEventType.THINKING_TEXT_MESSAGE_CONTENT] = AGUIEventType.THINKING_TEXT_MESSAGE_CONTENT
    content: str

class ThinkingEndEvent(BaseAGUIEvent):
    type: Literal[AGUIEventType.THINKING_END] = AGUIEventType.THINKING_END
```

---

## Part 2: CopilotKit Frontend Enhancements

### 2.1 Currently Used Hooks (Keep & Maintain)

| Hook | Location | Purpose |
|------|----------|---------|
| `useCopilotReadable` | `use-copilot-context.ts` | Share page/session/preferences |
| `useCoAgentStateRender` | `ThoughtTraceStepper.tsx`, `use-generative-ui.tsx` | Render agent state |
| `useFrontendTool` | `use-copilot-actions.ts` | 5 frontend tools |
| `useCopilotAction` | `use-generative-ui.tsx` | 3 render-only actions |
| `useCopilotChat` | `use-programmatic-chat.ts` | Headless chat control |
| `useHumanInTheLoop` | `use-source-validation.ts` | HITL validation |
| `useCopilotAdditionalInstructions` | `use-dynamic-instructions.ts` | Dynamic prompts |
| `useRenderToolCall` | `tool-renderers.tsx` | Tool visualization |

### 2.2 New Hooks to Implement (FREE)

#### 2.2.1 `useCoAgent` - Bidirectional State Sync

**Purpose:** Enable frontend → agent state updates (currently only agent → frontend)

**File:** `frontend/hooks/use-coagent-state.ts` (new)

```typescript
import { useCoAgent } from "@copilotkit/react-core";

export interface OrchestratorState {
  selectedSources: string[];
  filterCriteria: {
    dateRange?: { start: string; end: string };
    documentTypes?: string[];
    minSimilarity?: number;
  };
  userFeedback: {
    helpful?: boolean;
    accuracy?: number;
  };
}

export function useOrchestratorState(initialState?: Partial<OrchestratorState>) {
  const { agentState, setAgentState } = useCoAgent<OrchestratorState>({
    name: "orchestrator",
    initialState: {
      selectedSources: [],
      filterCriteria: {},
      userFeedback: {},
      ...initialState,
    },
  });

  const updateFilters = useCallback((filters: OrchestratorState['filterCriteria']) => {
    setAgentState(prev => ({ ...prev, filterCriteria: filters }));
  }, [setAgentState]);

  const selectSources = useCallback((sourceIds: string[]) => {
    setAgentState(prev => ({ ...prev, selectedSources: sourceIds }));
  }, [setAgentState]);

  const provideFeedback = useCallback((feedback: OrchestratorState['userFeedback']) => {
    setAgentState(prev => ({ ...prev, userFeedback: feedback }));
  }, [setAgentState]);

  return {
    agentState,
    updateFilters,
    selectSources,
    provideFeedback,
  };
}
```

**Usage in Components:**

```tsx
// In search results component
const { updateFilters, selectSources } = useOrchestratorState();

<FilterPanel onChange={updateFilters} />
<SourceList onSelect={selectSources} />
```

#### 2.2.2 Simplify HITL with `renderAndWaitForResponse`

**Purpose:** Simplify current custom HITL pattern

**File:** `frontend/hooks/use-source-validation-v2.ts` (new, simpler version)

```typescript
import { useCopilotAction } from "@copilotkit/react-core";

export function useSourceValidationV2() {
  useCopilotAction({
    name: "validate_sources",
    parameters: [
      { name: "sources", type: "object[]", required: true },
      { name: "query", type: "string", required: true },
      { name: "checkpoint_id", type: "string", required: true },
    ],
    renderAndWaitForResponse: ({ args, respond, status }) => {
      if (status === "complete") {
        return <div className="text-green-600">Sources validated</div>;
      }

      return (
        <SourceValidationDialog
          sources={args.sources}
          query={args.query}
          onApprove={(approvedIds) => respond?.({ approved: approvedIds })}
          onReject={() => respond?.({ approved: [] })}
          onSkip={() => respond?.({ approved: args.sources.map(s => s.id) })}
        />
      );
    },
  });
}
```

### 2.3 Enhanced Context Sharing

**File:** `frontend/hooks/use-copilot-context.ts`

Add more context for richer AI understanding:

```typescript
// Add document context when viewing specific documents
useCopilotReadable({
  description: "Currently viewed document",
  value: currentDocument ? {
    id: currentDocument.id,
    title: currentDocument.title,
    type: currentDocument.type,
    summary: currentDocument.summary?.slice(0, 500),
  } : null,
});

// Add search context
useCopilotReadable({
  description: "Current search filters and results summary",
  value: {
    activeFilters: searchFilters,
    resultCount: searchResults?.length ?? 0,
    topSources: searchResults?.slice(0, 3).map(r => r.title),
  },
});
```

### 2.4 Tool Renderers Enhancement

**File:** `frontend/components/copilot/tool-renderers.tsx`

Add specialized renderers for more tools:

```typescript
// Add renderer for graph operations
useRenderToolCall({
  name: "graph_traverse",
  render: ({ args, status, result }) => (
    <GraphTraversalCard
      startNode={args.start_node}
      relationship={args.relationship}
      depth={args.depth}
      status={status}
      paths={result?.paths}
    />
  ),
});

// Add renderer for document ingestion progress
useRenderToolCall({
  name: "ingest_document",
  render: ({ args, status, result }) => (
    <IngestionProgressCard
      documentName={args.filename}
      status={status}
      progress={result?.progress}
      chunks={result?.chunk_count}
    />
  ),
});
```

---

## Part 3: A2UI/AG-UI Composer Integration

### 3.1 Widget Opportunities

AG-UI Composer provides pre-built widgets that could replace custom components:

| Current Component | Composer Alternative | Benefit |
|-------------------|---------------------|---------|
| `ThoughtTraceStepper` | Step Progress Widget | Animated, polished |
| `SourceValidationDialog` | Approval Dialog Widget | Standard patterns |
| `StatusBadge` | Status Indicator Widget | Consistent styling |
| `GraphPreview` | Graph Widget | Interactive features |
| Custom charts | Chart Widgets | Declarative config |

### 3.2 Declarative UI from Backend

**File:** `backend/src/agentic_rag_backend/protocols/open_json_ui.py`

Enable backend to send declarative UI that frontend renders:

```python
# Backend can send UI descriptions
yield CustomEvent(
    name: "render_ui",
    value: {
        "type": "approval_dialog",
        "props": {
            "title": "Approve Sources",
            "items": sources,
            "actions": ["approve", "reject", "skip"]
        }
    }
)
```

**File:** `frontend/components/copilot/GenerativeUIRenderer.tsx`

```typescript
// Frontend renders declarative UI from backend
const widgetRegistry = {
  approval_dialog: ApprovalDialogWidget,
  step_progress: StepProgressWidget,
  data_table: DataTableWidget,
  chart: ChartWidget,
};

function GenerativeUIRenderer({ event }) {
  if (event.type === "CUSTOM" && event.name === "render_ui") {
    const Widget = widgetRegistry[event.value.type];
    return Widget ? <Widget {...event.value.props} /> : null;
  }
  // ... existing logic
}
```

### 3.3 Widget Registry

**File:** `frontend/lib/widget-registry.ts` (new)

```typescript
import { StepProgressWidget } from "@/components/widgets/StepProgressWidget";
import { ApprovalDialogWidget } from "@/components/widgets/ApprovalDialogWidget";
import { DataTableWidget } from "@/components/widgets/DataTableWidget";

export const widgetRegistry = {
  // Core widgets
  step_progress: StepProgressWidget,
  approval_dialog: ApprovalDialogWidget,
  data_table: DataTableWidget,

  // Data visualization
  bar_chart: BarChartWidget,
  line_chart: LineChartWidget,
  pie_chart: PieChartWidget,

  // Knowledge graph
  graph_view: GraphViewWidget,
  node_detail: NodeDetailWidget,

  // Forms
  filter_form: FilterFormWidget,
  search_form: SearchFormWidget,
} as const;
```

---

## Part 4: Implementation Phases

### Phase 1: AG-UI Protocol Compliance (Backend) ✅ COMPLETE

1. ✅ Add missing event type models (9 new types)
2. ✅ Implement STATE_DELTA for incremental updates
3. ✅ Add TOOL_CALL_RESULT emission
4. ✅ Update metrics for new event types
5. ✅ Add tests for new events (31 tests passing)

**Files Modified:**
- `backend/src/agentic_rag_backend/models/copilot.py`
- `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`
- `backend/src/agentic_rag_backend/protocols/ag_ui_metrics.py`
- `backend/tests/protocols/test_ag_ui_bridge.py`

### Phase 2: Frontend Hook Enhancements ✅ COMPLETE

1. ✅ Implement `useCoAgent` wrapper (`useOrchestratorState`)
2. ✅ Create simplified HITL with `renderAndWaitForResponse`
3. ✅ Enhance context sharing (DocumentContext, SearchContext)
4. ⏳ Add new tool renderers (deferred - existing renderers sufficient)

**Files Created:**
- `frontend/hooks/use-coagent-state.ts`
- `frontend/hooks/use-source-validation-v2.ts`

**Files Modified:**
- `frontend/hooks/use-copilot-context.ts`

### Phase 3: Widget System ✅ COMPLETE

1. ✅ Create widget registry
2. ✅ Implement core widgets (StepProgress, ApprovalDialog, DataTable)
3. ✅ Add backend CUSTOM event support
4. ✅ Connect GenerativeUIRenderer to widget registry

**Files Created:**
- `frontend/lib/widget-registry.ts`
- `frontend/lib/widget-init.ts`
- `frontend/components/widgets/StepProgressWidget.tsx`
- `frontend/components/widgets/ApprovalDialogWidget.tsx`
- `frontend/components/widgets/DataTableWidget.tsx`
- `frontend/components/copilot/CustomEventRenderer.tsx`

**Files Modified:**
- `frontend/components/copilot/GenerativeUIRenderer.tsx`

### Phase 4: THINKING Events ✅ COMPLETE (included in Phase 1)

1. ✅ Add THINKING_* event models
2. ✅ Emit thinking events during orchestrator reasoning
3. ⏳ Create ThinkingIndicator component (can use existing ThoughtTraceStepper)
4. ⏳ Integrate with ThoughtTraceStepper

**Files Modified:**
- `backend/src/agentic_rag_backend/models/copilot.py`
- `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py`

### Phase 5: ACTIVITY Events ✅ COMPLETE

1. ✅ Add ACTIVITY_* event models (done in Phase 1)
2. ✅ Emit activity events for long operations (ingestion, large searches)
3. ✅ Create ActivityTrackerWidget component
4. ✅ Show progress in UI with useActivityTracker hook

**Files Created:**
- `frontend/components/widgets/ActivityTrackerWidget.tsx`
- `frontend/hooks/use-activity-tracker.ts`

---

## Part 5: Testing Strategy

### Backend Tests

```python
# tests/protocols/test_ag_ui_events.py

async def test_state_delta_emission():
    """STATE_DELTA events use RFC 6902 JSON Patch format"""

async def test_tool_call_result_before_end():
    """TOOL_CALL_RESULT emitted before TOOL_CALL_END"""

async def test_thinking_events_wrapper():
    """Thoughts wrapped in THINKING_START/END"""

async def test_activity_progress_updates():
    """ACTIVITY_DELTA updates progress incrementally"""
```

### Frontend Tests

```typescript
// __tests__/hooks/use-coagent-state.test.ts

it("should sync state bidirectionally with agent")
it("should update filters and notify agent")
it("should handle agent state updates")

// __tests__/components/widgets/StepProgressWidget.test.tsx

it("should render steps from STATE_DELTA events")
it("should animate step transitions")
it("should handle completion state")
```

### Integration Tests

```typescript
// e2e/ag-ui-protocol.spec.ts

test("full AG-UI event flow with all event types")
test("HITL flow with renderAndWaitForResponse")
test("bidirectional state sync with useCoAgent")
test("widget rendering from CUSTOM events")
```

---

## Part 6: Files Summary

### New Files to Create

| File | Purpose |
|------|---------|
| `frontend/hooks/use-coagent-state.ts` | Bidirectional agent state |
| `frontend/hooks/use-source-validation-v2.ts` | Simplified HITL |
| `frontend/lib/widget-registry.ts` | Widget type registry |
| `frontend/components/widgets/StepProgressWidget.tsx` | Step progress widget |
| `frontend/components/widgets/ApprovalDialogWidget.tsx` | Approval dialog widget |
| `frontend/components/widgets/DataTableWidget.tsx` | Data table widget |

### Files to Modify

| File | Changes |
|------|---------|
| `backend/.../models/copilot.py` | Add 7 new event models |
| `backend/.../protocols/ag_ui_bridge.py` | Emit new events |
| `backend/.../protocols/ag_ui_metrics.py` | Track new events |
| `frontend/hooks/use-copilot-context.ts` | Enhanced context |
| `frontend/components/copilot/tool-renderers.tsx` | New renderers |
| `frontend/components/copilot/GenerativeUIRenderer.tsx` | Widget support |

---

## Part 7: Cost Analysis

| Feature | Cost | Notes |
|---------|------|-------|
| All AG-UI events | FREE | Protocol is open |
| useCoAgent | FREE | Open source |
| useHumanInTheLoop | FREE | Open source |
| renderAndWaitForResponse | FREE | Open source |
| Widget system | FREE | Custom implementation |
| Unlimited users | FREE | Self-hosted |
| **Premium (NOT included)** | | |
| Inspector | $1K+/mo | Not needed for production |
| Observability | $1K+/mo | Use custom metrics instead |
| useCopilotChatHeadless_c | $1K+/mo | useCopilotChat is sufficient |

---

## Part 8: Advanced AG-UI Capabilities

### 8.1 Sub-Agent Composition

**Purpose:** Enable nested agent delegation with scoped state and tracing

**Backend Implementation:**
```python
# backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py

class SubAgentContext:
    """Context for sub-agent delegation"""
    parent_run_id: str
    parent_thread_id: str
    scope: Dict[str, Any]

async def delegate_to_subagent(
    self,
    subagent_name: str,
    task: str,
    context: SubAgentContext
) -> AsyncIterator[AGUIEvent]:
    # Emit sub-agent start with parent context
    yield RunStartedEvent(
        runId=f"sub-{uuid.uuid4().hex[:12]}",
        threadId=context.parent_thread_id,
        metadata={"parent_run_id": context.parent_run_id, "agent": subagent_name}
    )

    # Sub-agent execution...

    yield RunFinishedEvent(...)
```

**Frontend Handling:**
```typescript
// Track nested agent runs
interface AgentRun {
  runId: string;
  parentRunId?: string;
  agentName: string;
  status: 'running' | 'completed' | 'error';
}

// Display sub-agent activity in UI
<SubAgentTracker runs={nestedRuns} />
```

### 8.2 Agent Steering (Real-Time User Input)

**Purpose:** Allow users to redirect agent execution mid-flow

**Backend Implementation:**
```python
# Support steering events from frontend
class SteeringInput(BaseModel):
    run_id: str
    instruction: str
    context: Optional[Dict[str, Any]] = None

@router.post("/copilot/steer")
async def steer_agent(steering: SteeringInput):
    """Inject user guidance into running agent"""
    # Signal orchestrator to incorporate steering
    await orchestrator.inject_steering(
        run_id=steering.run_id,
        instruction=steering.instruction
    )
    return {"status": "steering_applied"}
```

**Frontend Implementation:**
```typescript
// frontend/hooks/use-agent-steering.ts
export function useAgentSteering() {
  const steerAgent = useCallback(async (runId: string, instruction: string) => {
    await fetch('/api/v1/copilot/steer', {
      method: 'POST',
      body: JSON.stringify({ run_id: runId, instruction }),
    });
  }, []);

  return { steerAgent };
}

// Usage: Allow user to redirect while agent is working
<SteeringInput onSubmit={(instruction) => steerAgent(currentRunId, instruction)} />
```

### 8.3 Tool Output Streaming

**Purpose:** Stream tool results in real-time for long operations

**Backend Implementation:**
```python
# Emit tool output incrementally during execution
async def execute_tool_with_streaming(
    self,
    tool_name: str,
    args: Dict[str, Any]
) -> AsyncIterator[AGUIEvent]:
    tool_call_id = f"tool-{uuid.uuid4().hex[:12]}"

    yield ToolCallStartEvent(tool_call_id=tool_call_id, tool_name=tool_name)
    yield ToolCallArgsEvent(tool_call_id=tool_call_id, delta=json.dumps(args))

    # Stream partial results as they become available
    async for partial_result in tool.execute_streaming(args):
        yield ToolCallResultEvent(
            tool_call_id=tool_call_id,
            result=json.dumps({"partial": True, "data": partial_result})
        )

    # Final result
    yield ToolCallResultEvent(
        tool_call_id=tool_call_id,
        result=json.dumps({"partial": False, "data": final_result})
    )
    yield ToolCallEndEvent(tool_call_id=tool_call_id)
```

### 8.4 Multimodal Support

**Purpose:** Handle files, images, audio, and transcripts

**Backend Models:**
```python
# backend/src/agentic_rag_backend/models/copilot.py

class BinaryInputContent(BaseModel):
    type: Literal["binary"] = "binary"
    media_type: str  # e.g., "image/png", "audio/wav"
    data: str  # base64 encoded
    filename: Optional[str] = None

class TextInputContent(BaseModel):
    type: Literal["text"] = "text"
    content: str

class MultimodalMessage(BaseModel):
    role: MessageRole
    content: List[Union[TextInputContent, BinaryInputContent]]
```

**Frontend Support:**
```typescript
// frontend/hooks/use-multimodal-input.ts
export function useMultimodalInput() {
  const [attachments, setAttachments] = useState<Attachment[]>([]);

  const addFile = useCallback(async (file: File) => {
    const base64 = await fileToBase64(file);
    setAttachments(prev => [...prev, {
      type: 'binary',
      media_type: file.type,
      data: base64,
      filename: file.name,
    }]);
  }, []);

  const sendMultimodalMessage = useCallback(async (text: string) => {
    const content = [
      { type: 'text', content: text },
      ...attachments,
    ];
    // Send via CopilotKit with multimodal content
  }, [attachments]);

  return { attachments, addFile, sendMultimodalMessage, clearAttachments };
}
```

### 8.5 Cancel/Resume Agent Runs

**Purpose:** Allow users to pause, cancel, or resume agent execution

**Backend Implementation:**
```python
# backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py

class RunManager:
    """Manages cancellable agent runs"""
    _active_runs: Dict[str, asyncio.Event] = {}
    _run_states: Dict[str, Dict[str, Any]] = {}

    async def cancel_run(self, run_id: str) -> bool:
        if run_id in self._active_runs:
            self._active_runs[run_id].set()  # Signal cancellation
            return True
        return False

    async def get_run_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._run_states.get(run_id)

    async def resume_run(self, run_id: str) -> AsyncIterator[AGUIEvent]:
        state = self._run_states.get(run_id)
        if not state:
            raise KeyError(f"Run {run_id} not found")
        # Resume from saved state...

# API endpoints
@router.post("/copilot/cancel/{run_id}")
async def cancel_run(run_id: str):
    success = await run_manager.cancel_run(run_id)
    return {"cancelled": success}

@router.post("/copilot/resume/{run_id}")
async def resume_run(run_id: str):
    # Resume and stream remaining events
    return StreamingResponse(...)
```

**Frontend Implementation:**
```typescript
// frontend/hooks/use-run-control.ts
export function useRunControl() {
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);

  const cancelRun = useCallback(async () => {
    if (!currentRunId) return;
    await fetch(`/api/v1/copilot/cancel/${currentRunId}`, { method: 'POST' });
    setCurrentRunId(null);
  }, [currentRunId]);

  const resumeRun = useCallback(async (runId: string) => {
    const response = await fetch(`/api/v1/copilot/resume/${runId}`, { method: 'POST' });
    // Handle resumed event stream
  }, []);

  return { currentRunId, setCurrentRunId, cancelRun, resumeRun };
}
```

### 8.6 RAW Events (External Protocol Wrapping)

**Purpose:** Wrap events from external protocols (MCP, A2A) in AG-UI stream

**Backend Implementation:**
```python
class RawEvent(BaseAGUIEvent):
    type: Literal[AGUIEventType.RAW] = AGUIEventType.RAW
    event: Dict[str, Any]  # Original event data
    source: Optional[str] = None  # e.g., "mcp", "a2a"

# Wrap MCP events in AG-UI stream
async def process_mcp_tool_call(mcp_event: MCPToolResponse):
    yield RawEvent(
        event=mcp_event.model_dump(),
        source="mcp"
    )
```

---

## Part 9: Implementation Priority Matrix

| Feature | Priority | Complexity | Value | Phase |
|---------|----------|------------|-------|-------|
| STATE_DELTA | High | Low | High | 1 |
| TOOL_CALL_RESULT | High | Low | Medium | 1 |
| useCoAgent | High | Medium | High | 2 |
| Simplified HITL | High | Low | High | 2 |
| Widget Registry | Medium | Medium | High | 3 |
| THINKING events | Medium | Low | Medium | 4 |
| ACTIVITY events | Medium | Medium | Medium | 5 |
| Tool Output Streaming | Medium | Medium | High | 5 |
| Cancel/Resume | Medium | High | High | 6 |
| Agent Steering | Low | High | Medium | 6 |
| Sub-Agent Composition | Low | High | Medium | 7 |
| Multimodal Support | Low | High | Medium | 7 |
| RAW Events | Low | Low | Low | 7 |

---

## Part 10: Verification Plan

1. **Backend Events:** Run `pytest backend/tests/protocols/` - all new event tests pass
2. **Frontend Hooks:** Run `pnpm test` - all hook tests pass
3. **Integration:**
   - Start full stack: `docker compose up`
   - Open browser, send query
   - Verify STATE_DELTA events in network tab
   - Verify bidirectional state with filter changes
   - Test HITL flow with source validation
4. **Metrics:** Check Prometheus at `/metrics` for new event types

---

## Appendix: CopilotKit Pricing Reference

| Tier | Cost | MAU Limit | Notes |
|------|------|-----------|-------|
| Developer | Free | 50 MAU | 1 seat |
| Team | $1,000/seat/mo | 100 MAU/seat | +$100/100 MAU overage |
| Enterprise | From $5K/mo | Pooled | Custom terms |

**Self-Hosted = FREE unlimited users** (current configuration)

---

## Part 11: Code Review Fixes ✅ COMPLETE

Comprehensive code review of Phases 5-7 identified 16 issues across security, type safety, error handling, and code quality. All have been resolved.

### 11.1 Critical Issues (3)

| Issue | Fix | File |
|-------|-----|------|
| ✅ Tenant authorization on cancel/resume | Already in place (verified) | `copilot.py` |
| ✅ Tenant validation in steering | Already in place (verified) | `copilot.py` |
| ✅ Backend size validation | Added `MAX_BINARY_CONTENT_SIZE` (10MB) + `ALLOWED_BINARY_MEDIA_TYPES` validators | `models/copilot.py` |

### 11.2 Medium Issues (10)

| Issue | Fix | File |
|-------|-----|------|
| ✅ Race condition in run cancellation | Added `asyncio.Lock` to `RunManager` | `ag_ui_bridge.py` |
| ✅ Unsafe type casting in activity tracker | Added type guards in `applyPatchOperation()` | `use-activity-tracker.ts` |
| ✅ Loose typing in JSON Patch operations | Created shared types in `types/ag-ui.ts` | `types/ag-ui.ts` |
| ✅ Silent failure in run state retrieval | Created `GetRunStateResult` interface | `use-run-control.ts` |
| ✅ Steering injection after completion | Added `checkRunStatus()` method | `use-agent-steering.ts` |
| ✅ Activity completion race condition | Added activity ID tracking with `useRef` | `use-activity-tracker.ts` |
| ✅ Unnecessary re-renders | Optimized memoization with stable deps | `use-activity-tracker.ts` |
| ✅ Error boundary in widget rendering | Created `WidgetErrorBoundary` component | `WidgetErrorBoundary.tsx` |
| ✅ Memory leak in chart widget | Fixed memoization dependencies | `ChartWidget.tsx` |
| ✅ OpenAPI documentation | Added to all Phase 6 endpoints | `copilot.py` |

### 11.3 Low Issues (3)

| Issue | Fix | File |
|-------|-----|------|
| ✅ Extract shared types | Created `frontend/types/ag-ui.ts` | `types/ag-ui.ts` |
| ✅ Magic numbers | Added `ACTIVITY_RESET_DELAY_MS`, `MAX_FILE_SIZE`, `MAX_ATTACHMENTS` | `types/ag-ui.ts` |
| ✅ Invalid chart type handling | Added fallback UI for unknown types | `ChartWidget.tsx` |

### 11.4 New Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `use-activity-tracker.test.ts` | 10 | Snapshots, deltas, security, auto-reset |
| `use-run-control.test.ts` | 10 | Cancel, resume, state retrieval, errors |
| `use-agent-steering.test.ts` | 11 | Validation, status check, API calls |
| `WidgetErrorBoundary.test.tsx` | 9 | Error catching, retry, custom fallback |

### 11.5 Files Created

| File | Purpose |
|------|---------|
| `frontend/types/ag-ui.ts` | Shared types and constants for AG-UI |
| `frontend/components/widgets/WidgetErrorBoundary.tsx` | Error boundary for widget rendering |
| `frontend/__tests__/hooks/use-activity-tracker.test.ts` | Activity tracker hook tests |
| `frontend/__tests__/hooks/use-run-control.test.ts` | Run control hook tests |
| `frontend/__tests__/hooks/use-agent-steering.test.ts` | Agent steering hook tests |
| `frontend/__tests__/components/widgets/WidgetErrorBoundary.test.tsx` | Error boundary tests |

### 11.6 Files Modified

| File | Changes |
|------|---------|
| `backend/.../models/copilot.py` | Size/type validation for BinaryInputContent |
| `backend/.../protocols/ag_ui_bridge.py` | Thread-safe RunManager with asyncio.Lock |
| `backend/.../api/routes/copilot.py` | OpenAPI documentation for all endpoints |
| `frontend/hooks/use-activity-tracker.ts` | Type guards, memoization, activity ID tracking |
| `frontend/hooks/use-run-control.ts` | GetRunStateResult type, explicit error handling |
| `frontend/hooks/use-agent-steering.ts` | checkRunStatus(), status validation |
| `frontend/hooks/use-multimodal-input.ts` | Use shared constants from types/ag-ui.ts |
| `frontend/components/widgets/ActivityTrackerWidget.tsx` | Import from shared types |
| `frontend/components/widgets/ChartWidget.tsx` | Fixed memoization, invalid type handling |

### 11.7 Security Improvements

1. **Prototype Pollution Prevention**: JSON Patch operations now validate keys against `ACTIVITY_STATE_KEYS` set
2. **Type Validation**: All patch values validated against expected types before application
3. **Size Limits**: Backend enforces 10MB limit on binary content with allowed media types
4. **Race Condition Protection**: `asyncio.Lock` prevents concurrent state mutations

### 11.8 Verification

```bash
# Backend tests
cd backend && uv run pytest  # 48 passing

# Frontend tests
cd frontend && pnpm test     # 1043 passing (47 suites)
```
