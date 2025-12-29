# Story 6-2: Chat Sidebar Interface

Status: drafted
Epic: 6 - Interactive Copilot Experience
Priority: High
Depends on: Story 6-1 (CopilotKit React Integration)

## User Story

As an **end-user**,
I want **to interact with the AI through a pre-built chat sidebar**,
So that **I can ask questions and receive responses naturally with visible agent progress**.

## Acceptance Criteria

- Given the CopilotKit integration is configured (Story 6-1)
- When the user opens the chat sidebar
- Then they see a polished chat interface (shadcn/ui styling)
- And can type messages and submit queries
- And see streaming responses as they generate
- And view the "Thought Trace" stepper showing agent progress
- And the sidebar uses the design system colors (Indigo-600, Slate)
- And typography follows the specification (Inter for headings/body, JetBrains Mono for traces)

## Technical Approach

### 1. Create Chat Sidebar Component

**File:** `frontend/src/components/copilot/ChatSidebar.tsx`

Create the main chat interface using CopilotKit's UI components with the project's design system:

```typescript
"use client";

import { CopilotSidebar } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { ThoughtTraceStepper } from "./ThoughtTraceStepper";

export function ChatSidebar() {
  return (
    <CopilotSidebar
      defaultOpen={true}
      labels={{
        title: "AI Copilot",
        initial: "How can I help you today?",
      }}
      className="copilot-sidebar"
    >
      <ThoughtTraceStepper />
    </CopilotSidebar>
  );
}
```

Key implementation details:
- Use `CopilotSidebar` from `@copilotkit/react-ui` as the base component
- Apply custom CSS variables to override default CopilotKit styling with design system colors
- Include `ThoughtTraceStepper` component within the sidebar for agent progress visibility
- Support both controlled and uncontrolled open/close states

### 2. Create Thought Trace Stepper Component

**File:** `frontend/src/components/copilot/ThoughtTraceStepper.tsx`

Implement a vertical progress indicator that streams the agent's current task:

```typescript
"use client";

import { useCoAgentStateRender } from "@copilotkit/react-core";
import { cn } from "@/lib/utils";
import { ThoughtStep } from "@/types/copilot";
import { ChevronDown, ChevronRight, CheckCircle2, Loader2, Circle } from "lucide-react";
import { useState } from "react";

export function ThoughtTraceStepper() {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());

  useCoAgentStateRender<{ steps: ThoughtStep[] }>({
    name: "orchestrator",
    render: ({ state }) => {
      if (!state?.steps?.length) return null;

      return (
        <div className="flex flex-col gap-2 p-4 font-mono text-sm border-t border-slate-200">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Agent Progress
          </h3>
          {state.steps.map((step, idx) => (
            <StepIndicator
              key={idx}
              step={step}
              index={idx}
              isExpanded={expandedSteps.has(idx)}
              onToggle={() => toggleStep(idx)}
            />
          ))}
        </div>
      );
    },
  });

  const toggleStep = (index: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  return null; // Render is handled by useCoAgentStateRender hook
}
```

Visual design requirements:
- Vertical layout with step indicators
- Status icons: `Circle` (pending), `Loader2` (in_progress with animation), `CheckCircle2` (completed)
- Expandable steps showing raw logs/details when clicked
- Color coding: Slate for pending, Indigo-600 for in_progress, Emerald-500 for completed

### 3. Create Step Indicator Sub-Component

**File:** `frontend/src/components/copilot/ThoughtTraceStepper.tsx` (same file)

```typescript
interface StepIndicatorProps {
  step: ThoughtStep;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
}

function StepIndicator({ step, index, isExpanded, onToggle }: StepIndicatorProps) {
  const StatusIcon = {
    pending: Circle,
    in_progress: Loader2,
    completed: CheckCircle2,
  }[step.status];

  const statusColors = {
    pending: "text-slate-400",
    in_progress: "text-indigo-600 animate-spin",
    completed: "text-emerald-500",
  };

  return (
    <div className="flex flex-col">
      <button
        onClick={onToggle}
        className="flex items-center gap-2 text-left hover:bg-slate-50 rounded p-1 -m-1"
      >
        {step.details ? (
          isExpanded ? (
            <ChevronDown className="h-3 w-3 text-slate-400" />
          ) : (
            <ChevronRight className="h-3 w-3 text-slate-400" />
          )
        ) : (
          <span className="w-3" />
        )}
        <StatusIcon className={cn("h-4 w-4", statusColors[step.status])} />
        <span className={cn(
          "text-sm",
          step.status === "in_progress" && "text-indigo-600 font-medium",
          step.status === "completed" && "text-slate-600",
          step.status === "pending" && "text-slate-400"
        )}>
          {step.step}
        </span>
      </button>
      {isExpanded && step.details && (
        <pre className="mt-1 ml-7 p-2 bg-slate-100 rounded text-xs text-slate-600 font-mono overflow-x-auto">
          {step.details}
        </pre>
      )}
    </div>
  );
}
```

### 4. Implement Thought Trace Hook

**File:** `frontend/src/hooks/use-thought-trace.ts`

Create a custom hook to manage thought trace state:

```typescript
"use client";

import { useCopilotContext } from "@copilotkit/react-core";
import { useState, useEffect } from "react";
import { ThoughtStep } from "@/types/copilot";

interface UseThoughtTraceOptions {
  autoExpand?: boolean;
  maxVisibleSteps?: number;
}

interface UseThoughtTraceReturn {
  steps: ThoughtStep[];
  currentStep: ThoughtStep | null;
  isProcessing: boolean;
  clearSteps: () => void;
}

export function useThoughtTrace(options: UseThoughtTraceOptions = {}): UseThoughtTraceReturn {
  const { autoExpand = true, maxVisibleSteps = 10 } = options;
  const [steps, setSteps] = useState<ThoughtStep[]>([]);

  // Get the current processing state from context
  const context = useCopilotContext();
  const isProcessing = steps.some((s) => s.status === "in_progress");
  const currentStep = steps.find((s) => s.status === "in_progress") || null;

  const clearSteps = () => setSteps([]);

  return {
    steps: steps.slice(-maxVisibleSteps),
    currentStep,
    isProcessing,
    clearSteps,
  };
}
```

### 5. Create Custom CSS for Design System Integration

**File:** `frontend/src/app/globals.css` (append to existing)

```css
/* CopilotKit Design System Overrides */
:root {
  /* Map design system colors to CopilotKit variables */
  --copilot-kit-primary-color: #4F46E5; /* Indigo-600 */
  --copilot-kit-secondary-color: #10B981; /* Emerald-500 */
  --copilot-kit-background-color: #F8FAFC; /* Slate-50 */
  --copilot-kit-text-color: #0F172A; /* Slate-900 */
  --copilot-kit-border-color: #E2E8F0; /* Slate-200 */
  --copilot-kit-accent-color: #FBBF24; /* Amber-400 for HITL */
}

.copilot-sidebar {
  --font-heading: 'Inter', sans-serif;
  --font-body: 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

/* Sidebar container styling */
.copilot-sidebar {
  @apply bg-white border-l border-slate-200 shadow-lg;
}

/* Message styling */
.copilot-sidebar [data-role="user"] {
  @apply bg-indigo-50 text-slate-900 rounded-lg;
}

.copilot-sidebar [data-role="assistant"] {
  @apply bg-white text-slate-700 rounded-lg border border-slate-100;
}

/* Input styling */
.copilot-sidebar textarea,
.copilot-sidebar input {
  @apply border-slate-300 focus:border-indigo-500 focus:ring-indigo-500 rounded-lg;
}

/* Send button styling */
.copilot-sidebar button[type="submit"] {
  @apply bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg;
}

/* Thought trace styling */
.thought-trace {
  @apply font-mono text-sm text-slate-600;
}

.thought-trace-step {
  @apply flex items-center gap-2 py-1;
}

.thought-trace-step.completed {
  @apply text-emerald-600;
}

.thought-trace-step.in-progress {
  @apply text-indigo-600;
}

.thought-trace-step.pending {
  @apply text-slate-400;
}
```

### 6. Update Page to Include Chat Sidebar

**File:** `frontend/src/app/page.tsx` (modify)

Integrate the ChatSidebar component into the main page:

```typescript
import { ChatSidebar } from "@/components/copilot/ChatSidebar";

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-50">
      {/* Main content area */}
      <div className="container mx-auto py-8">
        {/* Existing page content */}
      </div>

      {/* Chat Sidebar - positioned by CopilotKit */}
      <ChatSidebar />
    </main>
  );
}
```

### 7. Add Font Configuration

**File:** `frontend/src/app/layout.tsx` (modify)

Ensure Inter and JetBrains Mono fonts are properly loaded:

```typescript
import { Inter, JetBrains_Mono } from "next/font/google";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

// In the body className:
<body className={`${inter.variable} ${jetbrainsMono.variable} font-sans`}>
```

### 8. Backend: Update AG-UI Bridge for Thought Steps

**File:** `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` (modify)

Enhance the state snapshot to include thought steps for the frontend:

```python
async def _create_state_snapshot(
    self,
    agent_state: AgentState,
) -> StateSnapshotEvent:
    """Create a state snapshot event with thought steps."""
    thought_steps = [
        {
            "step": thought.content,
            "status": "completed" if thought.completed else "in_progress",
            "timestamp": thought.timestamp.isoformat() if thought.timestamp else None,
            "details": thought.details,
        }
        for thought in agent_state.thoughts
    ]

    return StateSnapshotEvent(
        state={
            "currentStep": agent_state.current_step,
            "steps": thought_steps,
            "retrievedSources": agent_state.retrieved_sources,
            "answer": agent_state.answer,
            "trajectoryId": str(agent_state.trajectory_id) if agent_state.trajectory_id else None,
        }
    )
```

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `frontend/src/components/copilot/ChatSidebar.tsx` | Main chat sidebar component using CopilotKit |
| `frontend/src/components/copilot/ThoughtTraceStepper.tsx` | Vertical progress indicator showing agent steps |
| `frontend/src/hooks/use-thought-trace.ts` | Custom hook for managing thought trace state |

### Modified Files

| File | Change |
|------|--------|
| `frontend/src/app/page.tsx` | Integrate ChatSidebar component |
| `frontend/src/app/globals.css` | Add CopilotKit design system overrides |
| `frontend/src/app/layout.tsx` | Add Inter and JetBrains Mono font configuration |
| `frontend/tailwind.config.ts` | Add custom design system colors if not present |
| `backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py` | Enhance state snapshot with thought steps |

## Dependencies

### Frontend Dependencies (npm)

Already installed from Story 6-1:
```json
{
  "@copilotkit/react-core": "^1.50.1",
  "@copilotkit/react-ui": "^1.50.1",
  "@copilotkit/runtime": "^1.50.1"
}
```

Additional icons (already in project via shadcn/ui):
```json
{
  "lucide-react": "^0.x.x"
}
```

### Backend Dependencies (pip)

No new dependencies required. Uses existing:
- FastAPI + SSE from Story 6-1
- Existing OrchestratorAgent from Epic 2

### Environment Variables

No new environment variables required. Uses existing:
- `NEXT_PUBLIC_COPILOT_ENABLED=true` (from Story 6-1)

## Testing Requirements

### Unit Tests

| Test | Location |
|------|----------|
| ChatSidebar renders correctly | `frontend/__tests__/components/copilot/ChatSidebar.test.tsx` |
| ThoughtTraceStepper displays steps with correct statuses | `frontend/__tests__/components/copilot/ThoughtTraceStepper.test.tsx` |
| StepIndicator expands/collapses on click | `frontend/__tests__/components/copilot/ThoughtTraceStepper.test.tsx` |
| useThoughtTrace hook returns correct state | `frontend/__tests__/hooks/use-thought-trace.test.ts` |

### Integration Tests

| Test | Location |
|------|----------|
| Sidebar opens and closes correctly | `frontend/__tests__/integration/chat-sidebar.test.tsx` |
| Messages stream in real-time | `frontend/__tests__/integration/chat-sidebar.test.tsx` |
| Thought trace updates during agent processing | `frontend/__tests__/integration/chat-sidebar.test.tsx` |

### E2E Tests

| Test | Location |
|------|----------|
| Full chat flow with thought trace | `frontend/tests/e2e/chat-sidebar.spec.ts` |
| Design system colors applied correctly | `frontend/tests/e2e/chat-sidebar.spec.ts` |

### Manual Verification Steps

1. Start backend with `cd backend && uv run uvicorn agentic_rag_backend.main:app --reload`
2. Start frontend with `cd frontend && pnpm dev`
3. Open browser to `http://localhost:3000`
4. Verify chat sidebar is visible and uses Indigo-600 primary color
5. Verify Inter font is used for headings and body text
6. Type a message and verify it appears in the chat with proper styling
7. Submit a query and verify:
   - Streaming response appears progressively
   - Thought trace stepper shows agent progress
   - In-progress step animates with Indigo-600 color
   - Completed steps show Emerald-500 checkmark
8. Click on a thought step with details and verify it expands to show raw logs
9. Verify JetBrains Mono font is used in the thought trace area
10. Test sidebar open/close functionality

## Definition of Done

- [ ] All acceptance criteria met
- [ ] ChatSidebar component created and renders correctly
- [ ] ThoughtTraceStepper component displays agent progress
- [ ] Step details expand/collapse on click
- [ ] Design system colors applied (Indigo-600, Slate, Emerald-500)
- [ ] Typography follows spec (Inter, JetBrains Mono)
- [ ] Streaming responses display progressively
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] E2E tests passing
- [ ] Manual verification completed
- [ ] No TypeScript errors
- [ ] Code follows project naming conventions
- [ ] CSS follows project patterns (Tailwind utilities)

## Technical Notes

### CopilotKit UI Customization

CopilotKit provides several customization points:
- CSS variables for colors and typography
- `labels` prop for customizing text
- `className` prop for additional styling
- Child components for rendering custom UI within the sidebar

### State Management

The implementation uses CopilotKit's built-in state management:
- `useCoAgentStateRender` for rendering agent state changes
- State snapshots from backend provide `steps` array for thought trace
- No additional state management library needed for this story

### Design System Colors

Per UX Design Specification:
- **Primary (Indigo-600):** #4F46E5 - Intelligence/Brain
- **Secondary (Emerald-500):** #10B981 - Success/Validated
- **Neutral (Slate):** #0F172A to #F8FAFC - Readability
- **Accent (Amber-400):** #FBBF24 - HITL attention (used in Story 6-4)

### Accessibility Considerations

- Thought trace stepper should be keyboard navigable
- Expandable sections should use proper ARIA attributes
- Color contrast should meet WCAG AA standards
- Animation should respect `prefers-reduced-motion`

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CopilotKit styling conflicts | Use CSS specificity carefully, test with shadcn/ui components |
| State sync issues between agent and UI | Add error boundaries, implement reconnection logic |
| Font loading performance | Use Next.js font optimization with `next/font` |
| Thought trace overwhelming users | Implement `maxVisibleSteps` option, add collapse all |

## References

- [CopilotKit UI Documentation](https://docs.copilotkit.ai/reference/components/CopilotSidebar)
- [useCoAgentStateRender Hook](https://docs.copilotkit.ai/reference/hooks/useCoAgentStateRender)
- [Epic 6 Tech Spec](_bmad-output/implementation-artifacts/epic-6-tech-spec.md)
- [UX Design Specification](_bmad-output/project-planning-artifacts/ux-design-specification.md)
- [Story 6-1: CopilotKit React Integration](_bmad-output/implementation-artifacts/stories/6-1-copilotkit-react-integration.md)
- [Architecture Document](_bmad-output/architecture.md)
