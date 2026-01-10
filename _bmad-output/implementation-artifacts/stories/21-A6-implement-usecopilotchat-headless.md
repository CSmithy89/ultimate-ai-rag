# Story 21-A6: Implement useCopilotChat for Headless Control

Status: done

Epic: 21 - CopilotKit Full Integration
Priority: P2 - LOW
Story Points: 3
Owner: Frontend

## Story

As a **developer building custom interfaces or automated testing**,
I want **programmatic control over the chat conversation**,
So that **I can send messages programmatically, access chat history, clear conversations, and control generation without using the built-in chat UI**.

## Background

The Epic 21 frontend audit identified that `useCopilotChat` is not used anywhere in our codebase despite CopilotKit providing comprehensive headless chat capabilities. Currently, developers cannot:

- Send messages programmatically (e.g., from quick action buttons)
- Regenerate AI responses on demand
- Stop generation mid-stream
- Clear conversation history
- Access message history for custom UIs

### Why Headless Control Matters for RAG

1. **Quick Actions** - Preset buttons like "Summarize", "Extract Key Points" can trigger messages
2. **Test Automation** - E2E tests can send messages and verify responses programmatically
3. **Custom UIs** - Build entirely custom chat interfaces while using CopilotKit backend
4. **Onboarding Flows** - Guide users through scripted tutorials with preset messages
5. **Keyboard Shortcuts** - Power users can trigger regeneration with Cmd+R

### CopilotKit useCopilotChat API

The hook provides these capabilities:
- `visibleMessages` - Array of messages in the conversation
- `appendMessage` - Send a new message programmatically
- `reloadMessages` - Regenerate response for a specific message
- `stopGeneration` - Cancel ongoing generation
- `reset` - Clear all messages and reset chat state
- `isLoading` - Whether chat is generating a response
- `runChatCompletion` - Manually trigger completion for advanced usage

## Acceptance Criteria

1. **Given** the `useProgrammaticChat` hook is called, **when** it initializes, **then** it exposes convenient wrapper methods for chat control.

2. **Given** the `sendMessage` function is called with content, **when** executed, **then** a new user message is appended and AI responds.

3. **Given** the `regenerateLastResponse` function is called, **when** there are messages, **then** the last assistant message is regenerated.

4. **Given** the `stopGeneration` function is called during generation, **when** `isLoading` is true, **then** generation stops immediately.

5. **Given** the `clearHistory` function is called, **when** executed, **then** all messages are cleared and chat is reset.

6. **Given** the hook is active, **when** checking `messageCount`, **then** it returns the current number of visible messages.

7. **Given** any chat operation errors, **when** the error occurs, **then** the error is logged and the function fails gracefully without crashing.

8. **Given** the `QuickActions` component uses the hook, **when** buttons are clicked, **then** preset messages are sent to the chat.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: **N/A** - No data storage, uses CopilotKit context
- [x] Rate limiting / abuse protection: **Addressed** - Buttons disabled during loading
- [x] Input validation / schema enforcement: **Addressed** - Message content validated
- [x] Tests (unit/integration): **Addressed** - Unit tests for hook utilities
- [x] Error handling + logging: **Addressed** - Try/catch with console logging
- [x] Documentation updates: **Addressed** - JSDoc comments on hook and functions

## Security Checklist

- [x] **Cross-tenant isolation verified**: **N/A** - Uses CopilotKit context
- [x] **Authorization checked**: **N/A** - No direct API calls
- [x] **No information leakage**: **N/A** - No sensitive data handling
- [x] **Redis keys include tenant scope**: **N/A** - No Redis interactions
- [x] **Integration tests for access control**: **N/A** - Frontend-only hook
- [x] **RFC 7807 error responses**: **N/A** - No API changes
- [x] **File-path inputs scoped**: **N/A** - No file path handling

## Tasks / Subtasks

- [x] **Task 1: Create types for programmatic chat** (AC: 1)
  - [x] Define `ProgrammaticChatReturn` interface
  - [x] Define `ChatMessage` type for message structure
  - [x] Add types to `frontend/types/copilot.ts`

- [x] **Task 2: Create useProgrammaticChat hook** (AC: 1-6)
  - [x] Create `frontend/hooks/use-programmatic-chat.ts`
  - [x] Wrap `useCopilotChat` with convenient methods
  - [x] Implement `sendMessage(content: string)` function
  - [x] Implement `regenerateLastResponse()` function
  - [x] Implement `clearHistory()` function
  - [x] Expose `stopGeneration`, `isLoading`, `messages`, `messageCount`
  - [x] Add error handling with graceful degradation (AC: 7)
  - [x] Export hook and types

- [x] **Task 3: Create QuickActions component** (AC: 8)
  - [x] Create `frontend/components/copilot/QuickActions.tsx`
  - [x] Add preset action buttons (Summarize, Key Insights, Related Topics)
  - [x] Disable buttons during loading state
  - [x] Style with consistent design system

- [x] **Task 4: Add unit tests** (AC: 1-8)
  - [x] Test hook initialization
  - [x] Test message sending
  - [x] Test error handling
  - [x] Test QuickActions component

- [x] **Task 5: Documentation** (AC: all)
  - [x] Add JSDoc to all new functions
  - [x] Document use cases in comments
  - [x] Update story file with Dev Notes

## Technical Notes

### useCopilotChat Hook Pattern

```typescript
import { useCopilotChat } from "@copilotkit/react-core";
import { TextMessage, MessageRole } from "@copilotkit/runtime-client-gql";

export function useProgrammaticChat() {
  const {
    visibleMessages,
    appendMessage,
    reloadMessages,
    stopGeneration,
    reset,
    isLoading,
  } = useCopilotChat();

  // Send a message programmatically
  const sendMessage = async (content: string) => {
    try {
      await appendMessage(
        new TextMessage({
          role: MessageRole.User,
          content,
        })
      );
    } catch (error) {
      console.error("Failed to send message:", error);
    }
  };

  // Regenerate the last assistant response
  const regenerateLastResponse = async () => {
    const lastMessage = visibleMessages[visibleMessages.length - 1];
    if (lastMessage?.id) {
      try {
        await reloadMessages(lastMessage.id);
      } catch (error) {
        console.error("Failed to regenerate:", error);
      }
    }
  };

  // Clear chat history
  const clearHistory = () => {
    reset();
  };

  return {
    messages: visibleMessages,
    sendMessage,
    regenerateLastResponse,
    stopGeneration,
    clearHistory,
    isLoading,
    messageCount: visibleMessages.length,
  };
}
```

### QuickActions Component

```tsx
import { useProgrammaticChat } from "@/hooks/use-programmatic-chat";

export function QuickActions() {
  const { sendMessage, isLoading } = useProgrammaticChat();

  const actions = [
    { label: "Summarize", message: "Summarize the current document" },
    { label: "Key Insights", message: "Extract key insights" },
    { label: "Related Topics", message: "Find related topics" },
  ];

  return (
    <div className="flex gap-2 flex-wrap">
      {actions.map((action) => (
        <Button
          key={action.label}
          variant="outline"
          size="sm"
          disabled={isLoading}
          onClick={() => sendMessage(action.message)}
        >
          {action.label}
        </Button>
      ))}
    </div>
  );
}
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `frontend/hooks/use-programmatic-chat.ts` | Create | Main headless chat hook |
| `frontend/components/copilot/QuickActions.tsx` | Create | Quick action buttons component |
| `frontend/types/copilot.ts` | Modify | Add programmatic chat types |
| `frontend/__tests__/hooks/use-programmatic-chat.test.ts` | Create | Unit tests |
| `frontend/__tests__/components/QuickActions.test.tsx` | Create | Component tests |

### Error Handling Strategy

All operations are wrapped in try/catch to ensure graceful degradation:
- Errors are logged to console for debugging
- Operations fail silently without crashing the UI
- Loading state is properly managed even on errors

## Dependencies

- **CopilotKit v1.x+** - `useCopilotChat` hook available
- **@copilotkit/runtime-client-gql** - TextMessage, MessageRole types
- **Story 6-2 completed** - CopilotSidebar exists
- **Story 21-A1 to 21-A5 patterns** - Follows established hook patterns

## Definition of Done

- [x] `useProgrammaticChat` hook created with all methods
- [x] `QuickActions` component implemented
- [x] Types added to copilot.ts
- [x] Unit tests pass for hook utilities
- [x] Component renders correctly
- [x] Error handling works gracefully
- [x] Lint and type-check pass
- [x] Story file updated with Dev Notes

## Dev Notes

### Implementation Approach

The story implemented `useProgrammaticChat` as a wrapper around CopilotKit's `useCopilotChat` hook, providing convenient methods for common operations.

### Key Design Decisions

1. **Wrapper Pattern**: Created a wrapper hook rather than using `useCopilotChat` directly. This provides:
   - Simpler API for common use cases
   - Consistent error handling
   - Type-safe message creation abstraction
   - Easier testing and mocking

2. **TextMessage Abstraction**: The hook abstracts the `TextMessage` class creation, so consumers don't need to import from `@copilotkit/runtime-client-gql`.

3. **Error Isolation**: All async operations are wrapped in try/catch to prevent chat errors from crashing the app.

4. **Message Deduplication**: The `sendMessage` function checks for empty content and validates before sending.

5. **QuickActions Flexibility**: The component accepts custom actions via props, with sensible defaults for RAG-specific operations.

### Testing Strategy

Due to CopilotKit's ESM dependencies, tests mock the entire `@copilotkit/react-core` and `@copilotkit/runtime-client-gql` modules. Tests focus on:
- Type validation
- Utility function behavior
- Component rendering
- Button interaction states

### Integration Notes

The `QuickActions` component can be placed above or below the chat input area. It automatically disables during loading to prevent duplicate sends.

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No debug logs generated

### Completion Notes List

1. Added programmatic chat types to `frontend/types/copilot.ts`
2. Created `useProgrammaticChat` hook with wrapper methods
3. Created `QuickActions` component with configurable actions
4. Created comprehensive unit tests for hook utilities
5. Created component tests for QuickActions
6. All tests pass
7. TypeScript and ESLint checks pass

### File List

**Created:**
- `/home/chris/projects/work/Agentic Rag and Graphrag with copilot/frontend/hooks/use-programmatic-chat.ts`
- `/home/chris/projects/work/Agentic Rag and Graphrag with copilot/frontend/components/copilot/QuickActions.tsx`
- `/home/chris/projects/work/Agentic Rag and Graphrag with copilot/frontend/__tests__/hooks/use-programmatic-chat.test.ts`
- `/home/chris/projects/work/Agentic Rag and Graphrag with copilot/frontend/__tests__/components/QuickActions.test.tsx`

**Modified:**
- `/home/chris/projects/work/Agentic Rag and Graphrag with copilot/frontend/types/copilot.ts`

## Test Outcomes

```
Test Suites: 27 passed, 27 total
Tests:       546 passed, 546 total
Snapshots:   0 total
Time:        4.767 s
```

New tests added:
- `use-programmatic-chat.test.ts` - 58 tests covering utilities, types, and message handling
- `QuickActions.test.tsx` - Component tests for rendering, interactions, and accessibility

## Challenges Encountered

1. **CopilotKit Import Compatibility**: The `@copilotkit/runtime-client-gql` package was not installed in the project. Added it as a dependency to enable TextMessage and MessageRole imports for programmatic message creation.

2. **Package Discovery**: The Context7 documentation referenced `@copilotkit/runtime-client-gql` but the project only had `@copilotkit/react-core`, `@copilotkit/react-ui`, and `@copilotkit/runtime` installed. Resolved by adding the missing package.

## Future Enhancements

1. **Keyboard Shortcuts**: Add Cmd+R for regenerate, Cmd+K for clear.
2. **Custom Message Templates**: Allow templated messages with variable substitution.
3. **Message History Export**: Function to export chat history as JSON/Markdown.
