# Story 21-A7: Implement useCopilotAdditionalInstructions for Dynamic Prompts

Status: done

Epic: 21 - CopilotKit Full Integration
Priority: P2 - LOW
Story Points: 2
Owner: Frontend

## Story

As a **developer building context-aware AI experiences**,
I want **dynamic system prompt modifications based on application context**,
So that **the AI can receive page-specific, user preference, and feature-aware instructions without hardcoding behavior into the backend**.

## Background

The Epic 21 frontend audit identified that `useCopilotAdditionalInstructions` is not used anywhere in our codebase despite CopilotKit providing the capability to dynamically modify the system prompt. Currently:

- AI behavior is static regardless of which page the user is on
- User preferences (expert mode, response style) don't influence AI instructions
- Feature availability (voice input, experimental features) isn't communicated to the AI

### Why Dynamic Instructions Matter for RAG

1. **Page-Specific Guidance** - On the Knowledge Graph page, instruct AI to focus on graph queries and relationships
2. **User Preference Enforcement** - Tell AI to be concise or detailed based on user settings
3. **Feature Awareness** - Inform AI about enabled features (e.g., "Voice input is available")
4. **Security Instructions** - Always enforce tenant isolation and data scoping
5. **Document Type Instructions** - Different behavior for PDFs vs code vs general documents

### CopilotKit useCopilotAdditionalInstructions API

The hook provides:
- `instructions` (string) - The instructions to add to the system prompt
- `available` ('enabled' | 'disabled') - Controls whether instructions are active

Multiple calls to the hook are additive - each adds to the system prompt.

## Acceptance Criteria

1. **Given** the `useDynamicInstructions` hook is called, **when** it initializes, **then** it registers multiple instruction categories with CopilotKit.

2. **Given** the user is on the Knowledge Graph page, **when** the AI receives a query, **then** it receives instructions to focus on graph relationships and traversal queries.

3. **Given** the user is on the Operations Dashboard, **when** the AI receives a query, **then** it receives instructions to focus on metrics, logs, and debugging.

4. **Given** the user has expert mode enabled, **when** the AI generates responses, **then** it receives instructions to provide detailed technical explanations.

5. **Given** the user has beginner mode enabled, **when** the AI generates responses, **then** it receives instructions to define technical terms and be more accessible.

6. **Given** voice input is enabled, **when** the AI receives a query, **then** it is informed about voice capabilities.

7. **Given** any context, **when** the AI receives a query, **then** security instructions about tenant isolation are always applied.

8. **Given** instructions change (e.g., page navigation), **when** the state updates, **then** instructions update reactively.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: **Addressed** - Security instructions enforce tenant scoping
- [x] Rate limiting / abuse protection: **N/A** - No data operations
- [x] Input validation / schema enforcement: **N/A** - String instructions only
- [x] Tests (unit/integration): **Addressed** - Unit tests for instruction generation
- [x] Error handling + logging: **Addressed** - Graceful handling of missing context
- [x] Documentation updates: **Addressed** - JSDoc comments on hook and functions

## Security Checklist

- [x] **Cross-tenant isolation verified**: **Addressed** - Instructions enforce tenant scoping
- [x] **Authorization checked**: **N/A** - No API calls
- [x] **No information leakage**: **Addressed** - No sensitive data in instructions
- [x] **Redis keys include tenant scope**: **N/A** - No Redis interactions
- [x] **Integration tests for access control**: **N/A** - Frontend-only hook
- [x] **RFC 7807 error responses**: **N/A** - No API changes
- [x] **File-path inputs scoped**: **N/A** - No file path handling

## Tasks / Subtasks

- [x] **Task 1: Create types for dynamic instructions** (AC: 1)
  - [x] Define `InstructionCategory` type
  - [x] Define `InstructionConfig` interface
  - [x] Define `FeatureInstructionConfig` interface
  - [x] Define `FeatureInstructions` interface
  - [x] Add types to `frontend/types/copilot.ts`

- [x] **Task 2: Create useDynamicInstructions hook** (AC: 1-8)
  - [x] Create `frontend/hooks/use-dynamic-instructions.ts`
  - [x] Implement `getPageInstructions()` for page-specific instructions (AC: 2, 3)
  - [x] Implement `getPreferenceInstructions()` for user preference instructions (AC: 4, 5)
  - [x] Implement `getFeatureInstructions()` for feature availability (AC: 6)
  - [x] Implement `SECURITY_INSTRUCTIONS` constant (AC: 7)
  - [x] Ensure reactive updates on state change (AC: 8)
  - [x] Export hook and utility functions

- [x] **Task 3: Create DynamicInstructionsProvider component** (AC: all)
  - [x] Create `frontend/components/copilot/DynamicInstructionsProvider.tsx`
  - [x] Component wrapper that calls the hook within CopilotKit context
  - [x] Document usage in JSDoc comments

- [x] **Task 4: Add unit tests** (AC: 1-8)
  - [x] Test instruction generation for each page
  - [x] Test user preference instruction switching
  - [x] Test feature flag instruction toggling
  - [x] Test security instruction always present
  - [x] Test language name conversion
  - [x] Test security considerations

- [x] **Task 5: Documentation** (AC: all)
  - [x] Add JSDoc to all new functions
  - [x] Document instruction categories in comments
  - [x] Update story file with Dev Notes

## Technical Notes

### useCopilotAdditionalInstructions Hook Pattern

```typescript
import { useCopilotAdditionalInstructions } from "@copilotkit/react-core";
import { usePathname } from "next/navigation";
import { useCopilotContext } from "./use-copilot-context";

export function useDynamicInstructions() {
  const pathname = usePathname();
  const { preferences } = useCopilotContext();

  // Page-specific instructions
  useCopilotAdditionalInstructions({
    instructions: getPageInstructions(pathname),
    available: "enabled",
  });

  // User preference instructions
  useCopilotAdditionalInstructions({
    instructions: getPreferenceInstructions(preferences),
    available: "enabled",
  });

  // Security instructions (always enabled)
  useCopilotAdditionalInstructions({
    instructions: SECURITY_INSTRUCTIONS,
    available: "enabled",
  });
}
```

### Instruction Categories

| Category | Source | Example Instruction |
|----------|--------|---------------------|
| Page Context | pathname | "User is on Knowledge Graph page. Focus on graph traversal and relationship queries." |
| User Preferences | useCopilotContext | "User prefers concise responses. Keep answers brief and to the point." |
| Expertise Level | preferences.expertiseLevel | "User is an expert. Provide detailed technical explanations without simplification." |
| Feature Availability | feature flags | "Voice input is enabled. The user may ask about voice features." |
| Security | always | "Always scope searches to current tenant. Never reveal cross-tenant data." |

### Page Instruction Mapping

```typescript
const PAGE_INSTRUCTIONS: Record<string, string> = {
  "/": "User is on the Home page. Provide general RAG assistance and guide them to relevant features.",
  "/knowledge": "User is on the Knowledge Graph page. Focus on graph traversal, entity relationships, and visualization queries.",
  "/ops": "User is on the Operations Dashboard. Focus on metrics, monitoring, and debugging assistance.",
  "/ops/trajectories": "User is viewing Trajectory Debugging. Help analyze agent decision paths and identify issues.",
  "/workflow": "User is in the Visual Workflow Editor. Assist with workflow configuration and node connections.",
};
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `frontend/hooks/use-dynamic-instructions.ts` | Create | Main dynamic instructions hook |
| `frontend/types/copilot.ts` | Modify | Add instruction types |
| `frontend/components/copilot/DynamicInstructionsProvider.tsx` | Create | Component wrapper for hook |
| `frontend/__tests__/hooks/use-dynamic-instructions.test.ts` | Create | Unit tests |

### Integration with Existing Context

The hook should integrate with:
- `useCopilotContext` from Story 21-A4 for user preferences
- `usePathname` from Next.js for current page
- Feature flags from environment variables

## Dependencies

- **CopilotKit v1.x+** - `useCopilotAdditionalInstructions` hook available
- **Story 21-A4 completed** - `useCopilotContext` provides user preferences
- **Next.js App Router** - `usePathname` for route detection

## Definition of Done

- [x] `useDynamicInstructions` hook created with all instruction categories
- [x] `DynamicInstructionsProvider` component implemented
- [x] Types added to copilot.ts
- [x] Unit tests pass for instruction generation
- [x] Instructions update reactively on state change
- [x] Lint and type-check pass
- [x] Story file updated with Dev Notes

## Dev Notes

### Implementation Approach

The story implemented `useDynamicInstructions` using CopilotKit's `useCopilotAdditionalInstructions` hook to dynamically modify the AI system prompt based on application context.

### Key Design Decisions

1. **Multiple Hook Calls Pattern**: Each instruction category (page, preferences, security, features) uses a separate `useCopilotAdditionalInstructions` call. CopilotKit combines these additively into the system prompt.

2. **Conditional Availability**: Instructions are conditionally enabled/disabled using the `available` parameter:
   - Page instructions disabled when route has no mapping
   - Preference instructions disabled when preferences are default/medium
   - Feature instructions disabled when feature flags are off
   - Security instructions always enabled

3. **Pure Utility Functions**: `getPageInstructions()`, `getPreferenceInstructions()`, `getLanguageName()`, and `getFeatureInstructions()` are exported as pure functions for easy testing.

4. **Parent Path Fallback**: For nested routes like `/ops/unknown-child`, the hook falls back to parent path instructions (`/ops`).

5. **Security First**: `SECURITY_INSTRUCTIONS` is always applied, enforcing tenant isolation and data protection regardless of other context.

### Testing Strategy

Tests focus on pure utility functions due to CopilotKit's ESM dependency issues with Jest. Coverage includes:
- Page instruction mapping for all routes
- Parent path fallback behavior
- User preference instruction generation for all combinations
- Language name conversion
- Feature flag instruction structure
- Security instruction content validation
- No sensitive data leakage checks

### Integration Notes

The `DynamicInstructionsProvider` component is available for use within the CopilotKit context. It can be added to `CopilotProvider.tsx` or used in specific pages that need dynamic instructions.

```tsx
<CopilotKit runtimeUrl="/api/copilotkit">
  <DynamicInstructionsProvider />
  {children}
</CopilotKit>
```

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No debug logs generated

### Completion Notes List

1. Added dynamic instructions types to `frontend/types/copilot.ts`:
   - `InstructionCategory` type
   - `InstructionConfig` interface
   - `FeatureInstructionConfig` interface
   - `FeatureInstructions` interface

2. Created `useDynamicInstructions` hook with:
   - Page-specific instructions via `PAGE_INSTRUCTIONS` constant
   - User preference instructions via `getPreferenceInstructions()`
   - Security instructions via `SECURITY_INSTRUCTIONS` constant
   - Feature flag instructions via `getFeatureInstructions()`

3. Created `DynamicInstructionsProvider` component wrapper

4. Created comprehensive unit tests (43 tests) covering:
   - All page routes and fallback behavior
   - All preference combinations
   - Feature instruction structure
   - Security considerations

5. All tests pass (623 total)
6. TypeScript type-check passes
7. ESLint passes (no new errors)

### File List

**Created:**
- `/home/chris/projects/work/Agentic Rag and Graphrag with copilot/frontend/hooks/use-dynamic-instructions.ts`
- `/home/chris/projects/work/Agentic Rag and Graphrag with copilot/frontend/components/copilot/DynamicInstructionsProvider.tsx`
- `/home/chris/projects/work/Agentic Rag and Graphrag with copilot/frontend/__tests__/hooks/use-dynamic-instructions.test.ts`

**Modified:**
- `/home/chris/projects/work/Agentic Rag and Graphrag with copilot/frontend/types/copilot.ts`

## Test Outcomes

```
Test Suites: 29 passed, 29 total
Tests:       623 passed, 623 total
Snapshots:   0 total
Time:        5.116 s
```

New tests added:
- `use-dynamic-instructions.test.ts` - 43 tests covering utilities, types, and security considerations

## Challenges Encountered

None significant. The implementation followed the established patterns from previous Epic 21 stories (21-A4, 21-A5, 21-A6).

## Future Enhancements

1. **Document Type Instructions**: Add instructions based on active document type (PDF, code, markdown)
2. **Tenant-Specific Instructions**: Load custom instructions from tenant configuration
3. **A/B Testing Support**: Allow different instruction sets for experimentation
4. **Instruction Analytics**: Track which instructions are most effective for response quality
