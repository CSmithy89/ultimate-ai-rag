# Story 22-TD6: Prepare for CopilotKit 2.x Migration

Status: blocked
Blocked-Reason: CopilotKit 2.x not yet released as stable (current: 1.50.1)

Epic: 22 - Advanced Protocol Integration
Priority: P1 - MEDIUM (When CopilotKit 2.x stable)
Story Points: 3
Owner: Frontend
Origin: Epic 21 Retrospective (TD-21-1)

## Story

As a **frontend developer**,
I want **to migrate from the dual-format schema approach to native CopilotKit 2.x Zod schemas**,
So that **the codebase is simpler and leverages the latest CopilotKit features**.

## Background

During Epic 21 Story 21-A1, we discovered CopilotKit 1.x doesn't support direct Zod schemas in `useFrontendTool` - it requires `Parameter[]` arrays. We implemented a dual-format approach:

1. Zod schemas for TypeScript type inference
2. Parameter[] arrays for CopilotKit runtime

CopilotKit 2.x (currently `2.0.0-next.1`) supports Zod schemas directly. When 2.x becomes stable, we should migrate.

## Pre-requisites

- [ ] CopilotKit 2.x released as stable (not -next)
- [ ] CopilotKit 2.x breaking changes documented and reviewed
- [ ] Migration guide from CopilotKit team reviewed

## Acceptance Criteria

1. **Given** CopilotKit 2.x is stable, **when** upgrading, **then** package.json reflects new version.

2. **Given** dual-format schemas exist, **when** migrated, **then** Parameter[] arrays are removed and only Zod schemas remain.

3. **Given** useFrontendTool uses Parameter[], **when** migrated, **then** it uses Zod schemas directly.

4. **Given** tests mock useFrontendTool, **when** migrated, **then** mocks are updated for new signature.

5. **Given** migration is complete, **when** reviewing code, **then** no `Parameter[]` definitions remain in `tools.ts`.

## Migration Scope

### Files to Migrate

| File | Current State | After Migration |
|------|---------------|-----------------|
| `frontend/lib/schemas/tools.ts` | Zod schemas + Parameter[] | Zod schemas only |
| `frontend/hooks/use-copilot-actions.ts` | Uses Parameter[] | Uses Zod schemas |
| `frontend/hooks/use-source-validation.ts` | Uses Parameter[] | Uses Zod schemas |
| `frontend/__tests__/hooks/*.test.ts` | Mocks Parameter[] | Mocks Zod |

### Code Changes

**Before (CopilotKit 1.x):**
```typescript
// tools.ts
export const SaveToWorkspaceSchema = z.object({ ... });
export const saveToWorkspaceToolParams: Parameter[] = [ ... ];

// use-copilot-actions.ts
useFrontendTool({
  name: "save_to_workspace",
  parameters: saveToWorkspaceToolParams,  // Parameter[]
  handler: async (params) => { ... }
});
```

**After (CopilotKit 2.x):**
```typescript
// tools.ts
export const SaveToWorkspaceSchema = z.object({ ... });
// Remove: export const saveToWorkspaceToolParams

// use-copilot-actions.ts
useFrontendTool({
  name: "save_to_workspace",
  parameters: SaveToWorkspaceSchema,  // Zod schema directly
  handler: async (params) => { ... }  // params already typed!
});
```

## Tasks

- [ ] **Task 1: Upgrade CopilotKit**
  - [ ] Update package.json to CopilotKit 2.x
  - [ ] Run pnpm install
  - [ ] Fix any immediate breaking changes

- [ ] **Task 2: Remove Parameter[] Arrays**
  - [ ] Delete all `*ToolParams` exports from tools.ts
  - [ ] Update imports in consuming files

- [ ] **Task 3: Update useFrontendTool Calls**
  - [ ] Change `parameters: *ToolParams` to `parameters: *Schema`
  - [ ] Remove `as unknown as` type assertions (should be automatic now)

- [ ] **Task 4: Update Tests**
  - [ ] Update mocks for new useFrontendTool signature
  - [ ] Remove tests for Parameter[] arrays
  - [ ] Verify all tests pass

- [ ] **Task 5: Cleanup**
  - [ ] Remove any now-unused imports
  - [ ] Update JSDoc comments
  - [ ] Run lint and type-check

## Risks

1. **Breaking Changes:** CopilotKit 2.x may have other breaking changes beyond schema format
2. **Plugin Compatibility:** Third-party CopilotKit plugins may not support 2.x yet
3. **Documentation Lag:** CopilotKit docs may not be fully updated for 2.x

## Definition of Done

- [ ] CopilotKit upgraded to 2.x stable
- [ ] All Parameter[] arrays removed
- [ ] All useFrontendTool calls use Zod schemas directly
- [ ] All tests pass
- [ ] No TypeScript errors
- [ ] Code review approved

## Files to Modify

1. `frontend/package.json` - Upgrade CopilotKit
2. `frontend/lib/schemas/tools.ts` - Remove Parameter[] exports
3. `frontend/hooks/use-copilot-actions.ts` - Use Zod schemas
4. `frontend/hooks/use-source-validation.ts` - Use Zod schemas
5. `frontend/__tests__/hooks/*.test.ts` - Update mocks
