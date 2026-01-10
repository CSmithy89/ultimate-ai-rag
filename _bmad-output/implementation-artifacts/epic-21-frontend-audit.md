# Epic 21 Frontend Audit: CopilotKit Hook Migration

**Date:** 2026-01-07
**Status:** Completed
**Auditor:** Chris

## Overview

This audit identifies all usages of deprecated CopilotKit hooks (`useCopilotAction` with `handler` or `render`) and maps them to modern replacements (`useFrontendTool`, `useHumanInTheLoop`, `useRenderToolCall`).

## Findings

### 1. `frontend/hooks/use-copilot-actions.ts`

**Current State:**
- 5 instances of `useCopilotAction` with `handler` prop.
- Used for backend-like operations triggered by the frontend.

**Migration Plan:**
- **Goal:** Migrate all to `useFrontendTool`.
- **Reason:** `useCopilotAction` is deprecated. `useFrontendTool` provides better typing and schema validation via Zod.

| Action Name | Current Hook | New Hook | Parameters Schema (Zod) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `save_to_workspace` | `useCopilotAction` | `useFrontendTool` | `content_id` (str), `content_text` (str), `title` (str, opt), `query` (str, opt) | |
| `export_content` | `useCopilotAction` | `useFrontendTool` | `content_id` (str), `content_text` (str), `format` (enum: markdown, pdf, json), `title` (str, opt) | |
| `share_content` | `useCopilotAction` | `useFrontendTool` | `content_id` (str), `content_text` (str), `title` (str, opt) | |
| `bookmark_content` | `useCopilotAction` | `useFrontendTool` | `content_id` (str), `content_text` (str), `title` (str, opt) | |
| `suggest_follow_up` | `useCopilotAction` | `useFrontendTool` | `suggested_query` (str), `context` (str, opt) | Dispatches DOM event |

### 2. `frontend/hooks/use-source-validation.ts`

**Current State:**
- 1 instance of `useCopilotAction` with `render` prop.
- **Critical Hack Identified:** Uses `setTimeout(() => startValidation(sources), 0)` inside `render` to trigger state update. This is an anti-pattern.

**Migration Plan:**
- **Goal:** Migrate to `useHumanInTheLoop`.
- **Reason:** `useHumanInTheLoop` is designed exactly for this "ask user for input/confirmation" pattern. It provides a `respond` callback that eliminates the need for the state hack.

| Action Name | Current Hook | New Hook | Parameters Schema (Zod) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `validate_sources` | `useCopilotAction` | `useHumanInTheLoop` | `sources` (object[]), `query` (str, opt) | Pass `respond` to validation dialog. |

### 3. `frontend/hooks/use-generative-ui.tsx`

**Current State:**
- 3 instances of `useCopilotAction` with `render` prop.
- Used for Generative UI (rendering components in chat based on tool calls).

**Migration Plan:**
- **Goal:** Migrate to `useRenderToolCall` (or `CopilotKitProvider.renderToolCalls` prop).
- **Reason:** `useCopilotAction` with `render` is deprecated for GenUI. `useRenderToolCall` or the provider prop is the standard way to render custom UIs for tool calls.
- **Strategy:** Since we use `CopilotSidebar`, we can use the `renderToolCalls` prop on the provider OR use `useRenderToolCall` hook which automatically registers with the context. Hook is cleaner for modularity.

| Action Name | Current Hook | New Hook | Parameters Schema (Zod) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `show_sources` | `useCopilotAction` | `useRenderToolCall` | `sources` (object[]), `title` (str, opt) | Render `SourceCard` list |
| `show_answer` | `useCopilotAction` | `useRenderToolCall` | `answer` (str), `sources` (object[], opt), `title` (str, opt) | Render `AnswerPanel` |
| `show_knowledge_graph` | `useCopilotAction` | `useRenderToolCall` | `nodes` (object[]), `edges` (object[]), `title` (str, opt) | Render `GraphPreview` |

## Summary of Work

1.  **New File:** `frontend/hooks/use-frontend-tools.ts` - Implement migrated actions from `use-copilot-actions.ts`.
2.  **Refactor:** `frontend/hooks/use-source-validation.ts` - Switch to `useHumanInTheLoop`.
3.  **New File/Refactor:** `frontend/components/copilot/ToolCallRenderers.tsx` (or similar) - Implement GenUI renderers using `useRenderToolCall` and import them in the provider or a child component.
4.  **Cleanup:** Remove deprecated `useCopilotAction` calls.

## Risks & Mitigations

-   **Risk:** `useHumanInTheLoop` might have different lifecycle behavior than the `render` hack.
    -   *Mitigation:* Test approval/rejection flows thoroughly.
-   **Risk:** Zod schema mismatch with backend or previous loose typing.
    -   *Mitigation:* Ensure Zod schemas match the `parameters` previously defined in `useCopilotAction`.
