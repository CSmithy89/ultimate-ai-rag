"use client";

import { useMemo } from "react";
import { useCopilotChatSuggestions } from "@copilotkit/react-core";
import { usePathname } from "next/navigation";

/**
 * Context for generating page-specific suggestions.
 * Story 21-A5: Implement useCopilotChatSuggestions for Smart Follow-ups
 */
export interface PageSuggestionContext {
  /** Human-readable page name */
  pageName: string;
  /** Page-specific instructions for suggestion generation */
  specificInstructions: string;
  /** Example suggestions for this page context */
  exampleSuggestions: string[];
}

/**
 * Map of route paths to suggestion contexts.
 * Provides page-specific guidance for AI suggestion generation.
 */
const PAGE_SUGGESTION_MAP: Record<string, PageSuggestionContext> = {
  "/": {
    pageName: "Home",
    specificInstructions: `The user is on the home page. They may want to:
- Start a new search or query
- Import documents into the knowledge base
- View their recent activity
- Navigate to other features`,
    exampleSuggestions: [
      "Search the knowledge base",
      "Import a document",
      "View recent queries",
      "Explore the graph",
    ],
  },
  "/knowledge": {
    pageName: "Knowledge Graph",
    specificInstructions: `The user is viewing the Knowledge Graph visualization. They may want to:
- Explore relationships between entities
- Find specific nodes or connections
- Understand the structure of their knowledge base
- Navigate to related concepts`,
    exampleSuggestions: [
      "Show related entities",
      "Find connections",
      "Explore node details",
      "Filter by entity type",
    ],
  },
  "/ops": {
    pageName: "Operations Dashboard",
    specificInstructions: `The user is on the Operations Dashboard. They may want to:
- Monitor system performance
- View agent trajectories
- Check cost metrics
- Debug issues`,
    exampleSuggestions: [
      "Show recent trajectories",
      "View system metrics",
      "Check agent performance",
      "Analyze query costs",
    ],
  },
  "/ops/trajectories": {
    pageName: "Trajectory Debugging",
    specificInstructions: `The user is debugging agent trajectories. They may want to:
- Filter trajectories by status or time
- Examine specific trajectory details
- Compare execution patterns
- Identify failed operations`,
    exampleSuggestions: [
      "Filter by status",
      "Show failed runs",
      "Compare trajectories",
      "View step details",
    ],
  },
  "/workflow": {
    pageName: "Visual Workflow Editor",
    specificInstructions: `The user is in the Visual Workflow Editor. They may want to:
- Add or modify workflow nodes
- Connect workflow steps
- Test workflow execution
- Save or export configurations`,
    exampleSuggestions: [
      "Add a new node",
      "Connect steps",
      "Test the workflow",
      "Save configuration",
    ],
  },
};

/**
 * Default suggestion context for unknown pages.
 */
const DEFAULT_SUGGESTION_CONTEXT: PageSuggestionContext = {
  pageName: "Application",
  specificInstructions: `The user is exploring the application. They may want to:
- Search for information
- Get help with features
- Navigate to different sections
- Perform common actions`,
  exampleSuggestions: [
    "Search for a topic",
    "How can I help?",
    "Show available features",
    "Get started",
  ],
};

/**
 * Get page-specific suggestion context for a given pathname.
 *
 * This function determines the appropriate suggestion context based on the
 * current route. It first checks for exact matches, then falls back to
 * parent path matching for nested routes.
 *
 * @param pathname - The current route pathname
 * @returns PageSuggestionContext for the route
 *
 * @example
 * ```typescript
 * const context = getPageSuggestionContext("/knowledge");
 * // Returns Knowledge Graph specific context
 *
 * const context = getPageSuggestionContext("/ops/trajectories");
 * // Returns Trajectory Debugging specific context
 *
 * const context = getPageSuggestionContext("/unknown");
 * // Returns default application context
 * ```
 */
export function getPageSuggestionContext(pathname: string): PageSuggestionContext {
  // Direct match
  if (PAGE_SUGGESTION_MAP[pathname]) {
    return PAGE_SUGGESTION_MAP[pathname];
  }

  // Try to match parent paths for nested routes
  const segments = pathname.split("/").filter(Boolean);
  while (segments.length > 0) {
    const parentPath = "/" + segments.join("/");
    if (PAGE_SUGGESTION_MAP[parentPath]) {
      return PAGE_SUGGESTION_MAP[parentPath];
    }
    segments.pop();
  }

  return DEFAULT_SUGGESTION_CONTEXT;
}

/**
 * Build the complete instructions string for suggestion generation.
 *
 * @param context - The page suggestion context
 * @returns Formatted instruction string for AI
 */
function buildInstructions(context: PageSuggestionContext): string {
  const exampleList = context.exampleSuggestions
    .map((s, i) => `${i + 1}. "${s}"`)
    .join("\n");

  return `You are generating helpful follow-up suggestions for a RAG (Retrieval-Augmented Generation) copilot assistant.

Current page: ${context.pageName}

${context.specificInstructions}

Generate 2-4 concise, actionable suggestions that:
1. Are relevant to the "${context.pageName}" page context
2. Are under 50 characters each
3. Start with an action verb (Show, Find, Explore, View, Search, etc.)
4. Help the user accomplish tasks or explore features
5. Are specific enough to be immediately useful

Example suggestions for this context:
${exampleList}

Generate suggestions that match this style but vary based on the conversation context.`;
}

/**
 * useChatSuggestions hook registers AI-powered suggestion generation with CopilotKit.
 *
 * Story 21-A5: Implement useCopilotChatSuggestions for Smart Follow-ups
 *
 * This hook provides contextual suggestions that appear as clickable chips below
 * the chat input in CopilotSidebar. Suggestions are automatically generated:
 * - When the chat is first opened
 * - After each message exchange completes
 *
 * The suggestions are context-aware, changing based on the current page:
 * - Home: General navigation and search suggestions
 * - Knowledge Graph: Entity exploration and relationship suggestions
 * - Operations: Monitoring and debugging suggestions
 * - Workflow: Editing and configuration suggestions
 *
 * @example
 * ```tsx
 * // In GenerativeUIRenderer or CopilotProvider
 * function MyComponent() {
 *   useChatSuggestions();
 *   return null;
 * }
 * ```
 *
 * @example
 * ```tsx
 * // The hook works with CopilotSidebar's default suggestions="auto" mode.
 * // No additional configuration is needed - just call the hook.
 * <CopilotSidebar>
 *   <GenerativeUIRenderer />  // Calls useChatSuggestions internally
 * </CopilotSidebar>
 * ```
 */
export function useChatSuggestions(): void {
  const pathname = usePathname();

  // Get page-specific context
  const suggestionContext = useMemo(
    () => getPageSuggestionContext(pathname),
    [pathname]
  );

  // Build instructions for suggestion generation
  const instructions = useMemo(
    () => buildInstructions(suggestionContext),
    [suggestionContext]
  );

  // Register suggestions with CopilotKit
  useCopilotChatSuggestions({
    instructions,
    minSuggestions: 2,
    maxSuggestions: 4,
  });
}

export default useChatSuggestions;
