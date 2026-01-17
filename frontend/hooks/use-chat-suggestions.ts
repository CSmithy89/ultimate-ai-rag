"use client";

import { useMemo } from "react";
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
      "Search the knowledge base for a topic",
      "Ingest a URL into the knowledge base",
      "Upload a PDF for ingestion",
      "Show knowledge graph stats",
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
      "Show related entities for a node",
      "Find shortest path between two entities",
      "List orphan entities",
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
      "Summarize cost trends for the last 7 days",
      "Check alerts and thresholds",
      "Analyze query costs by model",
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
      "Filter trajectories by status",
      "Show failed runs",
      "Compare two trajectories",
      "Explain a timeline step",
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
      "Add an ingest node",
      "Connect retrieval and rerank steps",
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
    "Search the knowledge base",
    "Explain available features",
    "How do I ingest documents?",
    "Get started with a sample query",
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
 * Suggestion item format compatible with CopilotKit's suggestions prop.
 * Matches the CopilotChatSuggestion type from @copilotkit/react-ui.
 */
export interface SuggestionItem {
  /** Display title for the suggestion chip */
  title: string;
  /** Message sent to chat when suggestion is clicked */
  message: string;
}

/**
 * useChatSuggestions hook provides static, context-aware suggestions for CopilotKit.
 *
 * Story 21-A5: Implement useCopilotChatSuggestions for Smart Follow-ups
 *
 * This hook provides contextual suggestions that appear as clickable chips below
 * the chat input in CopilotSidebar/CopilotChat. Suggestions are based on the
 * current page context:
 * - Home: General navigation and search suggestions
 * - Knowledge Graph: Entity exploration and relationship suggestions
 * - Operations: Monitoring and debugging suggestions
 * - Workflow: Editing and configuration suggestions
 *
 * **Important:** This hook returns static suggestions instead of using
 * `useCopilotChatSuggestions` because the AI-powered suggestions feature
 * triggers internal CopilotKit API calls that bypass our AG-UI compliant backend,
 * causing ZodError validation failures. Static suggestions avoid this issue.
 *
 * @returns Array of suggestion items for the current page context
 *
 * @example
 * ```tsx
 * // Use with CopilotSidebar
 * function ChatWrapper() {
 *   const suggestions = useChatSuggestions();
 *   return (
 *     <CopilotSidebar suggestions={suggestions}>
 *       <GenerativeUIRenderer />
 *     </CopilotSidebar>
 *   );
 * }
 * ```
 *
 * @example
 * ```tsx
 * // Use with CopilotChat (embedded)
 * function EmbeddedChatWrapper() {
 *   const suggestions = useChatSuggestions();
 *   return (
 *     <CopilotChat suggestions={suggestions}>
 *       <GenerativeUIRenderer />
 *     </CopilotChat>
 *   );
 * }
 * ```
 */
export function useChatSuggestions(): SuggestionItem[] {
  // Handle null pathname (Issue 2.1)
  const rawPathname = usePathname();
  const pathname = rawPathname ?? "/";

  // Get page-specific context and convert to suggestion items
  const suggestions = useMemo(() => {
    const context = getPageSuggestionContext(pathname);
    return context.exampleSuggestions.map((suggestion) => ({
      title: suggestion,
      message: suggestion,
    }));
  }, [pathname]);

  return suggestions;
}

export default useChatSuggestions;
