"use client";

import { useMemo, useState, useEffect, useCallback } from "react";
import { useCopilotReadable } from "@copilotkit/react-core";
import { usePathname } from "next/navigation";
import { useQueryHistory } from "./use-query-history";
import type { PageContext, SessionContext, UserPreferences } from "@/types/copilot";
import { UserPreferencesSchema } from "@/types/copilot";

/**
 * localStorage key for user preferences persistence.
 *
 * WARNING: Preferences are stored unencrypted in localStorage.
 * Do not add sensitive data to UserPreferences without implementing encryption.
 * (Issue 2.12: User Preferences Stored Unencrypted)
 */
const USER_PREFERENCES_KEY = "rag-copilot-user-preferences";

/**
 * sessionStorage key for session start time.
 * Note: This persists across page refreshes within the same tab.
 * A new tab or browser restart will create a new session.
 * (Issue 3.9: Session Start Never Updates)
 */
const SESSION_START_KEY = "rag-copilot-session-start";

/**
 * Default user preferences when none are stored.
 */
const DEFAULT_PREFERENCES: UserPreferences = {
  responseLength: "medium",
  includeCitations: true,
  language: "en",
  expertiseLevel: "intermediate",
};

/**
 * Map of route paths to human-readable page names.
 * Used to provide meaningful context to the AI.
 */
const PAGE_NAME_MAP: Record<string, string> = {
  "/": "Home",
  "/knowledge": "Knowledge Graph",
  "/ops": "Operations Dashboard",
  "/ops/trajectories": "Trajectory Debugging",
  "/workflow": "Visual Workflow Editor",
};

/**
 * Get human-readable page name from route path.
 *
 * @param pathname - The current route pathname
 * @returns Human-readable page name
 */
export function getPageName(pathname: string): string {
  // Direct match
  if (PAGE_NAME_MAP[pathname]) {
    return PAGE_NAME_MAP[pathname];
  }

  // Try to match parent paths for dynamic routes
  const segments = pathname.split("/").filter(Boolean);
  while (segments.length > 0) {
    const parentPath = "/" + segments.join("/");
    if (PAGE_NAME_MAP[parentPath]) {
      return PAGE_NAME_MAP[parentPath];
    }
    segments.pop();
  }

  // Fallback: generate name from last path segment
  const lastSegment = pathname.split("/").filter(Boolean).pop();
  if (lastSegment) {
    // Convert kebab-case to Title Case
    return lastSegment
      .split("-")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  return "Unknown Page";
}

/**
 * Load user preferences from localStorage with Zod validation.
 * (Issue 2.7: localStorage Parsing Lacks Zod Validation)
 */
function loadPreferences(): UserPreferences {
  if (typeof window === "undefined") {
    return DEFAULT_PREFERENCES;
  }

  try {
    const stored = localStorage.getItem(USER_PREFERENCES_KEY);
    if (!stored) {
      return DEFAULT_PREFERENCES;
    }

    const parsed = JSON.parse(stored);

    // Validate with Zod schema (Issue 2.7)
    const result = UserPreferencesSchema.safeParse(parsed);
    if (!result.success) {
      console.warn("Invalid preferences in localStorage:", result.error.flatten());
      return DEFAULT_PREFERENCES;
    }

    return result.data;
  } catch (error) {
    console.warn("Failed to load preferences from localStorage:", error);
    return DEFAULT_PREFERENCES;
  }
}

/**
 * Save user preferences to localStorage.
 *
 * WARNING: Preferences are stored unencrypted. Do not add sensitive data.
 * (Issue 2.12: User Preferences Stored Unencrypted)
 */
export function savePreferences(preferences: UserPreferences): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    localStorage.setItem(USER_PREFERENCES_KEY, JSON.stringify(preferences));
  } catch {
    console.warn("Failed to save user preferences to localStorage");
  }
}

/**
 * Get session start time.
 * Uses a stable timestamp for the session, stored in sessionStorage.
 *
 * Note: Session persists across page refreshes in the same tab but resets
 * when the tab is closed or a new tab is opened.
 * (Issue 2.8: Wrap both getItem and setItem in try/catch)
 * (Issue 3.9: Session Start Never Updates - documented behavior)
 */
function getSessionStart(): string {
  if (typeof window === "undefined") {
    return new Date().toISOString();
  }

  // Wrap all sessionStorage operations in try/catch (Issue 2.8)
  try {
    let sessionStart = sessionStorage.getItem(SESSION_START_KEY);

    if (!sessionStart) {
      sessionStart = new Date().toISOString();
      sessionStorage.setItem(SESSION_START_KEY, sessionStart);
    }

    return sessionStart;
  } catch {
    // sessionStorage might be disabled or throw in private browsing
    return new Date().toISOString();
  }
}

// ============================================
// AG-UI Enhancement: Extended Context Types
// ============================================

/**
 * Currently viewed document context.
 * AG-UI Enhancement: Provides document-specific context to the AI.
 */
export interface DocumentContext {
  /** Document ID */
  id: string;
  /** Document title */
  title: string;
  /** Document type (pdf, markdown, etc.) */
  type: string;
  /** Brief summary (first 500 chars) */
  summary?: string;
  /** Document metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Current search context including filters and results.
 * AG-UI Enhancement: Provides search state to the AI.
 */
export interface SearchContext {
  /** Active filters */
  activeFilters: {
    dateRange?: { start: string; end: string };
    documentTypes?: string[];
    collections?: string[];
    minSimilarity?: number;
  };
  /** Number of results found */
  resultCount: number;
  /** Top source titles (first 3) */
  topSources: string[];
  /** Current query if any */
  currentQuery?: string;
}

/**
 * Return type for useCopilotContext hook.
 */
export interface UseCopilotContextReturn {
  /** Current page context */
  pageContext: PageContext;
  /** Current session context */
  sessionContext: SessionContext;
  /** User preferences */
  preferences: UserPreferences;
  /** Update user preferences */
  updatePreferences: (updates: Partial<UserPreferences>) => void;
  /** Add a query to history */
  addQueryToHistory: (query: string) => void;
  /** Set document context (when viewing a document) */
  setDocumentContext: (doc: DocumentContext | null) => void;
  /** Set search context (when search results are available) */
  setSearchContext: (search: SearchContext | null) => void;
}

/**
 * useCopilotContext hook exposes application state to CopilotKit AI.
 *
 * Story 21-A4: Implement useCopilotReadable for App Context
 *
 * This hook registers multiple readable contexts that help the AI understand:
 * - What page the user is currently viewing
 * - Session information (tenant, authentication state)
 * - Recent query history for continuity
 * - User preferences for response formatting
 *
 * Security: Only non-sensitive data is exposed. Passwords, tokens, and
 * API keys are NEVER included in readable context.
 *
 * Code Review Fixes:
 * - Issue 2.1: Handle null pathname from usePathname()
 * - Issue 2.7: Zod validation for localStorage
 * - Issue 2.8: Wrap sessionStorage in try/catch
 * - Issue 2.12: Document unencrypted storage warning
 * - Issue 3.9: Document session lifetime behavior
 * - Issue 3.10: useCopilotReadable calls are intentional per CopilotKit design
 *
 * @example
 * ```tsx
 * // In a component within CopilotKit context
 * function MyComponent() {
 *   useCopilotContext();
 *   return <div>...</div>;
 * }
 * ```
 *
 * @example
 * ```tsx
 * // Access context values and update preferences
 * function SettingsPanel() {
 *   const { preferences, updatePreferences } = useCopilotContext();
 *   return (
 *     <select
 *       value={preferences.responseLength}
 *       onChange={(e) => updatePreferences({ responseLength: e.target.value })}
 *     >
 *       <option value="brief">Brief</option>
 *       <option value="medium">Medium</option>
 *       <option value="detailed">Detailed</option>
 *     </select>
 *   );
 * }
 * ```
 */
export function useCopilotContext(): UseCopilotContextReturn {
  // Handle null pathname (Issue 2.1)
  const rawPathname = usePathname();
  const pathname = rawPathname ?? "/";

  const { queries: recentQueries, addQuery: addQueryToHistory } = useQueryHistory();
  const [preferences, setPreferences] = useState<UserPreferences>(DEFAULT_PREFERENCES);
  const [isPreferencesLoaded, setIsPreferencesLoaded] = useState(false);

  // AG-UI Enhancement: Document and search context state
  const [documentContext, setDocumentContext] = useState<DocumentContext | null>(null);
  const [searchContext, setSearchContext] = useState<SearchContext | null>(null);

  // Load preferences on mount
  useEffect(() => {
    const loaded = loadPreferences();
    setPreferences(loaded);
    setIsPreferencesLoaded(true);
  }, []);

  // Derive page context from pathname
  const pageContext = useMemo<PageContext>(
    () => ({
      route: pathname,
      pageName: getPageName(pathname),
    }),
    [pathname]
  );

  // Get tenant ID from environment (non-sensitive)
  const tenantId = typeof window !== "undefined"
    ? process.env.NEXT_PUBLIC_TENANT_ID ?? null
    : null;

  // Derive session context
  const sessionContext = useMemo<SessionContext>(
    () => ({
      tenantId,
      sessionStart: getSessionStart(),
      isAuthenticated: false, // Can be updated when auth is implemented
    }),
    [tenantId]
  );

  // Register page context with CopilotKit
  // Note: useCopilotReadable is designed to be called every render (Issue 3.10)
  useCopilotReadable({
    description: "Current page the user is viewing in the RAG application. Use this to understand what the user is looking at and tailor responses accordingly.",
    value: pageContext,
  });

  // Register session context with CopilotKit
  useCopilotReadable({
    description: "Current session information including tenant context. Use the tenant ID when referencing data scoping.",
    value: sessionContext,
  });

  // Register query history with CopilotKit
  // Only expose if loaded and has items
  useCopilotReadable({
    description: "Recent queries the user has made in this session. Use this for context continuity and to reference previous questions.",
    value: recentQueries.length > 0 ? recentQueries : null,
    available: recentQueries.length > 0 ? "enabled" : "disabled",
  });

  // Register user preferences with CopilotKit
  // Only expose when loaded from storage
  useCopilotReadable({
    description: "User preferences for AI response formatting. Adjust response length, citation inclusion, and complexity based on these preferences.",
    value: isPreferencesLoaded ? preferences : null,
    available: isPreferencesLoaded ? "enabled" : "disabled",
  });

  // AG-UI Enhancement: Register document context when viewing specific documents
  useCopilotReadable({
    description: "Currently viewed document. Use this to understand what specific document the user is examining and provide document-specific answers.",
    value: documentContext,
    available: documentContext ? "enabled" : "disabled",
  });

  // AG-UI Enhancement: Register search context with filters and results summary
  useCopilotReadable({
    description: "Current search filters and results summary. Use this to understand what the user has searched for and what filters they've applied.",
    value: searchContext,
    available: searchContext ? "enabled" : "disabled",
  });

  // Update preferences handler with useCallback for stable reference
  const updatePreferences = useCallback((updates: Partial<UserPreferences>) => {
    setPreferences((prev) => {
      const updated = { ...prev, ...updates };
      savePreferences(updated);
      return updated;
    });
  }, []);

  // AG-UI Enhancement: Callbacks for setting document and search context
  const setDocumentContextCallback = useCallback((doc: DocumentContext | null) => {
    setDocumentContext(doc);
  }, []);

  const setSearchContextCallback = useCallback((search: SearchContext | null) => {
    setSearchContext(search);
  }, []);

  return {
    pageContext,
    sessionContext,
    preferences,
    updatePreferences,
    addQueryToHistory,
    setDocumentContext: setDocumentContextCallback,
    setSearchContext: setSearchContextCallback,
  };
}

export default useCopilotContext;
