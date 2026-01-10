"use client";

import { useMemo, useState, useEffect } from "react";
import { useCopilotReadable } from "@copilotkit/react-core";
import { usePathname } from "next/navigation";
import { useQueryHistory } from "./use-query-history";
import type { PageContext, SessionContext, UserPreferences } from "@/types/copilot";

/**
 * localStorage key for user preferences persistence.
 */
const USER_PREFERENCES_KEY = "rag-copilot-user-preferences";

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
 * Load user preferences from localStorage.
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
    return {
      responseLength: parsed.responseLength ?? DEFAULT_PREFERENCES.responseLength,
      includeCitations: parsed.includeCitations ?? DEFAULT_PREFERENCES.includeCitations,
      language: parsed.language ?? DEFAULT_PREFERENCES.language,
      expertiseLevel: parsed.expertiseLevel ?? DEFAULT_PREFERENCES.expertiseLevel,
    };
  } catch {
    return DEFAULT_PREFERENCES;
  }
}

/**
 * Save user preferences to localStorage.
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
 * Uses a stable timestamp for the session.
 */
function getSessionStart(): string {
  if (typeof window === "undefined") {
    return new Date().toISOString();
  }

  const key = "rag-copilot-session-start";
  let sessionStart = sessionStorage.getItem(key);

  if (!sessionStart) {
    sessionStart = new Date().toISOString();
    try {
      sessionStorage.setItem(key, sessionStart);
    } catch {
      // sessionStorage might be disabled
    }
  }

  return sessionStart;
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
  const pathname = usePathname();
  const { queries: recentQueries, addQuery: addQueryToHistory } = useQueryHistory();
  const [preferences, setPreferences] = useState<UserPreferences>(DEFAULT_PREFERENCES);
  const [isPreferencesLoaded, setIsPreferencesLoaded] = useState(false);

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

  // Update preferences handler
  const updatePreferences = (updates: Partial<UserPreferences>) => {
    setPreferences((prev) => {
      const updated = { ...prev, ...updates };
      savePreferences(updated);
      return updated;
    });
  };

  return {
    pageContext,
    sessionContext,
    preferences,
    updatePreferences,
    addQueryToHistory,
  };
}

export default useCopilotContext;
