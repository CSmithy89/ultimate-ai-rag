"use client";

import { useState, useCallback, useEffect } from "react";
import type { QueryHistoryItem } from "@/types/copilot";

/**
 * localStorage key for query history persistence.
 * Prefixed to avoid conflicts with other localStorage usage.
 */
const QUERY_HISTORY_KEY = "rag-copilot-query-history";

/**
 * Maximum number of queries to retain in history.
 * Keeps context size manageable for AI.
 */
const MAX_QUERY_HISTORY = 5;

/**
 * Return type for the useQueryHistory hook.
 */
export interface UseQueryHistoryReturn {
  /** Array of recent query history items */
  queries: QueryHistoryItem[];
  /** Add a new query to history */
  addQuery: (query: string) => void;
  /** Clear all query history */
  clearHistory: () => void;
  /** Whether history is loaded from storage */
  isLoaded: boolean;
}

/**
 * Load query history from localStorage.
 * Returns empty array if not available or invalid.
 */
function loadQueryHistory(): QueryHistoryItem[] {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const stored = localStorage.getItem(QUERY_HISTORY_KEY);
    if (!stored) {
      return [];
    }

    const parsed = JSON.parse(stored);
    if (!Array.isArray(parsed)) {
      return [];
    }

    // Validate and filter valid items
    return parsed
      .filter(
        (item): item is QueryHistoryItem =>
          typeof item === "object" &&
          item !== null &&
          typeof item.query === "string" &&
          typeof item.timestamp === "string"
      )
      .slice(0, MAX_QUERY_HISTORY);
  } catch {
    // Invalid JSON or other error
    return [];
  }
}

/**
 * Save query history to localStorage.
 */
function saveQueryHistory(queries: QueryHistoryItem[]): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    localStorage.setItem(QUERY_HISTORY_KEY, JSON.stringify(queries));
  } catch {
    // localStorage might be full or disabled
    console.warn("Failed to save query history to localStorage");
  }
}

/**
 * useQueryHistory hook manages recent query history for AI context.
 *
 * Story 21-A4: Implement useCopilotReadable for App Context
 *
 * Features:
 * - Persists query history to localStorage
 * - Limits to last 5 queries to keep context manageable
 * - Deduplicates consecutive identical queries
 * - Provides add and clear functions
 *
 * @example
 * ```tsx
 * const { queries, addQuery, clearHistory } = useQueryHistory();
 *
 * // Add a query when user submits
 * const handleSubmit = (query: string) => {
 *   addQuery(query);
 *   // ... send to backend
 * };
 *
 * // Clear history
 * <button onClick={clearHistory}>Clear History</button>
 * ```
 */
export function useQueryHistory(): UseQueryHistoryReturn {
  const [queries, setQueries] = useState<QueryHistoryItem[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load history from localStorage on mount
  useEffect(() => {
    const loaded = loadQueryHistory();
    setQueries(loaded);
    setIsLoaded(true);
  }, []);

  // Persist to localStorage when queries change (Issue 2.9)
  // Moving saveQueryHistory out of setQueries callback prevents race conditions
  // when React batches multiple state updates
  useEffect(() => {
    if (isLoaded && queries.length > 0) {
      saveQueryHistory(queries);
    }
  }, [queries, isLoaded]);

  /**
   * Add a new query to history.
   * Deduplicates if same as most recent query.
   * Trims to MAX_QUERY_HISTORY items.
   * (Issue 2.9: Race Condition in useQueryHistory - Fixed by moving save to useEffect)
   */
  const addQuery = useCallback((query: string) => {
    if (!query.trim()) {
      return;
    }

    setQueries((prev) => {
      // Deduplicate consecutive identical queries
      if (prev.length > 0 && prev[0].query === query.trim()) {
        return prev;
      }

      const newItem: QueryHistoryItem = {
        query: query.trim(),
        timestamp: new Date().toISOString(),
      };

      // Add to front, limit to max
      // Note: saveQueryHistory is called in useEffect, not here (Issue 2.9)
      return [newItem, ...prev].slice(0, MAX_QUERY_HISTORY);
    });
  }, []);

  /**
   * Clear all query history.
   */
  const clearHistory = useCallback(() => {
    setQueries([]);
    if (typeof window !== "undefined") {
      try {
        localStorage.removeItem(QUERY_HISTORY_KEY);
      } catch {
        // Ignore errors
      }
    }
  }, []);

  return {
    queries,
    addQuery,
    clearHistory,
    isLoaded,
  };
}

export default useQueryHistory;
