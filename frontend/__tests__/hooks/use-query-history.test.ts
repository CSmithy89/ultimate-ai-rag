/**
 * Tests for use-query-history hook
 * Story 21-A4: Implement useCopilotReadable for App Context
 */

import { renderHook, act } from "@testing-library/react";
import { useQueryHistory } from "@/hooks/use-query-history";

describe("useQueryHistory", () => {
  // Mock localStorage
  const localStorageMock = (() => {
    let store: Record<string, string> = {};
    return {
      getItem: jest.fn((key: string) => store[key] || null),
      setItem: jest.fn((key: string, value: string) => {
        store[key] = value;
      }),
      removeItem: jest.fn((key: string) => {
        delete store[key];
      }),
      clear: jest.fn(() => {
        store = {};
      }),
    };
  })();

  beforeEach(() => {
    // Reset mocks
    Object.defineProperty(window, "localStorage", {
      value: localStorageMock,
      writable: true,
    });
    localStorageMock.clear();
    jest.clearAllMocks();
  });

  describe("Initial State", () => {
    it("returns empty queries array initially", () => {
      const { result } = renderHook(() => useQueryHistory());

      expect(result.current.queries).toEqual([]);
    });

    it("loads queries from localStorage on mount", () => {
      const storedQueries = [
        { query: "test query", timestamp: "2024-01-15T10:00:00.000Z" },
        { query: "another query", timestamp: "2024-01-15T09:00:00.000Z" },
      ];
      localStorageMock.setItem(
        "rag-copilot-query-history",
        JSON.stringify(storedQueries)
      );

      const { result } = renderHook(() => useQueryHistory());

      // Wait for useEffect
      expect(result.current.queries.length).toBe(2);
      expect(result.current.queries[0].query).toBe("test query");
    });

    it("handles invalid JSON in localStorage gracefully", () => {
      localStorageMock.setItem("rag-copilot-query-history", "invalid json{");

      const { result } = renderHook(() => useQueryHistory());

      expect(result.current.queries).toEqual([]);
    });

    it("handles non-array data in localStorage", () => {
      localStorageMock.setItem(
        "rag-copilot-query-history",
        JSON.stringify({ not: "an array" })
      );

      const { result } = renderHook(() => useQueryHistory());

      expect(result.current.queries).toEqual([]);
    });

    it("filters out invalid query items from localStorage", () => {
      const mixedData = [
        { query: "valid query", timestamp: "2024-01-15T10:00:00.000Z" },
        { query: 123, timestamp: "invalid" }, // Invalid: query is not string
        null, // Invalid: not an object
        { notQuery: "missing query field", timestamp: "2024-01-15T10:00:00.000Z" },
        { query: "another valid", timestamp: "2024-01-15T09:00:00.000Z" },
      ];
      localStorageMock.setItem(
        "rag-copilot-query-history",
        JSON.stringify(mixedData)
      );

      const { result } = renderHook(() => useQueryHistory());

      expect(result.current.queries.length).toBe(2);
      expect(result.current.queries[0].query).toBe("valid query");
      expect(result.current.queries[1].query).toBe("another valid");
    });
  });

  describe("addQuery()", () => {
    it("adds a query to the front of the list", () => {
      const { result } = renderHook(() => useQueryHistory());

      act(() => {
        result.current.addQuery("first query");
      });

      expect(result.current.queries.length).toBe(1);
      expect(result.current.queries[0].query).toBe("first query");

      act(() => {
        result.current.addQuery("second query");
      });

      expect(result.current.queries.length).toBe(2);
      expect(result.current.queries[0].query).toBe("second query");
      expect(result.current.queries[1].query).toBe("first query");
    });

    it("trims whitespace from queries", () => {
      const { result } = renderHook(() => useQueryHistory());

      act(() => {
        result.current.addQuery("  trimmed query  ");
      });

      expect(result.current.queries[0].query).toBe("trimmed query");
    });

    it("ignores empty queries", () => {
      const { result } = renderHook(() => useQueryHistory());

      act(() => {
        result.current.addQuery("");
      });

      expect(result.current.queries.length).toBe(0);

      act(() => {
        result.current.addQuery("   ");
      });

      expect(result.current.queries.length).toBe(0);
    });

    it("deduplicates consecutive identical queries", () => {
      const { result } = renderHook(() => useQueryHistory());

      act(() => {
        result.current.addQuery("same query");
      });

      act(() => {
        result.current.addQuery("same query");
      });

      expect(result.current.queries.length).toBe(1);
      expect(result.current.queries[0].query).toBe("same query");
    });

    it("allows duplicate queries if not consecutive", () => {
      const { result } = renderHook(() => useQueryHistory());

      act(() => {
        result.current.addQuery("query A");
      });

      act(() => {
        result.current.addQuery("query B");
      });

      act(() => {
        result.current.addQuery("query A");
      });

      expect(result.current.queries.length).toBe(3);
      expect(result.current.queries.map((q) => q.query)).toEqual([
        "query A",
        "query B",
        "query A",
      ]);
    });

    it("limits history to 5 queries", () => {
      const { result } = renderHook(() => useQueryHistory());

      act(() => {
        for (let i = 1; i <= 7; i++) {
          result.current.addQuery(`query ${i}`);
        }
      });

      expect(result.current.queries.length).toBe(5);
      expect(result.current.queries[0].query).toBe("query 7");
      expect(result.current.queries[4].query).toBe("query 3");
    });

    it("saves to localStorage after adding query", () => {
      const { result } = renderHook(() => useQueryHistory());

      act(() => {
        result.current.addQuery("persisted query");
      });

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "rag-copilot-query-history",
        expect.any(String)
      );

      const savedData = JSON.parse(
        localStorageMock.setItem.mock.calls[0][1]
      );
      expect(savedData[0].query).toBe("persisted query");
    });

    it("includes timestamp in ISO format", () => {
      const mockDate = new Date("2024-01-15T12:00:00.000Z");
      jest.spyOn(global, "Date").mockImplementation(() => mockDate);

      const { result } = renderHook(() => useQueryHistory());

      act(() => {
        result.current.addQuery("timestamped query");
      });

      expect(result.current.queries[0].timestamp).toBe(
        "2024-01-15T12:00:00.000Z"
      );

      jest.restoreAllMocks();
    });
  });

  describe("clearHistory()", () => {
    it("removes all queries from state", () => {
      const { result } = renderHook(() => useQueryHistory());

      act(() => {
        result.current.addQuery("query 1");
        result.current.addQuery("query 2");
      });

      expect(result.current.queries.length).toBe(2);

      act(() => {
        result.current.clearHistory();
      });

      expect(result.current.queries.length).toBe(0);
    });

    it("removes data from localStorage", () => {
      const { result } = renderHook(() => useQueryHistory());

      act(() => {
        result.current.addQuery("to be cleared");
      });

      act(() => {
        result.current.clearHistory();
      });

      expect(localStorageMock.removeItem).toHaveBeenCalledWith(
        "rag-copilot-query-history"
      );
    });
  });

  describe("isLoaded", () => {
    it("is true after initial load", () => {
      const { result } = renderHook(() => useQueryHistory());

      // After useEffect runs, isLoaded should be true
      expect(result.current.isLoaded).toBe(true);
    });
  });
});
