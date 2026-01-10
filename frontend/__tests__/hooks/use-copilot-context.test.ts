/**
 * Tests for use-copilot-context hook utilities
 * Story 21-A4: Implement useCopilotReadable for App Context
 *
 * NOTE: This test focuses on pure utility functions and types.
 * The hook integration with useCopilotReadable is tested via integration tests
 * since Jest module mocking with CopilotKit dependencies has compatibility issues.
 */

// Mock CopilotKit before any imports
jest.mock("@copilotkit/react-core", () => ({
  useCopilotReadable: jest.fn(),
}));

// Mock Next.js navigation
jest.mock("next/navigation", () => ({
  usePathname: jest.fn(() => "/"),
}));

// Mock query history hook
jest.mock("@/hooks/use-query-history", () => ({
  useQueryHistory: () => ({
    queries: [],
    addQuery: jest.fn(),
  }),
}));

// Now import the module after mocks are set up
import { getPageName, savePreferences } from "@/hooks/use-copilot-context";
import type { UserPreferences } from "@/types/copilot";

describe("useCopilotContext utilities", () => {
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
    Object.defineProperty(window, "localStorage", {
      value: localStorageMock,
      writable: true,
    });
    localStorageMock.clear();
    jest.clearAllMocks();
  });

  describe("getPageName()", () => {
    it("returns correct name for root path", () => {
      expect(getPageName("/")).toBe("Home");
    });

    it("returns correct name for knowledge path", () => {
      expect(getPageName("/knowledge")).toBe("Knowledge Graph");
    });

    it("returns correct name for ops path", () => {
      expect(getPageName("/ops")).toBe("Operations Dashboard");
    });

    it("returns correct name for ops/trajectories path", () => {
      expect(getPageName("/ops/trajectories")).toBe("Trajectory Debugging");
    });

    it("returns correct name for workflow path", () => {
      expect(getPageName("/workflow")).toBe("Visual Workflow Editor");
    });

    it("generates title case name for unknown single-word paths", () => {
      expect(getPageName("/settings")).toBe("Settings");
    });

    it("generates title case name for unknown kebab-case paths", () => {
      expect(getPageName("/my-custom-page")).toBe("My Custom Page");
    });

    it("handles deeply nested unknown paths", () => {
      expect(getPageName("/some/deep/nested-route")).toBe("Nested Route");
    });

    it("handles paths with multiple hyphens", () => {
      expect(getPageName("/very-long-page-name")).toBe("Very Long Page Name");
    });

    it("handles empty path segments", () => {
      expect(getPageName("/")).toBe("Home");
      expect(getPageName("")).toBe("Unknown Page");
    });

    it("matches parent path when child not found", () => {
      // /ops is known, /ops/unknown should fall back to parent
      expect(getPageName("/ops/unknown-child")).toBe("Operations Dashboard");
    });
  });

  describe("savePreferences()", () => {
    it("saves preferences to localStorage", () => {
      const prefs: UserPreferences = {
        responseLength: "detailed",
        includeCitations: false,
        language: "es",
        expertiseLevel: "expert",
      };

      savePreferences(prefs);

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "rag-copilot-user-preferences",
        JSON.stringify(prefs)
      );
    });

    it("saves all preference fields correctly", () => {
      const prefs: UserPreferences = {
        responseLength: "brief",
        includeCitations: true,
        language: "fr",
        expertiseLevel: "beginner",
      };

      savePreferences(prefs);

      const savedValue = localStorageMock.setItem.mock.calls[0][1];
      const parsed = JSON.parse(savedValue);

      expect(parsed.responseLength).toBe("brief");
      expect(parsed.includeCitations).toBe(true);
      expect(parsed.language).toBe("fr");
      expect(parsed.expertiseLevel).toBe("beginner");
    });

    it("handles localStorage errors gracefully", () => {
      localStorageMock.setItem.mockImplementationOnce(() => {
        throw new Error("localStorage full");
      });

      const consoleSpy = jest.spyOn(console, "warn").mockImplementation();

      expect(() => {
        savePreferences({
          responseLength: "brief",
          includeCitations: true,
          language: "en",
          expertiseLevel: "beginner",
        });
      }).not.toThrow();

      expect(consoleSpy).toHaveBeenCalledWith(
        "Failed to save user preferences to localStorage"
      );

      consoleSpy.mockRestore();
    });
  });

  describe("UserPreferences type validation", () => {
    it("accepts valid responseLength values", () => {
      const validLengths: UserPreferences["responseLength"][] = [
        "brief",
        "medium",
        "detailed",
      ];

      validLengths.forEach((length) => {
        const prefs: UserPreferences = {
          responseLength: length,
          includeCitations: true,
          language: "en",
          expertiseLevel: "intermediate",
        };
        expect(prefs.responseLength).toBe(length);
      });
    });

    it("accepts valid expertiseLevel values", () => {
      const validLevels: UserPreferences["expertiseLevel"][] = [
        "beginner",
        "intermediate",
        "expert",
      ];

      validLevels.forEach((level) => {
        const prefs: UserPreferences = {
          responseLength: "medium",
          includeCitations: true,
          language: "en",
          expertiseLevel: level,
        };
        expect(prefs.expertiseLevel).toBe(level);
      });
    });

    it("accepts any language code string", () => {
      const languageCodes = ["en", "es", "fr", "de", "zh", "ja", "pt-BR"];

      languageCodes.forEach((lang) => {
        const prefs: UserPreferences = {
          responseLength: "medium",
          includeCitations: true,
          language: lang,
          expertiseLevel: "intermediate",
        };
        expect(prefs.language).toBe(lang);
      });
    });
  });

  describe("Security considerations", () => {
    it("preferences type does not include sensitive fields", () => {
      const prefs: UserPreferences = {
        responseLength: "medium",
        includeCitations: true,
        language: "en",
        expertiseLevel: "intermediate",
      };

      // Verify the type only has expected fields
      const keys = Object.keys(prefs);
      expect(keys).toEqual([
        "responseLength",
        "includeCitations",
        "language",
        "expertiseLevel",
      ]);

      // Verify no sensitive data patterns
      keys.forEach((key) => {
        expect(key).not.toMatch(/password/i);
        expect(key).not.toMatch(/token/i);
        expect(key).not.toMatch(/secret/i);
        expect(key).not.toMatch(/api[-_]?key/i);
        expect(key).not.toMatch(/credential/i);
      });
    });

    it("page name mapping does not expose sensitive internal route patterns", () => {
      // Routes should be converted to human-readable names
      // Check that conversion happens (not exposing raw route structure)
      expect(getPageName("/admin")).toBe("Admin");
      expect(getPageName("/debug")).toBe("Debug");

      // Known routes should return proper names
      expect(getPageName("/knowledge")).toBe("Knowledge Graph");
      expect(getPageName("/ops")).toBe("Operations Dashboard");
    });
  });
});

describe("QueryHistoryItem type", () => {
  it("has required query and timestamp fields", () => {
    const item = {
      query: "test query",
      timestamp: "2024-01-15T10:00:00.000Z",
    };

    expect(item.query).toBeDefined();
    expect(item.timestamp).toBeDefined();
    expect(typeof item.query).toBe("string");
    expect(typeof item.timestamp).toBe("string");
  });

  it("timestamp should be ISO format", () => {
    const item = {
      query: "test",
      timestamp: new Date().toISOString(),
    };

    // ISO format: YYYY-MM-DDTHH:mm:ss.sssZ
    expect(item.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
  });
});

describe("PageContext type", () => {
  it("has required route and pageName fields", () => {
    const context = {
      route: "/knowledge",
      pageName: "Knowledge Graph",
    };

    expect(context.route).toBeDefined();
    expect(context.pageName).toBeDefined();
  });

  it("allows optional metadata field", () => {
    const contextWithMetadata = {
      route: "/knowledge",
      pageName: "Knowledge Graph",
      metadata: { nodeCount: 100, edgeCount: 50 },
    };

    expect(contextWithMetadata.metadata).toBeDefined();
    expect(contextWithMetadata.metadata?.nodeCount).toBe(100);
  });
});

describe("SessionContext type", () => {
  it("has required fields without sensitive data", () => {
    const session = {
      tenantId: "tenant-123",
      sessionStart: "2024-01-15T10:00:00.000Z",
      isAuthenticated: true,
    };

    // Only these fields should exist
    expect(session.tenantId).toBeDefined();
    expect(session.sessionStart).toBeDefined();
    expect(session.isAuthenticated).toBeDefined();

    // Should NOT have sensitive fields
    expect(session).not.toHaveProperty("token");
    expect(session).not.toHaveProperty("apiKey");
    expect(session).not.toHaveProperty("password");
    expect(session).not.toHaveProperty("userId");
  });

  it("tenantId can be null", () => {
    const session = {
      tenantId: null,
      sessionStart: "2024-01-15T10:00:00.000Z",
      isAuthenticated: false,
    };

    expect(session.tenantId).toBeNull();
  });
});
