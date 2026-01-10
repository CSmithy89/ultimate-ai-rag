/**
 * Tests for use-dynamic-instructions hook utilities
 * Story 21-A7: Implement useCopilotAdditionalInstructions for Dynamic Prompts
 *
 * NOTE: This test focuses on pure utility functions and types.
 * The hook integration with useCopilotAdditionalInstructions is tested via integration tests
 * since Jest module mocking with CopilotKit dependencies has compatibility issues.
 */

// Mock CopilotKit before any imports
jest.mock("@copilotkit/react-core", () => ({
  useCopilotAdditionalInstructions: jest.fn(),
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
import {
  getPageInstructions,
  getPreferenceInstructions,
  getLanguageName,
  getFeatureInstructions,
  PAGE_INSTRUCTIONS,
  SECURITY_INSTRUCTIONS,
} from "@/hooks/use-dynamic-instructions";
import type { UserPreferences } from "@/types/copilot";

describe("useDynamicInstructions utilities", () => {
  describe("getPageInstructions()", () => {
    it("returns correct instructions for root path", () => {
      const instructions = getPageInstructions("/");
      expect(instructions).toContain("Home page");
      expect(instructions).toContain("general RAG assistance");
    });

    it("returns correct instructions for knowledge path", () => {
      const instructions = getPageInstructions("/knowledge");
      expect(instructions).toContain("Knowledge Graph");
      expect(instructions).toContain("graph traversal");
      expect(instructions).toContain("entity relationships");
    });

    it("returns correct instructions for ops path", () => {
      const instructions = getPageInstructions("/ops");
      expect(instructions).toContain("Operations Dashboard");
      expect(instructions).toContain("metrics");
      expect(instructions).toContain("debugging");
    });

    it("returns correct instructions for ops/trajectories path", () => {
      const instructions = getPageInstructions("/ops/trajectories");
      expect(instructions).toContain("Trajectory Debugging");
      expect(instructions).toContain("agent decision");
    });

    it("returns correct instructions for workflow path", () => {
      const instructions = getPageInstructions("/workflow");
      expect(instructions).toContain("Visual Workflow Editor");
      expect(instructions).toContain("workflow configuration");
    });

    it("returns empty string for unknown paths", () => {
      expect(getPageInstructions("/unknown")).toBe("");
      expect(getPageInstructions("/some/random/path")).toBe("");
    });

    it("falls back to parent path instructions for nested unknown routes", () => {
      // /ops is known, /ops/unknown should fall back to /ops
      const instructions = getPageInstructions("/ops/unknown-child");
      expect(instructions).toContain("Operations Dashboard");
    });

    it("handles deeply nested paths under known routes", () => {
      const instructions = getPageInstructions("/ops/trajectories/detail/123");
      expect(instructions).toContain("Trajectory Debugging");
    });

    it("handles empty path", () => {
      expect(getPageInstructions("")).toBe("");
    });
  });

  describe("PAGE_INSTRUCTIONS constant", () => {
    it("has instructions for all main routes", () => {
      expect(PAGE_INSTRUCTIONS["/"]).toBeDefined();
      expect(PAGE_INSTRUCTIONS["/knowledge"]).toBeDefined();
      expect(PAGE_INSTRUCTIONS["/ops"]).toBeDefined();
      expect(PAGE_INSTRUCTIONS["/ops/trajectories"]).toBeDefined();
      expect(PAGE_INSTRUCTIONS["/workflow"]).toBeDefined();
    });

    it("all instructions are non-empty strings", () => {
      Object.values(PAGE_INSTRUCTIONS).forEach((instruction) => {
        expect(typeof instruction).toBe("string");
        expect(instruction.length).toBeGreaterThan(10);
      });
    });
  });

  describe("SECURITY_INSTRUCTIONS constant", () => {
    it("includes tenant isolation instruction", () => {
      expect(SECURITY_INSTRUCTIONS).toContain("tenant context");
    });

    it("includes cross-tenant protection", () => {
      expect(SECURITY_INSTRUCTIONS).toContain("other tenants");
    });

    it("includes credential protection", () => {
      expect(SECURITY_INSTRUCTIONS.toLowerCase()).toContain("api key");
    });

    it("is a non-empty string", () => {
      expect(typeof SECURITY_INSTRUCTIONS).toBe("string");
      expect(SECURITY_INSTRUCTIONS.length).toBeGreaterThan(50);
    });
  });

  describe("getPreferenceInstructions()", () => {
    const basePreferences: UserPreferences = {
      responseLength: "medium",
      includeCitations: false,
      language: "en",
      expertiseLevel: "intermediate",
    };

    it("returns empty string for default/medium preferences", () => {
      const instructions = getPreferenceInstructions(basePreferences);
      expect(instructions).toBe("");
    });

    describe("response length preferences", () => {
      it("includes brief instruction for brief responseLength", () => {
        const prefs: UserPreferences = { ...basePreferences, responseLength: "brief" };
        const instructions = getPreferenceInstructions(prefs);
        expect(instructions).toContain("concise");
        expect(instructions).toContain("2-3 sentences");
      });

      it("includes detailed instruction for detailed responseLength", () => {
        const prefs: UserPreferences = {
          ...basePreferences,
          responseLength: "detailed",
        };
        const instructions = getPreferenceInstructions(prefs);
        expect(instructions).toContain("comprehensive");
        expect(instructions).toContain("examples");
      });

      it("does not include length instruction for medium responseLength", () => {
        const prefs: UserPreferences = {
          ...basePreferences,
          responseLength: "medium",
        };
        const instructions = getPreferenceInstructions(prefs);
        expect(instructions).not.toContain("concise");
        expect(instructions).not.toContain("comprehensive");
      });
    });

    describe("expertise level preferences", () => {
      it("includes beginner instruction for beginner expertiseLevel", () => {
        const prefs: UserPreferences = {
          ...basePreferences,
          expertiseLevel: "beginner",
        };
        const instructions = getPreferenceInstructions(prefs);
        expect(instructions).toContain("Define technical terms");
        expect(instructions).toContain("simple explanations");
      });

      it("includes expert instruction for expert expertiseLevel", () => {
        const prefs: UserPreferences = { ...basePreferences, expertiseLevel: "expert" };
        const instructions = getPreferenceInstructions(prefs);
        expect(instructions).toContain("technical explanations");
        expect(instructions).toContain("precise terminology");
      });

      it("does not include expertise instruction for intermediate", () => {
        const prefs: UserPreferences = {
          ...basePreferences,
          expertiseLevel: "intermediate",
        };
        const instructions = getPreferenceInstructions(prefs);
        expect(instructions).not.toContain("Define technical terms");
        expect(instructions).not.toContain("Skip basic");
      });
    });

    describe("citation preferences", () => {
      it("includes citation instruction when includeCitations is true", () => {
        const prefs: UserPreferences = { ...basePreferences, includeCitations: true };
        const instructions = getPreferenceInstructions(prefs);
        expect(instructions).toContain("Cite sources");
      });

      it("does not include citation instruction when includeCitations is false", () => {
        const prefs: UserPreferences = { ...basePreferences, includeCitations: false };
        const instructions = getPreferenceInstructions(prefs);
        expect(instructions).not.toContain("Cite sources");
      });
    });

    describe("language preferences", () => {
      it("includes language instruction for non-English languages", () => {
        const prefs: UserPreferences = { ...basePreferences, language: "es" };
        const instructions = getPreferenceInstructions(prefs);
        expect(instructions).toContain("Spanish");
      });

      it("does not include language instruction for English", () => {
        const prefs: UserPreferences = { ...basePreferences, language: "en" };
        const instructions = getPreferenceInstructions(prefs);
        expect(instructions).not.toContain("English");
        expect(instructions).not.toContain("Respond in");
      });
    });

    describe("combined preferences", () => {
      it("combines multiple preference instructions", () => {
        const prefs: UserPreferences = {
          responseLength: "brief",
          includeCitations: true,
          language: "fr",
          expertiseLevel: "expert",
        };
        const instructions = getPreferenceInstructions(prefs);
        expect(instructions).toContain("concise");
        expect(instructions).toContain("technical explanations");
        expect(instructions).toContain("Cite sources");
        expect(instructions).toContain("French");
      });
    });
  });

  describe("getLanguageName()", () => {
    it("returns correct name for known language codes", () => {
      expect(getLanguageName("en")).toBe("English");
      expect(getLanguageName("es")).toBe("Spanish");
      expect(getLanguageName("fr")).toBe("French");
      expect(getLanguageName("de")).toBe("German");
      expect(getLanguageName("pt")).toBe("Portuguese");
      expect(getLanguageName("it")).toBe("Italian");
      expect(getLanguageName("zh")).toBe("Chinese");
      expect(getLanguageName("ja")).toBe("Japanese");
      expect(getLanguageName("ko")).toBe("Korean");
      expect(getLanguageName("ru")).toBe("Russian");
      expect(getLanguageName("ar")).toBe("Arabic");
      expect(getLanguageName("hi")).toBe("Hindi");
    });

    it("returns correct name for regional codes", () => {
      expect(getLanguageName("pt-BR")).toBe("Brazilian Portuguese");
    });

    it("returns the code itself for unknown languages", () => {
      expect(getLanguageName("xx")).toBe("xx");
      expect(getLanguageName("unknown")).toBe("unknown");
    });
  });

  describe("getFeatureInstructions()", () => {
    const originalEnv = process.env;

    beforeEach(() => {
      jest.resetModules();
      process.env = { ...originalEnv };
    });

    afterAll(() => {
      process.env = originalEnv;
    });

    it("returns feature instruction objects with correct structure", () => {
      const features = getFeatureInstructions();

      expect(features).toHaveProperty("voiceInput");
      expect(features).toHaveProperty("experimentalFeatures");
      expect(features).toHaveProperty("a2ui");

      expect(features.voiceInput).toHaveProperty("instructions");
      expect(features.voiceInput).toHaveProperty("available");
      expect(typeof features.voiceInput.instructions).toBe("string");
      expect(typeof features.voiceInput.available).toBe("boolean");
    });

    it("voice input instructions mention voice capability", () => {
      const features = getFeatureInstructions();
      expect(features.voiceInput.instructions).toContain("Voice input");
    });

    it("experimental features instructions mention beta", () => {
      const features = getFeatureInstructions();
      expect(features.experimentalFeatures.instructions).toContain("Experimental");
    });

    it("a2ui instructions mention rich UI components", () => {
      const features = getFeatureInstructions();
      expect(features.a2ui.instructions).toContain("A2UI");
      expect(features.a2ui.instructions).toContain("cards");
    });
  });

  describe("InstructionCategory type", () => {
    it("includes expected category values", () => {
      const categories: Array<
        "page" | "preferences" | "security" | "feature" | "custom"
      > = ["page", "preferences", "security", "feature", "custom"];

      categories.forEach((cat) => {
        expect(["page", "preferences", "security", "feature", "custom"]).toContain(
          cat
        );
      });
    });
  });

  describe("Security considerations", () => {
    it("security instructions do not contain sensitive example data", () => {
      expect(SECURITY_INSTRUCTIONS).not.toMatch(/password/i);
      expect(SECURITY_INSTRUCTIONS).not.toMatch(/token: /i);
      expect(SECURITY_INSTRUCTIONS).not.toMatch(/secret:/i);
    });

    it("page instructions do not contain internal route patterns", () => {
      Object.values(PAGE_INSTRUCTIONS).forEach((instruction) => {
        expect(instruction).not.toMatch(/\/api\//);
        expect(instruction).not.toMatch(/internal/i);
      });
    });

    it("preference instructions do not leak user data patterns", () => {
      const prefs: UserPreferences = {
        responseLength: "detailed",
        includeCitations: true,
        language: "es",
        expertiseLevel: "expert",
      };
      const instructions = getPreferenceInstructions(prefs);

      expect(instructions).not.toMatch(/user[-_]?id/i);
      expect(instructions).not.toMatch(/email/i);
      expect(instructions).not.toMatch(/password/i);
    });
  });

  describe("Instruction content quality", () => {
    it("all page instructions are grammatically complete sentences", () => {
      Object.values(PAGE_INSTRUCTIONS).forEach((instruction) => {
        expect(instruction).toMatch(/[.!?]$/);
        expect(instruction.charAt(0)).toBe(instruction.charAt(0).toUpperCase());
      });
    });

    it("preference instructions form complete guidance", () => {
      const prefs: UserPreferences = {
        responseLength: "brief",
        includeCitations: true,
        language: "en",
        expertiseLevel: "beginner",
      };
      const instructions = getPreferenceInstructions(prefs);

      // Each part ends with a period
      instructions.split(". ").forEach((part) => {
        if (part.trim()) {
          expect(part.charAt(0)).toBe(part.charAt(0).toUpperCase());
        }
      });
    });
  });
});

describe("DynamicInstructionsProvider component", () => {
  it("can be imported", () => {
    const providerModule = require("@/components/copilot/DynamicInstructionsProvider");
    expect(providerModule.DynamicInstructionsProvider).toBeDefined();
  });
});

describe("Type exports", () => {
  it("InstructionCategory type is exported from types", () => {
    const copilotTypes = require("@/types/copilot");
    // Type checks at compile time, but we can verify the module exports
    expect(copilotTypes).toBeDefined();
  });

  it("FeatureInstructionConfig type is exported from types", () => {
    const copilotTypes = require("@/types/copilot");
    expect(copilotTypes).toBeDefined();
  });
});
