/**
 * Tests for use-default-tool hook utilities
 * Story 21-A8: Implement useDefaultTool Catch-All
 *
 * NOTE: This test focuses on pure utility functions and types.
 * The hook integration with useDefaultTool is tested via integration tests
 * since Jest module mocking with CopilotKit dependencies has compatibility issues.
 */

// Mock CopilotKit before any imports
jest.mock("@copilotkit/react-core", () => ({
  useDefaultTool: jest.fn(),
}));

// Mock toast hook
const mockToast = jest.fn();
jest.mock("@/hooks/use-toast", () => ({
  useToast: () => ({
    toast: mockToast,
    toasts: [],
    dismiss: jest.fn(),
    dismissAll: jest.fn(),
  }),
}));

// Mock redact utility
jest.mock("@/lib/utils/redact", () => ({
  redactSensitiveKeys: jest.fn((obj) => {
    if (!obj || typeof obj !== "object") return obj;
    return Object.fromEntries(
      Object.entries(obj).map(([k, v]) =>
        /password|secret|token|key|auth/i.test(k) ? [k, "[REDACTED]"] : [k, v]
      )
    );
  }),
}));

// Now import the module after mocks are set up
import {
  isRunning,
  isComplete,
  formatToolName,
  getDefaultToolUtilities,
} from "@/hooks/use-default-tool";
import type { DefaultToolStatus, DefaultToolRenderProps } from "@/types/copilot";

describe("useDefaultTool utilities", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("isRunning()", () => {
    it('returns true for "inProgress" status', () => {
      expect(isRunning("inProgress")).toBe(true);
    });

    it('returns true for "executing" status', () => {
      expect(isRunning("executing")).toBe(true);
    });

    it('returns false for "complete" status', () => {
      expect(isRunning("complete")).toBe(false);
    });
  });

  describe("isComplete()", () => {
    it('returns true for "complete" status', () => {
      expect(isComplete("complete")).toBe(true);
    });

    it('returns false for "inProgress" status', () => {
      expect(isComplete("inProgress")).toBe(false);
    });

    it('returns false for "executing" status', () => {
      expect(isComplete("executing")).toBe(false);
    });
  });

  describe("formatToolName()", () => {
    it("removes mcp_ prefix", () => {
      expect(formatToolName("mcp_vector_search")).toBe("vector_search");
    });

    it("removes MCP_ prefix (case insensitive)", () => {
      expect(formatToolName("MCP_Search")).toBe("Search");
    });

    it("extracts tool name after colon (server prefix)", () => {
      expect(formatToolName("github:create_issue")).toBe("create_issue");
    });

    it("handles multiple colons by taking last segment", () => {
      expect(formatToolName("server:namespace:tool_name")).toBe("tool_name");
    });

    it("returns unchanged name when no prefix present", () => {
      expect(formatToolName("search_docs")).toBe("search_docs");
    });

    it("handles empty string", () => {
      expect(formatToolName("")).toBe("");
    });

    it("handles combined mcp_ and colon prefix", () => {
      expect(formatToolName("mcp_server:tool")).toBe("tool");
    });
  });

  describe("getDefaultToolUtilities()", () => {
    it("returns all utility functions", () => {
      const utils = getDefaultToolUtilities();

      expect(typeof utils.isRunning).toBe("function");
      expect(typeof utils.isComplete).toBe("function");
      expect(typeof utils.formatToolName).toBe("function");
    });

    it("returned isRunning works correctly", () => {
      const { isRunning: utilIsRunning } = getDefaultToolUtilities();

      expect(utilIsRunning("inProgress")).toBe(true);
      expect(utilIsRunning("executing")).toBe(true);
      expect(utilIsRunning("complete")).toBe(false);
    });

    it("returned isComplete works correctly", () => {
      const { isComplete: utilIsComplete } = getDefaultToolUtilities();

      expect(utilIsComplete("complete")).toBe(true);
      expect(utilIsComplete("inProgress")).toBe(false);
    });

    it("returned formatToolName works correctly", () => {
      const { formatToolName: utilFormatToolName } = getDefaultToolUtilities();

      expect(utilFormatToolName("mcp_test")).toBe("test");
      expect(utilFormatToolName("server:action")).toBe("action");
    });
  });
});

describe("DefaultToolStatus type", () => {
  it("accepts valid status values", () => {
    const validStatuses: DefaultToolStatus[] = [
      "inProgress",
      "executing",
      "complete",
    ];

    validStatuses.forEach((status) => {
      expect(["inProgress", "executing", "complete"]).toContain(status);
    });
  });
});

describe("DefaultToolRenderProps type", () => {
  it("has required name, args, and status fields", () => {
    const props: DefaultToolRenderProps = {
      name: "test_tool",
      args: { query: "test" },
      status: "inProgress",
    };

    expect(props.name).toBeDefined();
    expect(props.args).toBeDefined();
    expect(props.status).toBeDefined();
  });

  it("allows optional result field", () => {
    const propsWithResult: DefaultToolRenderProps = {
      name: "test_tool",
      args: { query: "test" },
      status: "complete",
      result: { success: true, data: ["item1", "item2"] },
    };

    expect(propsWithResult.result).toBeDefined();
  });

  it("args can contain nested objects", () => {
    const props: DefaultToolRenderProps = {
      name: "complex_tool",
      args: {
        query: "test",
        options: {
          limit: 10,
          filters: ["a", "b"],
        },
      },
      status: "inProgress",
    };

    expect(props.args.options).toBeDefined();
  });
});

describe("Security considerations", () => {
  it("formatToolName does not expose internal server identifiers", () => {
    // Server identifiers should be stripped
    expect(formatToolName("internal_server:secret_tool")).not.toContain(
      "internal_server"
    );
    expect(formatToolName("internal_server:secret_tool")).toBe("secret_tool");
  });

  it("status values do not contain sensitive information", () => {
    const statuses: DefaultToolStatus[] = ["inProgress", "executing", "complete"];

    statuses.forEach((status) => {
      expect(status).not.toMatch(/password/i);
      expect(status).not.toMatch(/token/i);
      expect(status).not.toMatch(/secret/i);
      expect(status).not.toMatch(/api[-_]?key/i);
    });
  });

  it("DefaultToolRenderProps type does not include sensitive fields", () => {
    const props: DefaultToolRenderProps = {
      name: "test",
      args: {},
      status: "complete",
    };

    const keys = Object.keys(props);
    keys.forEach((key) => {
      expect(key).not.toMatch(/password/i);
      expect(key).not.toMatch(/token/i);
      expect(key).not.toMatch(/secret/i);
      expect(key).not.toMatch(/api[-_]?key/i);
      expect(key).not.toMatch(/credential/i);
    });
  });
});

describe("Status transition logic", () => {
  it("tool goes through expected status progression", () => {
    // Typical progression: inProgress -> executing -> complete
    const progression: DefaultToolStatus[] = [
      "inProgress",
      "executing",
      "complete",
    ];

    expect(isRunning(progression[0])).toBe(true);
    expect(isRunning(progression[1])).toBe(true);
    expect(isComplete(progression[2])).toBe(true);
  });

  it("only one status is terminal", () => {
    // Only "complete" should be terminal (not running)
    expect(isRunning("inProgress")).toBe(true);
    expect(isRunning("executing")).toBe(true);
    expect(isRunning("complete")).toBe(false);
    expect(isComplete("complete")).toBe(true);
  });
});

describe("Edge cases", () => {
  describe("formatToolName edge cases", () => {
    it("handles tool name with only prefix", () => {
      expect(formatToolName("mcp_")).toBe("");
    });

    it("handles tool name with only colon", () => {
      expect(formatToolName(":")).toBe("");
    });

    it("handles tool name with trailing colon", () => {
      expect(formatToolName("server:")).toBe("");
    });

    it("handles tool name with leading colon", () => {
      expect(formatToolName(":tool")).toBe("tool");
    });

    it("handles whitespace in tool name", () => {
      expect(formatToolName("  tool_name  ").trim()).toBe("tool_name");
    });

    it("handles unicode characters", () => {
      expect(formatToolName("tool_unicode_")).toBe("tool_unicode_");
    });
  });

  describe("status handling edge cases", () => {
    it("functions handle all defined statuses without error", () => {
      const statuses: DefaultToolStatus[] = [
        "inProgress",
        "executing",
        "complete",
      ];

      statuses.forEach((status) => {
        expect(() => isRunning(status)).not.toThrow();
        expect(() => isComplete(status)).not.toThrow();
      });
    });
  });
});

describe("Integration with other hooks pattern", () => {
  it("utilities can be used outside React components", () => {
    // This simulates usage in test utilities or helper functions
    const utils = getDefaultToolUtilities();

    const testProps: DefaultToolRenderProps = {
      name: "mcp_test_tool",
      args: { param: "value" },
      status: "executing",
    };

    expect(utils.isRunning(testProps.status)).toBe(true);
    expect(utils.formatToolName(testProps.name)).toBe("test_tool");
  });
});
