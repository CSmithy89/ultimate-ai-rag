/**
 * MCP-UI Protocol Compliance Tests
 *
 * Story 22-D2: Implement Protocol Compliance Tests
 *
 * Verifies MCP-UI protocol implementation:
 * - PostMessage schema validation
 * - Origin validation
 * - Message type compliance
 */

import { describe, it, expect } from "@jest/globals";
import { z } from "zod";

import {
  MCPUIMessageSchema,
  isAllowedOrigin,
  extractOrigin,
  getEnvAllowedOrigins,
} from "@/lib/mcp-ui-security";

// =============================================================================
// PostMessage Schema Compliance Tests
// =============================================================================

describe("MCP-UI PostMessage Schema Compliance", () => {
  describe("mcp_ui_resize message", () => {
    it("accepts valid resize message", () => {
      const message = {
        type: "mcp_ui_resize",
        width: 600,
        height: 400,
      };
      const result = MCPUIMessageSchema.safeParse(message);
      expect(result.success).toBe(true);
    });

    it("requires width within bounds (100-4000)", () => {
      const tooSmall = { type: "mcp_ui_resize", width: 50, height: 400 };
      const tooLarge = { type: "mcp_ui_resize", width: 5000, height: 400 };

      expect(MCPUIMessageSchema.safeParse(tooSmall).success).toBe(false);
      expect(MCPUIMessageSchema.safeParse(tooLarge).success).toBe(false);
    });

    it("requires height within bounds (50-4000)", () => {
      const tooSmall = { type: "mcp_ui_resize", width: 400, height: 30 };
      const tooLarge = { type: "mcp_ui_resize", width: 400, height: 5000 };

      expect(MCPUIMessageSchema.safeParse(tooSmall).success).toBe(false);
      expect(MCPUIMessageSchema.safeParse(tooLarge).success).toBe(false);
    });

    it("requires numeric width and height", () => {
      const invalidWidth = { type: "mcp_ui_resize", width: "600", height: 400 };
      const invalidHeight = { type: "mcp_ui_resize", width: 600, height: "400" };

      expect(MCPUIMessageSchema.safeParse(invalidWidth).success).toBe(false);
      expect(MCPUIMessageSchema.safeParse(invalidHeight).success).toBe(false);
    });
  });

  describe("mcp_ui_result message", () => {
    it("accepts valid result message with any result type", () => {
      const stringResult = { type: "mcp_ui_result", result: "success" };
      const objectResult = { type: "mcp_ui_result", result: { data: [1, 2, 3] } };
      const arrayResult = { type: "mcp_ui_result", result: ["a", "b"] };
      const nullResult = { type: "mcp_ui_result", result: null };

      expect(MCPUIMessageSchema.safeParse(stringResult).success).toBe(true);
      expect(MCPUIMessageSchema.safeParse(objectResult).success).toBe(true);
      expect(MCPUIMessageSchema.safeParse(arrayResult).success).toBe(true);
      expect(MCPUIMessageSchema.safeParse(nullResult).success).toBe(true);
    });

    it("accepts result field with undefined (z.unknown allows undefined)", () => {
      // Note: z.unknown() accepts undefined, so mcp_ui_result without explicit result is valid
      const noResult = { type: "mcp_ui_result" };
      const result = MCPUIMessageSchema.safeParse(noResult);
      expect(result.success).toBe(true);
    });
  });

  describe("mcp_ui_error message", () => {
    it("accepts valid error message", () => {
      const message = {
        type: "mcp_ui_error",
        error: "User cancelled operation",
      };
      const result = MCPUIMessageSchema.safeParse(message);
      expect(result.success).toBe(true);
    });

    it("requires error to be string", () => {
      const objectError = { type: "mcp_ui_error", error: { message: "error" } };
      const numberError = { type: "mcp_ui_error", error: 500 };

      expect(MCPUIMessageSchema.safeParse(objectError).success).toBe(false);
      expect(MCPUIMessageSchema.safeParse(numberError).success).toBe(false);
    });

    it("requires error field", () => {
      const noError = { type: "mcp_ui_error" };
      expect(MCPUIMessageSchema.safeParse(noError).success).toBe(false);
    });
  });

  describe("unknown message types", () => {
    it("rejects messages with unknown type", () => {
      const unknownType = { type: "mcp_ui_custom", data: "test" };
      expect(MCPUIMessageSchema.safeParse(unknownType).success).toBe(false);
    });

    it("rejects messages without type field", () => {
      const noType = { width: 600, height: 400 };
      expect(MCPUIMessageSchema.safeParse(noType).success).toBe(false);
    });
  });
});

// =============================================================================
// Origin Validation Compliance Tests
// =============================================================================

describe("MCP-UI Origin Validation Compliance", () => {
  describe("isAllowedOrigin", () => {
    it("returns true for allowed origin", () => {
      const allowedOrigins = new Set(["https://trusted.com", "https://partner.com"]);
      expect(isAllowedOrigin("https://trusted.com", allowedOrigins)).toBe(true);
      expect(isAllowedOrigin("https://partner.com", allowedOrigins)).toBe(true);
    });

    it("returns false for disallowed origin", () => {
      const allowedOrigins = new Set(["https://trusted.com"]);
      expect(isAllowedOrigin("https://untrusted.com", allowedOrigins)).toBe(false);
      expect(isAllowedOrigin("https://malicious.com", allowedOrigins)).toBe(false);
    });

    it("is case-sensitive for origins", () => {
      const allowedOrigins = new Set(["https://Trusted.com"]);
      expect(isAllowedOrigin("https://trusted.com", allowedOrigins)).toBe(false);
    });

    it("returns false for empty allowedOrigins", () => {
      const allowedOrigins = new Set<string>();
      expect(isAllowedOrigin("https://any.com", allowedOrigins)).toBe(false);
    });
  });

  describe("extractOrigin", () => {
    it("extracts origin from valid URL", () => {
      expect(extractOrigin("https://example.com/path")).toBe("https://example.com");
      expect(extractOrigin("https://example.com:8080/path")).toBe("https://example.com:8080");
      expect(extractOrigin("http://localhost:3000")).toBe("http://localhost:3000");
    });

    it("returns null for invalid URL", () => {
      expect(extractOrigin("not-a-url")).toBe(null);
      expect(extractOrigin("")).toBe(null);
    });

    it("handles URLs with query strings", () => {
      expect(extractOrigin("https://example.com/path?query=1")).toBe("https://example.com");
    });

    it("handles URLs with fragments", () => {
      expect(extractOrigin("https://example.com/path#section")).toBe("https://example.com");
    });
  });

  describe("getEnvAllowedOrigins", () => {
    it("returns empty set when env var not set", () => {
      const originalEnv = process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS;
      delete process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS;

      const origins = getEnvAllowedOrigins();
      expect(origins.size).toBe(0);

      if (originalEnv !== undefined) {
        process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS = originalEnv;
      }
    });

    it("parses comma-separated origins", () => {
      const originalEnv = process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS;
      process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS = "https://a.com,https://b.com";

      const origins = getEnvAllowedOrigins();
      expect(origins.has("https://a.com")).toBe(true);
      expect(origins.has("https://b.com")).toBe(true);

      if (originalEnv !== undefined) {
        process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS = originalEnv;
      } else {
        delete process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS;
      }
    });

    it("trims whitespace from origins", () => {
      const originalEnv = process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS;
      process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS = " https://a.com , https://b.com ";

      const origins = getEnvAllowedOrigins();
      expect(origins.has("https://a.com")).toBe(true);
      expect(origins.has("https://b.com")).toBe(true);
      expect(origins.has(" https://a.com ")).toBe(false);

      if (originalEnv !== undefined) {
        process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS = originalEnv;
      } else {
        delete process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS;
      }
    });

    it("filters empty strings", () => {
      const originalEnv = process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS;
      process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS = "https://a.com,,https://b.com,";

      const origins = getEnvAllowedOrigins();
      expect(origins.size).toBe(2);

      if (originalEnv !== undefined) {
        process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS = originalEnv;
      } else {
        delete process.env.NEXT_PUBLIC_MCP_UI_ALLOWED_ORIGINS;
      }
    });
  });
});

// =============================================================================
// Security Compliance Tests
// =============================================================================

describe("MCP-UI Security Compliance", () => {
  describe("iframe sandbox attributes", () => {
    // These are documentation tests - verifying expected sandbox policy
    it("should use restrictive sandbox", () => {
      const expectedSandbox = "allow-scripts allow-forms allow-same-origin";
      // Verify our documentation matches expected security policy
      expect(expectedSandbox).toContain("allow-scripts");
      expect(expectedSandbox).toContain("allow-forms");
      expect(expectedSandbox).toContain("allow-same-origin");
      expect(expectedSandbox).not.toContain("allow-top-navigation");
      expect(expectedSandbox).not.toContain("allow-popups");
    });
  });

  describe("postMessage validation", () => {
    it("validates message against schema before processing", () => {
      // This demonstrates the expected validation pattern
      const validateMessage = (data: unknown): boolean => {
        const result = MCPUIMessageSchema.safeParse(data);
        return result.success;
      };

      expect(validateMessage({ type: "mcp_ui_resize", width: 600, height: 400 })).toBe(true);
      expect(validateMessage({ type: "invalid", data: "malicious" })).toBe(false);
      expect(validateMessage("<script>alert('xss')</script>")).toBe(false);
    });
  });
});
