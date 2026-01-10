/**
 * Tests for redactSensitiveKeys utility.
 * Story 21-A3: Implement Tool Call Visualization (AC7)
 */

import {
  redactSensitiveKeys,
  SENSITIVE_PATTERNS,
} from "../../../lib/utils/redact";

describe("SENSITIVE_PATTERNS", () => {
  it("matches password keys", () => {
    expect(SENSITIVE_PATTERNS.test("password")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("PASSWORD")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("user_password")).toBe(true);
  });

  it("matches secret keys", () => {
    expect(SENSITIVE_PATTERNS.test("secret")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("client_secret")).toBe(true);
  });

  it("matches token keys", () => {
    expect(SENSITIVE_PATTERNS.test("token")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("access_token")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("accessToken")).toBe(true);
  });

  it("matches key keys", () => {
    expect(SENSITIVE_PATTERNS.test("key")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("api_key")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("apiKey")).toBe(true);
  });

  it("matches auth keys", () => {
    expect(SENSITIVE_PATTERNS.test("auth")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("authorization")).toBe(true);
  });

  it("matches credential keys", () => {
    expect(SENSITIVE_PATTERNS.test("credential")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("credentials")).toBe(true);
  });

  it("matches api_key and api-key variants", () => {
    expect(SENSITIVE_PATTERNS.test("api_key")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("api-key")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("apikey")).toBe(true);
  });

  it("matches private_key and private-key variants", () => {
    expect(SENSITIVE_PATTERNS.test("private_key")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("private-key")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("privatekey")).toBe(true);
  });

  it("matches access_token and access-token variants", () => {
    expect(SENSITIVE_PATTERNS.test("access_token")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("access-token")).toBe(true);
    expect(SENSITIVE_PATTERNS.test("accesstoken")).toBe(true);
  });

  it("does not match non-sensitive keys", () => {
    expect(SENSITIVE_PATTERNS.test("username")).toBe(false);
    expect(SENSITIVE_PATTERNS.test("email")).toBe(false);
    expect(SENSITIVE_PATTERNS.test("name")).toBe(false);
    expect(SENSITIVE_PATTERNS.test("query")).toBe(false);
  });
});

describe("redactSensitiveKeys", () => {
  describe("basic redaction", () => {
    it("redacts password key", () => {
      const input = { password: "secret123", username: "admin" };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({ password: "[REDACTED]", username: "admin" });
    });

    it("redacts api_key key", () => {
      const input = { api_key: "sk-12345", endpoint: "https://api.example.com" };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({
        api_key: "[REDACTED]",
        endpoint: "https://api.example.com",
      });
    });

    it("redacts api-key key", () => {
      const input = { "api-key": "sk-12345" };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({ "api-key": "[REDACTED]" });
    });

    it("redacts token key", () => {
      const input = { token: "abc123", user_id: 1 };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({ token: "[REDACTED]", user_id: 1 });
    });

    it("redacts access_token key", () => {
      const input = { access_token: "jwt-token-here" };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({ access_token: "[REDACTED]" });
    });

    it("redacts multiple sensitive keys", () => {
      const input = {
        username: "user1",
        password: "pass123",
        api_key: "key456",
        secret: "shhh",
      };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({
        username: "user1",
        password: "[REDACTED]",
        api_key: "[REDACTED]",
        secret: "[REDACTED]",
      });
    });
  });

  describe("preserves non-sensitive keys", () => {
    it("preserves string values", () => {
      const input = { name: "test", query: "search text" };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual(input);
    });

    it("preserves number values", () => {
      const input = { count: 42, limit: 100 };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual(input);
    });

    it("preserves boolean values", () => {
      const input = { enabled: true, disabled: false };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual(input);
    });
  });

  describe("handles nested objects", () => {
    it("redacts keys in nested objects", () => {
      const input = {
        user: {
          name: "John",
          credentials: {
            password: "secret",
            api_key: "key123",
          },
        },
      };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({
        user: {
          name: "John",
          credentials: "[REDACTED]",
        },
      });
    });

    it("handles deeply nested objects", () => {
      const input = {
        level1: {
          level2: {
            level3: {
              token: "deep-secret",
              data: "visible",
            },
          },
        },
      };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({
        level1: {
          level2: {
            level3: {
              token: "[REDACTED]",
              data: "visible",
            },
          },
        },
      });
    });
  });

  describe("handles arrays", () => {
    it("processes arrays with objects", () => {
      const input = {
        items: [
          { id: 1, token: "abc" },
          { id: 2, token: "def" },
        ],
      };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({
        items: [
          { id: 1, token: "[REDACTED]" },
          { id: 2, token: "[REDACTED]" },
        ],
      });
    });

    it("handles arrays with primitive values", () => {
      const input = { tags: ["tag1", "tag2"], count: 2 };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({ tags: ["tag1", "tag2"], count: 2 });
    });

    it("handles mixed arrays", () => {
      const input = {
        mixed: [1, "string", { password: "secret" }, null],
      };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({
        mixed: [1, "string", { password: "[REDACTED]" }, null],
      });
    });
  });

  describe("handles edge cases", () => {
    it("handles null input", () => {
      const output = redactSensitiveKeys(null as unknown as Record<string, unknown>);
      expect(output).toBeNull();
    });

    it("handles undefined input", () => {
      const output = redactSensitiveKeys(undefined as unknown as Record<string, unknown>);
      expect(output).toBeUndefined();
    });

    it("handles empty object", () => {
      const output = redactSensitiveKeys({});
      expect(output).toEqual({});
    });

    it("handles primitive values", () => {
      // While not the intended use case, it should not throw
      const stringOutput = redactSensitiveKeys("string" as unknown as Record<string, unknown>);
      expect(stringOutput).toBe("string");

      const numberOutput = redactSensitiveKeys(42 as unknown as Record<string, unknown>);
      expect(numberOutput).toBe(42);
    });

    it("handles null values within objects", () => {
      const input = { name: null, password: "secret" };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({ name: null, password: "[REDACTED]" });
    });

    it("handles undefined values within objects", () => {
      const input = { name: undefined, password: "secret" };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({ name: undefined, password: "[REDACTED]" });
    });
  });

  describe("case insensitivity", () => {
    it("redacts uppercase sensitive keys", () => {
      const input = { PASSWORD: "secret", API_KEY: "key" };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({ PASSWORD: "[REDACTED]", API_KEY: "[REDACTED]" });
    });

    it("redacts mixed case sensitive keys", () => {
      const input = { PassWord: "secret", ApiKey: "key" };
      const output = redactSensitiveKeys(input);
      expect(output).toEqual({ PassWord: "[REDACTED]", ApiKey: "[REDACTED]" });
    });
  });

  describe("original object immutability", () => {
    it("does not modify the original object", () => {
      const original = { password: "secret", name: "test" };
      const originalCopy = { ...original };
      redactSensitiveKeys(original);
      expect(original).toEqual(originalCopy);
    });
  });
});
