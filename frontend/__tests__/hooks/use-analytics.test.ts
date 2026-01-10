/**
 * Tests for use-analytics hook
 * Story 21-B1: Configure Observability Hooks and Dev Console
 *
 * Tests the useAnalytics hook for:
 * - Tracking events via fetch to /api/telemetry
 * - Development console logging
 * - Error handling (non-blocking)
 * - PII redaction via redactSensitiveKeys
 * - Timestamp generation
 */

// Mock fetch before imports
const mockFetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ status: "accepted" }),
  })
) as jest.Mock;
global.fetch = mockFetch;

// Mock redact utility
const mockRedactSensitiveKeys = jest.fn((obj: Record<string, unknown>) => {
  if (!obj || typeof obj !== "object") return obj;
  return Object.fromEntries(
    Object.entries(obj).map(([k, v]) =>
      /password|secret|token|key|auth/i.test(k) ? [k, "[REDACTED]"] : [k, v]
    )
  );
});
jest.mock("@/lib/utils/redact", () => ({
  redactSensitiveKeys: (obj: Record<string, unknown>) =>
    mockRedactSensitiveKeys(obj),
}));

// Import after mocks are set up
import { renderHook, act } from "@testing-library/react";
import { useAnalytics, TelemetryEventPayload } from "@/hooks/use-analytics";

describe("useAnalytics hook", () => {
  // Store original console methods
  const originalConsoleLog = console.log;
  const originalConsoleError = console.error;
  let consoleLogSpy: jest.SpyInstance;
  let consoleErrorSpy: jest.SpyInstance;

  // Store original NODE_ENV
  const originalNodeEnv = process.env.NODE_ENV;

  beforeEach(() => {
    jest.clearAllMocks();
    consoleLogSpy = jest.spyOn(console, "log").mockImplementation(() => {});
    consoleErrorSpy = jest.spyOn(console, "error").mockImplementation(() => {});
    // Reset to development mode by default
    process.env.NODE_ENV = "development";
  });

  afterEach(() => {
    consoleLogSpy.mockRestore();
    consoleErrorSpy.mockRestore();
    process.env.NODE_ENV = originalNodeEnv;
  });

  describe("track() function", () => {
    it("calls fetch with correct endpoint and method", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("test_event");
      });

      // Wait for async fetch
      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/telemetry",
        expect.objectContaining({
          method: "POST",
          headers: { "Content-Type": "application/json" },
        })
      );
    });

    it("sends event name in payload", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("copilot_message_sent");
      });

      await new Promise((resolve) => setTimeout(resolve, 0));

      const callBody = JSON.parse(
        mockFetch.mock.calls[0][1].body
      ) as TelemetryEventPayload;
      expect(callBody.event).toBe("copilot_message_sent");
    });

    it("sends properties in payload", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("copilot_message_sent", { messageLength: 150 });
      });

      await new Promise((resolve) => setTimeout(resolve, 0));

      const callBody = JSON.parse(
        mockFetch.mock.calls[0][1].body
      ) as TelemetryEventPayload;
      expect(callBody.properties).toEqual({ messageLength: 150 });
    });

    it("includes timestamp in payload", async () => {
      const { result } = renderHook(() => useAnalytics());

      const beforeTime = new Date().toISOString();
      act(() => {
        result.current.track("test_event");
      });
      const afterTime = new Date().toISOString();

      await new Promise((resolve) => setTimeout(resolve, 0));

      const callBody = JSON.parse(
        mockFetch.mock.calls[0][1].body
      ) as TelemetryEventPayload;

      // Timestamp should be between before and after
      expect(callBody.timestamp).toBeDefined();
      expect(new Date(callBody.timestamp).getTime()).toBeGreaterThanOrEqual(
        new Date(beforeTime).getTime() - 1000
      );
      expect(new Date(callBody.timestamp).getTime()).toBeLessThanOrEqual(
        new Date(afterTime).getTime() + 1000
      );
    });

    it("handles empty properties gracefully", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("copilot_chat_expanded");
      });

      await new Promise((resolve) => setTimeout(resolve, 0));

      const callBody = JSON.parse(
        mockFetch.mock.calls[0][1].body
      ) as TelemetryEventPayload;
      expect(callBody.properties).toEqual({});
    });

    it("handles undefined properties", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("test_event", undefined);
      });

      await new Promise((resolve) => setTimeout(resolve, 0));

      const callBody = JSON.parse(
        mockFetch.mock.calls[0][1].body
      ) as TelemetryEventPayload;
      expect(callBody.properties).toEqual({});
    });
  });

  describe("development logging", () => {
    it("logs to console in development mode", () => {
      process.env.NODE_ENV = "development";

      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("test_event", { data: "test" });
      });

      expect(consoleLogSpy).toHaveBeenCalledWith(
        "[Analytics]",
        "test_event",
        expect.objectContaining({ data: "test" })
      );
    });

    it("does not log to console in production mode", () => {
      process.env.NODE_ENV = "production";

      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("test_event", { data: "test" });
      });

      expect(consoleLogSpy).not.toHaveBeenCalled();
    });
  });

  describe("error handling", () => {
    it("does not throw when fetch fails", async () => {
      mockFetch.mockImplementationOnce(() =>
        Promise.reject(new Error("Network error"))
      );

      const { result } = renderHook(() => useAnalytics());

      // Should not throw
      expect(() => {
        act(() => {
          result.current.track("test_event");
        });
      }).not.toThrow();

      // Wait for async error handling
      await new Promise((resolve) => setTimeout(resolve, 10));
    });

    it("logs error when fetch fails", async () => {
      const networkError = new Error("Network error");
      mockFetch.mockImplementationOnce(() => Promise.reject(networkError));

      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("test_event");
      });

      // Wait for async error handling
      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        "[Analytics] Failed to send event:",
        networkError
      );
    });

    it("continues to work after fetch failure", async () => {
      // First call fails
      mockFetch.mockImplementationOnce(() =>
        Promise.reject(new Error("Network error"))
      );

      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("first_event");
      });

      await new Promise((resolve) => setTimeout(resolve, 10));

      // Reset mock for success
      mockFetch.mockImplementation(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ status: "accepted" }),
        })
      );

      // Second call should still work
      act(() => {
        result.current.track("second_event");
      });

      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(mockFetch).toHaveBeenCalledTimes(2);
    });
  });

  describe("PII redaction", () => {
    it("calls redactSensitiveKeys with properties", async () => {
      const { result } = renderHook(() => useAnalytics());

      const testProps = { apiKey: "secret123", message: "hello" };

      act(() => {
        result.current.track("test_event", testProps);
      });

      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(mockRedactSensitiveKeys).toHaveBeenCalledWith(testProps);
    });

    it("sends redacted properties to backend", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("test_event", { apiKey: "secret123", safe: "ok" });
      });

      await new Promise((resolve) => setTimeout(resolve, 0));

      const callBody = JSON.parse(
        mockFetch.mock.calls[0][1].body
      ) as TelemetryEventPayload;
      expect(callBody.properties.apiKey).toBe("[REDACTED]");
      expect(callBody.properties.safe).toBe("ok");
    });

    it("redacts password keys", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("test_event", { password: "hunter2" });
      });

      await new Promise((resolve) => setTimeout(resolve, 0));

      const callBody = JSON.parse(
        mockFetch.mock.calls[0][1].body
      ) as TelemetryEventPayload;
      expect(callBody.properties.password).toBe("[REDACTED]");
    });

    it("redacts token keys", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("test_event", { access_token: "abc123" });
      });

      await new Promise((resolve) => setTimeout(resolve, 0));

      const callBody = JSON.parse(
        mockFetch.mock.calls[0][1].body
      ) as TelemetryEventPayload;
      expect(callBody.properties.access_token).toBe("[REDACTED]");
    });
  });

  describe("hook stability", () => {
    it("returns stable track function reference", () => {
      const { result, rerender } = renderHook(() => useAnalytics());

      const firstTrackRef = result.current.track;

      rerender();

      expect(result.current.track).toBe(firstTrackRef);
    });

    it("can be called multiple times in succession", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("event1");
        result.current.track("event2");
        result.current.track("event3");
      });

      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(mockFetch).toHaveBeenCalledTimes(3);
    });
  });

  describe("event types", () => {
    it("handles copilot_message_sent correctly", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("copilot_message_sent", { messageLength: 100 });
      });

      await new Promise((resolve) => setTimeout(resolve, 0));

      const callBody = JSON.parse(
        mockFetch.mock.calls[0][1].body
      ) as TelemetryEventPayload;
      expect(callBody.event).toBe("copilot_message_sent");
      expect(callBody.properties.messageLength).toBe(100);
    });

    it("handles copilot_error correctly", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("copilot_error", {
          type: "network",
          message: "Connection failed",
        });
      });

      await new Promise((resolve) => setTimeout(resolve, 0));

      const callBody = JSON.parse(
        mockFetch.mock.calls[0][1].body
      ) as TelemetryEventPayload;
      expect(callBody.event).toBe("copilot_error");
      expect(callBody.properties.type).toBe("network");
    });

    it("handles copilot_feedback correctly", async () => {
      const { result } = renderHook(() => useAnalytics());

      act(() => {
        result.current.track("copilot_feedback", {
          messageId: "msg-123",
          type: "positive",
        });
      });

      await new Promise((resolve) => setTimeout(resolve, 0));

      const callBody = JSON.parse(
        mockFetch.mock.calls[0][1].body
      ) as TelemetryEventPayload;
      expect(callBody.event).toBe("copilot_feedback");
      expect(callBody.properties.messageId).toBe("msg-123");
      expect(callBody.properties.type).toBe("positive");
    });

    it("handles simple events without properties", async () => {
      const { result } = renderHook(() => useAnalytics());

      const simpleEvents = [
        "copilot_chat_expanded",
        "copilot_chat_minimized",
        "copilot_generation_started",
        "copilot_generation_stopped",
      ];

      for (const event of simpleEvents) {
        act(() => {
          result.current.track(event);
        });
      }

      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(mockFetch).toHaveBeenCalledTimes(simpleEvents.length);
    });
  });
});

describe("TelemetryEventPayload type", () => {
  it("accepts valid payload structure", () => {
    const payload: TelemetryEventPayload = {
      event: "test_event",
      properties: { key: "value" },
      timestamp: new Date().toISOString(),
    };

    expect(payload.event).toBeDefined();
    expect(payload.properties).toBeDefined();
    expect(payload.timestamp).toBeDefined();
  });

  it("allows empty properties object", () => {
    const payload: TelemetryEventPayload = {
      event: "test_event",
      properties: {},
      timestamp: new Date().toISOString(),
    };

    expect(payload.properties).toEqual({});
  });
});

describe("Security considerations", () => {
  it("never sends raw message content when using messageLength pattern", async () => {
    mockFetch.mockClear();
    const { result } = renderHook(() => useAnalytics());

    // Simulate the way ChatSidebar uses analytics - only messageLength, not message
    act(() => {
      result.current.track("copilot_message_sent", {
        messageLength: 150,
        // NOT: message: "actual user message"
      });
    });

    await new Promise((resolve) => setTimeout(resolve, 0));

    // Verify fetch was called
    expect(mockFetch).toHaveBeenCalledTimes(1);
    const callBody = JSON.parse(
      mockFetch.mock.calls[0][1].body
    ) as TelemetryEventPayload;

    // Should only have length, not content
    expect(callBody.properties.messageLength).toBe(150);
    expect(callBody.properties.message).toBeUndefined();
    expect(callBody.properties.content).toBeUndefined();
  });

  it("redacts sensitive keys in error events", async () => {
    mockFetch.mockClear();
    const { result } = renderHook(() => useAnalytics());

    act(() => {
      result.current.track("copilot_error", {
        type: "auth_error",
        // This would be redacted if it were passed
        apiKey: "sk-secret-key",
        message: "Authentication failed",
      });
    });

    await new Promise((resolve) => setTimeout(resolve, 0));

    // Verify fetch was called
    expect(mockFetch).toHaveBeenCalledTimes(1);
    const callBody = JSON.parse(
      mockFetch.mock.calls[0][1].body
    ) as TelemetryEventPayload;

    // apiKey should be redacted by the mock redactSensitiveKeys function
    expect(callBody.properties.apiKey).toBe("[REDACTED]");
    // type and message should pass through (not sensitive)
    expect(callBody.properties.type).toBe("auth_error");
    expect(callBody.properties.message).toBe("Authentication failed");
  });
});
