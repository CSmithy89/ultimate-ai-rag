/**
 * Tests for useAgentSteering hook
 */

import { renderHook, act } from "@testing-library/react";
import { useAgentSteering } from "@/hooks/use-agent-steering";

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe("useAgentSteering", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  describe("initial state", () => {
    it("should return initial state", () => {
      const { result } = renderHook(() => useAgentSteering());

      expect(result.current.isSteering).toBe(false);
      expect(result.current.lastResult).toBeNull();
      expect(result.current.error).toBeNull();
    });
  });

  describe("steerAgent", () => {
    it("should return false when runId is missing", async () => {
      const { result } = renderHook(() => useAgentSteering());

      let success: boolean;
      await act(async () => {
        success = await result.current.steerAgent("", "Do something else");
      });

      expect(success!).toBe(false);
      expect(result.current.error).toBe("Run ID and instruction are required");
    });

    it("should return false when instruction is missing", async () => {
      const { result } = renderHook(() => useAgentSteering());

      let success: boolean;
      await act(async () => {
        success = await result.current.steerAgent("run-123", "");
      });

      expect(success!).toBe(false);
      expect(result.current.error).toBe("Run ID and instruction are required");
    });

    it("should check run status before steering", async () => {
      // First call: checkRunStatus returns not running
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: "completed" }),
      });

      const { result } = renderHook(() => useAgentSteering());

      let success: boolean;
      await act(async () => {
        success = await result.current.steerAgent("run-123", "Do something else");
      });

      expect(success!).toBe(false);
      expect(result.current.error).toBe("Cannot steer: run is not in 'running' status");
    });

    it("should steer successfully when run is active", async () => {
      // First call: checkRunStatus returns running
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: "running" }),
      });
      // Second call: steer
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            runId: "run-123",
            status: "steering_applied",
            message: "Instruction received",
          }),
      });

      const onSuccess = jest.fn();
      const { result } = renderHook(() => useAgentSteering({ onSuccess }));

      let success: boolean;
      await act(async () => {
        success = await result.current.steerAgent("run-123", "Focus on performance");
      });

      expect(success!).toBe(true);
      expect(onSuccess).toHaveBeenCalledWith({
        runId: "run-123",
        status: "steering_applied",
        message: "Instruction received",
      });
      expect(result.current.lastResult).toEqual({
        runId: "run-123",
        status: "steering_applied",
        message: "Instruction received",
      });
    });

    it("should handle API errors", async () => {
      // First call: checkRunStatus returns running
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: "running" }),
      });
      // Second call: steer fails
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: "Server error" }),
      });

      const onError = jest.fn();
      const { result } = renderHook(() => useAgentSteering({ onError }));

      let success: boolean;
      await act(async () => {
        success = await result.current.steerAgent("run-123", "Test instruction");
      });

      expect(success!).toBe(false);
      expect(result.current.error).toBe("Server error");
      expect(onError).toHaveBeenCalled();
    });

    it("should include tenant header when provided", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: "running" }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            runId: "run-123",
            status: "steering_applied",
            message: "OK",
          }),
      });

      const { result } = renderHook(() =>
        useAgentSteering({ tenantId: "tenant-xyz" })
      );

      await act(async () => {
        await result.current.steerAgent("run-123", "Test instruction");
      });

      // Check both calls have tenant header
      expect(mockFetch).toHaveBeenNthCalledWith(
        1, // checkRunStatus call
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            "X-Tenant-ID": "tenant-xyz",
          }),
        })
      );
      expect(mockFetch).toHaveBeenNthCalledWith(
        2, // steer call
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            "X-Tenant-ID": "tenant-xyz",
          }),
        })
      );
    });

    it("should pass context to API", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: "running" }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            runId: "run-123",
            status: "steering_applied",
            message: "OK",
          }),
      });

      const { result } = renderHook(() => useAgentSteering());

      await act(async () => {
        await result.current.steerAgent("run-123", "Focus on security", {
          priority: "high",
          reason: "vulnerability found",
        });
      });

      expect(mockFetch).toHaveBeenNthCalledWith(
        2,
        expect.any(String),
        expect.objectContaining({
          body: JSON.stringify({
            run_id: "run-123",
            instruction: "Focus on security",
            context: { priority: "high", reason: "vulnerability found" },
          }),
        })
      );
    });
  });

  describe("checkRunStatus", () => {
    it("should return true for running status", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: "running" }),
      });

      const { result } = renderHook(() => useAgentSteering());

      let canSteer: boolean;
      await act(async () => {
        canSteer = await result.current.checkRunStatus("run-123");
      });

      expect(canSteer!).toBe(true);
    });

    it("should return false for non-running status", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: "completed" }),
      });

      const { result } = renderHook(() => useAgentSteering());

      let canSteer: boolean;
      await act(async () => {
        canSteer = await result.current.checkRunStatus("run-123");
      });

      expect(canSteer!).toBe(false);
    });

    it("should return false on API error", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Network error"));

      const { result } = renderHook(() => useAgentSteering());

      let canSteer: boolean;
      await act(async () => {
        canSteer = await result.current.checkRunStatus("run-123");
      });

      expect(canSteer!).toBe(false);
    });
  });

  describe("clearError", () => {
    it("should clear error state", async () => {
      const { result } = renderHook(() => useAgentSteering());

      await act(async () => {
        await result.current.steerAgent("", "test");
      });

      expect(result.current.error).not.toBeNull();

      act(() => {
        result.current.clearError();
      });

      expect(result.current.error).toBeNull();
    });
  });
});
