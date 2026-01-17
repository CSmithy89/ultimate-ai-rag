/**
 * Tests for useRunControl hook
 */

import { renderHook, act, waitFor } from "@testing-library/react";
import { useRunControl } from "@/hooks/use-run-control";

// Mock useCopilotChat
jest.mock("@copilotkit/react-core", () => ({
  useCopilotChat: () => ({ isLoading: false }),
}));

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe("useRunControl", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  describe("initial state", () => {
    it("should return initial state", () => {
      const { result } = renderHook(() => useRunControl());

      expect(result.current.currentRunId).toBeNull();
      expect(result.current.isRunning).toBe(false);
      expect(result.current.canCancel).toBe(false);
      expect(result.current.error).toBeNull();
    });
  });

  describe("cancelRun", () => {
    it("should return false when no active run", async () => {
      const { result } = renderHook(() => useRunControl());

      let success: boolean;
      await act(async () => {
        success = await result.current.cancelRun();
      });

      expect(success!).toBe(false);
      expect(result.current.error).toBe("No active run to cancel");
    });

    it("should cancel run successfully", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ cancelled: true }),
      });

      const onCancel = jest.fn();
      const { result } = renderHook(() => useRunControl({ onCancel }));

      act(() => {
        result.current.setCurrentRunId("run-123");
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.cancelRun();
      });

      expect(success!).toBe(true);
      expect(onCancel).toHaveBeenCalledWith("run-123");
      expect(result.current.currentRunId).toBeNull();
    });

    it("should handle cancel failure", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ cancelled: false, message: "Run already finished" }),
      });

      const { result } = renderHook(() => useRunControl());

      act(() => {
        result.current.setCurrentRunId("run-123");
      });

      let success: boolean;
      await act(async () => {
        success = await result.current.cancelRun();
      });

      expect(success!).toBe(false);
      expect(result.current.error).toBe("Run already finished");
    });

    it("should include tenant header when provided", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ cancelled: true }),
      });

      const { result } = renderHook(() =>
        useRunControl({ tenantId: "tenant-abc" })
      );

      act(() => {
        result.current.setCurrentRunId("run-123");
      });

      await act(async () => {
        await result.current.cancelRun();
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            "X-Tenant-ID": "tenant-abc",
          }),
        })
      );
    });
  });

  describe("getRunState", () => {
    it("should return run state on success", async () => {
      const mockRunState = {
        run_id: "run-123",
        thread_id: "thread-456",
        status: "running",
        query: "test query",
        created_at: "2024-01-01T00:00:00Z",
        current_step: 2,
        total_steps: 4,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockRunState),
      });

      const { result } = renderHook(() => useRunControl());

      let response: Awaited<ReturnType<typeof result.current.getRunState>>;
      await act(async () => {
        response = await result.current.getRunState("run-123");
      });

      expect(response!.run).toEqual(mockRunState);
      expect(response!.notFound).toBe(false);
      expect(response!.error).toBeNull();
    });

    it("should return notFound for 404", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      const { result } = renderHook(() => useRunControl());

      let response: Awaited<ReturnType<typeof result.current.getRunState>>;
      await act(async () => {
        response = await result.current.getRunState("run-nonexistent");
      });

      expect(response!.run).toBeNull();
      expect(response!.notFound).toBe(true);
      expect(response!.error).toBeNull();
    });

    it("should return error for server errors", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: "Internal server error" }),
      });

      const onError = jest.fn();
      const { result } = renderHook(() => useRunControl({ onError }));

      let response: Awaited<ReturnType<typeof result.current.getRunState>>;
      await act(async () => {
        response = await result.current.getRunState("run-123");
      });

      expect(response!.run).toBeNull();
      expect(response!.notFound).toBe(false);
      expect(response!.error).not.toBeNull();
      expect(onError).toHaveBeenCalled();
    });

    it("should return error for network failures", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Network error"));

      const onError = jest.fn();
      const { result } = renderHook(() => useRunControl({ onError }));

      let response: Awaited<ReturnType<typeof result.current.getRunState>>;
      await act(async () => {
        response = await result.current.getRunState("run-123");
      });

      expect(response!.run).toBeNull();
      expect(response!.notFound).toBe(false);
      expect(response!.error).toBeInstanceOf(Error);
      expect(response!.error?.message).toBe("Network error");
      expect(onError).toHaveBeenCalled();
    });
  });

  describe("resumeRun", () => {
    it("should resume run successfully", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            run_id: "run-123",
            status: "running",
          }),
      });

      const onResume = jest.fn();
      const { result } = renderHook(() => useRunControl({ onResume }));

      let success: boolean;
      await act(async () => {
        success = await result.current.resumeRun("run-123");
      });

      expect(success!).toBe(true);
      expect(onResume).toHaveBeenCalledWith("run-123");
      expect(result.current.currentRunId).toBe("run-123");
    });
  });

  describe("listActiveRuns", () => {
    it("should return list of runs", async () => {
      const mockRuns = [
        { run_id: "run-1", status: "running" },
        { run_id: "run-2", status: "running" },
      ];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ runs: mockRuns }),
      });

      const { result } = renderHook(() =>
        useRunControl({ tenantId: "tenant-abc" })
      );

      let runs: Awaited<ReturnType<typeof result.current.listActiveRuns>>;
      await act(async () => {
        runs = await result.current.listActiveRuns();
      });

      expect(runs!).toEqual(mockRuns);
    });

    it("should return empty array on error", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Network error"));

      const { result } = renderHook(() => useRunControl());

      let runs: Awaited<ReturnType<typeof result.current.listActiveRuns>>;
      await act(async () => {
        runs = await result.current.listActiveRuns();
      });

      expect(runs!).toEqual([]);
      expect(result.current.error).toBe("Network error");
    });
  });

  describe("clearError", () => {
    it("should clear error state", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Test error"));

      const { result } = renderHook(() => useRunControl());

      await act(async () => {
        await result.current.listActiveRuns();
      });

      expect(result.current.error).toBe("Test error");

      act(() => {
        result.current.clearError();
      });

      expect(result.current.error).toBeNull();
    });
  });
});
