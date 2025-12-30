/**
 * Tests for use-copilot-actions hook
 * Story 6-5: Frontend Actions
 */

import { renderHook, act } from "@testing-library/react";
import { useCopilotAction } from "@copilotkit/react-core";
import { useCopilotActions } from "@/hooks/use-copilot-actions";
import { useToast } from "@/hooks/use-toast";

// Mock dependencies
jest.mock("@copilotkit/react-core", () => ({
  useCopilotAction: jest.fn(),
}));

jest.mock("@/hooks/use-toast", () => ({
  useToast: jest.fn(),
}));

// Mock fetch
global.fetch = jest.fn();

const mockUseCopilotAction = useCopilotAction as jest.MockedFunction<typeof useCopilotAction>;
const mockUseToast = useToast as jest.MockedFunction<typeof useToast>;
const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;

// Mock content for testing
const mockContent = {
  id: "content-1",
  content: "Test AI response content",
  title: "Test Response",
  query: "What is the answer?",
  sources: [
    { id: "source-1", title: "Source 1" },
    { id: "source-2", title: "Source 2", url: "https://example.com" },
  ],
  sessionId: "session-123",
  trajectoryId: "trajectory-456",
};

describe("useCopilotActions", () => {
  const mockToast = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    mockUseToast.mockReturnValue({
      toasts: [],
      toast: mockToast,
      dismiss: jest.fn(),
      dismissAll: jest.fn(),
    });

    // Default successful response
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: {} }),
      text: async () => "# Content",
      blob: async () => new Blob(["content"], { type: "application/pdf" }),
    } as Response);

    // Mock URL methods
    global.URL.createObjectURL = jest.fn().mockReturnValue("blob:test");
    global.URL.revokeObjectURL = jest.fn();

    // Mock clipboard
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: jest.fn().mockResolvedValue(undefined) },
      writable: true,
      configurable: true,
    });

    // Mock anchor click (prevent navigation issues)
    HTMLAnchorElement.prototype.click = jest.fn();
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  describe("Initial State", () => {
    it("initializes with all actions in idle state", () => {
      const { result } = renderHook(() => useCopilotActions());

      expect(result.current.actionStates).toEqual({
        save: "idle",
        export: "idle",
        share: "idle",
        bookmark: "idle",
        followUp: "idle",
      });
    });

    it("isLoading is false initially", () => {
      const { result } = renderHook(() => useCopilotActions());

      expect(result.current.isLoading).toBe(false);
    });

    it("provides all action functions", () => {
      const { result } = renderHook(() => useCopilotActions());

      expect(typeof result.current.saveToWorkspace).toBe("function");
      expect(typeof result.current.exportContent).toBe("function");
      expect(typeof result.current.shareContent).toBe("function");
      expect(typeof result.current.bookmarkContent).toBe("function");
      expect(typeof result.current.triggerFollowUp).toBe("function");
      expect(typeof result.current.resetStates).toBe("function");
    });
  });

  describe("saveToWorkspace", () => {
    it("sets loading state during save", async () => {
      const { result } = renderHook(() => useCopilotActions());

      let resolvePromise: () => void;
      mockFetch.mockReturnValueOnce(
        new Promise((resolve) => {
          resolvePromise = () =>
            resolve({
              ok: true,
              json: async () => ({ data: {} }),
            } as Response);
        })
      );

      act(() => {
        result.current.saveToWorkspace(mockContent);
      });

      expect(result.current.actionStates.save).toBe("loading");
      expect(result.current.isLoading).toBe(true);

      await act(async () => {
        resolvePromise!();
      });
    });

    it("calls API with correct payload", async () => {
      const { result } = renderHook(() =>
        useCopilotActions({ tenantId: "tenant-1" })
      );

      await act(async () => {
        await result.current.saveToWorkspace(mockContent);
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/workspace/save"),
        expect.objectContaining({
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: expect.stringContaining("content-1"),
        })
      );
    });

    it("shows success toast on successful save", async () => {
      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        await result.current.saveToWorkspace(mockContent);
      });

      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "default",
          title: "Saved to workspace",
        })
      );
    });

    it("shows error toast on failed save", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({ detail: "Save failed" }),
      } as Response);

      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        await result.current.saveToWorkspace(mockContent);
      });

      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "destructive",
          title: "Save failed",
        })
      );
    });

    it("calls onSaveComplete callback", async () => {
      const onSaveComplete = jest.fn();
      const { result } = renderHook(() =>
        useCopilotActions({ onSaveComplete })
      );

      await act(async () => {
        await result.current.saveToWorkspace(mockContent);
      });

      expect(onSaveComplete).toHaveBeenCalledWith(
        expect.objectContaining({ success: true })
      );
    });

    it("auto-resets state after success", async () => {
      const { result } = renderHook(() =>
        useCopilotActions({ successResetDelay: 1000 })
      );

      await act(async () => {
        await result.current.saveToWorkspace(mockContent);
      });

      expect(result.current.actionStates.save).toBe("success");

      act(() => {
        jest.advanceTimersByTime(1000);
      });

      expect(result.current.actionStates.save).toBe("idle");
    });
  });

  describe("exportContent", () => {
    it("calls export API with correct format", async () => {
      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        await result.current.exportContent(mockContent, "json");
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/workspace/export"),
        expect.objectContaining({
          method: "POST",
          body: expect.stringContaining("json"),
        })
      );
    });

    it("shows success toast on export", async () => {
      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        await result.current.exportContent(mockContent, "json");
      });

      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "default",
          title: "Export complete",
        })
      );
    });

    it("sets export state to success", async () => {
      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        await result.current.exportContent(mockContent, "json");
      });

      expect(result.current.actionStates.export).toBe("success");
    });

    it("creates downloadable blob for JSON format", async () => {
      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        await result.current.exportContent(mockContent, "json");
      });

      expect(global.URL.createObjectURL).toHaveBeenCalled();
    });
  });

  describe("shareContent", () => {
    it("copies share URL to clipboard", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: { share_url: "https://share.test/abc" } }),
      } as Response);

      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        const url = await result.current.shareContent(mockContent);
        expect(url).toBe("https://share.test/abc");
      });

      expect(navigator.clipboard.writeText).toHaveBeenCalledWith("https://share.test/abc");
    });

    it("shows success toast with link copied message", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: { share_url: "https://share.test/abc" } }),
      } as Response);

      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        await result.current.shareContent(mockContent);
      });

      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "default",
          title: "Link copied!",
        })
      );
    });

    it("calls onShareComplete with share URL", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: { share_url: "https://share.test/xyz" } }),
      } as Response);

      const onShareComplete = jest.fn();
      const { result } = renderHook(() => useCopilotActions({ onShareComplete }));

      await act(async () => {
        await result.current.shareContent(mockContent);
      });

      expect(onShareComplete).toHaveBeenCalledWith(
        expect.objectContaining({ success: true }),
        "https://share.test/xyz"
      );
    });
  });

  describe("bookmarkContent", () => {
    it("calls bookmark API", async () => {
      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        await result.current.bookmarkContent(mockContent);
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/workspace/bookmark"),
        expect.objectContaining({ method: "POST" })
      );
    });

    it("shows success toast", async () => {
      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        await result.current.bookmarkContent(mockContent);
      });

      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "default",
          title: "Bookmarked",
        })
      );
    });
  });

  describe("triggerFollowUp", () => {
    it("dispatches custom event when no callback provided", () => {
      const dispatchEventSpy = jest.spyOn(document, "dispatchEvent");

      const { result } = renderHook(() => useCopilotActions());

      act(() => {
        result.current.triggerFollowUp(mockContent);
      });

      expect(dispatchEventSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "copilot:follow-up",
        })
      );

      dispatchEventSpy.mockRestore();
    });

    it("calls onFollowUp callback when provided", () => {
      const onFollowUp = jest.fn();
      const { result } = renderHook(() => useCopilotActions({ onFollowUp }));

      act(() => {
        result.current.triggerFollowUp(mockContent);
      });

      expect(onFollowUp).toHaveBeenCalledWith(
        expect.stringContaining("Following up"),
        mockContent
      );
    });
  });

  describe("resetStates", () => {
    it("resets all action states to idle", async () => {
      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        await result.current.saveToWorkspace(mockContent);
      });

      expect(result.current.actionStates.save).toBe("success");

      act(() => {
        result.current.resetStates();
      });

      expect(result.current.actionStates).toEqual({
        save: "idle",
        export: "idle",
        share: "idle",
        bookmark: "idle",
        followUp: "idle",
      });
    });
  });

  describe("CopilotKit Integration", () => {
    it("registers 5 CopilotKit actions", () => {
      renderHook(() => useCopilotActions());

      expect(mockUseCopilotAction).toHaveBeenCalledTimes(5);
    });

    it("registers save_to_workspace action", () => {
      renderHook(() => useCopilotActions());

      expect(mockUseCopilotAction).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "save_to_workspace",
          description: expect.stringContaining("Save"),
        })
      );
    });

    it("registers export_content action", () => {
      renderHook(() => useCopilotActions());

      expect(mockUseCopilotAction).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "export_content",
          description: expect.stringContaining("Export"),
        })
      );
    });

    it("registers share_content action", () => {
      renderHook(() => useCopilotActions());

      expect(mockUseCopilotAction).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "share_content",
          description: expect.stringContaining("share"),
        })
      );
    });

    it("registers bookmark_content action", () => {
      renderHook(() => useCopilotActions());

      expect(mockUseCopilotAction).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "bookmark_content",
          description: expect.stringContaining("Bookmark"),
        })
      );
    });

    it("registers suggest_follow_up action", () => {
      renderHook(() => useCopilotActions());

      expect(mockUseCopilotAction).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "suggest_follow_up",
          description: expect.stringContaining("follow-up"),
        })
      );
    });
  });
});
