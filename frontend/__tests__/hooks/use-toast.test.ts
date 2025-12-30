/**
 * Tests for use-toast hook
 * Story 6-5: Frontend Actions
 */

import { renderHook, act } from "@testing-library/react";
import { useToast } from "@/hooks/use-toast";

describe("useToast", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    // Clear all toasts before each test
    const { result } = renderHook(() => useToast());
    act(() => {
      result.current.dismissAll();
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe("Initial State", () => {
    it("provides toast, dismiss, and dismissAll functions", () => {
      const { result } = renderHook(() => useToast());

      expect(typeof result.current.toast).toBe("function");
      expect(typeof result.current.dismiss).toBe("function");
      expect(typeof result.current.dismissAll).toBe("function");
    });
  });

  describe("toast()", () => {
    it("adds a toast with generated id", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({
          variant: "default",
          title: "Success",
          description: "Content saved",
        });
      });

      expect(result.current.toasts.length).toBe(1);
      expect(result.current.toasts[0].id).toBeDefined();
      expect(result.current.toasts[0].title).toBe("Success");
      expect(result.current.toasts[0].description).toBe("Content saved");
      expect(result.current.toasts[0].variant).toBe("default");
    });

    it("returns the toast id", () => {
      const { result } = renderHook(() => useToast());

      let toastId: string = "";
      act(() => {
        toastId = result.current.toast({
          variant: "default",
          title: "Test",
        });
      });

      expect(toastId).toBeDefined();
      expect(result.current.toasts[0].id).toBe(toastId);
    });

    it("adds multiple toasts", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ variant: "default", title: "Toast 1" });
        result.current.toast({ variant: "destructive", title: "Toast 2" });
        result.current.toast({ variant: "success", title: "Toast 3" });
      });

      expect(result.current.toasts.length).toBe(3);
      expect(result.current.toasts[0].title).toBe("Toast 1");
      expect(result.current.toasts[1].title).toBe("Toast 2");
      expect(result.current.toasts[2].title).toBe("Toast 3");
    });

    it("supports all variant types", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ variant: "default", title: "Default" });
        result.current.toast({ variant: "destructive", title: "Destructive" });
        result.current.toast({ variant: "success", title: "Success" });
        result.current.toast({ variant: "warning", title: "Warning" });
        result.current.toast({ variant: "info", title: "Info" });
      });

      expect(result.current.toasts.length).toBe(5);
      expect(result.current.toasts.map((t) => t.variant)).toEqual([
        "default",
        "destructive",
        "success",
        "warning",
        "info",
      ]);
    });

    it("auto-dismisses after default duration (5 seconds)", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({
          variant: "default",
          title: "Auto-dismiss test",
        });
      });

      expect(result.current.toasts.length).toBe(1);

      act(() => {
        jest.advanceTimersByTime(5000);
      });

      expect(result.current.toasts.length).toBe(0);
    });

    it("respects custom duration", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({
          variant: "default",
          title: "Custom duration",
          duration: 2000,
        });
      });

      expect(result.current.toasts.length).toBe(1);

      act(() => {
        jest.advanceTimersByTime(1999);
      });

      expect(result.current.toasts.length).toBe(1);

      act(() => {
        jest.advanceTimersByTime(1);
      });

      expect(result.current.toasts.length).toBe(0);
    });

    it("does not auto-dismiss when duration is 0", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({
          variant: "default",
          title: "Persistent toast",
          duration: 0,
        });
      });

      expect(result.current.toasts.length).toBe(1);

      act(() => {
        jest.advanceTimersByTime(10000);
      });

      expect(result.current.toasts.length).toBe(1);
    });

    it("limits maximum number of toasts to 5", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        for (let i = 1; i <= 7; i++) {
          result.current.toast({
            variant: "default",
            title: "Toast " + i,
            duration: 0,
          });
        }
      });

      expect(result.current.toasts.length).toBe(5);
      expect(result.current.toasts[0].title).toBe("Toast 3");
      expect(result.current.toasts[4].title).toBe("Toast 7");
    });
  });

  describe("dismiss()", () => {
    it("removes a specific toast by id", () => {
      const { result } = renderHook(() => useToast());

      let toastId: string = "";
      act(() => {
        toastId = result.current.toast({
          variant: "default",
          title: "To be dismissed",
          duration: 0,
        });
        result.current.toast({
          variant: "default",
          title: "Remaining",
          duration: 0,
        });
      });

      expect(result.current.toasts.length).toBe(2);

      act(() => {
        result.current.dismiss(toastId);
      });

      expect(result.current.toasts.length).toBe(1);
      expect(result.current.toasts[0].title).toBe("Remaining");
    });

    it("clears auto-dismiss timer when manually dismissed", () => {
      const { result } = renderHook(() => useToast());

      let toastId: string = "";
      act(() => {
        toastId = result.current.toast({
          variant: "default",
          title: "Manual dismiss",
          duration: 5000,
        });
      });

      act(() => {
        result.current.dismiss(toastId);
      });

      expect(result.current.toasts.length).toBe(0);

      act(() => {
        jest.advanceTimersByTime(5000);
      });

      expect(result.current.toasts.length).toBe(0);
    });

    it("handles dismissing non-existent toast gracefully", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({
          variant: "default",
          title: "Existing",
          duration: 0,
        });
      });

      expect(() => {
        act(() => {
          result.current.dismiss("non-existent-id");
        });
      }).not.toThrow();

      expect(result.current.toasts.length).toBe(1);
    });
  });

  describe("dismissAll()", () => {
    it("removes all toasts", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ variant: "default", title: "Toast 1", duration: 0 });
        result.current.toast({ variant: "default", title: "Toast 2", duration: 0 });
        result.current.toast({ variant: "default", title: "Toast 3", duration: 0 });
      });

      expect(result.current.toasts.length).toBe(3);

      act(() => {
        result.current.dismissAll();
      });

      expect(result.current.toasts.length).toBe(0);
    });

    it("clears all auto-dismiss timers", () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ variant: "default", title: "Toast 1", duration: 3000 });
        result.current.toast({ variant: "default", title: "Toast 2", duration: 5000 });
      });

      act(() => {
        result.current.dismissAll();
      });

      expect(result.current.toasts.length).toBe(0);

      act(() => {
        jest.advanceTimersByTime(10000);
      });

      expect(result.current.toasts.length).toBe(0);
    });
  });

  describe("Global State Sharing", () => {
    it("shares toasts across multiple hook instances", () => {
      const { result: result1 } = renderHook(() => useToast());
      const { result: result2 } = renderHook(() => useToast());

      act(() => {
        result1.current.toast({
          variant: "default",
          title: "Shared toast",
          duration: 0,
        });
      });

      // Both hooks should see the toast
      expect(result1.current.toasts.length).toBe(1);
      expect(result2.current.toasts.length).toBe(1);
      expect(result1.current.toasts[0].title).toBe("Shared toast");
      expect(result2.current.toasts[0].title).toBe("Shared toast");
    });

    it("updates all instances when toast is dismissed", () => {
      const { result: result1 } = renderHook(() => useToast());
      const { result: result2 } = renderHook(() => useToast());

      let toastId: string = "";
      act(() => {
        toastId = result1.current.toast({
          variant: "default",
          title: "To dismiss",
          duration: 0,
        });
      });

      // Dismiss from second hook
      act(() => {
        result2.current.dismiss(toastId);
      });

      // Both should be updated
      expect(result1.current.toasts.length).toBe(0);
      expect(result2.current.toasts.length).toBe(0);
    });
  });
});
