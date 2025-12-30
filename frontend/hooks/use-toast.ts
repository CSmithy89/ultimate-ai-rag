"use client";

import { useState, useCallback, useEffect, useRef } from "react";

/**
 * Toast notification variant.
 */
export type ToastVariant = "default" | "destructive" | "success" | "warning" | "info";

/**
 * Toast notification object.
 */
export interface Toast {
  id: string;
  variant: ToastVariant;
  title: string;
  description?: string;
  duration?: number;
}

/**
 * Options for creating a toast.
 */
export type ToastOptions = Omit<Toast, "id">;

/**
 * Return type for the useToast hook.
 */
export interface UseToastReturn {
  /** Currently active toasts */
  toasts: Toast[];
  /** Add a toast notification */
  toast: (options: ToastOptions) => string;
  /** Dismiss a specific toast by ID */
  dismiss: (id: string) => void;
  /** Dismiss all toasts */
  dismissAll: () => void;
}

/** Default toast duration in milliseconds */
const DEFAULT_DURATION = 5000;

/** Maximum number of toasts to display */
const MAX_TOASTS = 5;

/**
 * Global toast state for sharing across hook instances.
 * This allows any component to show toasts without prop drilling.
 */
type ToastListener = (toasts: Toast[]) => void;
const listeners: Set<ToastListener> = new Set();
let globalToasts: Toast[] = [];

function notifyListeners() {
  listeners.forEach((listener) => listener([...globalToasts]));
}

function addToast(toast: Toast): void {
  // Limit to max toasts
  if (globalToasts.length >= MAX_TOASTS) {
    globalToasts = globalToasts.slice(1);
  }
  globalToasts = [...globalToasts, toast];
  notifyListeners();
}

function removeToast(id: string): void {
  globalToasts = globalToasts.filter((t) => t.id !== id);
  notifyListeners();
}

function clearToasts(): void {
  globalToasts = [];
  notifyListeners();
}

/**
 * useToast hook provides toast notification functionality.
 *
 * Story 6-5: Frontend Actions
 *
 * Features:
 * - Queue-based toast management
 * - Auto-dismiss after configurable duration
 * - Variants: default (success), destructive (error), info, warning
 * - Global state sharing across components
 */
export function useToast(): UseToastReturn {
  const [toasts, setToasts] = useState<Toast[]>(globalToasts);
  const timersRef = useRef<Map<string, NodeJS.Timeout>>(new Map());

  // Subscribe to global toast state
  useEffect(() => {
    const listener: ToastListener = (newToasts) => {
      setToasts(newToasts);
    };

    listeners.add(listener);

    // Capture current ref value for cleanup
    const currentTimers = timersRef.current;

    return () => {
      listeners.delete(listener);
      // Clean up timers on unmount using captured ref
      currentTimers.forEach((timer) => clearTimeout(timer));
      currentTimers.clear();
    };
  }, []);

  const toast = useCallback((options: ToastOptions): string => {
    const timestamp = new Date().getTime();
    const random = Math.random().toString(36).substring(2, 9);
    const id = "toast-" + timestamp + "-" + random;
    const duration = options.duration ?? DEFAULT_DURATION;

    const newToast: Toast = {
      id,
      ...options,
      duration,
    };

    addToast(newToast);

    // Auto-dismiss after duration
    if (duration > 0) {
      const timer = setTimeout(() => {
        removeToast(id);
        timersRef.current.delete(id);
      }, duration);
      timersRef.current.set(id, timer);
    }

    return id;
  }, []);

  const dismiss = useCallback((id: string) => {
    // Clear timer if exists
    const timer = timersRef.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timersRef.current.delete(id);
    }
    removeToast(id);
  }, []);

  const dismissAll = useCallback(() => {
    // Clear all timers
    timersRef.current.forEach((timer) => clearTimeout(timer));
    timersRef.current.clear();
    clearToasts();
  }, []);

  return { toasts, toast, dismiss, dismissAll };
}

export default useToast;
