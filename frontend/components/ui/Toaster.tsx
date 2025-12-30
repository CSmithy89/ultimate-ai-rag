"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import { X, CheckCircle2, XCircle, AlertTriangle, Info } from "lucide-react";
import { useToast, type Toast, type ToastVariant } from "@/hooks/use-toast";

/**
 * Get icon for toast variant.
 */
function getVariantIcon(variant: ToastVariant): React.ReactNode {
  const iconClass = "h-5 w-5";
  switch (variant) {
    case "destructive":
      return <XCircle className={cn(iconClass, "text-red-500")} />;
    case "success":
      return <CheckCircle2 className={cn(iconClass, "text-emerald-500")} />;
    case "warning":
      return <AlertTriangle className={cn(iconClass, "text-amber-500")} />;
    case "info":
      return <Info className={cn(iconClass, "text-blue-500")} />;
    default:
      return <CheckCircle2 className={cn(iconClass, "text-emerald-500")} />;
  }
}

/**
 * Get variant styles.
 */
function getVariantStyles(variant: ToastVariant): string {
  switch (variant) {
    case "destructive":
      return "bg-red-50 border-red-200 text-red-900";
    case "success":
      return "bg-emerald-50 border-emerald-200 text-emerald-900";
    case "warning":
      return "bg-amber-50 border-amber-200 text-amber-900";
    case "info":
      return "bg-blue-50 border-blue-200 text-blue-900";
    default:
      return "bg-white border-slate-200 text-slate-900";
  }
}

/**
 * ToastItem displays a single toast notification.
 */
const ToastItem = memo(function ToastItem({
  toast,
  onDismiss,
}: {
  toast: Toast;
  onDismiss: (id: string) => void;
}) {
  const isError = toast.variant === "destructive";

  return (
    <div
      className={cn(
        "pointer-events-auto flex w-full max-w-md rounded-lg shadow-lg",
        "border p-4 transition-all duration-300 ease-in-out",
        getVariantStyles(toast.variant)
      )}
      role="alert"
      aria-live="assertive"
    >
      {/* Icon */}
      <div className="flex-shrink-0 mr-3">
        {getVariantIcon(toast.variant)}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium">{toast.title}</p>
        {toast.description && (
          <p
            className={cn(
              "mt-1 text-sm",
              isError ? "text-red-700" : "text-slate-600"
            )}
          >
            {toast.description}
          </p>
        )}
      </div>

      {/* Dismiss button */}
      <button
        type="button"
        onClick={() => onDismiss(toast.id)}
        className={cn(
          "flex-shrink-0 ml-4 inline-flex rounded-md p-1.5",
          "focus:outline-none focus:ring-2",
          isError
            ? "text-red-500 hover:bg-red-100 focus:ring-red-400"
            : "text-slate-400 hover:bg-slate-100 focus:ring-slate-400"
        )}
        aria-label="Dismiss"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
});

/**
 * Toaster renders toast notifications in a fixed position.
 *
 * Story 6-5: Frontend Actions
 *
 * Place this component at the root of your app:
 * ```tsx
 * <CopilotProvider>
 *   {children}
 *   <Toaster />
 * </CopilotProvider>
 * ```
 */
export function Toaster() {
  const { toasts, dismiss } = useToast();

  if (toasts.length === 0) return null;

  return (
    <div
      className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none"
      aria-label="Notifications"
    >
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={dismiss} />
      ))}
    </div>
  );
}

export default Toaster;
