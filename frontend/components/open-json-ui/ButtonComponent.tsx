/**
 * Open-JSON-UI Button Component
 *
 * Renders interactive buttons that trigger action callbacks.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

"use client";

import { memo, useCallback } from "react";
import { cn } from "@/lib/utils";
import { sanitizeToPlainText } from "@/lib/open-json-ui/sanitize";
import type { ButtonComponent as ButtonComponentType } from "@/lib/open-json-ui/schema";

/**
 * Props for ButtonComponent.
 */
export interface ButtonComponentProps {
  /** Button component data */
  component: ButtonComponentType;
  /** Callback when button is clicked */
  onAction?: (action: string) => void;
}

/**
 * Style variants for buttons, similar to shadcn/ui Button.
 */
const buttonVariants: Record<string, string> = {
  default: cn(
    "bg-slate-900 text-white hover:bg-slate-800",
    "focus-visible:ring-slate-950"
  ),
  destructive: cn(
    "bg-red-600 text-white hover:bg-red-700",
    "focus-visible:ring-red-600"
  ),
  outline: cn(
    "border border-slate-200 bg-white hover:bg-slate-100 hover:text-slate-900",
    "focus-visible:ring-slate-950"
  ),
  secondary: cn(
    "bg-slate-100 text-slate-900 hover:bg-slate-200",
    "focus-visible:ring-slate-950"
  ),
  ghost: cn(
    "hover:bg-slate-100 hover:text-slate-900",
    "focus-visible:ring-slate-950"
  ),
};

/**
 * ButtonComponent renders action buttons with variant styling.
 *
 * @param props - Component props
 * @returns Rendered button element
 *
 * @example
 * ```tsx
 * <ButtonComponent
 *   component={{
 *     type: "button",
 *     label: "Submit",
 *     action: "submit_form",
 *     variant: "default"
 *   }}
 *   onAction={(action) => console.log("Action:", action)}
 * />
 * ```
 */
export const ButtonComponent = memo(function ButtonComponent({
  component,
  onAction,
}: ButtonComponentProps) {
  const sanitizedLabel = sanitizeToPlainText(component.label);
  const variantClass =
    buttonVariants[component.variant ?? "default"] ?? buttonVariants.default;

  const handleClick = useCallback(() => {
    onAction?.(component.action);
  }, [component.action, onAction]);

  return (
    <button
      type="button"
      onClick={handleClick}
      className={cn(
        "inline-flex items-center justify-center gap-2",
        "whitespace-nowrap rounded-md text-sm font-medium",
        "ring-offset-white transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2",
        "disabled:pointer-events-none disabled:opacity-50",
        "h-9 px-4 py-2",
        variantClass
      )}
      data-testid="open-json-ui-button"
      data-action={component.action}
    >
      {sanitizedLabel}
    </button>
  );
});

export default ButtonComponent;
