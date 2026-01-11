/**
 * Open-JSON-UI Alert Component
 *
 * Renders alert/notification boxes with variant styling.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import { sanitizeContent, sanitizeToPlainText } from "@/lib/open-json-ui/sanitize";
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Info,
} from "lucide-react";
import type { AlertComponent as AlertComponentType } from "@/lib/open-json-ui/schema";

/**
 * Props for AlertComponent.
 */
export interface AlertComponentProps {
  /** Alert component data */
  component: AlertComponentType;
}

/**
 * Style and icon mappings for alert variants.
 */
const alertVariants: Record<
  string,
  { containerClass: string; icon: React.ReactNode }
> = {
  default: {
    containerClass: "bg-slate-50 border-slate-200 text-slate-900",
    icon: <Info className="h-4 w-4 text-slate-600" />,
  },
  destructive: {
    containerClass: "bg-red-50 border-red-200 text-red-900",
    icon: <AlertCircle className="h-4 w-4 text-red-600" />,
  },
  warning: {
    containerClass: "bg-amber-50 border-amber-200 text-amber-900",
    icon: <AlertTriangle className="h-4 w-4 text-amber-600" />,
  },
  success: {
    containerClass: "bg-emerald-50 border-emerald-200 text-emerald-900",
    icon: <CheckCircle2 className="h-4 w-4 text-emerald-600" />,
  },
};

/**
 * AlertComponent renders notification boxes similar to shadcn/ui Alert.
 *
 * @param props - Component props
 * @returns Rendered alert element
 *
 * @example
 * ```tsx
 * <AlertComponent
 *   component={{
 *     type: "alert",
 *     title: "Success",
 *     description: "Your changes have been saved.",
 *     variant: "success"
 *   }}
 * />
 * ```
 */
export const AlertComponent = memo(function AlertComponent({
  component,
}: AlertComponentProps) {
  const variant = component.variant ?? "default";
  const variantConfig = alertVariants[variant] ?? alertVariants.default;
  const sanitizedTitle = component.title
    ? sanitizeToPlainText(component.title)
    : undefined;
  const sanitizedDescription = sanitizeContent(component.description);

  return (
    <div
      className={cn(
        "relative w-full rounded-lg border p-4",
        "[&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4",
        "[&>svg~*]:pl-7",
        variantConfig.containerClass
      )}
      data-testid="open-json-ui-alert"
      role="alert"
    >
      {variantConfig.icon}
      <div className="space-y-1">
        {sanitizedTitle && (
          <h5 className="mb-1 font-medium leading-none tracking-tight">
            {sanitizedTitle}
          </h5>
        )}
        <div
          className="text-sm [&_p]:leading-relaxed"
          dangerouslySetInnerHTML={{ __html: sanitizedDescription }}
        />
      </div>
    </div>
  );
});

export default AlertComponent;
