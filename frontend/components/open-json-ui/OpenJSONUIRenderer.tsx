/**
 * Open-JSON-UI Renderer Component
 *
 * Main renderer component that dispatches Open-JSON-UI payloads to
 * the appropriate component renderers with proper validation.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 *
 * @example
 * ```tsx
 * import { OpenJSONUIRenderer } from "@/components/open-json-ui";
 *
 * function AgentResponse({ payload }) {
 *   return (
 *     <OpenJSONUIRenderer
 *       payload={payload}
 *       onAction={(action) => handleAction(action)}
 *     />
 *   );
 * }
 * ```
 */

"use client";

import { memo, useMemo } from "react";
import { cn } from "@/lib/utils";
import { AlertCircle } from "lucide-react";
import {
  OpenJSONUIPayloadSchema,
  type OpenJSONUIComponent,
  type OpenJSONUIPayload,
} from "@/lib/open-json-ui/schema";

// Component imports
import { TextComponent } from "./TextComponent";
import { HeadingComponent } from "./HeadingComponent";
import { CodeComponent } from "./CodeComponent";
import { ListComponent } from "./ListComponent";
import { TableComponent } from "./TableComponent";
import { ImageComponent } from "./ImageComponent";
import { ButtonComponent } from "./ButtonComponent";
import { DividerComponent } from "./DividerComponent";
import { LinkComponent } from "./LinkComponent";
import { ProgressComponent } from "./ProgressComponent";
import { AlertComponent } from "./AlertComponent";

/**
 * Props for OpenJSONUIRenderer.
 */
export interface OpenJSONUIRendererProps {
  /** Open-JSON-UI payload to render */
  payload: OpenJSONUIPayload;
  /** Callback when interactive elements trigger actions */
  onAction?: (action: string) => void;
  /** Additional CSS classes for the container */
  className?: string;
}

/**
 * Props for the individual component dispatcher.
 */
interface ComponentDispatcherProps {
  component: OpenJSONUIComponent;
  onAction?: (action: string) => void;
}

/**
 * Dispatches a single component to its renderer.
 */
const ComponentDispatcher = memo(function ComponentDispatcher({
  component,
  onAction,
}: ComponentDispatcherProps) {
  switch (component.type) {
    case "text":
      return <TextComponent component={component} />;
    case "heading":
      return <HeadingComponent component={component} />;
    case "code":
      return <CodeComponent component={component} />;
    case "list":
      return <ListComponent component={component} />;
    case "table":
      return <TableComponent component={component} />;
    case "image":
      return <ImageComponent component={component} />;
    case "button":
      return <ButtonComponent component={component} onAction={onAction} />;
    case "divider":
      return <DividerComponent />;
    case "link":
      return <LinkComponent component={component} />;
    case "progress":
      return <ProgressComponent component={component} />;
    case "alert":
      return <AlertComponent component={component} />;
    default:
      // Fallback for unsupported types
      return (
        <FallbackComponent type={(component as { type: string }).type} />
      );
  }
});

/**
 * Fallback component for unsupported types.
 */
interface FallbackComponentProps {
  type: string;
}

const FallbackComponent = memo(function FallbackComponent({
  type,
}: FallbackComponentProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 p-3 rounded-lg",
        "bg-amber-50 border border-amber-200 text-amber-800"
      )}
      data-testid="open-json-ui-fallback"
      role="alert"
    >
      <AlertCircle className="h-4 w-4 flex-shrink-0" aria-hidden="true" />
      <span className="text-sm">
        Unsupported component type:{" "}
        <code className="bg-amber-100 px-1 rounded text-xs">{type}</code>
      </span>
    </div>
  );
});

/**
 * OpenJSONUIRenderer validates and renders Open-JSON-UI payloads.
 *
 * Features:
 * - Zod schema validation before rendering
 * - Component dispatch to appropriate renderers
 * - Fallback for unsupported types
 * - Action callback for interactive elements
 *
 * @param props - Renderer props
 * @returns Rendered UI components or error state
 */
export const OpenJSONUIRenderer = memo(function OpenJSONUIRenderer({
  payload,
  onAction,
  className,
}: OpenJSONUIRendererProps) {
  // Validate payload
  const validationResult = useMemo(() => {
    return OpenJSONUIPayloadSchema.safeParse(payload);
  }, [payload]);

  // Show validation error
  if (!validationResult.success) {
    const errorMessage = validationResult.error.issues
      .map((e) => `${e.path.map(String).join(".")}: ${e.message}`)
      .join("; ");

    console.warn("Open-JSON-UI: Invalid payload", validationResult.error);

    return (
      <div
        className={cn(
          "flex items-start gap-2 p-4 rounded-lg",
          "bg-red-50 border border-red-200 text-red-800",
          className
        )}
        data-testid="open-json-ui-error"
        role="alert"
      >
        <AlertCircle
          className="h-5 w-5 flex-shrink-0 mt-0.5"
          aria-hidden="true"
        />
        <div>
          <h4 className="text-sm font-medium">Invalid UI payload</h4>
          <p className="text-sm mt-1 text-red-700">{errorMessage}</p>
        </div>
      </div>
    );
  }

  // Render validated components
  return (
    <div
      className={cn("space-y-3 my-2", className)}
      data-testid="open-json-ui-renderer"
    >
      {validationResult.data.components.map((component, index) => (
        <ComponentDispatcher
          key={`${component.type}-${index}`}
          component={component}
          onAction={onAction}
        />
      ))}
    </div>
  );
});

export default OpenJSONUIRenderer;
