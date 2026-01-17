"use client";

/**
 * CustomEventRenderer - AG-UI Enhancement
 *
 * Renders UI widgets from AG-UI CUSTOM events.
 * Supports the "render_ui" event name for declarative widget rendering.
 *
 * @example
 * ```tsx
 * // In a chat message renderer
 * {events.filter(isCustomEvent).map((event, i) => (
 *   <CustomEventRenderer key={i} event={event} />
 * ))}
 * ```
 */

import { useEffect, useState, memo, type ReactElement } from "react";
import {
  isRenderUIEvent,
  getWidget,
  type WidgetType,
  type RenderUIEventValue,
} from "@/lib/widget-registry";
import { initializeWidgetsSync, areWidgetsInitialized } from "@/lib/widget-init";

/**
 * AG-UI CUSTOM event structure.
 */
export interface CustomEvent {
  type: "CUSTOM";
  name: string;
  value: unknown;
}

/**
 * Props for CustomEventRenderer.
 */
interface CustomEventRendererProps {
  /** The CUSTOM event to render */
  event: CustomEvent;
  /** Callback when widget interaction occurs */
  onInteraction?: (widgetType: string, action: string, data: unknown) => void;
  /** Additional className for the wrapper */
  className?: string;
}

/**
 * Check if an event is a CUSTOM event.
 */
export function isCustomEvent(event: unknown): event is CustomEvent {
  return (
    typeof event === "object" &&
    event !== null &&
    (event as CustomEvent).type === "CUSTOM" &&
    typeof (event as CustomEvent).name === "string"
  );
}

/**
 * Render a widget from an AG-UI CUSTOM event.
 */
function renderWidgetFromEvent(value: RenderUIEventValue): ReactElement | null {
  const Widget = getWidget(value.type as WidgetType);

  if (!Widget) {
    console.warn(`Widget type "${value.type}" not registered`);
    return null;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return <Widget {...(value.props as any)} />;
}

/**
 * CustomEventRenderer handles CUSTOM events from the AG-UI protocol.
 *
 * Currently supports:
 * - "render_ui": Renders a widget from the widget registry
 *
 * @example
 * ```tsx
 * // Backend sends:
 * yield CustomEvent(
 *   name="render_ui",
 *   value={ "type": "step_progress", "props": { "steps": [...] } }
 * )
 *
 * // Frontend renders:
 * <CustomEventRenderer event={customEvent} />
 * ```
 */
function CustomEventRendererComponent({
  event,
  onInteraction,
  className,
}: CustomEventRendererProps): ReactElement | null {
  const [isInitialized, setIsInitialized] = useState(areWidgetsInitialized());

  // Ensure widgets are initialized
  useEffect(() => {
    if (!isInitialized) {
      initializeWidgetsSync();
      setIsInitialized(true);
    }
  }, [isInitialized]);

  // Handle "render_ui" events
  if (event.name === "render_ui" && isRenderUIEvent(event.value)) {
    // Store typed value for use in nested scopes
    const renderUIValue = event.value as RenderUIEventValue;
    const widget = renderWidgetFromEvent(renderUIValue);

    if (!widget) {
      return null;
    }

    // Wrap widget with interaction handler if provided
    if (onInteraction) {
      const widgetType = renderUIValue.type;
      const originalProps = renderUIValue.props as Record<string, unknown>;
      const wrappedProps = {
        ...originalProps,
        onRespond: (action: string, data: unknown) => {
          onInteraction(widgetType, action, data);
          // Also call original onRespond if present
          const originalOnRespond = originalProps?.onRespond;
          if (typeof originalOnRespond === "function") {
            originalOnRespond(action, data);
          }
        },
        onRowClick: (row: unknown, index: number) => {
          onInteraction(widgetType, "row_click", { row, index });
          const originalOnRowClick = originalProps?.onRowClick;
          if (typeof originalOnRowClick === "function") {
            originalOnRowClick(row, index);
          }
        },
      };

      const Widget = getWidget(renderUIValue.type as WidgetType);
      if (Widget) {
        return (
          <div className={className}>
            {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
            <Widget {...(wrappedProps as any)} />
          </div>
        );
      }
    }

    return className ? <div className={className}>{widget}</div> : widget;
  }

  // Unknown event type - render nothing but log for debugging
  if (process.env.NODE_ENV === "development") {
    console.debug(`CustomEventRenderer: Unhandled event name "${event.name}"`);
  }

  return null;
}

/**
 * Memoized CustomEventRenderer to prevent unnecessary re-renders.
 */
export const CustomEventRenderer = memo(CustomEventRendererComponent);

export default CustomEventRenderer;
