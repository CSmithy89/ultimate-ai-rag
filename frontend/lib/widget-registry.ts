/**
 * Widget Registry - AG-UI Enhancement
 *
 * Provides a centralized registry for declarative UI widgets that can be
 * rendered from backend CUSTOM events. This enables backend-driven UI
 * composition without tight coupling between frontend and backend code.
 *
 * Usage:
 * ```typescript
 * // Backend sends CUSTOM event:
 * yield CustomEvent(
 *   name="render_ui",
 *   value={
 *     "type": "step_progress",
 *     "props": { "steps": [...], "currentStep": 2 }
 *   }
 * )
 *
 * // Frontend renders via registry:
 * const Widget = widgetRegistry["step_progress"];
 * return <Widget {...props} />;
 * ```
 */

import type { ComponentType } from "react";

// ============================================
// WIDGET PROP TYPES
// ============================================

/**
 * Props for StepProgressWidget.
 */
export interface StepProgressWidgetProps {
  /** Array of step objects */
  steps: Array<{
    step: string;
    status: "pending" | "in_progress" | "completed";
    details?: string;
  }>;
  /** Current active step index (0-based) */
  currentStep?: number;
  /** Optional title for the progress section */
  title?: string;
  /** Whether to show step details */
  showDetails?: boolean;
}

/**
 * Props for ApprovalDialogWidget.
 */
export interface ApprovalDialogWidgetProps {
  /** Dialog title */
  title: string;
  /** Description or instructions */
  description?: string;
  /** Items to approve/reject */
  items: Array<{
    id: string;
    label: string;
    description?: string;
    metadata?: Record<string, unknown>;
  }>;
  /** Available actions */
  actions: Array<"approve" | "reject" | "skip">;
  /** Callback when user responds */
  onRespond?: (action: string, selectedIds: string[]) => void;
  /** Whether multiple items can be selected */
  multiSelect?: boolean;
}

/**
 * Props for DataTableWidget.
 */
export interface DataTableWidgetProps {
  /** Column definitions */
  columns: Array<{
    key: string;
    header: string;
    width?: string | number;
    align?: "left" | "center" | "right";
  }>;
  /** Row data */
  rows: Array<Record<string, unknown>>;
  /** Optional title */
  title?: string;
  /** Whether to show row numbers */
  showRowNumbers?: boolean;
  /** Max height before scrolling */
  maxHeight?: string | number;
  /** Callback when row is clicked */
  onRowClick?: (row: Record<string, unknown>, index: number) => void;
}

/**
 * Props for StatusIndicatorWidget.
 */
export interface StatusIndicatorWidgetProps {
  /** Current status */
  status: "idle" | "loading" | "success" | "error" | "warning";
  /** Status message */
  message?: string;
  /** Optional details */
  details?: string;
  /** Whether to show animated indicator */
  animated?: boolean;
}

/**
 * Props for ChartWidget (basic bar/line/pie charts).
 */
export interface ChartWidgetProps {
  /** Chart type */
  type: "bar" | "line" | "pie";
  /** Chart data */
  data: Array<{
    label: string;
    value: number;
    color?: string;
  }>;
  /** Chart title */
  title?: string;
  /** Whether to show legend */
  showLegend?: boolean;
  /** Chart dimensions */
  width?: number;
  height?: number;
}

/**
 * Props for GraphViewWidget.
 */
export interface GraphViewWidgetProps {
  /** Graph nodes */
  nodes: Array<{
    id: string;
    label: string;
    type?: string;
    properties?: Record<string, unknown>;
  }>;
  /** Graph edges */
  edges: Array<{
    id: string;
    source: string;
    target: string;
    label?: string;
    type?: string;
  }>;
  /** Optional title */
  title?: string;
  /** Whether to enable zoom/pan */
  interactive?: boolean;
  /** Callback when node is clicked */
  onNodeClick?: (nodeId: string) => void;
}

/**
 * Props for FilterFormWidget.
 */
export interface FilterFormWidgetProps {
  /** Filter field definitions */
  fields: Array<{
    key: string;
    label: string;
    type: "text" | "select" | "date" | "number" | "checkbox";
    options?: Array<{ value: string; label: string }>;
    placeholder?: string;
    defaultValue?: unknown;
  }>;
  /** Current filter values */
  values?: Record<string, unknown>;
  /** Callback when filters change */
  onChange?: (values: Record<string, unknown>) => void;
  /** Callback when filters are submitted */
  onSubmit?: (values: Record<string, unknown>) => void;
}

/**
 * Props for MarkdownContentWidget.
 */
export interface MarkdownContentWidgetProps {
  /** Markdown content */
  content: string;
  /** Optional title */
  title?: string;
  /** Whether to allow copying */
  copyable?: boolean;
  /** Max height before scrolling */
  maxHeight?: string | number;
}

/**
 * Props for CodeBlockWidget.
 */
export interface CodeBlockWidgetProps {
  /** Code content */
  code: string;
  /** Programming language for syntax highlighting */
  language?: string;
  /** Optional title/filename */
  title?: string;
  /** Whether to show line numbers */
  showLineNumbers?: boolean;
  /** Whether to allow copying */
  copyable?: boolean;
}

// ============================================
// WIDGET TYPE MAP
// ============================================

/**
 * Map of widget types to their prop types.
 * Used for type-safe widget rendering.
 */
export interface WidgetPropsMap {
  step_progress: StepProgressWidgetProps;
  approval_dialog: ApprovalDialogWidgetProps;
  data_table: DataTableWidgetProps;
  status_indicator: StatusIndicatorWidgetProps;
  bar_chart: ChartWidgetProps;
  line_chart: ChartWidgetProps;
  pie_chart: ChartWidgetProps;
  graph_view: GraphViewWidgetProps;
  filter_form: FilterFormWidgetProps;
  markdown_content: MarkdownContentWidgetProps;
  code_block: CodeBlockWidgetProps;
}

/**
 * Widget type names.
 */
export type WidgetType = keyof WidgetPropsMap;

/**
 * Widget component type.
 */
export type WidgetComponent<T extends WidgetType> = ComponentType<WidgetPropsMap[T]>;

// ============================================
// WIDGET REGISTRY
// ============================================

/**
 * Registry of widget components.
 * Components are lazily loaded to minimize bundle size.
 */
export const widgetRegistry: Partial<Record<WidgetType, ComponentType<unknown>>> = {};

/**
 * Register a widget component in the registry.
 *
 * @param type - Widget type name
 * @param component - React component to render
 */
export function registerWidget<T extends WidgetType>(
  type: T,
  component: WidgetComponent<T>
): void {
  widgetRegistry[type] = component as ComponentType<unknown>;
}

/**
 * Get a widget component from the registry.
 *
 * @param type - Widget type name
 * @returns Component if registered, undefined otherwise
 */
export function getWidget<T extends WidgetType>(
  type: T
): WidgetComponent<T> | undefined {
  return widgetRegistry[type] as WidgetComponent<T> | undefined;
}

/**
 * Check if a widget type is registered.
 *
 * @param type - Widget type name
 * @returns true if registered
 */
export function hasWidget(type: string): type is WidgetType {
  return type in widgetRegistry;
}

/**
 * Get all registered widget types.
 *
 * @returns Array of registered widget type names
 */
export function getRegisteredWidgets(): WidgetType[] {
  return Object.keys(widgetRegistry) as WidgetType[];
}

// ============================================
// CUSTOM EVENT HANDLER
// ============================================

/**
 * Custom event value structure for widget rendering.
 */
export interface RenderUIEventValue {
  /** Widget type to render */
  type: WidgetType;
  /** Props to pass to the widget */
  props: Record<string, unknown>;
}

/**
 * Check if a value is a valid RenderUI event.
 *
 * @param value - Event value to check
 * @returns true if valid RenderUI event
 */
export function isRenderUIEvent(value: unknown): value is RenderUIEventValue {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const obj = value as Record<string, unknown>;
  return typeof obj.type === "string" && typeof obj.props === "object";
}

/**
 * Render a widget from a CUSTOM event value.
 *
 * @param value - RenderUI event value
 * @returns React element or null if widget not found
 */
export function renderWidget(value: RenderUIEventValue): React.ReactElement | null {
  const Widget = getWidget(value.type);

  if (!Widget) {
    console.warn(`Widget type "${value.type}" not registered`);
    return null;
  }

  // Use React.createElement to avoid JSX in .ts file
  const React = require("react");
  return React.createElement(Widget, value.props);
}

// ============================================
// DEFAULT EXPORTS
// ============================================

const widgetRegistryAPI = {
  widgetRegistry,
  registerWidget,
  getWidget,
  hasWidget,
  getRegisteredWidgets,
  isRenderUIEvent,
  renderWidget,
};

export default widgetRegistryAPI;
