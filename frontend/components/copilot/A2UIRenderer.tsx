"use client";

/**
 * A2UI Widget Renderer Component
 *
 * Story 21-D2: Implement A2UI Widget Renderer
 *
 * This component renders A2UI widgets emitted by the backend agent
 * via STATE_SNAPSHOT events. It maps widget types to React components
 * following the project's Tailwind CSS design patterns.
 */

import { useCoAgentStateRender } from "@copilotkit/react-core";
import { clsx } from "clsx";
import {
  FileText,
  Table as TableIcon,
  FormInput,
  BarChart3,
  Image as ImageIcon,
  List,
  AlertCircle,
  ExternalLink,
  Check,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// ============================================
// TYPES
// ============================================

type A2UIWidgetType = "card" | "table" | "form" | "chart" | "image" | "list";

interface A2UIAction {
  label: string;
  action: string;
  variant?: "primary" | "secondary" | "destructive";
  disabled?: boolean;
}

interface A2UIWidget {
  type: A2UIWidgetType;
  properties: Record<string, unknown>;
  id?: string;
}

interface A2UIState {
  a2ui_widgets?: A2UIWidget[];
}

interface CardProperties {
  title: string;
  content: string;
  subtitle?: string;
  actions?: A2UIAction[];
  footer?: string;
  imageUrl?: string;
}

interface TableProperties {
  headers: string[];
  rows: unknown[][];
  caption?: string;
  sortable?: boolean;
}

interface FormField {
  name: string;
  label: string;
  type: "text" | "number" | "email" | "password" | "textarea" | "select" | "checkbox";
  placeholder?: string;
  required?: boolean;
  options?: Array<{ value: string; label: string }>;
  defaultValue?: unknown;
  /** Minimum length for text inputs */
  minLength?: number;
  /** Maximum length for text inputs */
  maxLength?: number;
  /** Minimum value for number inputs */
  min?: number;
  /** Maximum value for number inputs */
  max?: number;
  /** Custom regex pattern for validation */
  pattern?: string;
}

interface FormProperties {
  title: string;
  fields: FormField[];
  submitLabel?: string;
  submitAction?: string;
  description?: string;
}

interface ChartProperties {
  chartType: "bar" | "line" | "pie" | "area" | "scatter";
  data: Array<Record<string, unknown>>;
  xKey: string;
  yKey: string;
  title?: string;
  xLabel?: string;
  yLabel?: string;
}

interface ImageProperties {
  url: string;
  alt: string;
  caption?: string;
  width?: number;
  height?: number;
}

interface ListItem {
  text: string;
  icon?: string;
  description?: string;
  badge?: string;
  href?: string;
}

interface ListProperties {
  items: ListItem[];
  title?: string;
  ordered?: boolean;
  selectable?: boolean;
}

// ============================================
// ACTION HANDLER
// ============================================

interface A2UIRendererProps {
  /** Callback when an A2UI action is triggered */
  onAction?: (action: string, data?: Record<string, unknown>) => void;
  /** Callback when a form is submitted */
  onFormSubmit?: (action: string, formData: Record<string, unknown>) => void;
}

/**
 * A2UI Widget Renderer
 *
 * Renders A2UI widgets from agent state snapshots. Place this component
 * within your CopilotKit context to enable A2UI widget rendering.
 *
 * @example
 * ```tsx
 * <CopilotSidebar>
 *   <A2UIRenderer
 *     onAction={(action) => console.log("Action:", action)}
 *     onFormSubmit={(action, data) => handleForm(action, data)}
 *   />
 * </CopilotSidebar>
 * ```
 */
export function A2UIRenderer({ onAction, onFormSubmit }: A2UIRendererProps) {
  useCoAgentStateRender<A2UIState>({
    name: "orchestrator",
    render: ({ state }) => {
      if (!state?.a2ui_widgets?.length) return null;

      return (
        <div className="space-y-3 my-3">
          {state.a2ui_widgets.map((widget, idx) => (
            <A2UIWidgetComponent
              key={widget.id || idx}
              widget={widget}
              onAction={onAction}
              onFormSubmit={onFormSubmit}
            />
          ))}
        </div>
      );
    },
  });

  return null;
}

// ============================================
// WIDGET DISPATCHER
// ============================================

interface WidgetComponentProps {
  widget: A2UIWidget;
  onAction?: (action: string, data?: Record<string, unknown>) => void;
  onFormSubmit?: (action: string, formData: Record<string, unknown>) => void;
}

function A2UIWidgetComponent({ widget, onAction, onFormSubmit }: WidgetComponentProps) {
  const props = widget.properties as unknown;

  switch (widget.type) {
    case "card":
      return <A2UICard {...(props as CardProperties)} onAction={onAction} />;
    case "table":
      return <A2UITable {...(props as TableProperties)} />;
    case "form":
      return <A2UIForm {...(props as FormProperties)} onSubmit={onFormSubmit} />;
    case "chart":
      return <A2UIChart {...(props as ChartProperties)} />;
    case "image":
      return <A2UIImage {...(props as ImageProperties)} />;
    case "list":
      return <A2UIList {...(props as ListProperties)} onAction={onAction} />;
    default:
      return <A2UIFallback widget={widget} />;
  }
}

// ============================================
// CARD WIDGET
// ============================================

function A2UICard({
  title,
  content,
  subtitle,
  actions,
  footer,
  imageUrl,
  onAction,
}: CardProperties & { onAction?: (action: string, data?: Record<string, unknown>) => void }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm overflow-hidden">
      {imageUrl && (
        <div className="w-full h-32 bg-gray-100">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={imageUrl} alt={title} className="w-full h-full object-cover" />
        </div>
      )}
      <div className="p-4">
        <div className="flex items-start gap-2">
          <FileText className="w-5 h-5 text-gray-500 flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-gray-900">{title}</h3>
            {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
          </div>
        </div>
        <div className="mt-3 text-sm text-gray-700 prose prose-sm max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
        {actions && actions.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {actions.map((action, idx) => (
              <button
                key={idx}
                onClick={() => onAction?.(action.action)}
                disabled={action.disabled}
                className={clsx(
                  "px-3 py-1.5 text-sm font-medium rounded-md transition-colors",
                  action.variant === "primary" &&
                    "bg-blue-600 text-white hover:bg-blue-700 disabled:bg-blue-300",
                  action.variant === "destructive" &&
                    "bg-red-600 text-white hover:bg-red-700 disabled:bg-red-300",
                  (!action.variant || action.variant === "secondary") &&
                    "bg-gray-100 text-gray-700 hover:bg-gray-200 disabled:bg-gray-50 disabled:text-gray-400"
                )}
              >
                {action.label}
              </button>
            ))}
          </div>
        )}
        {footer && <div className="mt-4 pt-3 border-t text-xs text-gray-500">{footer}</div>}
      </div>
    </div>
  );
}

// ============================================
// TABLE WIDGET
// ============================================

function A2UITable({ headers, rows, caption }: TableProperties) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b bg-gray-50">
        <TableIcon className="w-4 h-4 text-gray-500" />
        {caption && <span className="text-sm font-medium text-gray-700">{caption}</span>}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              {headers.map((header, idx) => (
                <th key={idx} className="px-4 py-2 text-left font-medium text-gray-600">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {rows.map((row, rowIdx) => (
              <tr key={rowIdx} className="hover:bg-gray-50">
                {(row as unknown[]).map((cell, cellIdx) => (
                  <td key={cellIdx} className="px-4 py-2 text-gray-700">
                    {String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ============================================
// FORM WIDGET
// ============================================

function A2UIForm({
  title,
  fields,
  submitLabel = "Submit",
  submitAction = "form_submit",
  description,
  onSubmit,
}: FormProperties & {
  onSubmit?: (action: string, formData: Record<string, unknown>) => void;
}) {
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const data: Record<string, unknown> = {};
    formData.forEach((value, key) => {
      data[key] = value;
    });
    onSubmit?.(submitAction, data);
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b bg-gray-50">
        <FormInput className="w-4 h-4 text-gray-500" />
        <span className="text-sm font-medium text-gray-700">{title}</span>
      </div>
      <form onSubmit={handleSubmit} className="p-4 space-y-4">
        {description && <p className="text-sm text-gray-500">{description}</p>}
        {fields.map((field) => (
          <div key={field.name} className="space-y-1">
            <label htmlFor={field.name} className="block text-sm font-medium text-gray-700">
              {field.label}
              {field.required && <span className="text-red-500 ml-1">*</span>}
            </label>
            {field.type === "textarea" ? (
              <textarea
                id={field.name}
                name={field.name}
                placeholder={field.placeholder}
                required={field.required}
                defaultValue={String(field.defaultValue || "")}
                minLength={field.minLength}
                maxLength={field.maxLength}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 invalid:border-red-500 invalid:focus:ring-red-500"
                rows={3}
              />
            ) : field.type === "select" ? (
              <select
                id={field.name}
                name={field.name}
                required={field.required}
                defaultValue={String(field.defaultValue || "")}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select...</option>
                {field.options?.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            ) : field.type === "checkbox" ? (
              <input
                id={field.name}
                name={field.name}
                type="checkbox"
                defaultChecked={Boolean(field.defaultValue)}
                className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
            ) : (
              <input
                id={field.name}
                name={field.name}
                type={field.type}
                placeholder={field.placeholder}
                required={field.required}
                defaultValue={String(field.defaultValue || "")}
                // Validation attributes based on field type
                minLength={field.minLength}
                maxLength={field.maxLength}
                min={field.type === "number" ? field.min : undefined}
                max={field.type === "number" ? field.max : undefined}
                // Use custom pattern if provided, otherwise use defaults for email
                pattern={
                  field.pattern ||
                  (field.type === "email"
                    ? "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
                    : undefined)
                }
                // Improve mobile experience with inputMode
                inputMode={
                  field.type === "number"
                    ? "numeric"
                    : field.type === "email"
                      ? "email"
                      : undefined
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 invalid:border-red-500 invalid:focus:ring-red-500"
              />
            )}
          </div>
        ))}
        <button
          type="submit"
          className="w-full px-4 py-2 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 transition-colors"
        >
          {submitLabel}
        </button>
      </form>
    </div>
  );
}

// ============================================
// CHART WIDGET (Fallback without Recharts)
// ============================================

function A2UIChart({ chartType, data, xKey, yKey, title, xLabel, yLabel }: ChartProperties) {
  // Simple bar chart visualization using CSS (fallback without Recharts)
  const maxValue = Math.max(...data.map((d) => Number(d[yKey]) || 0));

  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b bg-gray-50">
        <BarChart3 className="w-4 h-4 text-gray-500" />
        <span className="text-sm font-medium text-gray-700">{title || `${chartType} chart`}</span>
      </div>
      <div className="p-4">
        {chartType === "bar" && maxValue > 0 ? (
          <div className="space-y-2">
            {data.map((item, idx) => {
              const value = Number(item[yKey]) || 0;
              const percentage = (value / maxValue) * 100;
              return (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-xs text-gray-600">
                    <span>{String(item[xKey])}</span>
                    <span>{value}</span>
                  </div>
                  <div className="h-4 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
            <div className="flex justify-between text-xs text-gray-400 mt-2">
              {xLabel && <span>{xLabel}</span>}
              {yLabel && <span>{yLabel}</span>}
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-500">
            <p className="mb-2">Chart data ({chartType}):</p>
            <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto">
              {JSON.stringify(data, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================
// IMAGE WIDGET
// ============================================

function A2UIImage({ url, alt, caption, width, height }: ImageProperties) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b bg-gray-50">
        <ImageIcon className="w-4 h-4 text-gray-500" />
        <span className="text-sm font-medium text-gray-700 truncate">{alt}</span>
      </div>
      <div className="p-2">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={url}
          alt={alt}
          width={width}
          height={height}
          className="max-w-full h-auto rounded"
          loading="lazy"
        />
        {caption && <p className="mt-2 text-xs text-gray-500 text-center">{caption}</p>}
      </div>
    </div>
  );
}

// ============================================
// LIST WIDGET
// ============================================

function A2UIList({
  items,
  title,
  ordered,
  selectable,
  onAction,
}: ListProperties & { onAction?: (action: string, data?: Record<string, unknown>) => void }) {
  const ListTag = ordered ? "ol" : "ul";

  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b bg-gray-50">
        <List className="w-4 h-4 text-gray-500" />
        {title && <span className="text-sm font-medium text-gray-700">{title}</span>}
      </div>
      <ListTag
        className={clsx(
          "divide-y divide-gray-100",
          ordered && "list-decimal list-inside"
        )}
      >
        {items.map((item, idx) => (
          <li
            key={idx}
            className={clsx(
              "px-4 py-2 flex items-start gap-3",
              selectable && "hover:bg-gray-50 cursor-pointer",
              item.href && "hover:bg-blue-50"
            )}
            onClick={() => {
              if (selectable) {
                onAction?.("list_item_select", { index: idx, item });
              }
            }}
          >
            {!ordered && (
              <Check className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
            )}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-700">{item.text}</span>
                {item.badge && (
                  <span className="px-1.5 py-0.5 text-xs bg-blue-100 text-blue-700 rounded">
                    {item.badge}
                  </span>
                )}
              </div>
              {item.description && (
                <p className="text-xs text-gray-500 mt-0.5">{item.description}</p>
              )}
            </div>
            {item.href && (
              <a
                href={item.href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 hover:text-blue-600"
                onClick={(e) => e.stopPropagation()}
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            )}
          </li>
        ))}
      </ListTag>
    </div>
  );
}

// ============================================
// FALLBACK WIDGET
// ============================================

function A2UIFallback({ widget }: { widget: A2UIWidget }) {
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-amber-200 bg-amber-100">
        <AlertCircle className="w-4 h-4 text-amber-600" />
        <span className="text-sm font-medium text-amber-700">
          Unsupported widget: {widget.type}
        </span>
      </div>
      <div className="p-4">
        <pre className="text-xs text-amber-800 bg-amber-100/50 p-2 rounded overflow-x-auto">
          {JSON.stringify(widget.properties, null, 2)}
        </pre>
      </div>
    </div>
  );
}

export default A2UIRenderer;
