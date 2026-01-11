/**
 * Open-JSON-UI Schema Definitions
 *
 * Zod schemas for all Open-JSON-UI component types.
 * Provides type-safe validation for declarative UI payloads from agents.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 *
 * @version 1.0.0-internal
 */

import { z } from "zod";

/**
 * Text component for displaying plain text with optional styling.
 */
export const TextComponentSchema = z.object({
  type: z.literal("text"),
  /** Text content to display */
  content: z.string(),
  /** Optional text style variant */
  style: z.enum(["normal", "muted", "error", "success"]).optional(),
});

/**
 * Heading component for section headers (h1-h6).
 */
export const HeadingComponentSchema = z.object({
  type: z.literal("heading"),
  /** Heading level (1-6) */
  level: z.number().min(1).max(6),
  /** Heading text content */
  content: z.string(),
});

/**
 * Code component for syntax-highlighted code blocks.
 */
export const CodeComponentSchema = z.object({
  type: z.literal("code"),
  /** Code content */
  content: z.string(),
  /** Optional programming language for syntax highlighting */
  language: z.string().optional(),
});

/**
 * Table component for tabular data display.
 */
export const TableComponentSchema = z.object({
  type: z.literal("table"),
  /** Column headers */
  headers: z.array(z.string()),
  /** Table rows (array of cell values) */
  rows: z.array(z.array(z.string())),
  /** Optional table caption */
  caption: z.string().optional(),
});

/**
 * Image component for displaying images with alt text.
 */
export const ImageComponentSchema = z.object({
  type: z.literal("image"),
  /** Image source URL */
  src: z.string().url(),
  /** Alt text for accessibility */
  alt: z.string(),
  /** Optional image width */
  width: z.number().optional(),
  /** Optional image height */
  height: z.number().optional(),
});

/**
 * Button component for interactive actions.
 */
export const ButtonComponentSchema = z.object({
  type: z.literal("button"),
  /** Button label text */
  label: z.string(),
  /** Action identifier sent to onAction callback */
  action: z.string(),
  /** Optional button variant */
  variant: z
    .enum(["default", "destructive", "outline", "ghost", "secondary"])
    .optional(),
});

/**
 * List component for ordered/unordered lists.
 */
export const ListComponentSchema = z.object({
  type: z.literal("list"),
  /** List item contents */
  items: z.array(z.string()),
  /** Whether the list is ordered (numbered) */
  ordered: z.boolean().default(false),
});

/**
 * Link component for hyperlinks.
 */
export const LinkComponentSchema = z.object({
  type: z.literal("link"),
  /** Link text */
  text: z.string(),
  /** Link URL */
  href: z.string().url(),
  /** Optional target for link opening behavior */
  target: z.enum(["_self", "_blank"]).optional(),
});

/**
 * Divider component for visual separation.
 */
export const DividerComponentSchema = z.object({
  type: z.literal("divider"),
});

/**
 * Progress component for progress bars.
 */
export const ProgressComponentSchema = z.object({
  type: z.literal("progress"),
  /** Progress value (0-100) */
  value: z.number().min(0).max(100),
  /** Optional label for the progress bar */
  label: z.string().optional(),
});

/**
 * Alert component for notifications and messages.
 */
export const AlertComponentSchema = z.object({
  type: z.literal("alert"),
  /** Optional alert title */
  title: z.string().optional(),
  /** Alert description/message */
  description: z.string(),
  /** Optional alert variant */
  variant: z.enum(["default", "destructive", "warning", "success"]).optional(),
});

/**
 * Discriminated union of all Open-JSON-UI component types.
 */
export const OpenJSONUIComponentSchema = z.discriminatedUnion("type", [
  TextComponentSchema,
  HeadingComponentSchema,
  CodeComponentSchema,
  TableComponentSchema,
  ImageComponentSchema,
  ButtonComponentSchema,
  ListComponentSchema,
  LinkComponentSchema,
  DividerComponentSchema,
  ProgressComponentSchema,
  AlertComponentSchema,
]);

/**
 * Full Open-JSON-UI payload wrapper.
 */
export const OpenJSONUIPayloadSchema = z.object({
  /** Payload type identifier */
  type: z.literal("open_json_ui"),
  /** Array of UI components to render */
  components: z.array(OpenJSONUIComponentSchema),
});

// Type exports
export type TextComponent = z.infer<typeof TextComponentSchema>;
export type HeadingComponent = z.infer<typeof HeadingComponentSchema>;
export type CodeComponent = z.infer<typeof CodeComponentSchema>;
export type TableComponent = z.infer<typeof TableComponentSchema>;
export type ImageComponent = z.infer<typeof ImageComponentSchema>;
export type ButtonComponent = z.infer<typeof ButtonComponentSchema>;
export type ListComponent = z.infer<typeof ListComponentSchema>;
export type LinkComponent = z.infer<typeof LinkComponentSchema>;
export type DividerComponent = z.infer<typeof DividerComponentSchema>;
export type ProgressComponent = z.infer<typeof ProgressComponentSchema>;
export type AlertComponent = z.infer<typeof AlertComponentSchema>;
export type OpenJSONUIComponent = z.infer<typeof OpenJSONUIComponentSchema>;
export type OpenJSONUIPayload = z.infer<typeof OpenJSONUIPayloadSchema>;

/**
 * Validates an Open-JSON-UI payload and returns parsed result with error details.
 *
 * @param payload - Unknown payload to validate
 * @returns Parsed payload on success, or error details on failure
 */
export function validatePayload(payload: unknown): {
  success: true;
  data: OpenJSONUIPayload;
} | {
  success: false;
  error: z.ZodError;
  message: string;
} {
  const result = OpenJSONUIPayloadSchema.safeParse(payload);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    error: result.error,
    message: result.error.issues
      .map((e) => `${e.path.map(String).join(".")}: ${e.message}`)
      .join("; "),
  };
}

/**
 * Type guard for checking if a component is of a specific type.
 */
export function isComponentType<T extends OpenJSONUIComponent["type"]>(
  component: OpenJSONUIComponent,
  type: T
): component is Extract<OpenJSONUIComponent, { type: T }> {
  return component.type === type;
}
