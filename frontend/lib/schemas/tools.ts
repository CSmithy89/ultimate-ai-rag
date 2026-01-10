/**
 * Shared Zod schemas for frontend tools.
 *
 * Story 21-A1: Migrate to useFrontendTool Pattern
 *
 * These schemas define parameters for useFrontendTool hooks.
 * The .describe() annotations provide context for the AI agent.
 *
 * NOTE: CopilotKit 1.x uses Parameter[] format. These Zod schemas are provided
 * for type inference and validation. The corresponding Parameter[] definitions
 * are exported for use with useFrontendTool. When upgrading to CopilotKit 2.x,
 * the Zod schemas can be used directly.
 *
 * @see {@link https://docs.copilotkit.ai/reference/use-frontend-tool}
 */
import { z } from "zod";

// Re-export ExportFormat from types/copilot for consistency
import { ExportFormatSchema, SourceSchema } from "@/types/copilot";

/**
 * CopilotKit Parameter type for tool definitions.
 * Used by useFrontendTool in CopilotKit 1.x.
 */
export interface ToolParameter {
  name: string;
  type: "string" | "number" | "boolean" | "object" | "string[]" | "number[]";
  description: string;
  required: boolean;
  enum?: string[];
}

/**
 * Schema for save_to_workspace tool parameters.
 *
 * Saves AI response content to the user's workspace for later reference.
 */
export const SaveToWorkspaceSchema = z.object({
  content_id: z.string().describe("Unique ID of the content to save"),
  content_text: z.string().describe("The actual content/response text to save"),
  title: z.string().optional().describe("Optional title for the saved content"),
  query: z
    .string()
    .optional()
    .describe("Original query that generated this response"),
});

/** Inferred TypeScript type for save_to_workspace parameters */
export type SaveToWorkspaceParams = z.infer<typeof SaveToWorkspaceSchema>;

/**
 * Schema for export_content tool parameters.
 *
 * Exports AI response content in a specified format (markdown, pdf, or json).
 */
export const ExportContentSchema = z.object({
  content_id: z.string().describe("Unique ID of the content to export"),
  content_text: z
    .string()
    .describe("The actual content/response text to export"),
  format: ExportFormatSchema.describe("Export format: markdown, pdf, or json"),
  title: z.string().optional().describe("Optional title for the export"),
});

/** Inferred TypeScript type for export_content parameters */
export type ExportContentParams = z.infer<typeof ExportContentSchema>;

/**
 * Schema for share_content tool parameters.
 *
 * Generates a shareable link for AI response content.
 */
export const ShareContentSchema = z.object({
  content_id: z.string().describe("Unique ID of the content to share"),
  content_text: z.string().describe("The actual content/response text to share"),
  title: z
    .string()
    .optional()
    .describe("Optional title for the shared content"),
});

/** Inferred TypeScript type for share_content parameters */
export type ShareContentParams = z.infer<typeof ShareContentSchema>;

/**
 * Schema for bookmark_content tool parameters.
 *
 * Bookmarks AI response content for quick access later.
 */
export const BookmarkContentSchema = z.object({
  content_id: z.string().describe("Unique ID of the content to bookmark"),
  content_text: z
    .string()
    .describe("The actual content/response text to bookmark"),
  title: z.string().optional().describe("Optional title for the bookmark"),
});

/** Inferred TypeScript type for bookmark_content parameters */
export type BookmarkContentParams = z.infer<typeof BookmarkContentSchema>;

/**
 * Schema for suggest_follow_up tool parameters.
 *
 * Suggests a follow-up query based on the current AI response.
 */
export const SuggestFollowUpSchema = z.object({
  suggested_query: z.string().describe("The suggested follow-up query"),
  context: z
    .string()
    .optional()
    .describe("Context from the current response"),
});

/** Inferred TypeScript type for suggest_follow_up parameters */
export type SuggestFollowUpParams = z.infer<typeof SuggestFollowUpSchema>;

// ============================================
// CopilotKit 1.x Parameter[] Definitions
// ============================================
// These match the Zod schemas above but in Parameter[] format
// required by useFrontendTool in CopilotKit 1.x

/**
 * Parameter definitions for save_to_workspace tool.
 * Compatible with CopilotKit 1.x useFrontendTool.
 */
export const saveToWorkspaceToolParams: ToolParameter[] = [
  {
    name: "content_id",
    type: "string",
    description: "Unique ID of the content to save",
    required: true,
  },
  {
    name: "content_text",
    type: "string",
    description: "The actual content/response text to save",
    required: true,
  },
  {
    name: "title",
    type: "string",
    description: "Optional title for the saved content",
    required: false,
  },
  {
    name: "query",
    type: "string",
    description: "Original query that generated this response",
    required: false,
  },
];

/**
 * Parameter definitions for export_content tool.
 * Compatible with CopilotKit 1.x useFrontendTool.
 */
export const exportContentToolParams: ToolParameter[] = [
  {
    name: "content_id",
    type: "string",
    description: "Unique ID of the content to export",
    required: true,
  },
  {
    name: "content_text",
    type: "string",
    description: "The actual content/response text to export",
    required: true,
  },
  {
    name: "format",
    type: "string",
    description: "Export format: markdown, pdf, or json",
    required: true,
    enum: ["markdown", "pdf", "json"],
  },
  {
    name: "title",
    type: "string",
    description: "Optional title for the export",
    required: false,
  },
];

/**
 * Parameter definitions for share_content tool.
 * Compatible with CopilotKit 1.x useFrontendTool.
 */
export const shareContentToolParams: ToolParameter[] = [
  {
    name: "content_id",
    type: "string",
    description: "Unique ID of the content to share",
    required: true,
  },
  {
    name: "content_text",
    type: "string",
    description: "The actual content/response text to share",
    required: true,
  },
  {
    name: "title",
    type: "string",
    description: "Optional title for the shared content",
    required: false,
  },
];

/**
 * Parameter definitions for bookmark_content tool.
 * Compatible with CopilotKit 1.x useFrontendTool.
 */
export const bookmarkContentToolParams: ToolParameter[] = [
  {
    name: "content_id",
    type: "string",
    description: "Unique ID of the content to bookmark",
    required: true,
  },
  {
    name: "content_text",
    type: "string",
    description: "The actual content/response text to bookmark",
    required: true,
  },
  {
    name: "title",
    type: "string",
    description: "Optional title for the bookmark",
    required: false,
  },
];

/**
 * Parameter definitions for suggest_follow_up tool.
 * Compatible with CopilotKit 1.x useFrontendTool.
 */
export const suggestFollowUpToolParams: ToolParameter[] = [
  {
    name: "suggested_query",
    type: "string",
    description: "The suggested follow-up query",
    required: true,
  },
  {
    name: "context",
    type: "string",
    description: "Context from the current response",
    required: false,
  },
];

// ============================================
// Human-in-the-Loop (HITL) Schemas
// Story 21-A2: Migrate to useHumanInTheLoop Pattern
// ============================================

/**
 * Schema for validate_sources tool parameters.
 *
 * Used by useHumanInTheLoop for Human-in-the-Loop source validation.
 * The agent triggers this tool to request human approval of retrieved sources
 * before generating an answer.
 */
export const ValidateSourcesSchema = z.object({
  sources: z.array(SourceSchema).describe("Array of sources requiring human validation"),
  query: z.string().optional().describe("The original user query for context"),
});

/** Inferred TypeScript type for validate_sources parameters */
export type ValidateSourcesParams = z.infer<typeof ValidateSourcesSchema>;

/**
 * Parameter definitions for validate_sources tool.
 * Compatible with CopilotKit 1.x useHumanInTheLoop.
 */
export const validateSourcesToolParams: ToolParameter[] = [
  {
    name: "sources",
    type: "object",
    description: "Array of sources requiring human validation",
    required: true,
  },
  {
    name: "query",
    type: "string",
    description: "The original user query for context",
    required: false,
  },
];
