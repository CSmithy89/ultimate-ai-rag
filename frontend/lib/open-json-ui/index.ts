/**
 * Open-JSON-UI Library
 *
 * Provides schemas, types, and utilities for Open-JSON-UI rendering.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 *
 * @example
 * ```ts
 * import {
 *   validatePayload,
 *   sanitizeContent,
 *   type OpenJSONUIPayload,
 * } from "@/lib/open-json-ui";
 *
 * const result = validatePayload(payload);
 * if (result.success) {
 *   // render components
 * }
 * ```
 */

// Schema exports
export {
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
  OpenJSONUIComponentSchema,
  OpenJSONUIPayloadSchema,
  validatePayload,
  isComponentType,
} from "./schema";

// Type exports
export type {
  TextComponent,
  HeadingComponent,
  CodeComponent,
  TableComponent,
  ImageComponent,
  ButtonComponent,
  ListComponent,
  LinkComponent,
  DividerComponent,
  ProgressComponent,
  AlertComponent,
  OpenJSONUIComponent,
  OpenJSONUIPayload,
} from "./schema";

// Sanitization exports
export {
  sanitizeContent,
  sanitizeToPlainText,
  isValidUrl,
  sanitizeUrl,
} from "./sanitize";
