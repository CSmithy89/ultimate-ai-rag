/**
 * Open-JSON-UI Components
 *
 * Declarative UI component renderers for Open-JSON-UI payloads.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 *
 * @example
 * ```tsx
 * import { OpenJSONUIRenderer } from "@/components/open-json-ui";
 *
 * <OpenJSONUIRenderer
 *   payload={{
 *     type: "open_json_ui",
 *     components: [
 *       { type: "heading", level: 1, content: "Hello" },
 *       { type: "text", content: "World" }
 *     ]
 *   }}
 *   onAction={(action) => console.log(action)}
 * />
 * ```
 */

// Main renderer
export { OpenJSONUIRenderer } from "./OpenJSONUIRenderer";
export type { OpenJSONUIRendererProps } from "./OpenJSONUIRenderer";

// Individual components (for advanced usage)
export { TextComponent } from "./TextComponent";
export { HeadingComponent } from "./HeadingComponent";
export { CodeComponent } from "./CodeComponent";
export { ListComponent } from "./ListComponent";
export { TableComponent } from "./TableComponent";
export { ImageComponent } from "./ImageComponent";
export { ButtonComponent } from "./ButtonComponent";
export { DividerComponent } from "./DividerComponent";
export { LinkComponent } from "./LinkComponent";
export { ProgressComponent } from "./ProgressComponent";
export { AlertComponent } from "./AlertComponent";

// Re-export types and schemas from lib
export type {
  TextComponent as TextComponentData,
  HeadingComponent as HeadingComponentData,
  CodeComponent as CodeComponentData,
  ListComponent as ListComponentData,
  TableComponent as TableComponentData,
  ImageComponent as ImageComponentData,
  ButtonComponent as ButtonComponentData,
  LinkComponent as LinkComponentData,
  DividerComponent as DividerComponentData,
  ProgressComponent as ProgressComponentData,
  AlertComponent as AlertComponentData,
  OpenJSONUIComponent,
  OpenJSONUIPayload,
} from "@/lib/open-json-ui";
