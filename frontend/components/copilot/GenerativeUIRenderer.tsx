"use client";

import { useGenerativeUI, type GraphPreviewNode } from "@/hooks/use-generative-ui";
import type { Source } from "@/types/copilot";

interface GenerativeUIRendererProps {
  /** Callback when a source card is clicked */
  onSourceClick?: (source: Source) => void;
  /** Callback when a graph node is clicked */
  onGraphNodeClick?: (node: GraphPreviewNode) => void;
  /** Callback when the graph expand button is clicked */
  onGraphExpand?: () => void;
}

/**
 * GenerativeUIRenderer initializes the generative UI action handlers
 * for rendering dynamic components within the chat flow.
 *
 * Include this component within your CopilotKit context to enable
 * generative UI capabilities.
 *
 * Story 6-3: Generative UI Components
 *
 * @example
 * ```tsx
 * <CopilotSidebar>
 *   <GenerativeUIRenderer
 *     onSourceClick={(source) => openSourceModal(source)}
 *     onGraphExpand={() => openGraphModal()}
 *   />
 * </CopilotSidebar>
 * ```
 */
export function GenerativeUIRenderer({
  onSourceClick,
  onGraphNodeClick,
  onGraphExpand,
}: GenerativeUIRendererProps) {
  // Initialize generative UI hooks
  useGenerativeUI({
    onSourceClick,
    onGraphNodeClick,
    onGraphExpand,
  });

  // This component doesn't render anything itself;
  // it just registers the action handlers
  return null;
}

export default GenerativeUIRenderer;
