"use client";

import { useGenerativeUI, type GraphPreviewNode } from "@/hooks/use-generative-ui";
import { useSourceValidation } from "@/hooks/use-source-validation";
import { SourceValidationDialog } from "./SourceValidationDialog";
import { SourceValidationPanel } from "./SourceValidationPanel";
import type { Source } from "@/types/copilot";

interface GenerativeUIRendererProps {
  /** Callback when a source card is clicked */
  onSourceClick?: (source: Source) => void;
  /** Callback when a graph node is clicked */
  onGraphNodeClick?: (node: GraphPreviewNode) => void;
  /** Callback when the graph expand button is clicked */
  onGraphExpand?: () => void;
  /** Use modal dialog (true) or inline panel (false) for HITL */
  useModalForValidation?: boolean;
  /** Callback when HITL validation completes */
  onValidationComplete?: (approvedIds: string[]) => void;
  /** Auto-approve sources at or above this confidence threshold */
  autoApproveThreshold?: number;
  /** Auto-reject sources below this confidence threshold */
  autoRejectThreshold?: number;
}

/**
 * GenerativeUIRenderer initializes the generative UI action handlers
 * including Human-in-the-Loop source validation.
 *
 * Include this component within your CopilotKit context to enable
 * generative UI capabilities.
 *
 * Story 6-3: Generative UI Components
 * Story 6-4: Human-in-the-Loop Source Validation
 *
 * @example
 * ```tsx
 * <CopilotSidebar>
 *   <GenerativeUIRenderer
 *     useModalForValidation={true}
 *     onValidationComplete={(ids) => console.log("Approved:", ids)}
 *   />
 * </CopilotSidebar>
 * ```
 */
export function GenerativeUIRenderer({
  onSourceClick,
  onGraphNodeClick,
  onGraphExpand,
  useModalForValidation = true,
  onValidationComplete,
  autoApproveThreshold,
  autoRejectThreshold,
}: GenerativeUIRendererProps) {
  // Initialize generative UI hooks (Story 6-3)
  useGenerativeUI({
    onSourceClick,
    onGraphNodeClick,
    onGraphExpand,
  });

  // Initialize source validation hooks (Story 6-4)
  const {
    state: validationState,
    isDialogOpen,
    submitValidation,
    cancelValidation,
  } = useSourceValidation({
    onValidationComplete,
    autoApproveThreshold,
    autoRejectThreshold,
  });

  return (
    <>
      {/* Modal dialog for HITL validation */}
      {useModalForValidation && (
        <SourceValidationDialog
          open={isDialogOpen}
          sources={validationState.pendingSources}
          onSubmit={submitValidation}
          onCancel={cancelValidation}
          isSubmitting={validationState.isSubmitting}
        />
      )}

      {/* Inline panel for non-modal HITL (rendered in chat flow) */}
      {!useModalForValidation && isDialogOpen && (
        <SourceValidationPanel
          sources={validationState.pendingSources}
          onSubmit={submitValidation}
          onSkip={() => submitValidation(validationState.pendingSources.map((s) => s.id))}
        />
      )}
    </>
  );
}

export default GenerativeUIRenderer;
