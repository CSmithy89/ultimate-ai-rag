"use client";

import { useGenerativeUI, type GraphPreviewNode } from "@/hooks/use-generative-ui";
import { useSourceValidation } from "@/hooks/use-source-validation";
import { useCopilotActions } from "@/hooks/use-copilot-actions";
import { SourceValidationDialog } from "./SourceValidationDialog";
import { SourceValidationPanel } from "./SourceValidationPanel";
import type { Source, ActionableContent } from "@/types/copilot";

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
  /** Tenant ID for multi-tenant operations */
  tenantId?: string;
  /** Callback when content is saved to workspace */
  onSaveComplete?: (content: ActionableContent) => void;
  /** Callback when content is exported */
  onExportComplete?: (content: ActionableContent, format: string) => void;
  /** Callback when content is shared */
  onShareComplete?: (content: ActionableContent, shareUrl: string) => void;
  /** Callback when content is bookmarked */
  onBookmarkComplete?: (content: ActionableContent) => void;
  /** Callback to handle follow-up query */
  onFollowUp?: (query: string, context: ActionableContent) => void;
}

/**
 * GenerativeUIRenderer initializes the generative UI action handlers
 * including Human-in-the-Loop source validation and frontend actions.
 *
 * Include this component within your CopilotKit context to enable
 * generative UI capabilities.
 *
 * Story 6-3: Generative UI Components
 * Story 6-4: Human-in-the-Loop Source Validation
 * Story 6-5: Frontend Actions
 *
 * @example
 * ```tsx
 * <CopilotSidebar>
 *   <GenerativeUIRenderer
 *     useModalForValidation={true}
 *     onValidationComplete={(ids) => console.log("Approved:", ids)}
 *     onSaveComplete={(content) => console.log("Saved:", content)}
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
  tenantId,
  onSaveComplete,
  onExportComplete,
  onShareComplete,
  onBookmarkComplete,
  onFollowUp,
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

  // Initialize frontend actions hooks (Story 6-5)
  // This registers CopilotKit actions for agent-triggered operations
  useCopilotActions({
    tenantId,
    onSaveComplete: onSaveComplete
      ? (result) => {
          if (result.success && result.data) {
            onSaveComplete(result.data as ActionableContent);
          }
        }
      : undefined,
    onExportComplete: onExportComplete
      ? (result) => {
          if (result.success && result.data) {
            const data = result.data as { format?: string };
            onExportComplete({} as ActionableContent, data.format || "");
          }
        }
      : undefined,
    onShareComplete: onShareComplete
      ? (result, shareUrl) => {
          if (result.success) {
            onShareComplete({} as ActionableContent, shareUrl);
          }
        }
      : undefined,
    onBookmarkComplete: onBookmarkComplete
      ? (result) => {
          if (result.success) {
            onBookmarkComplete({} as ActionableContent);
          }
        }
      : undefined,
    onFollowUp,
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
