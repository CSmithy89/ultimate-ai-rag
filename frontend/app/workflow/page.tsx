/**
 * Visual Workflow Editor page.
 * Story 20-H6: Implement Visual Workflow Editor
 */

'use client';

import { WorkflowEditor } from '../../components/workflow';

/**
 * Get feature flag from environment variable.
 * In a real app, this might come from a config API or server component.
 */
function isWorkflowEnabled(): boolean {
  // Check for client-side env var or default to true for development
  if (typeof window !== 'undefined') {
    // Could check localStorage or API for feature flag
    return true; // Default enabled for now
  }
  return process.env.NEXT_PUBLIC_VISUAL_WORKFLOW_ENABLED === 'true';
}

/**
 * Workflow editor page component.
 */
export default function WorkflowPage() {
  const enabled = isWorkflowEnabled();

  return (
    <div className="h-screen w-full">
      <WorkflowEditor enabled={enabled} />
    </div>
  );
}
