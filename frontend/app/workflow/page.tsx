/**
 * Visual Workflow Editor page.
 * Story 20-H6: Implement Visual Workflow Editor
 */

'use client';

import { WorkflowEditor } from '../../components/workflow';

/**
 * Get feature flag from environment variable.
 * NEXT_PUBLIC_VISUAL_WORKFLOW_ENABLED must be set to 'true' to enable.
 * Default: disabled (false) per AC4.
 */
function isWorkflowEnabled(): boolean {
  // Next.js exposes NEXT_PUBLIC_ vars at build time
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
