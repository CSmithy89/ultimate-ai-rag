/**
 * Open-JSON-UI Divider Component
 *
 * Renders a horizontal rule separator.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

"use client";

import { memo } from "react";

/**
 * DividerComponent renders a horizontal separator line.
 *
 * @returns Rendered hr element
 *
 * @example
 * ```tsx
 * <DividerComponent />
 * ```
 */
export const DividerComponent = memo(function DividerComponent() {
  return (
    <hr
      className="my-4 border-t border-slate-200"
      data-testid="open-json-ui-divider"
      role="separator"
    />
  );
});

export default DividerComponent;
