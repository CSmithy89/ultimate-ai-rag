"use client";

import { useDynamicInstructions } from "@/hooks/use-dynamic-instructions";

/**
 * DynamicInstructionsProvider - Component wrapper for dynamic AI instructions.
 *
 * Story 21-A7: Implement useCopilotAdditionalInstructions for Dynamic Prompts
 *
 * This component should be placed inside the CopilotKit context to register
 * dynamic instructions based on:
 * - Current page context
 * - User preferences
 * - Feature availability
 * - Security requirements
 *
 * The component renders nothing but ensures the useDynamicInstructions hook
 * is called within the proper context.
 *
 * @example
 * ```tsx
 * import { CopilotKit } from "@copilotkit/react-core";
 * import { DynamicInstructionsProvider } from "@/components/copilot/DynamicInstructionsProvider";
 *
 * function App({ children }) {
 *   return (
 *     <CopilotKit runtimeUrl="/api/copilotkit">
 *       <DynamicInstructionsProvider />
 *       {children}
 *     </CopilotKit>
 *   );
 * }
 * ```
 */
export function DynamicInstructionsProvider(): null {
  useDynamicInstructions();
  return null;
}

export default DynamicInstructionsProvider;
