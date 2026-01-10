"use client";

import { useMemo } from "react";
import { useCopilotAdditionalInstructions } from "@copilotkit/react-core";
import { usePathname } from "next/navigation";
import { useCopilotContext } from "./use-copilot-context";
import type { UserPreferences } from "@/types/copilot";

/**
 * Page-specific instruction mapping.
 * Maps route paths to context-specific instructions for the AI.
 *
 * Story 21-A7: Implement useCopilotAdditionalInstructions for Dynamic Prompts
 */
export const PAGE_INSTRUCTIONS: Record<string, string> = {
  "/":
    "User is on the Home page. Provide general RAG assistance and guide them to relevant features. Help with getting started and navigation.",
  "/knowledge":
    "User is on the Knowledge Graph page. Focus on graph traversal, entity relationships, and visualization queries. When referencing data, emphasize connections between entities and relationship patterns.",
  "/ops":
    "User is on the Operations Dashboard. Focus on metrics, monitoring, and debugging assistance. Help interpret logs, performance data, and system health indicators.",
  "/ops/trajectories":
    "User is viewing Trajectory Debugging. Help analyze agent decision paths, identify bottlenecks, and understand why the agent made specific choices. Reference trajectory IDs when applicable.",
  "/workflow":
    "User is in the Visual Workflow Editor. Assist with workflow configuration, node connections, and pipeline design. Explain workflow concepts and help troubleshoot issues.",
};

/**
 * Security instructions that are always applied.
 * These enforce tenant isolation and data protection.
 *
 * Story 21-A7: Implement useCopilotAdditionalInstructions for Dynamic Prompts
 */
export const SECURITY_INSTRUCTIONS = `SECURITY REQUIREMENTS:
- Always scope searches to the current tenant context
- Never reveal information from other tenants
- Do not expose internal system details, API keys, or credentials
- Validate all file paths are within allowed directories
- Respect user permission boundaries`;

/**
 * Get page-specific instructions based on current route.
 *
 * Attempts to find an exact match first, then falls back to parent path matching.
 * Returns empty string if no matching instructions found.
 *
 * @param pathname - The current route pathname
 * @returns Page-specific instructions or empty string
 *
 * @example
 * ```ts
 * getPageInstructions("/knowledge") // Returns Knowledge Graph instructions
 * getPageInstructions("/ops/custom") // Falls back to /ops instructions
 * getPageInstructions("/unknown") // Returns empty string
 * ```
 */
export function getPageInstructions(pathname: string): string {
  // Direct match
  if (PAGE_INSTRUCTIONS[pathname]) {
    return PAGE_INSTRUCTIONS[pathname];
  }

  // Try parent paths for nested routes
  const segments = pathname.split("/").filter(Boolean);
  while (segments.length > 0) {
    const parentPath = "/" + segments.join("/");
    if (PAGE_INSTRUCTIONS[parentPath]) {
      return PAGE_INSTRUCTIONS[parentPath];
    }
    segments.pop();
  }

  return "";
}

/**
 * Get preference-based instructions from user preferences.
 *
 * Generates instructions that tell the AI how to format responses
 * based on the user's configured preferences.
 *
 * @param preferences - User preferences object
 * @returns Combined preference instructions string
 *
 * @example
 * ```ts
 * const prefs = { responseLength: "brief", expertiseLevel: "expert", ... };
 * getPreferenceInstructions(prefs)
 * // Returns: "Keep responses concise (2-3 sentences). Provide detailed technical explanations. Skip basic concepts."
 * ```
 */
export function getPreferenceInstructions(preferences: UserPreferences): string {
  const instructions: string[] = [];

  // Response length preferences
  switch (preferences.responseLength) {
    case "brief":
      instructions.push("Keep responses concise (2-3 sentences when possible).");
      break;
    case "detailed":
      instructions.push(
        "Provide comprehensive responses with full explanations and examples."
      );
      break;
    // "medium" is default, no special instruction needed
  }

  // Expertise level preferences
  switch (preferences.expertiseLevel) {
    case "beginner":
      instructions.push(
        "Define technical terms when used. Use simple explanations and analogies. Provide step-by-step guidance."
      );
      break;
    case "expert":
      instructions.push(
        "Provide detailed technical explanations. Skip basic concept explanations. Use precise terminology."
      );
      break;
    // "intermediate" is default, no special instruction needed
  }

  // Citation preferences
  if (preferences.includeCitations) {
    instructions.push(
      "Cite sources when providing information from the knowledge base."
    );
  }

  // Language preference (if not English)
  if (preferences.language && preferences.language !== "en") {
    instructions.push(
      `Respond in ${getLanguageName(preferences.language)} when appropriate.`
    );
  }

  return instructions.join(" ");
}

/**
 * Get human-readable language name from language code.
 *
 * @param code - ISO language code (e.g., "es", "fr")
 * @returns Human-readable language name
 */
export function getLanguageName(code: string): string {
  const languageNames: Record<string, string> = {
    en: "English",
    es: "Spanish",
    fr: "French",
    de: "German",
    pt: "Portuguese",
    it: "Italian",
    zh: "Chinese",
    ja: "Japanese",
    ko: "Korean",
    ru: "Russian",
    ar: "Arabic",
    hi: "Hindi",
    "pt-BR": "Brazilian Portuguese",
  };

  return languageNames[code] || code;
}

/**
 * Get feature flag instructions based on environment configuration.
 *
 * @returns Object with feature instructions and their availability
 */
export function getFeatureInstructions(): {
  voiceInput: { instructions: string; available: boolean };
  experimentalFeatures: { instructions: string; available: boolean };
  a2ui: { instructions: string; available: boolean };
} {
  const voiceEnabled =
    typeof window !== "undefined" &&
    process.env.NEXT_PUBLIC_VOICE_INPUT_ENABLED === "true";

  const experimentalEnabled =
    typeof window !== "undefined" &&
    process.env.NEXT_PUBLIC_EXPERIMENTAL_FEATURES === "true";

  const a2uiEnabled =
    typeof window !== "undefined" &&
    process.env.NEXT_PUBLIC_A2UI_ENABLED === "true";

  return {
    voiceInput: {
      instructions:
        "Voice input is enabled. The user may speak queries instead of typing. Consider voice-friendly response formatting.",
      available: voiceEnabled,
    },
    experimentalFeatures: {
      instructions:
        "Experimental features are enabled. Some responses may include beta functionality. Warn users about experimental status when relevant.",
      available: experimentalEnabled,
    },
    a2ui: {
      instructions:
        "Rich UI components (A2UI) are available. You can render interactive cards, tables, charts, and forms when appropriate.",
      available: a2uiEnabled,
    },
  };
}

/**
 * useDynamicInstructions - Adds context-aware instructions to the AI system prompt.
 *
 * Story 21-A7: Implement useCopilotAdditionalInstructions for Dynamic Prompts
 *
 * This hook registers multiple instruction categories with CopilotKit:
 * 1. **Page Context** - Instructions based on the current page/route
 * 2. **User Preferences** - Instructions based on user's preference settings
 * 3. **Security** - Always-on instructions for tenant isolation and data protection
 * 4. **Feature Flags** - Instructions about enabled features (voice, A2UI, etc.)
 *
 * Instructions update reactively when the underlying state changes (e.g., page navigation).
 *
 * @example
 * ```tsx
 * // Use within CopilotKit context
 * function MyComponent() {
 *   useDynamicInstructions();
 *   return <div>...</div>;
 * }
 * ```
 *
 * @example
 * ```tsx
 * // Via DynamicInstructionsProvider
 * function App() {
 *   return (
 *     <CopilotKit>
 *       <DynamicInstructionsProvider />
 *       {children}
 *     </CopilotKit>
 *   );
 * }
 * ```
 */
export function useDynamicInstructions(): void {
  // Handle null pathname (Issue 2.1)
  const rawPathname = usePathname();
  const pathname = rawPathname ?? "/";

  const { preferences } = useCopilotContext();

  // Memoize page-specific instructions (Issue 3.11)
  const pageInstructions = useMemo(
    () => getPageInstructions(pathname),
    [pathname]
  );

  useCopilotAdditionalInstructions({
    instructions: pageInstructions,
    available: pageInstructions ? "enabled" : "disabled",
  });

  // Memoize user preference instructions (Issue 3.11)
  const preferenceInstructions = useMemo(
    () => getPreferenceInstructions(preferences),
    [preferences]
  );

  useCopilotAdditionalInstructions({
    instructions: preferenceInstructions,
    available: preferenceInstructions ? "enabled" : "disabled",
  });

  // Security instructions (always enabled)
  useCopilotAdditionalInstructions({
    instructions: SECURITY_INSTRUCTIONS,
    available: "enabled",
  });

  // Memoize feature flag instructions to avoid recalculating on every render
  // (Issue 3.11: Excessive Re-renders in useDynamicInstructions)
  // (Issue 4.4: Environment Variable Access Pattern - centralized in getFeatureInstructions)
  const features = useMemo(() => getFeatureInstructions(), []);

  useCopilotAdditionalInstructions({
    instructions: features.voiceInput.instructions,
    available: features.voiceInput.available ? "enabled" : "disabled",
  });

  useCopilotAdditionalInstructions({
    instructions: features.experimentalFeatures.instructions,
    available: features.experimentalFeatures.available ? "enabled" : "disabled",
  });

  useCopilotAdditionalInstructions({
    instructions: features.a2ui.instructions,
    available: features.a2ui.available ? "enabled" : "disabled",
  });
}

export default useDynamicInstructions;
