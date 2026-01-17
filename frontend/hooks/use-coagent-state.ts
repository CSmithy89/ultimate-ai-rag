"use client";

/**
 * useOrchestratorState - Bidirectional Agent State Sync
 *
 * AG-UI Enhancement: Provides bidirectional state synchronization between
 * the frontend and the orchestrator agent using CopilotKit's useCoAgent hook.
 *
 * This enables:
 * - Frontend → Agent: User selections, filters, feedback
 * - Agent → Frontend: Processing state, results, thinking
 *
 * @example
 * ```tsx
 * function SearchResults() {
 *   const { agentState, updateFilters, selectSources, provideFeedback } = useOrchestratorState();
 *
 *   return (
 *     <FilterPanel onChange={updateFilters} />
 *     <SourceList
 *       sources={agentState.retrievedSources}
 *       onSelect={selectSources}
 *     />
 *     <FeedbackButtons onFeedback={provideFeedback} />
 *   );
 * }
 * ```
 */

import { useCallback, useMemo } from "react";
import { useCoAgent, useCoAgentStateRender } from "@copilotkit/react-core";
import type { ThoughtStep, Source } from "@/types/copilot";

// ============================================
// TYPES
// ============================================

/**
 * Filter criteria for source filtering.
 */
export interface FilterCriteria {
  /** Filter by date range */
  dateRange?: {
    start: string;
    end: string;
  };
  /** Filter by document types */
  documentTypes?: string[];
  /** Minimum similarity threshold */
  minSimilarity?: number;
  /** Maximum number of sources */
  maxSources?: number;
  /** Search within specific collections */
  collections?: string[];
}

/**
 * User feedback on agent response quality.
 */
export interface UserFeedback {
  /** Overall helpfulness rating */
  helpful?: boolean;
  /** Accuracy rating (0-5) */
  accuracy?: number;
  /** Specific feedback text */
  comment?: string;
  /** Timestamp of feedback */
  timestamp?: string;
}

/**
 * Orchestrator agent state shared with frontend.
 */
export interface OrchestratorState {
  /** Current processing step */
  currentStep: string;
  /** Agent thought steps (for ThoughtTraceStepper) */
  steps: ThoughtStep[];
  /** Retrieved sources from RAG */
  retrievedSources: Source[];
  /** User-selected source IDs */
  selectedSources: string[];
  /** Active filter criteria */
  filterCriteria: FilterCriteria;
  /** User feedback on current response */
  userFeedback: UserFeedback;
  /** Retrieval strategy being used */
  retrievalStrategy: string;
  /** Trajectory ID for debugging */
  trajectoryId: string | null;
  /** Whether agent is currently thinking */
  isThinking: boolean;
  /** Current thinking content (for display) */
  thinkingContent: string[];
}

/**
 * Default initial state for the orchestrator.
 */
const DEFAULT_ORCHESTRATOR_STATE: OrchestratorState = {
  currentStep: "idle",
  steps: [],
  retrievedSources: [],
  selectedSources: [],
  filterCriteria: {},
  userFeedback: {},
  retrievalStrategy: "hybrid",
  trajectoryId: null,
  isThinking: false,
  thinkingContent: [],
};

/**
 * Options for the useOrchestratorState hook.
 */
export interface UseOrchestratorStateOptions {
  /** Initial state override */
  initialState?: Partial<OrchestratorState>;
  /** Callback when agent state changes */
  onStateChange?: (state: OrchestratorState) => void;
  /** Callback when filters are applied */
  onFiltersApplied?: (filters: FilterCriteria) => void;
  /** Callback when sources are selected */
  onSourcesSelected?: (sourceIds: string[]) => void;
}

/**
 * Return type for useOrchestratorState hook.
 */
export interface UseOrchestratorStateReturn {
  /** Current agent state */
  agentState: OrchestratorState;
  /** Update filter criteria (syncs to agent) */
  updateFilters: (filters: FilterCriteria) => void;
  /** Select specific sources (syncs to agent) */
  selectSources: (sourceIds: string[]) => void;
  /** Toggle source selection */
  toggleSource: (sourceId: string) => void;
  /** Clear all selected sources */
  clearSelection: () => void;
  /** Provide feedback on response (syncs to agent) */
  provideFeedback: (feedback: UserFeedback) => void;
  /** Reset all state to defaults */
  resetState: () => void;
  /** Check if a source is selected */
  isSourceSelected: (sourceId: string) => boolean;
  /** Get selected sources as objects */
  selectedSourceObjects: Source[];
  /** Whether agent is processing */
  isProcessing: boolean;
  /** Whether agent is in thinking phase */
  isThinking: boolean;
}

// ============================================
// HOOK IMPLEMENTATION
// ============================================

/**
 * useOrchestratorState provides bidirectional state sync with the orchestrator agent.
 *
 * AG-UI Enhancement: This hook wraps CopilotKit's useCoAgent to provide
 * typed, convenient access to orchestrator state with helper methods for
 * common operations like filtering, selection, and feedback.
 *
 * State Changes Flow:
 * 1. User interacts with UI (filter, select, feedback)
 * 2. Hook calls setAgentState to update state
 * 3. CopilotKit syncs state to backend agent
 * 4. Agent receives updated state for next turn
 *
 * Agent State Updates Flow:
 * 1. Agent emits STATE_SNAPSHOT or STATE_DELTA events
 * 2. CopilotKit updates local agentState
 * 3. Hook consumers re-render with new state
 *
 * @param options - Configuration options
 * @returns Orchestrator state and helper functions
 */
export function useOrchestratorState(
  options: UseOrchestratorStateOptions = {}
): UseOrchestratorStateReturn {
  const {
    initialState,
    onStateChange,
    onFiltersApplied,
    onSourcesSelected,
  } = options;

  // Merge initial state with defaults (ensuring type safety)
  const mergedInitialState: OrchestratorState = useMemo(
    () => ({
      ...DEFAULT_ORCHESTRATOR_STATE,
      ...initialState,
    }),
    [initialState]
  );

  // Helper to get current state with fallback to defaults
  const getState = useCallback(
    (prev: OrchestratorState | undefined): OrchestratorState =>
      prev ?? mergedInitialState,
    [mergedInitialState]
  );

  // Use CopilotKit's useCoAgent for bidirectional sync
  const { state: agentState, setState: setAgentState } = useCoAgent<OrchestratorState>({
    name: "orchestrator",
    initialState: mergedInitialState,
  });

  // Ensure we have a valid state (fallback to defaults if undefined)
  const safeAgentState = agentState ?? mergedInitialState;

  // ============================================
  // STATE UPDATE HELPERS
  // ============================================

  /**
   * Update filter criteria and notify agent.
   */
  const updateFilters = useCallback(
    (filters: FilterCriteria) => {
      setAgentState((prev) => {
        const currentState = getState(prev);
        const newState: OrchestratorState = {
          ...currentState,
          filterCriteria: { ...currentState.filterCriteria, ...filters },
        };
        onStateChange?.(newState);
        onFiltersApplied?.(newState.filterCriteria);
        return newState;
      });
    },
    [setAgentState, onStateChange, onFiltersApplied, getState]
  );

  /**
   * Select specific sources.
   */
  const selectSources = useCallback(
    (sourceIds: string[]) => {
      setAgentState((prev) => {
        const currentState = getState(prev);
        const newState: OrchestratorState = {
          ...currentState,
          selectedSources: sourceIds,
        };
        onStateChange?.(newState);
        onSourcesSelected?.(sourceIds);
        return newState;
      });
    },
    [setAgentState, onStateChange, onSourcesSelected, getState]
  );

  /**
   * Toggle a single source selection.
   */
  const toggleSource = useCallback(
    (sourceId: string) => {
      setAgentState((prev) => {
        const currentState = getState(prev);
        const isSelected = currentState.selectedSources.includes(sourceId);
        const newSelectedSources = isSelected
          ? currentState.selectedSources.filter((id) => id !== sourceId)
          : [...currentState.selectedSources, sourceId];

        const newState: OrchestratorState = {
          ...currentState,
          selectedSources: newSelectedSources,
        };
        onStateChange?.(newState);
        onSourcesSelected?.(newSelectedSources);
        return newState;
      });
    },
    [setAgentState, onStateChange, onSourcesSelected, getState]
  );

  /**
   * Clear all selected sources.
   */
  const clearSelection = useCallback(() => {
    setAgentState((prev) => {
      const currentState = getState(prev);
      const newState: OrchestratorState = {
        ...currentState,
        selectedSources: [] as string[],
      };
      onStateChange?.(newState);
      onSourcesSelected?.([] as string[]);
      return newState;
    });
  }, [setAgentState, onStateChange, onSourcesSelected, getState]);

  /**
   * Provide feedback on the current response.
   */
  const provideFeedback = useCallback(
    (feedback: UserFeedback) => {
      setAgentState((prev) => {
        const currentState = getState(prev);
        const newFeedback: UserFeedback = {
          ...currentState.userFeedback,
          ...feedback,
          timestamp: new Date().toISOString(),
        };
        const newState: OrchestratorState = {
          ...currentState,
          userFeedback: newFeedback,
        };
        onStateChange?.(newState);
        return newState;
      });
    },
    [setAgentState, onStateChange, getState]
  );

  /**
   * Reset all state to defaults.
   */
  const resetState = useCallback(() => {
    setAgentState(mergedInitialState);
    onStateChange?.(mergedInitialState);
  }, [setAgentState, mergedInitialState, onStateChange]);

  // ============================================
  // DERIVED STATE
  // ============================================

  /**
   * Check if a specific source is selected.
   */
  const isSourceSelected = useCallback(
    (sourceId: string) => safeAgentState.selectedSources.includes(sourceId),
    [safeAgentState.selectedSources]
  );

  /**
   * Get selected sources as full objects.
   */
  const selectedSourceObjects = useMemo(
    () =>
      safeAgentState.retrievedSources.filter((source) =>
        safeAgentState.selectedSources.includes(source.id)
      ),
    [safeAgentState.retrievedSources, safeAgentState.selectedSources]
  );

  /**
   * Whether agent is currently processing.
   */
  const isProcessing = useMemo(
    () =>
      safeAgentState.currentStep !== "idle" &&
      safeAgentState.currentStep !== "completed",
    [safeAgentState.currentStep]
  );

  return {
    agentState: safeAgentState,
    updateFilters,
    selectSources,
    toggleSource,
    clearSelection,
    provideFeedback,
    resetState,
    isSourceSelected,
    selectedSourceObjects,
    isProcessing,
    isThinking: safeAgentState.isThinking,
  };
}

// ============================================
// STATE RENDER HOOK
// ============================================

/**
 * useOrchestratorStateRender - Render component based on orchestrator state.
 *
 * This is a convenience wrapper around useCoAgentStateRender that provides
 * typed access to orchestrator state for rendering purposes.
 *
 * @example
 * ```tsx
 * function ProcessingIndicator() {
 *   useOrchestratorStateRender({
 *     render: (state) => {
 *       if (state.isThinking) {
 *         return <ThinkingIndicator thoughts={state.thinkingContent} />;
 *       }
 *       if (state.steps.length > 0) {
 *         return <ThoughtTraceStepper steps={state.steps} />;
 *       }
 *       return null;
 *     },
 *   });
 *   return null;
 * }
 * ```
 */
export function useOrchestratorStateRender(options: {
  render: (state: OrchestratorState) => React.ReactElement | string | null | undefined;
}): void {
  useCoAgentStateRender<OrchestratorState>({
    name: "orchestrator",
    render: (info) => {
      // Ensure we have valid state
      const state = info.state ?? DEFAULT_ORCHESTRATOR_STATE;
      return options.render(state);
    },
  });
}

export default useOrchestratorState;
