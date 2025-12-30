/**
 * Tests for useGenerativeUI hook.
 * Story 6-3: Generative UI Components
 */

import { renderHook, act } from "@testing-library/react";

// Mock CopilotKit hooks
const mockUseCopilotAction = jest.fn();
const mockUseCoAgentStateRender = jest.fn();

jest.mock("@copilotkit/react-core", () => ({
  useCopilotAction: (config: any) => mockUseCopilotAction(config),
  useCoAgentStateRender: (config: any) => mockUseCoAgentStateRender(config),
}));

// Mock the components
jest.mock("@/components/copilot/components/SourceCard", () => ({
  SourceCard: () => null,
}));

jest.mock("@/components/copilot/components/AnswerPanel", () => ({
  AnswerPanel: () => null,
}));

jest.mock("@/components/copilot/components/GraphPreview", () => ({
  GraphPreview: () => null,
}));

// Import after mocks are set up
import { useGenerativeUI } from "../../hooks/use-generative-ui";

describe("useGenerativeUI", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("registers show_sources action", () => {
    renderHook(() => useGenerativeUI());

    const registeredActions = mockUseCopilotAction.mock.calls.map(
      (call) => call[0].name
    );
    expect(registeredActions).toContain("show_sources");
  });

  it("registers show_answer action", () => {
    renderHook(() => useGenerativeUI());

    const registeredActions = mockUseCopilotAction.mock.calls.map(
      (call) => call[0].name
    );
    expect(registeredActions).toContain("show_answer");
  });

  it("registers show_knowledge_graph action", () => {
    renderHook(() => useGenerativeUI());

    const registeredActions = mockUseCopilotAction.mock.calls.map(
      (call) => call[0].name
    );
    expect(registeredActions).toContain("show_knowledge_graph");
  });

  it("registers agent state renderer", () => {
    renderHook(() => useGenerativeUI());

    expect(mockUseCoAgentStateRender).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "orchestrator",
      })
    );
  });

  describe("show_sources action", () => {
    it("has correct parameters", () => {
      renderHook(() => useGenerativeUI());

      const showSourcesAction = mockUseCopilotAction.mock.calls.find(
        (call) => call[0].name === "show_sources"
      )?.[0];

      expect(showSourcesAction).toBeDefined();
      expect(showSourcesAction.parameters).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            name: "sources",
            type: "object[]",
            required: true,
          }),
          expect.objectContaining({
            name: "title",
            type: "string",
            required: false,
          }),
        ])
      );
    });

    it("has render function", () => {
      renderHook(() => useGenerativeUI());

      const showSourcesAction = mockUseCopilotAction.mock.calls.find(
        (call) => call[0].name === "show_sources"
      )?.[0];

      expect(showSourcesAction.render).toBeDefined();
      expect(typeof showSourcesAction.render).toBe("function");
    });
  });

  describe("show_answer action", () => {
    it("has correct parameters", () => {
      renderHook(() => useGenerativeUI());

      const showAnswerAction = mockUseCopilotAction.mock.calls.find(
        (call) => call[0].name === "show_answer"
      )?.[0];

      expect(showAnswerAction).toBeDefined();
      expect(showAnswerAction.parameters).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            name: "answer",
            type: "string",
            required: true,
          }),
          expect.objectContaining({
            name: "sources",
            type: "object[]",
            required: false,
          }),
          expect.objectContaining({
            name: "title",
            type: "string",
            required: false,
          }),
        ])
      );
    });

    it("has render function", () => {
      renderHook(() => useGenerativeUI());

      const showAnswerAction = mockUseCopilotAction.mock.calls.find(
        (call) => call[0].name === "show_answer"
      )?.[0];

      expect(showAnswerAction.render).toBeDefined();
      expect(typeof showAnswerAction.render).toBe("function");
    });
  });

  describe("show_knowledge_graph action", () => {
    it("has correct parameters", () => {
      renderHook(() => useGenerativeUI());

      const showKGAction = mockUseCopilotAction.mock.calls.find(
        (call) => call[0].name === "show_knowledge_graph"
      )?.[0];

      expect(showKGAction).toBeDefined();
      expect(showKGAction.parameters).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            name: "nodes",
            type: "object[]",
            required: true,
          }),
          expect.objectContaining({
            name: "edges",
            type: "object[]",
            required: true,
          }),
          expect.objectContaining({
            name: "title",
            type: "string",
            required: false,
          }),
        ])
      );
    });

    it("has render function", () => {
      renderHook(() => useGenerativeUI());

      const showKGAction = mockUseCopilotAction.mock.calls.find(
        (call) => call[0].name === "show_knowledge_graph"
      )?.[0];

      expect(showKGAction.render).toBeDefined();
      expect(typeof showKGAction.render).toBe("function");
    });
  });

  describe("callback options", () => {
    it("accepts onSourceClick callback", () => {
      const onSourceClick = jest.fn();
      renderHook(() => useGenerativeUI({ onSourceClick }));

      // Hook should not throw
      expect(mockUseCopilotAction).toHaveBeenCalled();
    });

    it("accepts onGraphNodeClick callback", () => {
      const onGraphNodeClick = jest.fn();
      renderHook(() => useGenerativeUI({ onGraphNodeClick }));

      expect(mockUseCopilotAction).toHaveBeenCalled();
    });

    it("accepts onGraphExpand callback", () => {
      const onGraphExpand = jest.fn();
      renderHook(() => useGenerativeUI({ onGraphExpand }));

      expect(mockUseCopilotAction).toHaveBeenCalled();
    });
  });

  describe("return value", () => {
    it("returns state object", () => {
      const { result } = renderHook(() => useGenerativeUI());

      expect(result.current.state).toBeDefined();
      expect(result.current.state).toEqual({
        sources: [],
        answer: null,
        graphData: null,
      });
    });

    it("returns setState function", () => {
      const { result } = renderHook(() => useGenerativeUI());

      expect(result.current.setState).toBeDefined();
      expect(typeof result.current.setState).toBe("function");
    });
  });

  describe("input validation", () => {
    describe("show_sources action validation", () => {
      it("handles undefined sources array gracefully", () => {
        renderHook(() => useGenerativeUI());

        const showSourcesAction = mockUseCopilotAction.mock.calls.find(
          (call) => call[0].name === "show_sources"
        )?.[0];

        // Render with undefined sources should not throw
        let result: unknown;
        act(() => {
          result = showSourcesAction.render({
            status: "complete",
            args: { sources: undefined, title: "Test" },
          });
        });

        // Should render empty or error state, not crash
        expect(result).toBeDefined();
      });

      it("handles malformed sources array gracefully", () => {
        renderHook(() => useGenerativeUI());

        const showSourcesAction = mockUseCopilotAction.mock.calls.find(
          (call) => call[0].name === "show_sources"
        )?.[0];

        // Render with invalid sources should not throw
        let result: unknown;
        act(() => {
          result = showSourcesAction.render({
            status: "complete",
            args: { sources: [{ invalid: true }], title: "Test" },
          });
        });

        expect(result).toBeDefined();
      });

      it("handles empty sources array", () => {
        renderHook(() => useGenerativeUI());

        const showSourcesAction = mockUseCopilotAction.mock.calls.find(
          (call) => call[0].name === "show_sources"
        )?.[0];

        let result: unknown;
        act(() => {
          result = showSourcesAction.render({
            status: "complete",
            args: { sources: [], title: "Test" },
          });
        });

        expect(result).toBeDefined();
      });
    });

    describe("show_answer action validation", () => {
      it("handles empty answer string", () => {
        renderHook(() => useGenerativeUI());

        const showAnswerAction = mockUseCopilotAction.mock.calls.find(
          (call) => call[0].name === "show_answer"
        )?.[0];

        let result: unknown;
        act(() => {
          result = showAnswerAction.render({
            status: "complete",
            args: { answer: "", sources: [] },
          });
        });

        expect(result).toBeDefined();
      });

      it("handles undefined answer", () => {
        renderHook(() => useGenerativeUI());

        const showAnswerAction = mockUseCopilotAction.mock.calls.find(
          (call) => call[0].name === "show_answer"
        )?.[0];

        let result: unknown;
        act(() => {
          result = showAnswerAction.render({
            status: "complete",
            args: { answer: undefined, sources: [] },
          });
        });

        expect(result).toBeDefined();
      });
    });

    describe("show_knowledge_graph action validation", () => {
      it("handles invalid node data", () => {
        renderHook(() => useGenerativeUI());

        const showKGAction = mockUseCopilotAction.mock.calls.find(
          (call) => call[0].name === "show_knowledge_graph"
        )?.[0];

        let result: unknown;
        act(() => {
          result = showKGAction.render({
            status: "complete",
            args: {
              nodes: [{ notAnId: "test" }], // Missing required 'id' field
              edges: [],
            },
          });
        });

        expect(result).toBeDefined();
      });

      it("handles invalid edge data", () => {
        renderHook(() => useGenerativeUI());

        const showKGAction = mockUseCopilotAction.mock.calls.find(
          (call) => call[0].name === "show_knowledge_graph"
        )?.[0];

        let result: unknown;
        act(() => {
          result = showKGAction.render({
            status: "complete",
            args: {
              nodes: [{ id: "1", label: "Test" }],
              edges: [{ invalid: true }], // Missing required fields
            },
          });
        });

        expect(result).toBeDefined();
      });

      it("handles undefined nodes and edges", () => {
        renderHook(() => useGenerativeUI());

        const showKGAction = mockUseCopilotAction.mock.calls.find(
          (call) => call[0].name === "show_knowledge_graph"
        )?.[0];

        let result: unknown;
        act(() => {
          result = showKGAction.render({
            status: "complete",
            args: { nodes: undefined, edges: undefined },
          });
        });

        expect(result).toBeDefined();
      });
    });
  });

  describe("state updates on action completion", () => {
    it("updates state when show_sources action completes", () => {
      const { result } = renderHook(() => useGenerativeUI());

      // Get the agent state render callback
      const stateRenderCall = mockUseCoAgentStateRender.mock.calls[0]?.[0];
      expect(stateRenderCall).toBeDefined();

      // Simulate state update from agent
      act(() => {
        stateRenderCall.render({
          state: {
            generativeUI: {
              sources: [{ id: "1", title: "Test", preview: "Preview", similarity: 0.9 }],
              answer: "Test answer",
              graphData: null,
            },
          },
        });
      });

      // State should be updated
      expect(result.current.state.sources).toHaveLength(1);
      expect(result.current.state.answer).toBe("Test answer");
    });

    it("handles missing generativeUI in agent state", () => {
      renderHook(() => useGenerativeUI());

      const stateRenderCall = mockUseCoAgentStateRender.mock.calls[0]?.[0];

      // Should not throw when generativeUI is missing
      let result: unknown;
      act(() => {
        result = stateRenderCall.render({ state: {} });
      });
      expect(result).toBeDefined();
    });
  });
});
