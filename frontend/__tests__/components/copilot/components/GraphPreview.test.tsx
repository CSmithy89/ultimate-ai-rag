/**
 * Tests for GraphPreview component.
 * Story 6-3: Generative UI Components
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { GraphPreview, GraphPreviewNode, GraphPreviewEdge } from "../../../../components/copilot/components/GraphPreview";

// Mock lucide-react icons
jest.mock("lucide-react", () => ({
  Maximize2: () => <span data-testid="icon-maximize">Maximize2</span>,
}));

// Track mock function calls
const mockFns = {
  setNodes: jest.fn(),
  setEdges: jest.fn(),
};

// Mock reactflow with state sync support
jest.mock("reactflow", () => ({
  __esModule: true,
  default: ({ nodes, edges, onNodeClick, className }: any) => (
    <div data-testid="react-flow" className={className}>
      <div data-testid="nodes-count">{nodes?.length || 0}</div>
      <div data-testid="edges-count">{edges?.length || 0}</div>
      {nodes?.map((node: any) => (
        <div 
          key={node.id} 
          data-testid={`node-${node.id}`}
          onClick={(e) => onNodeClick?.(e, node)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              onNodeClick?.(e, node);
            }
          }}
        >
          {node.data.label}
        </div>
      ))}
    </div>
  ),
  Background: () => <div data-testid="react-flow-background" />,
  Controls: () => <div data-testid="react-flow-controls" />,
  useNodesState: (initialNodes: any[]) => {
    const setNodes = jest.fn();
    mockFns.setNodes = setNodes;
    return [initialNodes, setNodes, jest.fn()];
  },
  useEdgesState: (initialEdges: any[]) => {
    const setEdges = jest.fn();
    mockFns.setEdges = setEdges;
    return [initialEdges, setEdges, jest.fn()];
  },
  MarkerType: { ArrowClosed: "arrowclosed" },
  Position: { Right: "right", Left: "left" },
}));

const mockNodes: GraphPreviewNode[] = [
  { id: "node-1", label: "Center Entity", type: "Person" },
  { id: "node-2", label: "Related Entity 1", type: "Organization" },
  { id: "node-3", label: "Related Entity 2", type: "Technology" },
];

const mockEdges: GraphPreviewEdge[] = [
  { id: "edge-1", source: "node-1", target: "node-2", label: "WORKS_FOR" },
  { id: "edge-2", source: "node-1", target: "node-3", label: "USES" },
];

describe("GraphPreview", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders React Flow component", () => {
    render(<GraphPreview nodes={mockNodes} edges={mockEdges} />);
    expect(screen.getByTestId("react-flow")).toBeInTheDocument();
  });

  it("renders the default title", () => {
    render(<GraphPreview nodes={mockNodes} edges={mockEdges} />);
    expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();
  });

  it("renders custom title when provided", () => {
    render(<GraphPreview nodes={mockNodes} edges={mockEdges} title="Custom Graph" />);
    expect(screen.getByText("Custom Graph")).toBeInTheDocument();
  });

  it("displays node and edge counts", () => {
    render(<GraphPreview nodes={mockNodes} edges={mockEdges} />);
    expect(screen.getByText("3 nodes, 2 edges")).toBeInTheDocument();
  });

  describe("node transformation", () => {
    it("transforms nodes correctly", () => {
      render(<GraphPreview nodes={mockNodes} edges={mockEdges} />);
      
      expect(screen.getByTestId("nodes-count")).toHaveTextContent("3");
      expect(screen.getByText("Center Entity")).toBeInTheDocument();
      expect(screen.getByText("Related Entity 1")).toBeInTheDocument();
      expect(screen.getByText("Related Entity 2")).toBeInTheDocument();
    });
  });

  describe("edge transformation", () => {
    it("transforms edges correctly", () => {
      render(<GraphPreview nodes={mockNodes} edges={mockEdges} />);
      expect(screen.getByTestId("edges-count")).toHaveTextContent("2");
    });
  });

  describe("single node handling", () => {
    it("handles single node gracefully", () => {
      const singleNode = [{ id: "node-1", label: "Only Node", type: "Person" }];
      render(<GraphPreview nodes={singleNode} edges={[]} />);
      
      expect(screen.getByText("Only Node")).toBeInTheDocument();
      expect(screen.getByText("1 nodes, 0 edges")).toBeInTheDocument();
    });
  });

  describe("empty data handling", () => {
    it("handles empty nodes gracefully", () => {
      render(<GraphPreview nodes={[]} edges={[]} />);
      expect(screen.getByText("0 nodes, 0 edges")).toBeInTheDocument();
    });
  });

  describe("expand button", () => {
    it("renders expand button when onExpand is provided", () => {
      const handleExpand = jest.fn();
      render(<GraphPreview nodes={mockNodes} edges={mockEdges} onExpand={handleExpand} />);
      
      expect(screen.getByTestId("icon-maximize")).toBeInTheDocument();
    });

    it("does not render expand button when onExpand is not provided", () => {
      render(<GraphPreview nodes={mockNodes} edges={mockEdges} />);
      expect(screen.queryByTestId("icon-maximize")).not.toBeInTheDocument();
    });

    it("calls onExpand when expand button is clicked", () => {
      const handleExpand = jest.fn();
      render(<GraphPreview nodes={mockNodes} edges={mockEdges} onExpand={handleExpand} />);
      
      const expandButton = screen.getByRole("button", { name: /expand/i });
      fireEvent.click(expandButton);
      
      expect(handleExpand).toHaveBeenCalled();
    });
  });

  describe("node click handling", () => {
    it("calls onNodeClick when a node is clicked", () => {
      const handleNodeClick = jest.fn();
      render(
        <GraphPreview 
          nodes={mockNodes} 
          edges={mockEdges} 
          onNodeClick={handleNodeClick} 
        />
      );
      
      const node = screen.getByTestId("node-node-1");
      fireEvent.click(node);
      
      expect(handleNodeClick).toHaveBeenCalledWith(mockNodes[0]);
    });
  });

  describe("styling", () => {
    it("applies custom className when provided", () => {
      const { container } = render(
        <GraphPreview nodes={mockNodes} edges={mockEdges} className="custom-class" />
      );
      
      expect(container.firstChild).toHaveClass("custom-class");
    });

    it("applies slate background to React Flow", () => {
      render(<GraphPreview nodes={mockNodes} edges={mockEdges} />);
      expect(screen.getByTestId("react-flow")).toHaveClass("bg-slate-50");
    });
  });

  describe("height prop", () => {
    it("uses default height of 200", () => {
      const { container } = render(<GraphPreview nodes={mockNodes} edges={mockEdges} />);
      const graphContainer = container.querySelector('[style*="height"]');
      expect(graphContainer).toHaveStyle({ height: "200px" });
    });

    it("uses custom height when provided", () => {
      const { container } = render(
        <GraphPreview nodes={mockNodes} edges={mockEdges} height={300} />
      );
      const graphContainer = container.querySelector('[style*="height"]');
      expect(graphContainer).toHaveStyle({ height: "300px" });
    });
  });

  describe("accessibility", () => {
    it("has proper aria-label on expand button", () => {
      const handleExpand = jest.fn();
      render(<GraphPreview nodes={mockNodes} edges={mockEdges} onExpand={handleExpand} />);
      
      const expandButton = screen.getByRole("button", { name: /expand/i });
      expect(expandButton).toHaveAttribute("aria-label");
    });
  });

  describe("keyboard accessibility", () => {
    it("allows node activation via Enter key", () => {
      const handleNodeClick = jest.fn();
      render(
        <GraphPreview 
          nodes={mockNodes} 
          edges={mockEdges} 
          onNodeClick={handleNodeClick} 
        />
      );
      
      const node = screen.getByTestId("node-node-1");
      fireEvent.keyDown(node, { key: "Enter" });
      
      expect(handleNodeClick).toHaveBeenCalledWith(mockNodes[0]);
    });

    it("allows node activation via Space key", () => {
      const handleNodeClick = jest.fn();
      render(
        <GraphPreview 
          nodes={mockNodes} 
          edges={mockEdges} 
          onNodeClick={handleNodeClick} 
        />
      );
      
      const node = screen.getByTestId("node-node-1");
      fireEvent.keyDown(node, { key: " " });
      
      expect(handleNodeClick).toHaveBeenCalledWith(mockNodes[0]);
    });

    it("has focusable nodes with tabIndex", () => {
      render(<GraphPreview nodes={mockNodes} edges={mockEdges} />);
      
      const nodes = screen.getAllByRole("button");
      // Filter to get only the graph nodes (not expand button)
      const graphNodes = nodes.filter(node => 
        node.getAttribute("data-testid")?.startsWith("node-")
      );
      
      graphNodes.forEach(node => {
        expect(node).toHaveAttribute("tabIndex", "0");
      });
    });
  });

  describe("state synchronization", () => {
    it("syncs nodes when props change", () => {
      const { rerender } = render(
        <GraphPreview nodes={mockNodes} edges={mockEdges} />
      );

      const newNodes = [
        { id: "node-new", label: "New Node", type: "Concept" },
      ];

      rerender(<GraphPreview nodes={newNodes} edges={[]} />);

      // setNodes should have been called with the new transformed nodes
      expect(mockFns.setNodes).toHaveBeenCalled();
    });

    it("syncs edges when props change", () => {
      const { rerender } = render(
        <GraphPreview nodes={mockNodes} edges={mockEdges} />
      );

      const newEdges = [
        { id: "edge-new", source: "node-1", target: "node-2", label: "NEW_REL" },
      ];

      rerender(<GraphPreview nodes={mockNodes} edges={newEdges} />);

      // setEdges should have been called with the new transformed edges
      expect(mockFns.setEdges).toHaveBeenCalled();
    });
  });
});
