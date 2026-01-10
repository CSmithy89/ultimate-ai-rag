/**
 * Tests for A2UI Widget Renderer Component
 *
 * Story 21-D2: Implement A2UI Widget Renderer
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";

// Mock react-markdown (ESM module)
jest.mock("react-markdown", () => {
  return {
    __esModule: true,
    default: ({ children }: { children: string }) => <div>{children}</div>,
  };
});

// Mock remark-gfm (ESM module)
jest.mock("remark-gfm", () => ({
  __esModule: true,
  default: () => {},
}));

// Mock lucide-react icons
jest.mock("lucide-react", () => ({
  FileText: () => <span data-testid="icon-file-text" />,
  Table: () => <span data-testid="icon-table" />,
  FormInput: () => <span data-testid="icon-form-input" />,
  BarChart3: () => <span data-testid="icon-bar-chart" />,
  Image: () => <span data-testid="icon-image" />,
  List: () => <span data-testid="icon-list" />,
  AlertCircle: () => <span data-testid="icon-alert-circle" />,
  ExternalLink: () => <span data-testid="icon-external-link" />,
  Check: () => <span data-testid="icon-check" />,
}));

// Mock the CopilotKit hook
const mockState: { a2ui_widgets?: unknown[] } = {};
jest.mock("@copilotkit/react-core", () => ({
  useCoAgentStateRender: jest.fn(({ render }) => {
    // Call render with mock state
    const result = render({ state: mockState });
    if (result) {
      // Store rendered component for testing
      (global as unknown as { __a2uiRendered: React.ReactNode }).__a2uiRendered = result;
    }
    return null;
  }),
}));

// Import after mocks
import { A2UIRenderer } from "@/components/copilot/A2UIRenderer";

// Helper to render with state
function renderWithState(widgets: unknown[]) {
  mockState.a2ui_widgets = widgets;
  render(<A2UIRenderer />);
  const rendered = (global as unknown as { __a2uiRendered: React.ReactNode }).__a2uiRendered;
  if (rendered) {
    const { unmount } = render(<>{rendered}</>);
    return unmount;
  }
}

describe("A2UIRenderer", () => {
  beforeEach(() => {
    mockState.a2ui_widgets = undefined;
  });

  describe("Card Widget", () => {
    it("renders a card widget with title and content", () => {
      renderWithState([
        {
          type: "card",
          properties: {
            title: "Test Card",
            content: "This is card content",
          },
        },
      ]);

      expect(screen.getByText("Test Card")).toBeInTheDocument();
      expect(screen.getByText("This is card content")).toBeInTheDocument();
    });

    it("renders card with subtitle and footer", () => {
      renderWithState([
        {
          type: "card",
          properties: {
            title: "Full Card",
            content: "Content",
            subtitle: "Subtitle text",
            footer: "Footer text",
          },
        },
      ]);

      expect(screen.getByText("Subtitle text")).toBeInTheDocument();
      expect(screen.getByText("Footer text")).toBeInTheDocument();
    });

    it("renders card action buttons", () => {
      const onAction = jest.fn();
      mockState.a2ui_widgets = [
        {
          type: "card",
          properties: {
            title: "Card with Actions",
            content: "Content",
            actions: [
              { label: "Primary", action: "primary_action", variant: "primary" },
              { label: "Secondary", action: "secondary_action" },
            ],
          },
        },
      ];
      render(<A2UIRenderer onAction={onAction} />);
      const rendered = (global as unknown as { __a2uiRendered: React.ReactNode }).__a2uiRendered;
      render(<>{rendered}</>);

      const primaryBtn = screen.getByText("Primary");
      const secondaryBtn = screen.getByText("Secondary");

      expect(primaryBtn).toBeInTheDocument();
      expect(secondaryBtn).toBeInTheDocument();

      fireEvent.click(primaryBtn);
      expect(onAction).toHaveBeenCalledWith("primary_action");
    });
  });

  describe("Table Widget", () => {
    it("renders a table with headers and rows", () => {
      renderWithState([
        {
          type: "table",
          properties: {
            headers: ["Name", "Value"],
            rows: [
              ["Item 1", "100"],
              ["Item 2", "200"],
            ],
            caption: "Test Table",
          },
        },
      ]);

      expect(screen.getByText("Test Table")).toBeInTheDocument();
      expect(screen.getByText("Name")).toBeInTheDocument();
      expect(screen.getByText("Value")).toBeInTheDocument();
      expect(screen.getByText("Item 1")).toBeInTheDocument();
      expect(screen.getByText("200")).toBeInTheDocument();
    });
  });

  describe("Form Widget", () => {
    it("renders a form with fields", () => {
      renderWithState([
        {
          type: "form",
          properties: {
            title: "Contact Form",
            fields: [
              { name: "name", label: "Name", type: "text", required: true },
              { name: "email", label: "Email", type: "email" },
            ],
            submitLabel: "Send",
          },
        },
      ]);

      expect(screen.getByText("Contact Form")).toBeInTheDocument();
      expect(screen.getByLabelText(/Name/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Email/)).toBeInTheDocument();
      expect(screen.getByText("Send")).toBeInTheDocument();
    });

    it("calls onFormSubmit when form is submitted", () => {
      const onFormSubmit = jest.fn();
      mockState.a2ui_widgets = [
        {
          type: "form",
          properties: {
            title: "Test Form",
            fields: [{ name: "query", label: "Query", type: "text" }],
            submitAction: "search",
          },
        },
      ];
      render(<A2UIRenderer onFormSubmit={onFormSubmit} />);
      const rendered = (global as unknown as { __a2uiRendered: React.ReactNode }).__a2uiRendered;
      render(<>{rendered}</>);

      const input = screen.getByLabelText(/Query/);
      fireEvent.change(input, { target: { value: "test query" } });

      const submitBtn = screen.getByText("Submit");
      fireEvent.click(submitBtn);

      expect(onFormSubmit).toHaveBeenCalledWith("search", expect.objectContaining({
        query: "test query",
      }));
    });
  });

  describe("Chart Widget", () => {
    it("renders a bar chart", () => {
      renderWithState([
        {
          type: "chart",
          properties: {
            chartType: "bar",
            data: [
              { month: "Jan", sales: 100 },
              { month: "Feb", sales: 200 },
            ],
            xKey: "month",
            yKey: "sales",
            title: "Sales Chart",
          },
        },
      ]);

      expect(screen.getByText("Sales Chart")).toBeInTheDocument();
      expect(screen.getByText("Jan")).toBeInTheDocument();
      expect(screen.getByText("100")).toBeInTheDocument();
    });

    it("renders chart data as fallback for non-bar types", () => {
      renderWithState([
        {
          type: "chart",
          properties: {
            chartType: "line",
            data: [{ x: 1, y: 2 }],
            xKey: "x",
            yKey: "y",
          },
        },
      ]);

      expect(screen.getByText(/Chart data/)).toBeInTheDocument();
    });
  });

  describe("Image Widget", () => {
    it("renders an image with alt text and caption", () => {
      renderWithState([
        {
          type: "image",
          properties: {
            url: "https://example.com/image.png",
            alt: "Example Image",
            caption: "Image caption",
          },
        },
      ]);

      const img = screen.getByAltText("Example Image");
      expect(img).toBeInTheDocument();
      expect(img).toHaveAttribute("src", "https://example.com/image.png");
      expect(screen.getByText("Image caption")).toBeInTheDocument();
    });
  });

  describe("List Widget", () => {
    it("renders a list with items", () => {
      renderWithState([
        {
          type: "list",
          properties: {
            title: "Task List",
            items: [
              { text: "Task 1", description: "Description 1" },
              { text: "Task 2", badge: "New" },
            ],
          },
        },
      ]);

      expect(screen.getByText("Task List")).toBeInTheDocument();
      expect(screen.getByText("Task 1")).toBeInTheDocument();
      expect(screen.getByText("Description 1")).toBeInTheDocument();
      expect(screen.getByText("Task 2")).toBeInTheDocument();
      expect(screen.getByText("New")).toBeInTheDocument();
    });

    it("renders ordered list when ordered prop is true", () => {
      renderWithState([
        {
          type: "list",
          properties: {
            items: [{ text: "Step 1" }, { text: "Step 2" }],
            ordered: true,
          },
        },
      ]);

      // The list should be an ordered list
      expect(screen.getByRole("list")).toHaveClass("list-decimal");
    });
  });

  describe("Fallback Widget", () => {
    it("renders fallback for unknown widget types", () => {
      renderWithState([
        {
          type: "unknown_type",
          properties: { foo: "bar" },
        },
      ]);

      expect(screen.getByText(/Unsupported widget: unknown_type/)).toBeInTheDocument();
    });
  });

  describe("Empty State", () => {
    it("renders nothing when no widgets are present", () => {
      // Reset global before test
      (global as unknown as { __a2uiRendered: React.ReactNode }).__a2uiRendered = null;
      mockState.a2ui_widgets = [];
      render(<A2UIRenderer />);

      // The rendered result should be null when no widgets
      const rendered = (global as unknown as { __a2uiRendered: React.ReactNode }).__a2uiRendered;
      expect(rendered).toBeNull();
    });

    it("renders nothing when widgets is undefined", () => {
      // Reset global before test
      (global as unknown as { __a2uiRendered: React.ReactNode }).__a2uiRendered = null;
      mockState.a2ui_widgets = undefined;
      render(<A2UIRenderer />);

      const rendered = (global as unknown as { __a2uiRendered: React.ReactNode }).__a2uiRendered;
      expect(rendered).toBeNull();
    });
  });
});
