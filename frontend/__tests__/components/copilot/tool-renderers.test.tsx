import React from "react";
import { render, screen } from "@testing-library/react";
import { useToolCallRenderers } from "@/components/copilot/tool-renderers";

const renderers = new Map<string, (props: any) => React.ReactElement>();

jest.mock("@copilotkit/react-core", () => ({
  useRenderToolCall: ({ name, render }: { name: string; render: any }) => {
    renderers.set(name, render);
  },
}));

jest.mock("@/components/mcp-ui/MCPUIRenderer", () => ({
  MCPUIRenderer: ({ payload }: { payload: { tool_name: string } }) => (
    <div data-testid="mcp-ui-renderer">{payload.tool_name}</div>
  ),
}));

jest.mock("@/components/open-json-ui/OpenJSONUIRenderer", () => ({
  OpenJSONUIRenderer: () => <div data-testid="open-json-ui-renderer" />,
}));

jest.mock("@/components/copilot/MCPToolCallCard", () => ({
  MCPToolCallCard: ({ name }: { name: string }) => (
    <div data-testid="mcp-tool-card">{name}</div>
  ),
}));

jest.mock("@/components/copilot/CopilotErrorBoundary", () => ({
  CopilotErrorBoundary: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
}));

function Harness() {
  useToolCallRenderers();
  return null;
}

describe("tool renderers", () => {
  beforeEach(() => {
    renderers.clear();
  });

  it("renders MCP-UI payloads with MCPUIRenderer", () => {
    render(<Harness />);
    const renderer = renderers.get("*");
    expect(renderer).toBeDefined();

    const element = renderer!({
      name: "custom_tool",
      args: { tenant_id: "tenant-1" },
      status: "Complete",
      result: {
        type: "mcp_ui",
        tool_name: "example-ui",
        ui_url: "https://example.com/ui",
        ui_type: "iframe",
        sandbox: ["allow-scripts"],
        size: { width: 600, height: 400 },
        allow: [],
        data: {},
      },
    });

    render(element);
    expect(screen.getByTestId("mcp-ui-renderer")).toHaveTextContent("example-ui");
  });

  it("renders Open-JSON-UI payloads with OpenJSONUIRenderer", () => {
    render(<Harness />);
    const renderer = renderers.get("*");
    expect(renderer).toBeDefined();

    const element = renderer!({
      name: "custom_tool",
      args: {},
      status: "Complete",
      result: {
        type: "open_json_ui",
        components: [{ type: "text", content: "hello" }],
      },
    });

    render(element);
    expect(screen.getByTestId("open-json-ui-renderer")).toBeInTheDocument();
  });

  it("falls back to MCPToolCallCard for malformed Open-JSON-UI payloads", () => {
    render(<Harness />);
    const renderer = renderers.get("*");
    expect(renderer).toBeDefined();

    const element = renderer!({
      name: "custom_tool",
      args: {},
      status: "Complete",
      result: {
        type: "open_json_ui",
        components: [{ type: "unknown", content: "bad" }],
      },
    });

    render(element);
    expect(screen.getByTestId("mcp-tool-card")).toHaveTextContent("custom_tool");
  });

  it("falls back to MCPToolCallCard for malformed MCP-UI payloads", () => {
    render(<Harness />);
    const renderer = renderers.get("*");
    expect(renderer).toBeDefined();

    const element = renderer!({
      name: "custom_tool",
      args: {},
      status: "Complete",
      result: {
        type: "mcp_ui",
        tool_name: "bad-ui",
        ui_type: "iframe",
      },
    });

    render(element);
    expect(screen.getByTestId("mcp-tool-card")).toHaveTextContent("custom_tool");
  });
});
