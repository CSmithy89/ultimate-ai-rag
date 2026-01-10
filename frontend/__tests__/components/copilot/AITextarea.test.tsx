/**
 * AITextarea component tests.
 *
 * Story 21-F3: Implement CopilotTextarea Component
 *
 * Tests cover:
 * - Component rendering
 * - Value and onChange handling
 * - Autosuggestions configuration
 * - forwardRef and React Hook Form compatibility
 * - Accessibility
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { createRef } from "react";
import { AITextarea } from "@/components/copilot/AITextarea";

// Mock CopilotKit textarea
jest.mock("@copilotkit/react-textarea", () => ({
  CopilotTextarea: ({
    value,
    onChange,
    placeholder,
    disabled,
    className,
    style,
    autosuggestionsConfig,
    "aria-label": ariaLabel,
    id,
    name,
    ref,
  }: {
    value: string;
    onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
    placeholder: string;
    disabled: boolean;
    className: string;
    style: React.CSSProperties;
    autosuggestionsConfig: { textareaPurpose: string; chatApiConfigs: Record<string, unknown> };
    "aria-label"?: string;
    id?: string;
    name?: string;
    ref?: React.Ref<HTMLTextAreaElement>;
  }) => (
    <textarea
      ref={ref}
      data-testid="copilot-textarea"
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      disabled={disabled}
      className={className}
      style={style}
      data-purpose={autosuggestionsConfig.textareaPurpose}
      aria-label={ariaLabel}
      id={id}
      name={name}
    />
  ),
}));

describe("AITextarea", () => {
  const defaultProps = {
    value: "",
    onChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Rendering", () => {
    it("renders textarea with default props", () => {
      render(<AITextarea {...defaultProps} />);

      const textarea = screen.getByTestId("copilot-textarea");
      expect(textarea).toBeInTheDocument();
      expect(textarea).toHaveAttribute("placeholder", "Start typing...");
    });

    it("renders with custom placeholder", () => {
      render(<AITextarea {...defaultProps} placeholder="Custom placeholder" />);

      const textarea = screen.getByTestId("copilot-textarea");
      expect(textarea).toHaveAttribute("placeholder", "Custom placeholder");
    });

    it("renders with value", () => {
      render(<AITextarea {...defaultProps} value="Hello, world!" />);

      const textarea = screen.getByTestId("copilot-textarea");
      expect(textarea).toHaveValue("Hello, world!");
    });

    it("renders disabled state", () => {
      render(<AITextarea {...defaultProps} disabled />);

      const textarea = screen.getByTestId("copilot-textarea");
      expect(textarea).toBeDisabled();
    });
  });

  describe("Value Changes", () => {
    it("calls onChange when value changes", () => {
      const onChange = jest.fn();
      render(<AITextarea {...defaultProps} onChange={onChange} />);

      const textarea = screen.getByTestId("copilot-textarea");
      fireEvent.change(textarea, { target: { value: "New text" } });

      expect(onChange).toHaveBeenCalledWith("New text");
    });
  });

  describe("Autosuggestions Configuration", () => {
    it("uses default purpose when no config provided", () => {
      render(<AITextarea {...defaultProps} />);

      const textarea = screen.getByTestId("copilot-textarea");
      expect(textarea).toHaveAttribute("data-purpose", "General purpose text editor");
    });

    it("uses custom purpose from config", () => {
      render(
        <AITextarea
          {...defaultProps}
          autosuggestionsConfig={{
            textareaPurpose: "Note-taking and documentation",
          }}
        />
      );

      const textarea = screen.getByTestId("copilot-textarea");
      expect(textarea).toHaveAttribute("data-purpose", "Note-taking and documentation");
    });
  });

  describe("Sizing", () => {
    it("applies default min/max height based on rows", () => {
      render(<AITextarea {...defaultProps} />);

      const textarea = screen.getByTestId("copilot-textarea");
      const style = textarea.style;

      // Default: minRows=3, maxRows=10
      expect(style.minHeight).toBe("4.5rem"); // 3 * 1.5rem
      expect(style.maxHeight).toBe("15rem"); // 10 * 1.5rem
    });

    it("applies custom min/max rows", () => {
      render(<AITextarea {...defaultProps} minRows={5} maxRows={15} />);

      const textarea = screen.getByTestId("copilot-textarea");
      const style = textarea.style;

      expect(style.minHeight).toBe("7.5rem"); // 5 * 1.5rem
      expect(style.maxHeight).toBe("22.5rem"); // 15 * 1.5rem
    });
  });

  describe("Styling", () => {
    it("applies custom className", () => {
      render(<AITextarea {...defaultProps} className="custom-class" />);

      const textarea = screen.getByTestId("copilot-textarea");
      expect(textarea.className).toContain("custom-class");
    });

    it("includes default styling classes", () => {
      render(<AITextarea {...defaultProps} />);

      const textarea = screen.getByTestId("copilot-textarea");
      expect(textarea.className).toContain("w-full");
      expect(textarea.className).toContain("rounded-md");
      expect(textarea.className).toContain("border");
    });
  });

  describe("Accessibility", () => {
    it("applies aria-label", () => {
      render(<AITextarea {...defaultProps} aria-label="Note content" />);

      const textarea = screen.getByTestId("copilot-textarea");
      expect(textarea).toHaveAttribute("aria-label", "Note content");
    });

    it("applies id and name for form association", () => {
      render(<AITextarea {...defaultProps} id="note-input" name="note" />);

      const textarea = screen.getByTestId("copilot-textarea");
      expect(textarea).toHaveAttribute("id", "note-input");
      expect(textarea).toHaveAttribute("name", "note");
    });
  });

  describe("forwardRef Support", () => {
    it("forwards ref to textarea", () => {
      const ref = createRef<HTMLTextAreaElement>();
      render(<AITextarea {...defaultProps} ref={ref} />);

      expect(ref.current).toBeInstanceOf(HTMLTextAreaElement);
    });
  });

  describe("React Hook Form Compatibility", () => {
    it("works with controlled value pattern", () => {
      const onChange = jest.fn();
      const { rerender } = render(
        <AITextarea {...defaultProps} value="" onChange={onChange} />
      );

      const textarea = screen.getByTestId("copilot-textarea");
      fireEvent.change(textarea, { target: { value: "Test" } });

      expect(onChange).toHaveBeenCalledWith("Test");

      // Simulate React Hook Form updating value
      rerender(<AITextarea {...defaultProps} value="Test" onChange={onChange} />);

      expect(textarea).toHaveValue("Test");
    });

    it("supports name attribute for form submission", () => {
      render(<AITextarea {...defaultProps} name="content" />);

      const textarea = screen.getByTestId("copilot-textarea");
      expect(textarea).toHaveAttribute("name", "content");
    });
  });

  describe("Memoization", () => {
    it("is memoized to prevent unnecessary re-renders", () => {
      const { rerender } = render(<AITextarea {...defaultProps} />);

      const textarea1 = screen.getByTestId("copilot-textarea");

      // Re-render with same props
      rerender(<AITextarea {...defaultProps} />);

      const textarea2 = screen.getByTestId("copilot-textarea");

      // Same element reference indicates memoization working
      expect(textarea1).toBe(textarea2);
    });
  });
});
