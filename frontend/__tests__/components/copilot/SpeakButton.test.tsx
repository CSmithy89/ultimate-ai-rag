/**
 * SpeakButton component tests.
 *
 * Story 21-E2: Implement Voice Output (Text-to-Speech)
 *
 * Tests cover:
 * - Component rendering
 * - Audio playback
 * - Stop functionality
 * - Error handling
 * - Accessibility
 * - Loading states
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SpeakButton } from "@/components/copilot/SpeakButton";

// Mock Audio
const mockAudio = {
  play: jest.fn().mockResolvedValue(undefined),
  pause: jest.fn(),
  currentTime: 0,
  onended: null as (() => void) | null,
  onerror: null as (() => void) | null,
};

beforeAll(() => {
  // @ts-expect-error - Mock Audio
  global.Audio = jest.fn().mockImplementation(() => mockAudio);

  // Mock URL.createObjectURL and revokeObjectURL
  global.URL.createObjectURL = jest.fn(() => "blob:mock-url");
  global.URL.revokeObjectURL = jest.fn();
});

beforeEach(() => {
  jest.clearAllMocks();
  mockAudio.currentTime = 0;
  mockAudio.onended = null;
  mockAudio.onerror = null;
});

describe("SpeakButton", () => {
  const defaultProps = {
    text: "Hello, world!",
  };

  describe("Rendering", () => {
    it("renders speaker button", () => {
      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button", { name: /read aloud/i });
      expect(button).toBeInTheDocument();
    });

    it("renders disabled when disabled prop is true", () => {
      render(<SpeakButton {...defaultProps} disabled />);

      const button = screen.getByRole("button");
      expect(button).toBeDisabled();
    });

    it("renders disabled when text is empty", () => {
      render(<SpeakButton text="" />);

      const button = screen.getByRole("button");
      expect(button).toBeDisabled();
    });

    it("applies custom className", () => {
      render(<SpeakButton {...defaultProps} className="custom-class" />);

      const container = screen.getByRole("button").parentElement;
      expect(container).toHaveClass("custom-class");
    });
  });

  describe("Audio Playback", () => {
    beforeEach(() => {
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        blob: async () => new Blob(["audio data"], { type: "audio/mpeg" }),
      });
    });

    it("starts playback on click", async () => {
      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith("/api/copilot/tts", expect.any(Object));
      });

      await waitFor(() => {
        expect(mockAudio.play).toHaveBeenCalled();
      });
    });

    it("shows loading state during API call", async () => {
      let resolvePromise: (value: unknown) => void;
      const pendingPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      global.fetch = jest.fn().mockReturnValueOnce(pendingPromise);

      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /loading/i })).toBeInTheDocument();
      });

      // Cleanup
      resolvePromise!({
        ok: true,
        blob: async () => new Blob(["audio"], { type: "audio/mpeg" }),
      });
    });

    it("shows playing state during playback", async () => {
      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /stop speaking/i })).toBeInTheDocument();
      });
    });

    it("stops playback on second click", async () => {
      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button); // Start

      await waitFor(() => {
        expect(mockAudio.play).toHaveBeenCalled();
      });

      await userEvent.click(button); // Stop

      expect(mockAudio.pause).toHaveBeenCalled();
    });

    it("resets state when audio ends", async () => {
      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(mockAudio.play).toHaveBeenCalled();
      });

      // Trigger audio ended event
      mockAudio.onended?.();

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /read aloud/i })).toBeInTheDocument();
      });
    });
  });

  describe("Voice Configuration", () => {
    beforeEach(() => {
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        blob: async () => new Blob(["audio"], { type: "audio/mpeg" }),
      });
    });

    it("sends voice parameter to API", async () => {
      render(<SpeakButton {...defaultProps} voice="nova" />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          "/api/copilot/tts",
          expect.objectContaining({
            body: expect.stringContaining('"voice":"nova"'),
          })
        );
      });
    });

    it("sends speed parameter to API", async () => {
      render(<SpeakButton {...defaultProps} speed={1.5} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          "/api/copilot/tts",
          expect.objectContaining({
            body: expect.stringContaining('"speed":1.5'),
          })
        );
      });
    });
  });

  describe("Error Handling", () => {
    it("shows error when API fails", async () => {
      global.fetch = jest.fn().mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ detail: "TTS service unavailable" }),
      });

      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(screen.getByRole("alert")).toHaveTextContent(/TTS service unavailable/i);
      });
    });

    it("shows error when audio playback fails", async () => {
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        blob: async () => new Blob(["audio"], { type: "audio/mpeg" }),
      });

      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(mockAudio.play).toHaveBeenCalled();
      });

      // Trigger audio error
      mockAudio.onerror?.();

      await waitFor(() => {
        expect(screen.getByRole("alert")).toHaveTextContent(/audio playback failed/i);
      });
    });

    it("dismisses error on click", async () => {
      global.fetch = jest.fn().mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ detail: "Error" }),
      });

      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      const errorAlert = await screen.findByRole("alert");
      await userEvent.click(errorAlert);

      await waitFor(() => {
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      });
    });

    it("auto-dismisses error after 5 seconds", async () => {
      jest.useFakeTimers();
      global.fetch = jest.fn().mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ detail: "Error" }),
      });

      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      expect(await screen.findByRole("alert")).toBeInTheDocument();

      jest.advanceTimersByTime(5000);

      await waitFor(() => {
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      });

      jest.useRealTimers();
    });

    it("revokes object URL on error to prevent memory leak", async () => {
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        blob: async () => new Blob(["audio"], { type: "audio/mpeg" }),
      });

      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(mockAudio.play).toHaveBeenCalled();
      });

      // Trigger audio error
      mockAudio.onerror?.();

      expect(global.URL.revokeObjectURL).toHaveBeenCalled();
    });
  });

  describe("Accessibility", () => {
    it("has accessible button label", () => {
      render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      expect(button).toHaveAccessibleName(/read aloud/i);
    });

    it("has ARIA live region for status updates", () => {
      render(<SpeakButton {...defaultProps} />);

      const liveRegion = screen.getByRole("status");
      expect(liveRegion).toBeInTheDocument();
    });
  });

  describe("Cleanup", () => {
    it("cleans up audio on unmount", async () => {
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        blob: async () => new Blob(["audio"], { type: "audio/mpeg" }),
      });

      const { unmount } = render(<SpeakButton {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(mockAudio.play).toHaveBeenCalled();
      });

      unmount();

      expect(mockAudio.pause).toHaveBeenCalled();
      expect(global.URL.revokeObjectURL).toHaveBeenCalled();
    });
  });
});
