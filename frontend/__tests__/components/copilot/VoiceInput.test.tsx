/**
 * VoiceInput component tests.
 *
 * Story 21-E1: Implement Voice Input (Speech-to-Text)
 *
 * Tests cover:
 * - Component rendering
 * - Recording state management
 * - Transcription flow
 * - Error handling
 * - Accessibility
 * - Keyboard interactions
 */

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { VoiceInput } from "@/components/copilot/VoiceInput";

// Mock MediaRecorder
const mockMediaRecorder = {
  start: jest.fn(),
  stop: jest.fn(),
  ondataavailable: null as ((event: { data: Blob }) => void) | null,
  onstop: null as (() => void) | null,
  state: "inactive",
};

const mockMediaStream = {
  getTracks: jest.fn(() => [{ stop: jest.fn() }]),
};

// Mock navigator.mediaDevices
const mockGetUserMedia = jest.fn();

// Store original globals for restoration
const originalMediaDevices = global.navigator.mediaDevices;
// @ts-expect-error - MediaRecorder may not exist in jsdom
const originalMediaRecorder = global.MediaRecorder;
const originalFetch = global.fetch;

beforeAll(() => {
  Object.defineProperty(global.navigator, "mediaDevices", {
    value: {
      getUserMedia: mockGetUserMedia,
    },
    writable: true,
  });

  // @ts-expect-error - Mock MediaRecorder
  global.MediaRecorder = jest.fn().mockImplementation(() => mockMediaRecorder);
  // @ts-expect-error - Mock isTypeSupported
  global.MediaRecorder.isTypeSupported = jest.fn(() => true);
});

afterAll(() => {
  // Restore original globals
  Object.defineProperty(global.navigator, "mediaDevices", {
    value: originalMediaDevices,
    writable: true,
  });
  // @ts-expect-error - Restore original MediaRecorder
  global.MediaRecorder = originalMediaRecorder;
  global.fetch = originalFetch;
});

beforeEach(() => {
  jest.clearAllMocks();
  mockGetUserMedia.mockResolvedValue(mockMediaStream);
  mockMediaRecorder.state = "inactive";
  // Reset fetch to original before each test
  global.fetch = originalFetch;
});

describe("VoiceInput", () => {
  const defaultProps = {
    onTranscription: jest.fn(),
  };

  describe("Rendering", () => {
    it("renders microphone button", () => {
      render(<VoiceInput {...defaultProps} />);

      const button = screen.getByRole("button", { name: /start voice input/i });
      expect(button).toBeInTheDocument();
    });

    it("renders disabled when disabled prop is true", () => {
      render(<VoiceInput {...defaultProps} disabled />);

      const button = screen.getByRole("button");
      expect(button).toBeDisabled();
    });

    it("applies custom className", () => {
      render(<VoiceInput {...defaultProps} className="custom-class" />);

      const container = screen.getByRole("button").parentElement;
      expect(container).toHaveClass("custom-class");
    });
  });

  describe("Recording", () => {
    it("starts recording on click", async () => {
      render(<VoiceInput {...defaultProps} />);

      const button = screen.getByRole("button", { name: /start voice input/i });
      await userEvent.click(button);

      expect(mockGetUserMedia).toHaveBeenCalledWith({ audio: true });
      expect(global.MediaRecorder).toHaveBeenCalled();
      expect(mockMediaRecorder.start).toHaveBeenCalled();
    });

    it("shows recording indicator when recording", async () => {
      render(<VoiceInput {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      // Button should show stop label
      await waitFor(() => {
        expect(screen.getByRole("button", { name: /stop recording/i })).toBeInTheDocument();
      });
    });

    it("stops recording on second click", async () => {
      render(<VoiceInput {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button); // Start

      mockMediaRecorder.state = "recording";
      await userEvent.click(button); // Stop

      expect(mockMediaRecorder.stop).toHaveBeenCalled();
    });
  });

  describe("Error Handling", () => {
    it("shows error when microphone permission denied", async () => {
      const permissionError = new DOMException("Permission denied", "NotAllowedError");
      mockGetUserMedia.mockRejectedValueOnce(permissionError);

      render(<VoiceInput {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(screen.getByRole("alert")).toHaveTextContent(/microphone permission denied/i);
      });
    });

    it("shows error when no microphone found", async () => {
      const notFoundError = new DOMException("No microphone", "NotFoundError");
      mockGetUserMedia.mockRejectedValueOnce(notFoundError);

      render(<VoiceInput {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      await waitFor(() => {
        expect(screen.getByRole("alert")).toHaveTextContent(/no microphone found/i);
      });
    });

    it("dismisses error on click", async () => {
      const permissionError = new DOMException("Permission denied", "NotAllowedError");
      mockGetUserMedia.mockRejectedValueOnce(permissionError);

      render(<VoiceInput {...defaultProps} />);

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
      const permissionError = new DOMException("Permission denied", "NotAllowedError");
      mockGetUserMedia.mockRejectedValueOnce(permissionError);

      render(<VoiceInput {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      expect(await screen.findByRole("alert")).toBeInTheDocument();

      jest.advanceTimersByTime(5000);

      await waitFor(() => {
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      });

      jest.useRealTimers();
    });
  });

  describe("Keyboard Accessibility", () => {
    it("cancels recording on Escape key", async () => {
      render(<VoiceInput {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button); // Start recording

      mockMediaRecorder.state = "recording";

      fireEvent.keyDown(button, { key: "Escape" });

      expect(mockMediaRecorder.stop).toHaveBeenCalled();
    });
  });

  describe("Accessibility", () => {
    it("has accessible button label", () => {
      render(<VoiceInput {...defaultProps} />);

      const button = screen.getByRole("button");
      expect(button).toHaveAccessibleName(/voice input/i);
    });

    it("has ARIA live region for status updates", () => {
      render(<VoiceInput {...defaultProps} />);

      const liveRegion = screen.getByRole("status");
      expect(liveRegion).toBeInTheDocument();
    });
  });

  describe("Transcription", () => {
    it("calls onTranscription with transcribed text", async () => {
      const onTranscription = jest.fn();
      global.fetch = jest.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => ({ text: "Hello, world!" }),
      });

      render(<VoiceInput onTranscription={onTranscription} />);

      const button = screen.getByRole("button");
      await userEvent.click(button); // Start

      // Simulate recording data available
      const blob = new Blob(["audio data"], { type: "audio/webm" });
      mockMediaRecorder.ondataavailable?.({ data: blob });

      // Simulate recording stop
      mockMediaRecorder.onstop?.();

      await waitFor(() => {
        expect(onTranscription).toHaveBeenCalledWith("Hello, world!");
      });
    });

    it("shows transcribing state during API call", async () => {
      let resolvePromise: (value: unknown) => void;
      const pendingPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      global.fetch = jest.fn().mockReturnValueOnce(pendingPromise);

      render(<VoiceInput {...defaultProps} />);

      const button = screen.getByRole("button");
      await userEvent.click(button);

      const blob = new Blob(["audio data"], { type: "audio/webm" });
      mockMediaRecorder.ondataavailable?.({ data: blob });
      mockMediaRecorder.onstop?.();

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /transcribing/i })).toBeInTheDocument();
      });

      // Cleanup
      resolvePromise!({ ok: true, json: async () => ({ text: "test" }) });
    });
  });
});
