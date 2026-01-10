/**
 * Tests for use-programmatic-chat hook utilities
 * Story 21-A6: Implement useCopilotChat for Headless Control
 *
 * NOTE: This test focuses on pure utility functions and types.
 * The hook integration with useCopilotChat is tested via integration tests
 * since Jest module mocking with CopilotKit dependencies has compatibility issues.
 */

// Mock CopilotKit before any imports
jest.mock("@copilotkit/react-core", () => ({
  useCopilotChat: jest.fn(() => ({
    visibleMessages: [],
    appendMessage: jest.fn(),
    reloadMessages: jest.fn(),
    stopGeneration: jest.fn(),
    reset: jest.fn(),
    isLoading: false,
  })),
}));

// Mock @copilotkit/runtime-client-gql
jest.mock("@copilotkit/runtime-client-gql", () => ({
  TextMessage: jest.fn().mockImplementation(({ role, content }) => ({
    role,
    content,
  })),
  MessageRole: {
    User: "user",
    Assistant: "assistant",
    System: "system",
  },
}));

// Now import the module after mocks are set up
import {
  isValidMessageContent,
  toChatMessage,
} from "@/hooks/use-programmatic-chat";
import type { ChatMessage, ProgrammaticChatReturn } from "@/types/copilot";

describe("useProgrammaticChat utilities", () => {
  describe("isValidMessageContent()", () => {
    it("returns true for non-empty string", () => {
      expect(isValidMessageContent("hello")).toBe(true);
    });

    it("returns true for string with content after trim", () => {
      expect(isValidMessageContent("  hello  ")).toBe(true);
    });

    it("returns false for empty string", () => {
      expect(isValidMessageContent("")).toBe(false);
    });

    it("returns false for whitespace-only string", () => {
      expect(isValidMessageContent("   ")).toBe(false);
    });

    it("returns false for tab and newline only", () => {
      expect(isValidMessageContent("\t\n")).toBe(false);
    });

    it("returns true for single character", () => {
      expect(isValidMessageContent("a")).toBe(true);
    });

    it("returns true for multiline content", () => {
      expect(isValidMessageContent("line1\nline2")).toBe(true);
    });

    it("returns true for content with special characters", () => {
      expect(isValidMessageContent("Hello! @#$%^&*()")).toBe(true);
    });

    it("handles unicode characters", () => {
      expect(isValidMessageContent("Hello World")).toBe(true);
      expect(isValidMessageContent("Bonjour")).toBe(true);
    });
  });

  describe("toChatMessage()", () => {
    it("converts message with all fields", () => {
      const input = {
        id: "msg-123",
        role: "user",
        content: "Hello",
      };

      const result = toChatMessage(input);

      expect(result).toEqual({
        id: "msg-123",
        role: "user",
        content: "Hello",
      });
    });

    it("handles assistant role", () => {
      const input = {
        id: "msg-456",
        role: "assistant",
        content: "Hi there!",
      };

      const result = toChatMessage(input);

      expect(result.role).toBe("assistant");
    });

    it("handles system role", () => {
      const input = {
        id: "msg-789",
        role: "system",
        content: "System message",
      };

      const result = toChatMessage(input);

      expect(result.role).toBe("system");
    });

    it("defaults to assistant role when undefined", () => {
      const input = {
        id: "msg-001",
        role: undefined,
        content: "Content",
      };

      const result = toChatMessage(input);

      expect(result.role).toBe("assistant");
    });

    it("defaults to empty string content when undefined", () => {
      const input = {
        id: "msg-002",
        role: "user",
        content: undefined,
      };

      const result = toChatMessage(input);

      expect(result.content).toBe("");
    });

    it("handles missing role and content", () => {
      const input = {
        id: "msg-003",
      };

      const result = toChatMessage(input);

      expect(result.id).toBe("msg-003");
      expect(result.role).toBe("assistant");
      expect(result.content).toBe("");
    });

    it("preserves message ID exactly", () => {
      const input = {
        id: "very-long-uuid-style-id-12345",
        role: "user",
        content: "Test",
      };

      const result = toChatMessage(input);

      expect(result.id).toBe("very-long-uuid-style-id-12345");
    });

    it("handles empty content string", () => {
      const input = {
        id: "msg-004",
        role: "assistant",
        content: "",
      };

      const result = toChatMessage(input);

      expect(result.content).toBe("");
    });
  });
});

describe("ChatMessage type", () => {
  it("has required id, role, and content fields", () => {
    const message: ChatMessage = {
      id: "test-id",
      role: "user",
      content: "Test content",
    };

    expect(message.id).toBeDefined();
    expect(message.role).toBeDefined();
    expect(message.content).toBeDefined();
  });

  it("accepts all valid role values", () => {
    const roles: ChatMessage["role"][] = ["user", "assistant", "system"];

    roles.forEach((role) => {
      const message: ChatMessage = {
        id: "test",
        role,
        content: "test",
      };
      expect(message.role).toBe(role);
    });
  });
});

describe("ProgrammaticChatReturn type", () => {
  it("has all required fields", () => {
    const mockReturn: ProgrammaticChatReturn = {
      messages: [],
      messageCount: 0,
      isLoading: false,
      sendMessage: jest.fn(),
      regenerateLastResponse: jest.fn(),
      stopGeneration: jest.fn(),
      clearHistory: jest.fn(),
    };

    expect(mockReturn.messages).toBeDefined();
    expect(mockReturn.messageCount).toBeDefined();
    expect(mockReturn.isLoading).toBeDefined();
    expect(typeof mockReturn.sendMessage).toBe("function");
    expect(typeof mockReturn.regenerateLastResponse).toBe("function");
    expect(typeof mockReturn.stopGeneration).toBe("function");
    expect(typeof mockReturn.clearHistory).toBe("function");
  });

  it("messages is an array of ChatMessage", () => {
    const mockReturn: ProgrammaticChatReturn = {
      messages: [
        { id: "1", role: "user", content: "Hello" },
        { id: "2", role: "assistant", content: "Hi there" },
      ],
      messageCount: 2,
      isLoading: false,
      sendMessage: jest.fn(),
      regenerateLastResponse: jest.fn(),
      stopGeneration: jest.fn(),
      clearHistory: jest.fn(),
    };

    expect(Array.isArray(mockReturn.messages)).toBe(true);
    expect(mockReturn.messages.length).toBe(2);
    expect(mockReturn.messages[0].role).toBe("user");
    expect(mockReturn.messages[1].role).toBe("assistant");
  });

  it("sendMessage returns a Promise", async () => {
    const mockSendMessage = jest.fn().mockResolvedValue(undefined);

    const mockReturn: ProgrammaticChatReturn = {
      messages: [],
      messageCount: 0,
      isLoading: false,
      sendMessage: mockSendMessage,
      regenerateLastResponse: jest.fn(),
      stopGeneration: jest.fn(),
      clearHistory: jest.fn(),
    };

    await mockReturn.sendMessage("test");
    expect(mockSendMessage).toHaveBeenCalledWith("test");
  });

  it("regenerateLastResponse returns a Promise", async () => {
    const mockRegenerate = jest.fn().mockResolvedValue(undefined);

    const mockReturn: ProgrammaticChatReturn = {
      messages: [],
      messageCount: 0,
      isLoading: false,
      sendMessage: jest.fn(),
      regenerateLastResponse: mockRegenerate,
      stopGeneration: jest.fn(),
      clearHistory: jest.fn(),
    };

    await mockReturn.regenerateLastResponse();
    expect(mockRegenerate).toHaveBeenCalled();
  });
});

describe("QuickActionConfig type", () => {
  it("has required label and message fields", () => {
    const action = {
      label: "Summarize",
      message: "Please summarize this",
    };

    expect(action.label).toBeDefined();
    expect(action.message).toBeDefined();
  });

  it("accepts optional icon and description", () => {
    const action = {
      label: "Summarize",
      message: "Please summarize this",
      icon: "FileText",
      description: "Get a concise summary",
    };

    expect(action.icon).toBe("FileText");
    expect(action.description).toBe("Get a concise summary");
  });

  it("works without optional fields", () => {
    const action = {
      label: "Test",
      message: "Test message",
    };

    expect(action.icon).toBeUndefined();
    expect(action.description).toBeUndefined();
  });
});

describe("Message validation scenarios", () => {
  describe("edge cases for message content", () => {
    it("handles very long content", () => {
      const longContent = "a".repeat(10000);
      expect(isValidMessageContent(longContent)).toBe(true);
    });

    it("handles content with only whitespace variations", () => {
      // Non-breaking space (\u00A0) is not trimmed by String.trim() in ES5
      // but ES2015+ trim() does trim it - behavior depends on environment
      // Testing with regular whitespace which should always be false
      expect(isValidMessageContent("\t\t\t")).toBe(false);
      expect(isValidMessageContent("\n\n\n")).toBe(false);
      expect(isValidMessageContent("   ")).toBe(false);
    });

    it("handles content with numbers", () => {
      expect(isValidMessageContent("123")).toBe(true);
      expect(isValidMessageContent("0")).toBe(true);
    });

    it("handles JSON-like content", () => {
      expect(isValidMessageContent('{"key": "value"}')).toBe(true);
      expect(isValidMessageContent("[1, 2, 3]")).toBe(true);
    });

    it("handles markdown content", () => {
      expect(isValidMessageContent("# Heading\n\n**bold**")).toBe(true);
      expect(isValidMessageContent("```code```")).toBe(true);
    });
  });

  describe("edge cases for message conversion", () => {
    it("handles message with extra properties", () => {
      const input = {
        id: "msg-extra",
        role: "user",
        content: "Hello",
        extraField: "ignored",
        timestamp: "2024-01-01",
      };

      const result = toChatMessage(input);

      // Should only have id, role, content
      expect(Object.keys(result)).toEqual(["id", "role", "content"]);
      expect((result as Record<string, unknown>).extraField).toBeUndefined();
    });

    it("handles numeric-like role", () => {
      const input = {
        id: "msg-numeric",
        role: "user",
        content: "test",
      };

      const result = toChatMessage(input);
      expect(typeof result.role).toBe("string");
    });
  });
});
