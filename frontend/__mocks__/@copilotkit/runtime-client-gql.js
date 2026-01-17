/**
 * Mock for @copilotkit/runtime-client-gql
 *
 * This mock provides stub implementations for CopilotKit runtime types
 * to allow tests to run without the ESM transformation issues.
 */

// MessageRole enum
const MessageRole = {
  User: 'user',
  Assistant: 'assistant',
  System: 'system',
};

// TextMessage class mock
class TextMessage {
  constructor({ content, role }) {
    this.content = content;
    this.role = role;
  }
}

module.exports = {
  MessageRole,
  TextMessage,
};
