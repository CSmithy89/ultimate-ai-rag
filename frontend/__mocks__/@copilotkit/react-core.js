/**
 * Mock for @copilotkit/react-core
 *
 * This mock provides stub implementations for CopilotKit hooks
 * to allow tests to run without the ESM transformation issues.
 */

const React = require('react');

// Mock hooks that return stable values
const useCopilotChat = jest.fn(() => ({
  appendMessage: jest.fn(),
  setMessages: jest.fn(),
  messages: [],
  isLoading: false,
  error: null,
}));

const useCopilotReadable = jest.fn();

const useCopilotAction = jest.fn();

const useCoAgent = jest.fn(() => ({
  state: undefined,
  setState: jest.fn(),
}));

const useCoAgentStateRender = jest.fn();

const useFrontendTool = jest.fn();

const useCopilotAdditionalInstructions = jest.fn();

const useHumanInTheLoop = jest.fn(() => ({
  state: null,
  respond: jest.fn(),
  isLoading: false,
}));

// Mock CopilotKit provider component
const CopilotKit = ({ children }) => React.createElement('div', { 'data-testid': 'copilotkit-provider' }, children);

module.exports = {
  useCopilotChat,
  useCopilotReadable,
  useCopilotAction,
  useCoAgent,
  useCoAgentStateRender,
  useFrontendTool,
  useCopilotAdditionalInstructions,
  useHumanInTheLoop,
  CopilotKit,
};
