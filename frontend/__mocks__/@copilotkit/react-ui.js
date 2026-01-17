/**
 * Mock for @copilotkit/react-ui
 *
 * This mock provides stub implementations for CopilotKit UI components
 * to allow tests to run without the ESM transformation issues.
 */

const React = require('react');

// Mock CopilotSidebar component
const CopilotSidebar = ({ children, ...props }) =>
  React.createElement('div', { 'data-testid': 'copilot-sidebar', ...props }, children);

// Mock CopilotPopup component
const CopilotPopup = ({ children, ...props }) =>
  React.createElement('div', { 'data-testid': 'copilot-popup', ...props }, children);

// Mock CopilotChat component
const CopilotChat = ({ children, ...props }) =>
  React.createElement('div', { 'data-testid': 'copilot-chat', ...props }, children);

module.exports = {
  CopilotSidebar,
  CopilotPopup,
  CopilotChat,
};
