/** @type {import('jest').Config} */
const config = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
    '\\.(css|less|scss|sass)$': '<rootDir>/__mocks__/styleMock.js',
    // Mock CopilotKit packages to avoid ESM transformation issues
    '^@copilotkit/react-core$': '<rootDir>/__mocks__/@copilotkit/react-core.js',
    '^@copilotkit/react-ui$': '<rootDir>/__mocks__/@copilotkit/react-ui.js',
    '^@copilotkit/runtime-client-gql$': '<rootDir>/__mocks__/@copilotkit/runtime-client-gql.js',
  },
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', {
      tsconfig: 'tsconfig.jest.json',
    }],
    // Transform ESM modules in node_modules
    '^.+\\.(js|jsx|mjs)$': ['babel-jest', {
      presets: ['@babel/preset-env'],
    }],
  },
  // Allow transforming ESM packages from node_modules
  // CopilotKit and its dependencies use ESM which Jest doesn't handle by default
  // Pattern handles both npm and pnpm (.pnpm) directory structures
  transformIgnorePatterns: [
    'node_modules/(?!(\\.pnpm/.+/node_modules/)?' +
      '(' +
      // Unist/Unified ecosystem (markdown processing)
      'unist-util-[^/]+|' +
      'unified|' +
      'vfile[^/]*|' +
      'bail|' +
      'trough|' +
      // Mdast/Hast (AST utilities)
      'mdast-util-[^/]+|' +
      'hast-util-[^/]+|' +
      'estree-util-[^/]+|' +
      // Micromark (markdown parser)
      'micromark[^/]*|' +
      // Rehype/Remark ecosystem
      'rehype[^/]*|' +
      'remark[^/]*|' +
      // Property utilities
      'property-information|' +
      'comma-separated-tokens|' +
      'space-separated-tokens|' +
      // Misc utilities
      'devlop|' +
      'decode-named-character-reference|' +
      'character-entities[^/]*|' +
      'ccount|' +
      'escape-string-regexp|' +
      'markdown-table|' +
      'zwitch|' +
      'longest-streak|' +
      'trim-lines|' +
      'stringify-entities|' +
      'is-plain-obj|' +
      // CopilotKit packages
      '@copilotkit[^/]*|' +
      '@copilotkitnext[^/]*|' +
      'streamdown' +
    ')(/|$))',
  ],
  testMatch: ['**/__tests__/**/*.test.{ts,tsx}'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'mjs'],
  // Coverage configuration (Story 22-TD4)
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
  collectCoverageFrom: [
    'components/**/*.{ts,tsx}',
    'hooks/**/*.{ts,tsx}',
    'lib/**/*.{ts,tsx}',
    '!**/*.d.ts',
    '!**/node_modules/**',
    '!**/__tests__/**',
    '!**/__mocks__/**',
  ],
};

module.exports = config;
