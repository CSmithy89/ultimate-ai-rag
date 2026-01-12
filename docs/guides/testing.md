# Testing Framework Documentation

This guide documents the testing patterns, frameworks, and best practices for the Agentic RAG + GraphRAG platform.

## Testing Stack Overview

### Backend (Python)

| Framework | Purpose | Configuration |
|-----------|---------|---------------|
| **pytest** | Unit and integration testing | `pyproject.toml` |
| **pytest-asyncio** | Async test support | Automatic fixtures |
| **pytest-cov** | Code coverage | 80% threshold |
| **pytest-timeout** | Test timeouts | 60s default |

### Frontend (TypeScript/React)

| Framework | Purpose | Configuration |
|-----------|---------|---------------|
| **Jest** | Unit and component testing | `jest.config.js` |
| **Testing Library** | React component testing | `@testing-library/react` |
| **ts-jest** | TypeScript transformation | `tsconfig.jest.json` |

### End-to-End (Planned)

| Framework | Purpose | Status |
|-----------|---------|--------|
| **Playwright** | Browser automation | Planned |

---

## Backend Testing

### Directory Structure

```
backend/tests/
├── conftest.py              # Root fixtures
├── fixtures/                # Test data files (PDFs, etc.)
├── unit/                    # Unit tests
│   ├── models/              # Pydantic model tests
│   ├── protocols/           # Protocol implementation tests
│   └── codebase/            # Codebase intelligence tests
├── integration/             # Integration tests
│   └── conftest.py          # Integration fixtures
├── security/                # Security tests
│   └── conftest.py          # Security fixtures
├── api/                     # API endpoint tests
│   └── routes/              # Route-specific tests
├── retrieval/               # Retrieval system tests
├── indexing/                # Ingestion pipeline tests
├── protocols/               # Protocol compliance tests
│   └── compliance/          # Spec compliance tests
├── benchmarks/              # Performance benchmarks
│   ├── data/                # Benchmark datasets
│   └── results/             # Benchmark output
└── mcp_server/              # MCP server tests
```

### Running Backend Tests

```bash
# Navigate to backend directory
cd backend

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov --cov-report=term-missing

# Run with coverage threshold enforcement (CI default)
uv run pytest --cov --cov-report=term-missing --cov-fail-under=80

# Run specific test file
uv run pytest tests/api/routes/test_copilot.py

# Run tests by marker
uv run pytest -m integration           # Integration tests only
uv run pytest -m "not integration"     # Exclude integration tests
uv run pytest -m security              # Security tests only
uv run pytest -m compliance            # Compliance tests only

# Run with verbose output
uv run pytest -v

# Run with timeout (for CI)
uv run pytest --timeout=60
```

### Unit Test Patterns

#### Basic Test Structure

```python
"""Tests for feature X."""

import os

# Set environment variables BEFORE any imports
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestFeatureX:
    """Tests for FeatureX component."""

    @pytest.mark.asyncio
    async def test_feature_does_something(self):
        """Test that feature performs expected action."""
        # Arrange
        input_data = {"key": "value"}

        # Act
        result = await some_async_function(input_data)

        # Assert
        assert result.status == "success"
        assert result.data is not None
```

#### Using Fixtures

```python
@pytest.fixture
def sample_tenant_id():
    """Provide a sample tenant ID."""
    from uuid import uuid4
    return uuid4()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.xadd.return_value = b"1234567890-0"
    redis_mock.xreadgroup.return_value = []
    redis_mock.xack.return_value = True
    return redis_mock


class TestWithFixtures:
    @pytest.mark.asyncio
    async def test_uses_fixtures(self, sample_tenant_id, mock_redis):
        """Test using injected fixtures."""
        assert sample_tenant_id is not None
        assert mock_redis.xadd.return_value == b"1234567890-0"
```

### Database Fixtures

The project provides comprehensive database fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def mock_postgres_client(sample_job_id, sample_tenant_id):
    """Mock PostgresClient wrapper."""
    from agentic_rag_backend.db.postgres import PostgresClient

    client = MagicMock(spec=PostgresClient)
    client.create_job = AsyncMock(return_value=sample_job_id)
    client.get_job = AsyncMock(return_value=JobStatus(...))
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    return client


@pytest.fixture
def mock_neo4j_client():
    """Mock Neo4jClient wrapper."""
    from agentic_rag_backend.db.neo4j import Neo4jClient

    client = MagicMock(spec=Neo4jClient)
    client.find_similar_entity = AsyncMock(return_value=None)
    client.create_entity = AsyncMock(return_value={"id": "test-id"})
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    return client
```

### Mocking LLM Calls

#### Mock Orchestrator Pattern

```python
class DummyOrchestrator:
    """Stub orchestrator that returns a fixed response."""

    def __init__(self, answer: str = "Test answer"):
        self.answer = answer
        self.call_count = 0
        self.last_call_args = None

    async def run(
        self, query: str, tenant_id: str, session_id: str | None = None
    ) -> OrchestratorResult:
        self.call_count += 1
        self.last_call_args = {
            "query": query,
            "tenant_id": tenant_id,
            "session_id": session_id
        }
        return OrchestratorResult(
            answer=self.answer,
            plan=[PlanStep(step="Analyze", status="completed")],
            thoughts=["Analyzed query"],
            retrieval_strategy=RetrievalStrategy.HYBRID,
            trajectory_id=uuid4(),
        )


class ErrorOrchestrator:
    """Stub orchestrator that raises an error."""

    async def run(self, query: str, tenant_id: str, **kwargs):
        raise RuntimeError("Internal database error")
```

#### Mock Embedding Generator

```python
class MockEmbeddingGenerator:
    """Mock embedding generator for testing."""

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def generate_embedding(
        self,
        text: str,
        tenant_id: str | None = None,
    ) -> list[float]:
        """Generate a deterministic embedding based on text hash."""
        import hashlib
        hash_val = hashlib.md5(text.encode()).hexdigest()
        seed = int(hash_val[:8], 16) / (16**8)
        vector = [0.0] * self._dimension
        vector[0] = seed
        return vector
```

#### Graphiti Mock Helpers

```python
def make_mock_graphiti_node(
    uuid: str = "node-1",
    name: str = "Test Node",
    summary: str = "A test node",
    labels: list | None = None,
):
    """Create a mock Graphiti node with proper name attribute."""
    if labels is None:
        labels = ["Entity"]
    node = MagicMock()
    node.uuid = uuid
    node.configure_mock(name=name)  # Handle reserved attribute
    node.summary = summary
    node.labels = labels
    return node


def make_mock_graphiti_edge(
    uuid: str = "edge-1",
    source_node_uuid: str = "node-1",
    target_node_uuid: str = "node-2",
    name: str = "RELATES_TO",
    fact: str = "Node 1 relates to Node 2",
):
    """Create a mock Graphiti edge with temporal validity."""
    edge = MagicMock()
    edge.uuid = uuid
    edge.source_node_uuid = source_node_uuid
    edge.target_node_uuid = target_node_uuid
    edge.configure_mock(name=name)  # Handle reserved attribute
    edge.fact = fact
    return edge
```

### Integration Tests

Integration tests require real database services. They are gated by the `INTEGRATION_TESTS` environment variable.

#### Integration Test Configuration

```python
# tests/integration/conftest.py

REQUIRED_ENV = [
    "DATABASE_URL",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "REDIS_URL",
]


def _require_integration_env() -> None:
    if os.getenv("INTEGRATION_TESTS") != "1":
        pytest.skip("INTEGRATION_TESTS=1 required for integration tests")
    missing = [key for key in REQUIRED_ENV if not os.getenv(key)]
    if missing:
        pytest.skip(f"Missing env for integration tests: {', '.join(missing)}")


@pytest_asyncio.fixture(scope="session")
async def integration_env() -> dict[str, str]:
    _require_integration_env()
    # ... validate and return environment
```

#### Integration Test Example

```python
pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(60),  # Prevent hanging tests in CI
]


class TestRetrievalPipeline:
    """Integration tests for the full retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_vector_only(
        self,
        postgres_client: PostgresClient,
        integration_cleanup: str,  # Auto-cleanup fixture
    ) -> None:
        """Vector search -> rerank -> grade -> response."""
        tenant_id = integration_cleanup

        # Insert test documents
        doc_id = await postgres_client.create_document(...)
        chunk_id = await postgres_client.create_chunk(...)

        # Perform vector search
        hits = await vector_service.search(query, tenant_id)

        # Verify results
        assert len(hits) >= 1
```

#### Integration Cleanup Fixtures

```python
async def cleanup_postgres(client: PostgresClient, tenant_id: str) -> None:
    """Clean up PostgreSQL test data."""
    tenant_uuid = UUID(tenant_id)
    statements = [
        "DELETE FROM chunks WHERE tenant_id = $1",
        "DELETE FROM ingestion_jobs WHERE tenant_id = $1",
        "DELETE FROM documents WHERE tenant_id = $1",
    ]
    async with client.pool.acquire() as conn:
        for statement in statements:
            await conn.execute(statement, tenant_uuid)


async def cleanup_neo4j(client: Neo4jClient, tenant_id: str) -> None:
    """Clean up Neo4j test data."""
    async with client.driver.session() as session:
        await session.run(
            "MATCH (n {tenant_id: $tenant_id}) DETACH DELETE n",
            tenant_id=tenant_id,
        )


@pytest_asyncio.fixture
async def integration_cleanup(
    integration_tenant_id: str,
    postgres_client: PostgresClient,
    neo4j_client: Neo4jClient,
    redis_client: RedisClient,
) -> str:
    try:
        yield integration_tenant_id
    finally:
        await cleanup_postgres(postgres_client, integration_tenant_id)
        await cleanup_neo4j(neo4j_client, integration_tenant_id)
        await cleanup_redis(redis_client, integration_tenant_id)
```

### Security Tests

Security tests verify tenant isolation and attack prevention.

#### Security Test Markers

```python
# tests/security/conftest.py

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for security tests."""
    config.addinivalue_line(
        "markers",
        "security: marks tests as security-related",
    )
    config.addinivalue_line(
        "markers",
        "tenant_isolation: marks tests for tenant isolation enforcement",
    )
    config.addinivalue_line(
        "markers",
        "cross_tenant: marks adversarial cross-tenant access tests",
    )
    config.addinivalue_line(
        "markers",
        "attack_simulation: marks tests that simulate real attack patterns",
    )
    config.addinivalue_line(
        "markers",
        "injection: marks SQL/Cypher injection attack tests",
    )
```

#### Attack Simulation Fixtures

```python
@pytest.fixture
def sql_injection_payloads() -> list[str]:
    """Common SQL injection attack payloads."""
    return [
        "'; DROP TABLE chunks; --",
        "' OR '1'='1",
        "' UNION SELECT * FROM chunks WHERE tenant_id='victim' --",
        "1; UPDATE chunks SET tenant_id='attacker' --",
    ]


@pytest.fixture
def cypher_injection_payloads() -> list[str]:
    """Common Cypher (Neo4j) injection attack payloads."""
    return [
        "') RETURN * UNION MATCH (n) RETURN n //",
        "' RETURN n UNION MATCH (m) RETURN m //",
        "' OR 1=1 WITH n MATCH (m) RETURN m //",
    ]
```

---

## Frontend Testing

### Directory Structure

```
frontend/__tests__/
├── components/
│   ├── copilot/           # CopilotKit components
│   │   └── components/    # Nested components
│   ├── mcp-ui/            # MCP-UI components
│   ├── open-json-ui/      # Open-JSON-UI components
│   └── ui/                # Base UI components
├── hooks/                 # Custom hook tests
├── lib/                   # Utility tests
│   ├── open-json-ui/      # Open-JSON-UI utilities
│   └── utils/             # General utilities
└── protocols/             # Protocol compliance tests
```

### Running Frontend Tests

```bash
# Navigate to frontend directory
cd frontend

# Run all tests
pnpm test

# Run with coverage
pnpm test -- --coverage

# Run specific test file
pnpm test -- __tests__/components/copilot/ChatSidebar.test.tsx

# Run in watch mode
pnpm test -- --watch

# Via Turbo (from repo root)
pnpm turbo test
pnpm turbo test -- --coverage
```

### Jest Configuration

```javascript
// jest.config.js
const config = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
    '\\.(css|less|scss|sass)$': '<rootDir>/__mocks__/styleMock.js',
  },
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', {
      tsconfig: 'tsconfig.jest.json',
    }],
  },
  testMatch: ['**/__tests__/**/*.test.{ts,tsx}'],
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
  ],
};
```

### Component Test Patterns

#### Basic Component Test

```tsx
import React from "react";
import { render, screen } from "@testing-library/react";
import { ChatSidebar } from "../../../components/copilot/ChatSidebar";

// Mock external dependencies
jest.mock("@copilotkit/react-ui", () => ({
  CopilotSidebar: ({
    children,
    labels,
    className,
    defaultOpen,
  }: {
    children?: React.ReactNode;
    labels?: { title?: string; initial?: string };
    className?: string;
    defaultOpen?: boolean;
  }) => (
    <div
      data-testid="copilot-sidebar"
      data-default-open={defaultOpen}
      className={className}
    >
      {labels?.title && <h2 data-testid="sidebar-title">{labels.title}</h2>}
      {children}
    </div>
  ),
}));

describe("ChatSidebar", () => {
  it("renders the CopilotSidebar component", () => {
    render(<ChatSidebar />);
    expect(screen.getByTestId("copilot-sidebar")).toBeInTheDocument();
  });

  it("has correct title label", () => {
    render(<ChatSidebar />);
    expect(screen.getByTestId("sidebar-title")).toHaveTextContent("AI Copilot");
  });

  it("is open by default", () => {
    render(<ChatSidebar />);
    expect(screen.getByTestId("copilot-sidebar")).toHaveAttribute(
      "data-default-open",
      "true"
    );
  });
});
```

### Hook Test Patterns

```typescript
import { renderHook, act } from "@testing-library/react";
import { useCopilotActions } from "@/hooks/use-copilot-actions";

// Mock dependencies
jest.mock("@copilotkit/react-core", () => ({
  useFrontendTool: jest.fn(),
}));

jest.mock("@/hooks/use-toast", () => ({
  useToast: jest.fn(),
}));

describe("useCopilotActions", () => {
  const mockToast = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    mockUseToast.mockReturnValue({
      toasts: [],
      toast: mockToast,
      dismiss: jest.fn(),
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe("Initial State", () => {
    it("initializes with all actions in idle state", () => {
      const { result } = renderHook(() => useCopilotActions());

      expect(result.current.actionStates).toEqual({
        save: "idle",
        export: "idle",
        share: "idle",
        bookmark: "idle",
        followUp: "idle",
      });
    });

    it("isLoading is false initially", () => {
      const { result } = renderHook(() => useCopilotActions());
      expect(result.current.isLoading).toBe(false);
    });
  });

  describe("saveToWorkspace", () => {
    it("sets loading state during save", async () => {
      const { result } = renderHook(() => useCopilotActions());

      act(() => {
        result.current.saveToWorkspace(mockContent);
      });

      expect(result.current.actionStates.save).toBe("loading");
      expect(result.current.isLoading).toBe(true);
    });

    it("shows success toast on successful save", async () => {
      const { result } = renderHook(() => useCopilotActions());

      await act(async () => {
        await result.current.saveToWorkspace(mockContent);
      });

      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "default",
          title: "Saved to workspace",
        })
      );
    });
  });
});
```

### Mocking Fetch

```typescript
// Mock fetch globally
global.fetch = jest.fn();
const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;

beforeEach(() => {
  // Default successful response
  mockFetch.mockResolvedValue({
    ok: true,
    json: async () => ({ data: {} }),
    text: async () => "# Content",
    blob: async () => new Blob(["content"], { type: "application/pdf" }),
  } as Response);
});

// Test with specific response
it("handles API error", async () => {
  mockFetch.mockResolvedValueOnce({
    ok: false,
    json: async () => ({ detail: "Save failed" }),
  } as Response);

  // ... test code
});
```

### Zod Schema Validation Tests

```typescript
import {
  SaveToWorkspaceSchema,
  ExportContentSchema,
} from "@/lib/schemas/tools";

describe("Zod Schema Validation", () => {
  describe("SaveToWorkspaceSchema", () => {
    it("accepts valid params with required fields", () => {
      const valid = { content_id: "abc", content_text: "test content" };
      expect(SaveToWorkspaceSchema.parse(valid)).toEqual(valid);
    });

    it("rejects missing required content_id", () => {
      const invalid = { content_text: "test" };
      expect(() => SaveToWorkspaceSchema.parse(invalid)).toThrow();
    });
  });

  describe("ExportContentSchema", () => {
    it("rejects invalid format enum value", () => {
      const invalid = {
        content_id: "abc",
        content_text: "test",
        format: "docx",  // Not in enum
      };
      expect(() => ExportContentSchema.parse(invalid)).toThrow();
    });
  });
});
```

---

## CI Pipeline Integration

### Backend CI Workflow

```yaml
# .github/workflows/ci-backend.yml
name: CI - Backend

on:
  pull_request:
  push:
    branches: ["main"]

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Setup uv
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        working-directory: backend
        run: uv sync

      - name: Lint
        working-directory: backend
        run: uv run ruff check .

      - name: Type check
        working-directory: backend
        run: uv run mypy src

      - name: Run tests with coverage
        working-directory: backend
        run: uv run pytest --cov --cov-report=term-missing --cov-fail-under=80

  backend-integration:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_DB: agentic_rag
          POSTGRES_USER: agentic_rag
          POSTGRES_PASSWORD: agentic_rag
        ports:
          - 5432:5432
      neo4j:
        image: neo4j:5-community
        env:
          NEO4J_AUTH: neo4j/neo4j_password
        ports:
          - 7687:7687
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    env:
      INTEGRATION_TESTS: "1"
      DATABASE_URL: postgresql://agentic_rag:agentic_rag@localhost:5432/agentic_rag
      NEO4J_URI: bolt://localhost:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: neo4j_password
      REDIS_URL: redis://localhost:6379
    steps:
      - name: Run integration tests
        working-directory: backend
        run: uv run pytest -m integration
```

### Frontend CI Workflow

```yaml
# .github/workflows/ci-frontend.yml
name: CI - Frontend

on:
  pull_request:
  push:
    branches: ["main"]

jobs:
  frontend:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "pnpm"

      - name: Enable Corepack
        run: corepack enable

      - name: Install dependencies
        run: pnpm install

      - name: Lint
        run: pnpm turbo lint

      - name: Type check
        run: pnpm turbo type-check

      - name: Test with coverage
        run: pnpm turbo test -- --coverage
```

---

## Test Data Management

### Fixtures Directory

Test fixtures are stored in `backend/tests/fixtures/`:

```
backend/tests/fixtures/
├── sample_simple.pdf       # Simple PDF for basic parsing tests
├── sample_tables.pdf       # PDF with tables for layout tests
└── sample_complex.pdf      # Complex PDF for comprehensive tests
```

### Benchmark Data

Benchmark datasets are stored in `backend/tests/benchmarks/data/`:

```
backend/tests/benchmarks/
├── data/
│   └── eval_dataset.json   # Evaluation dataset for retrieval quality
└── results/
    └── *.json              # Benchmark run results (gitignored)
```

### Test Database Setup

For integration tests, databases are provisioned via Docker Compose or CI service containers:

```yaml
# docker-compose.test.yml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: agentic_rag_test
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    ports:
      - "5433:5432"

  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/test
    ports:
      - "7688:7687"

  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
```

---

## Best Practices

### Multi-Tenancy in Tests

**Critical:** Every database query MUST include `tenant_id` filtering.

```python
# Good: Tenant-scoped query
async def test_vector_search_tenant_filter(self):
    """Vector search only returns tenant's documents."""
    tenant_a_id = str(uuid4())
    tenant_b_id = str(uuid4())

    # Insert documents for tenant A
    await service.insert(content="Secret A", tenant_id=tenant_a_id)

    # Search as tenant B - should return nothing
    hits = await service.search("secret", tenant_id=tenant_b_id)
    assert len(hits) == 0

    # Search as tenant A - should return their data
    hits = await service.search("secret", tenant_id=tenant_a_id)
    assert len(hits) == 1
```

### Async Testing Patterns

```python
import pytest


class TestAsyncOperations:
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations don't interfere."""
        import asyncio

        async def grade_query(query_id: int):
            return await grader.grade(query=f"query {query_id}")

        # Run multiple operations concurrently
        results = await asyncio.gather(
            grade_query(1),
            grade_query(2),
            grade_query(3),
        )

        # Verify isolation
        assert len(results) == 3
        assert all(r.score >= 0 for r in results)

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test pipeline handles timeout gracefully."""
        import asyncio

        with pytest.raises(asyncio.TimeoutError):
            await vector_service.search(
                query="test",
                tenant_id=tenant_id,
                timeout=0.001,  # Very short timeout
            )
```

### Test Isolation

```python
@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test."""
    # Setup
    yield
    # Teardown - reset any global state


@pytest.fixture
def isolated_cache():
    """Provide isolated cache for test."""
    cache = InMemoryCache()
    yield cache
    cache.clear()
```

### Parameterized Tests

```python
@pytest.mark.parametrize("format_type,expected_ext", [
    ("json", ".json"),
    ("markdown", ".md"),
    ("pdf", ".pdf"),
])
async def test_export_formats(format_type: str, expected_ext: str):
    """Test all export formats."""
    result = await exporter.export(content, format=format_type)
    assert result.filename.endswith(expected_ext)
```

### Coverage Requirements

Both backend and frontend enforce 80% coverage threshold:

```toml
# pyproject.toml
[tool.coverage.report]
fail_under = 80
show_missing = true
```

```javascript
// jest.config.js
coverageThreshold: {
  global: {
    branches: 80,
    functions: 80,
    lines: 80,
    statements: 80,
  },
},
```

---

## Related Documentation

- [Troubleshooting Guide](./troubleshooting.md) - Common test failures and solutions
- [Configuration Profiles](./configuration-profiles.md) - Environment configuration for tests
- [CI/CD Pipeline](./deployment-production.md#ci-cd-pipeline) - CI workflow details
