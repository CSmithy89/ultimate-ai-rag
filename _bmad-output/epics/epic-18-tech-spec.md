# Epic 18 Tech Spec: Enhanced Documentation & DevOps

**Date:** 2025-12-31
**Updated:** 2026-01-04 (Comprehensive Documentation Plan)
**Status:** Backlog
**Epic Owner:** Product and Engineering

---

## Overview

Epic 18 produces missing documentation and DevOps automation to make the system maintainable and enterprise-ready. It covers observability docs, security automation, and guides for the new universal engine features.

### Documentation Status

Some documentation has already been created during earlier epics:

| Document | Status | Location |
|----------|--------|----------|
| Advanced Retrieval Config | ✅ EXISTS | `docs/guides/advanced-retrieval-configuration.md` |
| MCP Wrapper Architecture | ✅ EXISTS | `docs/guides/mcp-wrapper-architecture.md` |
| Roadmap Decisions | ✅ EXISTS | `docs/roadmap-decisions-2026-01-03.md` |
| Provider Config Guide | ❌ NEEDED | `docs/guides/provider-configuration.md` |
| Observability Guide | ❌ NEEDED | `docs/guides/observability.md` |
| Headless Protocol Docs | ❌ NEEDED | `docs/guides/headless-agent-protocol.md` |
| CLI Installation Manual | ❌ NEEDED | `docs/guides/cli-installation.md` |

### Goals

- Complete documentation coverage for all new features.
- Add security automation via Dependabot and CodeQL.
- Ensure documentation stays in sync with implementation.

### Scope

**In scope**
- Documentation for remaining guides (provider config, observability, protocol, CLI).
- Dependabot and CodeQL configuration.
- Documentation templates and contribution guidelines.

**Out of scope**
- New product features beyond documentation and automation.

---

## Stories

### Story 18-1: Document Observability Metrics

**Objective:** Create a comprehensive observability guide for operators.

**Document Structure:**

```markdown
# Observability Guide

## Key Metrics
- LLM token usage and cost per request
- Retrieval latency (p50, p95, p99)
- Ingestion throughput (docs/minute)
- Cache hit rates
- Agent trajectory duration

## Logging
- Structured logging with structlog
- Correlation IDs for request tracing
- Trajectory logging format

## Dashboards
- Grafana dashboard templates
- Key panels and alerts

## Alert Thresholds
- LLM cost > $X/hour
- Retrieval latency > 5s
- Error rate > 5%
```

**Metrics to Document:**

| Metric | Type | Source | Alert Threshold |
|--------|------|--------|-----------------|
| `llm_tokens_total` | Counter | LLM calls | - |
| `llm_cost_usd` | Counter | Token * price | > $10/hour |
| `retrieval_latency_seconds` | Histogram | Retrieval | p95 > 3s |
| `ingestion_docs_total` | Counter | Ingestion | - |
| `cache_hit_ratio` | Gauge | Redis | < 0.5 |
| `agent_trajectory_duration` | Histogram | Orchestrator | p95 > 10s |

**Acceptance Criteria**
- Guide describes all key metrics with their meaning.
- Dashboard JSON templates are provided.
- Alert thresholds are documented with rationale.
- Logging format and correlation IDs are explained.

### Story 18-2: Configure Dependabot Security Updates

**Objective:** Automate dependency updates for security and maintenance.

**Dependabot Configuration:**

```yaml
# .github/dependabot.yml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/backend"
    schedule:
      interval: "weekly"
      day: "monday"
    labels:
      - "dependencies"
      - "python"
    groups:
      production:
        patterns:
          - "*"
        exclude-patterns:
          - "pytest*"
          - "ruff*"
      dev:
        patterns:
          - "pytest*"
          - "ruff*"
    open-pull-requests-limit: 10

  # Node.js dependencies
  - package-ecosystem: "npm"
    directory: "/frontend"
    schedule:
      interval: "weekly"
      day: "monday"
    labels:
      - "dependencies"
      - "javascript"
    groups:
      production:
        patterns:
          - "*"
        exclude-patterns:
          - "@types/*"
          - "eslint*"
          - "typescript"
      dev:
        patterns:
          - "@types/*"
          - "eslint*"
          - "typescript"
    open-pull-requests-limit: 10

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "ci"

  # Docker
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "monthly"
    labels:
      - "dependencies"
      - "docker"
```

**Acceptance Criteria**
- Dependabot configuration exists for Python, Node.js, GitHub Actions, and Docker.
- Update cadence is weekly for code, monthly for Docker.
- Labels are defined for easy filtering.
- Grouped updates reduce PR noise.

### Story 18-3: Configure CodeQL Analysis

**Objective:** Add automated security scanning to CI.

**CodeQL Workflow:**

```yaml
# .github/workflows/codeql.yml
name: "CodeQL"

on:
  push:
    branches: [main, epic/*]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: ['python', 'javascript']

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}
        queries: security-extended,security-and-quality

    - name: Autobuild
      uses: github/codeql-action/autobuild@v3

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
      with:
        category: "/language:${{ matrix.language }}"
```

**Security Queries:**
- SQL injection
- Cross-site scripting (XSS)
- Path traversal
- Command injection
- Insecure deserialization

**Acceptance Criteria**
- CodeQL workflow runs on PRs and weekly schedule.
- Python and JavaScript are both analyzed.
- Security-extended queries are enabled.
- Findings appear in GitHub Security tab.

### Story 18-4: Document Provider Configuration Guide

**Objective:** Create a comprehensive guide for configuring LLM and embedding providers.

**Document Structure:**

```markdown
# Provider Configuration Guide

## LLM Providers

### OpenAI
- API key: `OPENAI_API_KEY`
- Models: gpt-4o, gpt-4o-mini
- Cost: $2.50-$10/1M tokens

### Anthropic
- API key: `ANTHROPIC_API_KEY`
- Models: claude-3-5-sonnet, claude-3-haiku
- Cost: $3-$15/1M tokens

### Google Gemini
- API key: `GOOGLE_API_KEY`
- Models: gemini-1.5-pro, gemini-1.5-flash
- Cost: $1.25-$5/1M tokens

### OpenRouter
- API key: `OPENROUTER_API_KEY`
- 100+ models via unified API
- Cost: varies by model

### Ollama (Local)
- No API key required
- Models: llama3.2, mistral, qwen2.5
- Cost: free (hardware only)

## Embedding Providers

### OpenAI
- Model: text-embedding-3-small (1536 dims)
- Cost: $0.02/1M tokens

### Voyage AI
- Model: voyage-code-3 (best for code)
- Cost: $0.06/1M tokens

### Ollama
- Model: nomic-embed-text
- Cost: free

## Common Pitfalls
- OpenRouter requires model prefix
- Anthropic keys start with sk-ant-
- Ollama must be running locally
```

**Acceptance Criteria**
- All 5 LLM providers are documented with setup steps.
- All 4 embedding providers are documented.
- Cost estimates are included.
- Common pitfalls section helps avoid errors.

### Story 18-5: Update Advanced Retrieval Tuning Guide

**Objective:** Enhance existing guide with benchmarking and tuning tips.

**NOTE:** Base guide already exists at `docs/guides/advanced-retrieval-configuration.md`.

**Additions Needed:**

```markdown
## Tuning Recommendations

### Reranking
- Start with FlashRank (free) for testing
- Switch to Cohere for production (better accuracy)
- RERANKER_TOP_K: 10 is optimal for most cases

### Contextual Retrieval
- Enable prompt caching to reduce costs 90%
- Use claude-3-haiku for cost-effective enrichment
- Batch during ingestion, not query time

### CRAG Grader
- Threshold 0.5 is a good starting point
- Lower threshold = more fallbacks = higher cost
- Monitor fallback rate in production

## Benchmarking
- Use evaluation dataset with known relevant docs
- Measure: MRR@10, NDCG@10, Precision@10
- Compare with/without each feature
```

**Acceptance Criteria**
- Existing guide is extended with tuning section.
- Benchmarking methodology is described.
- Recommended defaults are documented with rationale.

### Story 18-6: Document Headless Agent Protocol

**Objective:** Create protocol specification for framework adapters.

**Document Structure:**

```markdown
# Headless Agent Protocol Specification

## Overview
The headless agent protocol defines a framework-agnostic interface
for agent execution. Any agent framework can implement this protocol.

## Protocol Interface

```python
from typing import Protocol, AsyncIterator
from pydantic import BaseModel

class AgentInput(BaseModel):
    query: str
    history: list[dict]
    context: dict = {}

class AgentResponse(BaseModel):
    content: str
    sources: list[dict]
    trajectory: list[dict]
    metadata: dict = {}

class AgentProtocol(Protocol):
    async def run(self, input: AgentInput) -> AgentResponse:
        """Execute agent and return complete response."""
        ...

    async def stream(self, input: AgentInput) -> AsyncIterator[str]:
        """Execute agent and stream response tokens."""
        ...
```

## Implementing an Adapter

1. Implement `AgentProtocol` interface
2. Map framework-specific constructs to protocol
3. Ensure trajectory logging is captured
4. Register adapter in factory

## Existing Adapters
- `AgnoAdapter` - Default, battle-tested
- `PydanticAIAdapter` - Type-safe outputs
- `CrewAIAdapter` - Multi-agent orchestration
- `LangGraphAdapter` - Stateful workflows
- `AnthropicAdapter` - Agent Skills integration
```

**Acceptance Criteria**
- Protocol interface is fully documented with types.
- Adapter implementation guide is included.
- Each framework adapter is described with its strengths.

### Story 18-7: Update MCP Server Usage Guide

**Objective:** Enhance existing guide with usage examples.

**NOTE:** Base guide already exists at `docs/guides/mcp-wrapper-architecture.md`.

**Additions Needed:**

```markdown
## Client Integration Examples

### Claude Desktop
Add to claude_desktop_config.json:
```json
{
  "mcpServers": {
    "agentic-rag": {
      "command": "uvx",
      "args": ["agentic-rag-mcp"]
    }
  }
}
```

### Cursor
Add to cursor settings.json:
```json
{
  "mcp.servers": {
    "agentic-rag": {
      "command": "uvx",
      "args": ["agentic-rag-mcp"]
    }
  }
}
```

### Programmatic Usage
```python
from agentic_rag import MCPClient

async with MCPClient("http://localhost:8000/mcp") as client:
    # Search
    results = await client.call("hybrid_retrieve", {
        "query": "authentication flow",
        "top_k": 10
    })

    # Ingest
    await client.call("ingest_url", {
        "url": "https://docs.example.com",
        "max_depth": 2
    })
```

## Authentication
- API key in header: `X-API-Key: your-key`
- Rate limiting: 100 requests/minute default
```

**Acceptance Criteria**
- Existing guide is extended with client examples.
- Claude Desktop and Cursor integration documented.
- Programmatic usage examples provided.
- Authentication and rate limiting explained.

### Story 18-8: Create CLI Installation Manual

**Objective:** Create comprehensive CLI documentation.

**Document Structure:**

```markdown
# CLI Installation Manual

## Installation

### via pipx (Recommended)
```bash
pipx install agentic-rag-cli
```

### From Source
```bash
git clone https://github.com/example/agentic-rag
cd agentic-rag/backend
uv run rag-install
```

## Quick Start
```bash
# Interactive setup
rag-install

# Non-interactive (CI/CD)
rag-install --profile standard --llm openai --yes
```

## Commands

### rag-install
Main installation wizard.

Options:
- `--profile` - minimal, standard, enterprise
- `--llm` - openai, anthropic, gemini, openrouter, ollama
- `--embedding` - openai, voyage, gemini, ollama
- `--framework` - agno, pydanticai, crewai, langgraph, anthropic
- `--yes` - Skip confirmations

### rag-install validate
Validate existing .env configuration.

### rag-install upgrade
Upgrade configuration to new version.

## Troubleshooting

### Docker not running
Error: "Cannot connect to Docker daemon"
Fix: Start Docker Desktop

### Port in use
Error: "Port 8000 already in use"
Fix: Stop existing service or set BACKEND_PORT=8001

### Out of memory
Error: "Killed" or OOM
Fix: Use `--profile minimal` or increase Docker memory
```

**Acceptance Criteria**
- Installation methods are documented.
- All CLI commands and options are listed.
- Troubleshooting section covers common issues.
- Examples for interactive and non-interactive usage.

---

## Technical Notes

### Documentation Standards

- Use CommonMark markdown format.
- Include code examples for all features.
- Keep docs in sync with code via CI checks.
- Use `docs/guides/` for user guides.
- Use `docs/api/` for API reference (auto-generated).

### Documentation CI

```yaml
# .github/workflows/docs.yml
name: Documentation
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Lint markdown
        uses: DavidAnson/markdownlint-cli2-action@v14
      - name: Check links
        uses: lycheeverse/lychee-action@v1
```

## Dependencies

- Features from Epics 11-17 to document.
- Existing documentation in `docs/guides/`.

## Risks

- Documentation can drift from implementation.
  - *Mitigation:* CI checks for doc sync, include docs in story DoD.
- Security workflows may require repo permissions.
  - *Mitigation:* Document required permissions, test in fork first.

## Success Metrics

- All features have corresponding documentation.
- Security automation runs successfully in CI.
- Zero broken links in documentation.
- Docs contribute to <15 minute first response goal.

## References

- `docs/guides/advanced-retrieval-configuration.md` - Already exists
- `docs/guides/mcp-wrapper-architecture.md` - Already exists
- `docs/roadmap-decisions-2026-01-03.md` - Already exists
- `_bmad-output/prd.md`
- `_bmad-output/architecture.md`
- `_bmad-output/project-planning-artifacts/epics.md`
- `docs/recommendations_2025.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
