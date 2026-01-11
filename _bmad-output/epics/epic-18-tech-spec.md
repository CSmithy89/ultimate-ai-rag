# Epic 18 Tech Spec: Enhanced Documentation & DevOps

**Date:** 2025-12-31
**Updated:** 2026-01-04 (Comprehensive Documentation Plan)
**Status:** Backlog
**Epic Owner:** Product and Engineering

---

## Overview

Epic 18 produces missing documentation and DevOps automation to make the system maintainable and enterprise-ready. It covers observability docs, security automation, and guides for the new universal engine features.

### Documentation Status

**Updated:** 2026-01-11 (Post-Epic 22 Review)

Documentation created during earlier epics, including comprehensive protocol docs from Epic 22:

| Document | Status | Location |
|----------|--------|----------|
| Advanced Retrieval Config | ✅ EXISTS | `docs/guides/advanced-retrieval-configuration.md` |
| MCP Wrapper Architecture | ✅ EXISTS | `docs/guides/mcp-wrapper-architecture.md` |
| Roadmap Decisions | ✅ EXISTS | `docs/roadmap-decisions-2026-01-03.md` |
| Voice I/O Configuration | ✅ EXISTS | `docs/guides/voice-io-configuration.md` |
| Frontend Security Checklist | ✅ EXISTS | `docs/checklists/frontend-security-checklist.md` |
| Protocol Integration (7 docs) | ✅ EXISTS | `docs/guides/protocol-integration/*.md` |
| - Overview | ✅ EXISTS | `docs/guides/protocol-integration/overview.md` |
| - AG-UI Protocol | ✅ EXISTS | `docs/guides/protocol-integration/ag-ui-protocol.md` |
| - A2A Protocol | ✅ EXISTS | `docs/guides/protocol-integration/a2a-protocol.md` |
| - MCP Integration | ✅ EXISTS | `docs/guides/protocol-integration/mcp-integration.md` |
| - A2UI Widgets | ✅ EXISTS | `docs/guides/protocol-integration/a2ui-widgets.md` |
| - MCP-UI Rendering | ✅ EXISTS | `docs/guides/protocol-integration/mcp-ui-rendering.md` |
| - Open-JSON-UI | ✅ EXISTS | `docs/guides/protocol-integration/open-json-ui.md` |
| Provider Config Guide | ❌ NEEDED | `docs/guides/provider-configuration.md` |
| Observability Guide | ❌ NEEDED | `docs/guides/observability.md` |
| CLI Installation Manual | ❌ NEEDED | `docs/guides/cli-installation.md` (depends on Epic 17) |

**Note:** Stories 18-6 (Headless Agent Protocol) and 18-7 (MCP Server Usage) are now largely covered by the Epic 22 protocol integration documentation. These stories should be reviewed for any remaining gaps.

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

**Status:** ⚠️ LARGELY COVERED by Epic 22-D1

**Note:** The A2A protocol documentation at `docs/guides/protocol-integration/a2a-protocol.md` covers middleware setup, delegation patterns, and agent collaboration. Review for any remaining gaps specific to "headless" agent scenarios.

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

**Status:** ⚠️ LARGELY COVERED by Epic 22-D1

**Note:** The MCP integration documentation at `docs/guides/protocol-integration/mcp-integration.md` covers tool registration, invocation patterns, and error handling. The existing `docs/guides/mcp-wrapper-architecture.md` provides the architectural overview. Review for any remaining client integration examples (Claude Desktop, Cursor) that may still be needed.

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

## NEW STORIES (Added 2026-01-11 - Comprehensive Audit)

The following stories were added after a comprehensive system audit revealed significant documentation gaps for the 200+ configurable features across the platform.

### Story 18-12: Create Provider Configuration Guide

**Objective:** Document all LLM and embedding provider configurations comprehensively.

**Why Needed:** System supports 5 LLM providers and 5 embedding providers with 30+ configuration variables. Users need clear guidance on setup, costs, and trade-offs.

**Document Structure:**

```markdown
# Provider Configuration Guide

## Overview
This guide covers configuration for all supported LLM and embedding providers.

## LLM Providers

### OpenAI
**Setup:**
- API Key: `OPENAI_API_KEY` (starts with `sk-`)
- Base URL: `OPENAI_BASE_URL` (optional, for proxies)
- Model: `OPENAI_MODEL_ID` or `LLM_MODEL_ID`

**Available Models:**
| Model | Input Cost | Output Cost | Context | Best For |
|-------|------------|-------------|---------|----------|
| gpt-4o | $2.50/1M | $10/1M | 128K | Complex reasoning |
| gpt-4o-mini | $0.15/1M | $0.60/1M | 128K | Cost-effective |
| gpt-4-turbo | $10/1M | $30/1M | 128K | Legacy support |

**Example Configuration:**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL_ID=gpt-4o-mini
```

### Anthropic
**Setup:**
- API Key: `ANTHROPIC_API_KEY` (starts with `sk-ant-`)
- No base URL override (native client)

**Available Models:**
| Model | Input Cost | Output Cost | Context | Best For |
|-------|------------|-------------|---------|----------|
| claude-3-5-sonnet | $3/1M | $15/1M | 200K | Balanced |
| claude-3-opus | $15/1M | $75/1M | 200K | Complex tasks |
| claude-3-haiku | $0.25/1M | $1.25/1M | 200K | Fast, cheap |

### Google Gemini
**Setup:**
- API Key: `GEMINI_API_KEY`

**Available Models:**
| Model | Cost | Context | Best For |
|-------|------|---------|----------|
| gemini-1.5-pro | $1.25-$5/1M | 2M | Long context |
| gemini-1.5-flash | $0.075/1M | 1M | Speed |

### OpenRouter
**Setup:**
- API Key: `OPENROUTER_API_KEY`
- Base URL: `OPENROUTER_BASE_URL` (default: https://openrouter.ai/api/v1)

**Benefits:**
- Access to 100+ models via single API
- Automatic fallback between providers
- Usage tracking across providers

**Model Naming:**
```bash
# Format: provider/model-name
LLM_MODEL_ID=anthropic/claude-3.5-sonnet
LLM_MODEL_ID=meta-llama/llama-3.2-90b-instruct
```

### Ollama (Local)
**Setup:**
- Base URL: `OLLAMA_BASE_URL` (default: http://localhost:11434/v1)
- API Key: `OLLAMA_API_KEY` (optional)

**Requirements:**
- Ollama installed locally
- GPU recommended for larger models

**Popular Models:**
| Model | Size | RAM Required | Quality |
|-------|------|--------------|---------|
| llama3.2:3b | 3B | 4GB | Basic |
| llama3.2:8b | 8B | 8GB | Good |
| mistral:7b | 7B | 8GB | Good |
| qwen2.5:32b | 32B | 32GB | Excellent |

---

## Embedding Providers

### OpenAI Embeddings
**Models:**
| Model | Dimensions | Cost | Best For |
|-------|------------|------|----------|
| text-embedding-3-small | 1536 | $0.02/1M | General use |
| text-embedding-3-large | 3072 | $0.13/1M | High precision |
| text-embedding-ada-002 | 1536 | $0.10/1M | Legacy |

### Voyage AI
**Best for code repositories.**

**Models:**
| Model | Dimensions | Best For |
|-------|------------|----------|
| voyage-code-3 | 1536 | Code |
| voyage-3 | 1536 | General |
| voyage-3-lite | 1536 | Cost-effective |

### Ollama Embeddings
**Local, free option.**
- Model: `nomic-embed-text` (768 dimensions)

---

## Common Configuration Patterns

### Cost-Optimized (Standard Profile)
```bash
LLM_PROVIDER=openai
LLM_MODEL_ID=gpt-4o-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

### Quality-Optimized (Enterprise Profile)
```bash
LLM_PROVIDER=openrouter
LLM_MODEL_ID=anthropic/claude-3.5-sonnet
EMBEDDING_PROVIDER=voyage
EMBEDDING_MODEL=voyage-code-3
```

### Local/Offline
```bash
LLM_PROVIDER=ollama
LLM_MODEL_ID=llama3.2:8b
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
```

---

## Troubleshooting

### "Invalid API key" errors
- OpenAI keys start with `sk-`
- Anthropic keys start with `sk-ant-`
- Check for trailing whitespace

### Model not found
- OpenRouter requires `provider/model` format
- Ollama requires `ollama pull <model>` first

### Rate limiting
- Implement exponential backoff
- Consider OpenRouter for automatic failover
```

**Acceptance Criteria:**
- All 5 LLM providers documented with setup instructions.
- All 5 embedding providers documented with model comparisons.
- Cost estimates included for each model.
- Common configuration patterns provided.
- Troubleshooting section covers frequent issues.

**File to Create:** `docs/guides/provider-configuration.md`

---

### Story 18-13: Create Memory Platform Guide

**Objective:** Document the memory platform including scopes, consolidation, and best practices.

**Why Needed:** Memory platform (Epic 20) has 15+ configuration variables. Users need to understand scope hierarchy, consolidation strategies, and when to use each feature.

**Document Structure:**

```markdown
# Memory Platform Guide

## Overview
The memory platform provides persistent memory with four scope levels,
automatic consolidation, and decay-based relevance management.

## Memory Scopes

### Scope Hierarchy
```
GLOBAL
  └── AGENT
        └── USER
              └── SESSION
```

**Inheritance:** Lower scopes inherit from parent scopes when
`MEMORY_INCLUDE_PARENT_SCOPES=true`.

### Scope Types

| Scope | Persistence | Use Case | Example |
|-------|-------------|----------|---------|
| SESSION | Conversation | Short-term context | "Remember I'm working on auth" |
| USER | Per user | Preferences, history | "I prefer concise answers" |
| AGENT | Per agent type | Agent-specific knowledge | "This agent handles billing" |
| GLOBAL | Shared | Organization knowledge | "Company uses Python 3.11" |

### Configuration
```bash
MEMORY_SCOPES_ENABLED=true
MEMORY_DEFAULT_SCOPE=session  # session|user|agent|global
MEMORY_INCLUDE_PARENT_SCOPES=true
MEMORY_CACHE_TTL_SECONDS=3600
MEMORY_MAX_PER_SCOPE=10000
```

---

## Memory Consolidation

### What It Does
- **Deduplication:** Merges similar memories (configurable threshold)
- **Decay:** Reduces importance of old memories over time
- **Cleanup:** Removes memories below importance threshold

### Configuration
```bash
MEMORY_CONSOLIDATION_ENABLED=true
MEMORY_CONSOLIDATION_SCHEDULE="0 2 * * *"  # Daily at 2 AM
MEMORY_SIMILARITY_THRESHOLD=0.9  # 0.0-1.0, higher = stricter
MEMORY_DECAY_HALF_LIFE_DAYS=30   # Days until importance halves
MEMORY_MIN_IMPORTANCE=0.1        # Below this, memory is deleted
MEMORY_CONSOLIDATION_BATCH_SIZE=100
```

### Decay Formula
```
current_importance = original_importance * (0.5 ^ (days / half_life))
```

**Example:** Memory with importance 1.0, half-life 30 days:
- Day 0: 1.0
- Day 30: 0.5
- Day 60: 0.25
- Day 90: 0.125 (deleted if MIN_IMPORTANCE=0.1)

---

## Best Practices

### When to Use Each Scope

| Scenario | Recommended Scope |
|----------|-------------------|
| User preferences | USER |
| Conversation context | SESSION |
| Agent personality | AGENT |
| Company policies | GLOBAL |

### Consolidation Tuning

| Use Case | Similarity | Decay | Min Importance |
|----------|------------|-------|----------------|
| High retention | 0.95 | 90 days | 0.05 |
| Balanced | 0.9 | 30 days | 0.1 |
| Aggressive cleanup | 0.8 | 14 days | 0.2 |

---

## API Endpoints

### Create Memory
```bash
POST /memories
{
  "content": "User prefers Python examples",
  "scope": "user",
  "importance": 0.8,
  "metadata": {"source": "explicit_preference"}
}
```

### Query Memories
```bash
GET /memories?scope=user&query=python&limit=10
```

### Trigger Consolidation
```bash
POST /memories/consolidate
```
```

**Acceptance Criteria:**
- All 4 scope levels documented with use cases.
- Consolidation mechanics explained with formulas.
- Configuration reference complete.
- Best practices for different scenarios.
- API endpoint documentation.

**File to Create:** `docs/guides/memory-platform.md`

---

### Story 18-14: Create Ingestion Pipeline Guide

**Objective:** Document all ingestion sources, configurations, and best practices.

**Why Needed:** System supports 6+ ingestion sources (URL, PDF, YouTube, Codebase, External Sync) with 40+ configuration variables.

**Document Structure:**

```markdown
# Ingestion Pipeline Guide

## Overview
The ingestion pipeline supports multiple document sources with
configurable processing, chunking, and entity extraction.

## Ingestion Sources

### 1. URL Crawling (Crawl4AI)

**Profiles:**
| Profile | JS Wait | Concurrency | Use Case |
|---------|---------|-------------|----------|
| fast | 0.5s | 20 | Static sites, documentation |
| thorough | 2s | 10 | SPAs, dynamic content |
| stealth | 3s | 5 | Anti-bot protected sites |

**Configuration:**
```bash
CRAWL4AI_PROFILE=fast  # fast|thorough|stealth
CRAWL4AI_HEADLESS=true
CRAWL4AI_MAX_CONCURRENT=10
CRAWL4AI_CACHE_ENABLED=true
CRAWL4AI_JS_WAIT_SECONDS=2.0
CRAWL4AI_PAGE_TIMEOUT_MS=60000
```

**Fallback Providers:**
```bash
CRAWL_FALLBACK_ENABLED=true
CRAWL_FALLBACK_PROVIDERS=["apify", "brightdata"]
APIFY_API_TOKEN=apify_api_...
BRIGHTDATA_USERNAME=...
BRIGHTDATA_PASSWORD=...
```

### 2. PDF Processing (Docling)

**Modes:**
| Mode | Speed | Table Quality | Use Case |
|------|-------|---------------|----------|
| fast | 2x | Basic | Simple PDFs |
| accurate | 1x | High | Complex tables, forms |

**Configuration:**
```bash
DOCLING_TABLE_MODE=accurate  # accurate|fast
ENHANCED_DOCLING_ENABLED=true
DOCLING_TABLE_EXTRACTION=true
DOCLING_PRESERVE_LAYOUT=true
DOCLING_TABLE_AS_MARKDOWN=true
MAX_UPLOAD_SIZE_MB=100
```

### 3. YouTube Transcripts

**Configuration:**
```bash
YOUTUBE_PREFERRED_LANGUAGES=["en", "en-US"]
YOUTUBE_CHUNK_DURATION_SECONDS=120
```

**Supported URLs:**
- `https://youtube.com/watch?v=...`
- `https://youtu.be/...`
- Playlists (each video processed separately)

### 4. Codebase Indexing

**Configuration:**
```bash
CODEBASE_RAG_ENABLED=true
CODEBASE_LANGUAGES=python,typescript,javascript
CODEBASE_EXCLUDE_PATTERNS=["**/node_modules/**", "**/.git/**"]
CODEBASE_MAX_CHUNK_SIZE=1000
CODEBASE_INCLUDE_CLASS_CONTEXT=true
CODEBASE_INCREMENTAL_INDEXING=true
```

**Supported Languages:**
- Python (.py)
- TypeScript (.ts, .tsx)
- JavaScript (.js, .jsx)
- Go (.go)
- Rust (.rs)
- Java (.java)

### 5. External Data Sources

**S3:**
```bash
EXTERNAL_SYNC_ENABLED=true
S3_SYNC_BUCKET=my-bucket
S3_SYNC_PREFIX=documents/
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

**Confluence:**
```bash
CONFLUENCE_URL=https://company.atlassian.net
CONFLUENCE_API_TOKEN=...
CONFLUENCE_SPACES=["DEV", "DOCS"]
```

**Notion:**
```bash
NOTION_API_KEY=secret_...
NOTION_DATABASE_IDS=["db-id-1", "db-id-2"]
```

---

## Chunking Configuration

```bash
CHUNK_SIZE=512       # Characters per chunk
CHUNK_OVERLAP=64     # Overlap between chunks
ENTITY_SIMILARITY_THRESHOLD=0.95  # Entity deduplication
```

### Hierarchical Chunking (Enterprise)
```bash
HIERARCHICAL_CHUNKS_ENABLED=true
HIERARCHICAL_CHUNK_LEVELS=256,512,1024,2048
HIERARCHICAL_OVERLAP_RATIO=0.1
```

---

## API Endpoints

### Ingest URL
```bash
POST /ingest/url
{
  "url": "https://docs.example.com",
  "max_depth": 2,
  "profile": "thorough"
}
```

### Upload Document
```bash
POST /ingest/file
Content-Type: multipart/form-data
file: @document.pdf
```

### Check Job Status
```bash
GET /ingest/jobs/{job_id}
```
```

**Acceptance Criteria:**
- All 6 ingestion sources documented.
- Configuration options explained with examples.
- Chunking strategies documented.
- API endpoints with request/response examples.
- Best practices for each source type.

**File to Create:** `docs/guides/ingestion-pipeline.md`

---

### Story 18-15: Create Graph Intelligence Guide

**Objective:** Document graph intelligence features including community detection, LazyRAG, and query routing.

**Why Needed:** Graph features (Epic 20) include community detection, LazyRAG pattern, query routing, and graph rerankers with 25+ configuration variables.

**Document Structure:**

```markdown
# Graph Intelligence Guide

## Overview
Graph intelligence extends the knowledge graph with community detection,
LazyRAG for efficient querying, and intelligent query routing.

## Community Detection

### What It Does
Groups related entities into communities for efficient summarization.

**Algorithms:**
| Algorithm | Speed | Quality | Best For |
|-----------|-------|---------|----------|
| Louvain | Fast | Good | Large graphs |
| Leiden | Slower | Better | Precision |

**Configuration:**
```bash
COMMUNITY_DETECTION_ENABLED=true
COMMUNITY_ALGORITHM=louvain
COMMUNITY_MIN_SIZE=3
COMMUNITY_MAX_LEVELS=3
COMMUNITY_SUMMARY_MODEL=gpt-4o-mini
COMMUNITY_REFRESH_SCHEDULE="0 3 * * 0"  # Weekly
```

### API Endpoints
```bash
# Trigger detection
POST /communities/detect

# List communities
GET /communities

# Get community summary
POST /communities/{id}/summarize
```

---

## LazyRAG Pattern

### What It Does
Defers entity summarization to query time, reducing indexing costs by 99%.

**How It Works:**
1. Indexing: Store raw entities without summaries
2. Query: Find relevant entities via graph traversal
3. Summarize: Generate on-demand summary for query context

**Configuration:**
```bash
LAZY_RAG_ENABLED=true
LAZY_RAG_MAX_ENTITIES=50   # Max entities to traverse
LAZY_RAG_MAX_HOPS=2        # Graph traversal depth
LAZY_RAG_SUMMARY_MODEL=gpt-4o-mini
LAZY_RAG_USE_COMMUNITIES=true  # Use community summaries
```

### API Endpoints
```bash
POST /lazy-rag/query
{
  "query": "How does authentication work?",
  "max_entities": 30,
  "use_communities": true
}
```

---

## Query Routing

### What It Does
Automatically routes queries to optimal retrieval strategy:
- **Global queries:** "What are all the authentication methods?" → Community summaries
- **Local queries:** "How does JWT validation work?" → Entity-level search

**Configuration:**
```bash
QUERY_ROUTING_ENABLED=true
QUERY_ROUTING_USE_LLM=false      # Heuristic by default
QUERY_ROUTING_LLM_MODEL=gpt-4o-mini
QUERY_ROUTING_CONFIDENCE_THRESHOLD=0.7
```

### Routing Strategies
| Query Type | Strategy | Example |
|------------|----------|---------|
| What/How/Why broad | GLOBAL | "What authentication methods exist?" |
| Specific entity | LOCAL | "What is the JWT secret format?" |
| Relationship | GRAPH | "How does User relate to Session?" |

---

## Graph Rerankers

### Types
| Type | Description | Best For |
|------|-------------|----------|
| episode | Recent episodes weighted higher | Time-sensitive queries |
| distance | Closer nodes weighted higher | Relationship queries |
| hybrid | Combined scoring | General use |

**Configuration:**
```bash
GRAPH_RERANKER_ENABLED=true
GRAPH_RERANKER_TYPE=hybrid
GRAPH_RERANKER_EPISODE_WEIGHT=0.3
GRAPH_RERANKER_DISTANCE_WEIGHT=0.3
GRAPH_RERANKER_ORIGINAL_WEIGHT=0.4
GRAPH_RERANKER_EPISODE_WINDOW_DAYS=30
GRAPH_RERANKER_MAX_DISTANCE=3
```
```

**Acceptance Criteria:**
- Community detection explained with algorithm comparison.
- LazyRAG pattern documented with cost savings.
- Query routing strategies explained.
- Graph rerankers documented.
- All configurations with examples.

**File to Create:** `docs/guides/graph-intelligence.md`

---

### Story 18-16: Create Database Administration Guide

**Objective:** Document database configuration, operations, and maintenance for PostgreSQL, Neo4j, and Redis.

**Why Needed:** Three database systems with 25+ configuration variables, pooling, and operational concerns.

**Document Structure:**

```markdown
# Database Administration Guide

## PostgreSQL (pgvector)

### Connection
```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname
DB_POOL_MIN=1
DB_POOL_MAX=50
```

### pgvector Extension
Required for vector similarity search.
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Index Management
```sql
-- Create HNSW index for fast similarity search
CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Maintenance
```sql
-- Vacuum for performance
VACUUM ANALYZE embeddings;

-- Check index health
SELECT * FROM pg_stat_user_indexes
WHERE relname = 'embeddings';
```

---

## Neo4j

### Connection
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### Pool Configuration
```bash
NEO4J_POOL_MIN=1
NEO4J_POOL_MAX=50
NEO4J_POOL_ACQUIRE_TIMEOUT_SECONDS=30
NEO4J_CONNECTION_TIMEOUT_SECONDS=30
NEO4J_MAX_CONNECTION_LIFETIME_SECONDS=3600
NEO4J_TRANSACTION_TIMEOUT_SECONDS=300
```

### Index Management
```cypher
-- Create full-text index for entity search
CREATE FULLTEXT INDEX entity_names FOR (n:Entity)
ON EACH [n.name, n.description];

-- Create index for tenant isolation
CREATE INDEX entity_tenant FOR (n:Entity) ON (n.tenant_id);
```

### Maintenance
```cypher
-- Check database stats
CALL db.stats.retrieve("GRAPH COUNTS");

-- Clear query cache
CALL db.clearQueryCaches();
```

---

## Redis

### Connection
```bash
REDIS_URL=redis://localhost:6379
```

### Usage Patterns
| Feature | Key Pattern | TTL |
|---------|-------------|-----|
| A2A Sessions | `a2a:session:{id}` | 24h |
| Rate Limiting | `rate:{endpoint}:{ip}` | 60s |
| Reranker Cache | `rerank:{hash}` | 5min |
| HITL Checkpoints | `hitl:{id}` | 1h |

### Monitoring
```bash
# Memory usage
redis-cli INFO memory

# Key count by pattern
redis-cli KEYS "a2a:*" | wc -l
```

---

## Backup & Recovery

### PostgreSQL
```bash
# Backup
pg_dump -Fc dbname > backup.dump

# Restore
pg_restore -d dbname backup.dump
```

### Neo4j
```bash
# Online backup (Enterprise)
neo4j-admin database dump neo4j --to-path=/backups/

# Restore
neo4j-admin database load neo4j --from-path=/backups/neo4j.dump
```

### Redis
```bash
# Trigger RDB snapshot
redis-cli BGSAVE

# Check last save
redis-cli LASTSAVE
```
```

**Acceptance Criteria:**
- All three databases documented.
- Connection and pooling configuration.
- Index management and optimization.
- Backup and recovery procedures.
- Monitoring and maintenance commands.

**File to Create:** `docs/guides/database-administration.md`

---

### Story 18-17: Generate OpenAPI Reference Documentation

**Objective:** Auto-generate comprehensive API reference from FastAPI OpenAPI spec.

**Why Needed:** System has 50+ API endpoints. Manual documentation is error-prone and quickly outdated.

**Implementation:**

```python
# scripts/generate_api_docs.py
import json
from pathlib import Path
import httpx

def generate_api_docs():
    """Fetch OpenAPI spec and generate markdown documentation."""
    # Fetch OpenAPI spec
    response = httpx.get("http://localhost:8000/openapi.json")
    spec = response.json()

    # Generate markdown
    md = generate_markdown(spec)

    # Write to docs
    Path("docs/api/reference.md").write_text(md)

def generate_markdown(spec: dict) -> str:
    """Convert OpenAPI spec to markdown."""
    lines = [
        "# API Reference",
        "",
        f"Version: {spec['info']['version']}",
        "",
        "## Endpoints",
        ""
    ]

    for path, methods in spec["paths"].items():
        for method, details in methods.items():
            lines.append(f"### {method.upper()} {path}")
            lines.append(f"**{details.get('summary', 'No summary')}**")
            lines.append("")
            # Add parameters, request body, responses...

    return "\n".join(lines)
```

**CI Integration:**
```yaml
# .github/workflows/docs.yml
- name: Generate API docs
  run: |
    docker compose up -d backend
    sleep 10
    python scripts/generate_api_docs.py
    docker compose down
```

**Acceptance Criteria:**
- OpenAPI spec automatically converted to markdown.
- All endpoints documented with parameters and responses.
- CI workflow generates docs on release.
- Docs include example requests/responses.

**Files to Create:**
- `scripts/generate_api_docs.py`
- `docs/api/reference.md` (generated)

---

### Story 18-18: Create Deployment & Production Guide

**Objective:** Document production deployment, scaling, and hardening.

**Why Needed:** No production deployment documentation exists. Users need guidance for secure, scalable deployments.

**Document Structure:**

```markdown
# Deployment & Production Guide

## Deployment Options

### Docker Compose (Development/Small Teams)
```bash
docker compose -f docker-compose.prod.yml up -d
```

### Kubernetes (Production)
See `deploy/kubernetes/` for Helm charts.

---

## Security Hardening

### Environment Variables
```bash
# REQUIRED for production
TRACE_ENCRYPTION_KEY=<64-char-hex>  # Generate with: openssl rand -hex 32
SHARE_SECRET=<random-string>

# Rate limiting
RATE_LIMIT_BACKEND=redis  # Don't use memory in production
RATE_LIMIT_PER_MINUTE=60
```

### Network Security
- Backend should not be directly exposed
- Use reverse proxy (nginx, Traefik)
- Enable HTTPS only
- Configure CORS appropriately

### Secret Management
- Use environment variables, not .env files in production
- Consider: AWS Secrets Manager, HashiCorp Vault, Kubernetes Secrets

---

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.prod.yml
services:
  backend:
    deploy:
      replicas: 3
```

### Database Scaling
- PostgreSQL: Connection pooling via PgBouncer
- Neo4j: Read replicas for query distribution
- Redis: Cluster mode for high availability

### Load Balancing
```nginx
upstream backend {
    least_conn;
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

---

## Monitoring

### Health Checks
```bash
# Backend health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/db
```

### Prometheus Metrics
```bash
PROMETHEUS_ENABLED=true
PROMETHEUS_PATH=/metrics
```

### Alerting
See `docs/observability/prometheus-alerts.yaml` for alert rules.

---

## Troubleshooting

### High Memory Usage
- Check Neo4j heap: `NEO4J_HEAP_SIZE`
- Review A2A session limits
- Enable memory profiling

### Slow Queries
- Check trajectory logs for slow operations
- Review database indexes
- Consider query routing optimization

### Connection Errors
- Verify database connection strings
- Check pool sizes vs worker count
- Review firewall rules
```

**Acceptance Criteria:**
- Multiple deployment options documented.
- Security hardening checklist.
- Scaling guidance with examples.
- Monitoring and alerting setup.
- Troubleshooting guide for common issues.

**File to Create:** `docs/guides/deployment-production.md`

---

### Story 18-19: Create Configuration Profiles Documentation

**Objective:** Document the profile-based configuration system introduced in Epic 17.

**Why Needed:** Profile-based configuration (Epic 17-8) requires comprehensive documentation for users to understand and customize profiles.

**Note:** This story depends on Epic 17-8 completion.

**Document Structure:**

```markdown
# Configuration Profiles Guide

## Overview
Configuration profiles provide pre-tested configurations for different
deployment scenarios, reducing setup complexity from 200+ variables
to a single profile selection.

## Available Profiles

### Minimal Profile
**Target:** Development, testing, resource-constrained environments
**Resources:** 4GB RAM, no GPU required

**Features Enabled:**
- Vector-only retrieval
- Basic ingestion (URL, PDF, YouTube)
- Session-scoped memory
- Cost tracking

**Features Disabled:**
- Graph intelligence (LazyRAG, communities)
- Advanced retrieval (reranking, CRAG)
- Voice I/O
- Codebase indexing

### Standard Profile
**Target:** Production deployments, small-medium teams
**Resources:** 8GB RAM, GPU optional

**Features Enabled:**
- Hybrid retrieval (vector + graph)
- Reranking (FlashRank)
- Heuristic grading with web fallback
- Session memory with scopes
- Prometheus metrics
- Query routing (heuristic)

**Features Disabled:**
- Contextual retrieval (cost consideration)
- Community detection
- LazyRAG
- Voice I/O
- External sync

### Enterprise Profile
**Target:** Large deployments, maximum capabilities
**Resources:** 16GB+ RAM, GPU recommended

**All features enabled including:**
- Cross-encoder reranking (Cohere)
- Contextual retrieval with prompt caching
- Community detection and LazyRAG
- Voice I/O
- External data sync
- Codebase intelligence
- Graph rerankers

## Customization

### Environment Overrides
Environment variables override profile defaults:
```bash
CONFIG_PROFILE=standard
LLM_PROVIDER=anthropic  # Override profile's openai default
RERANKER_PROVIDER=cohere  # Upgrade reranker
```

### Custom Profile
Create `config/profiles/custom.yaml`:
```yaml
# Start from standard, customize as needed
_extends: standard

llm:
  provider: anthropic
  model: claude-3-5-sonnet

retrieval:
  reranker:
    provider: cohere
```

## Migration

### From Flat .env
```bash
# Old way (200+ variables)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
RERANKER_ENABLED=true
RERANKER_PROVIDER=flashrank
# ... 190+ more

# New way
CONFIG_PROFILE=standard
OPENAI_API_KEY=sk-...
# Done! Profile handles the rest
```
```

**Acceptance Criteria:**
- All three profiles documented with features.
- Customization patterns explained.
- Migration guide from flat .env.
- Profile selection guidance.

**File to Create:** `docs/guides/configuration-profiles.md`

---

## ADDITIONAL STORIES (Added 2026-01-11 - Party Mode Critical Audit)

The following stories were added after a comprehensive party mode audit revealed critical documentation gaps that affect user success.

### Story 18-20: Create Advanced Retrieval Deep Dive Guide

**Objective:** Create comprehensive documentation covering all retrieval enhancements in depth.

**Why Needed:** Story 18-5 provides tuning tips but lacks deep technical documentation for:
- Reranking implementations (Cohere, FlashRank, ColBERT)
- CRAG grader patterns and fallback strategies
- Contextual retrieval chunking
- Sparse vectors (BM42) and hybrid search
- Cross-language query support
- Normalization strategies

**Priority:** HIGH

**Document Structure:**

```markdown
# Advanced Retrieval Deep Dive

## Reranking Implementations

### Architecture
```
Query → Initial Retrieval → Reranking → Final Results
                           ↓
                    [Cohere|FlashRank|ColBERT]
```

### Cohere Reranker
- Model: `rerank-v3.5`
- 100+ languages supported
- 32K context window
- Best for: Production, multilingual

**Configuration:**
```bash
RERANKER_ENABLED=true
RERANKER_PROVIDER=cohere
COHERE_API_KEY=...
RERANKER_TOP_K=10
RERANKER_CACHE_ENABLED=true
RERANKER_CACHE_TTL_SECONDS=300
```

### FlashRank Reranker
- Model: Local CPU-optimized
- No API costs
- Fast inference
- Best for: Development, cost-sensitive

**Configuration:**
```bash
RERANKER_PROVIDER=flashrank
RERANKER_PRELOAD_MODEL=true  # Eager loading
```

### ColBERT Reranker
- Model: `colbert-ir/colbertv2.0`
- Token-level embeddings (late interaction)
- MaxSim scoring algorithm
- Best for: Precision-critical use cases

**Configuration:**
```bash
COLBERT_ENABLED=true
COLBERT_MODEL=colbert-ir/colbertv2.0
COLBERT_MAX_LENGTH=512
```

---

## CRAG (Corrective RAG) Pattern

### How It Works
```
Query → Retrieve → Grade → [Pass: Return | Fail: Fallback]
                   ↓
         [Heuristic|CrossEncoder]
                   ↓
         [WebSearch|ExpandedQuery]
```

### Grader Types

| Grader | Speed | Accuracy | Cost | Use Case |
|--------|-------|----------|------|----------|
| Heuristic | Fast | Moderate | Free | Default |
| CrossEncoder | Slower | High | Free (local) | Precision |

### CrossEncoder Models
| Model | Size | Accuracy | Speed |
|-------|------|----------|-------|
| ms-marco-MiniLM-L-6-v2 | 22M | Good | Fast |
| ms-marco-MiniLM-L-12-v2 | 33M | Better | Medium |
| bge-reranker-base | 278M | High | Slower |
| bge-reranker-large | 560M | Highest | Slowest |

**Configuration:**
```bash
GRADER_ENABLED=true
GRADER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
GRADER_THRESHOLD=0.5
GRADER_PRELOAD_MODEL=true
GRADER_NORMALIZATION_STRATEGY=min_max  # min_max|z_score|softmax|percentile

# Fallback configuration
GRADER_FALLBACK_ENABLED=true
GRADER_FALLBACK_STRATEGY=web_search  # web_search|expanded_query
TAVILY_API_KEY=tvly-...
```

---

## Sparse Vectors & Hybrid Search

### BM42 Sparse Encoding
- Model: `Qdrant/bm42-all-minilm-l6-v2-attentions`
- Lexical matching via sparse vectors
- Complements dense embeddings

### Reciprocal Rank Fusion (RRF)
Combines dense and sparse results:
```
RRF(d) = Σ 1/(k + rank(d))
```
where k=60 (default)

**Configuration:**
```bash
SPARSE_VECTORS_ENABLED=true
SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions
RRF_K=60
DENSE_WEIGHT=0.7
SPARSE_WEIGHT=0.3
```

---

## Cross-Language Query Support

### Language Detection
- Unicode-based detection for non-Latin scripts
- Word marker detection for Latin languages
- Confidence scoring

### Multilingual Embeddings
- Model: `intfloat/multilingual-e5-base`
- Unified vector space across languages

### Query Translation
- LLM-based translation with caching
- LRU cache (1000 entries)

**Configuration:**
```bash
CROSS_LANGUAGE_ENABLED=true
CROSS_LANGUAGE_MODEL=intfloat/multilingual-e5-base
CROSS_LANGUAGE_TRANSLATE=true
CROSS_LANGUAGE_TARGET=en
```

---

## Normalization Strategies

### Available Strategies
| Strategy | Formula | Best For |
|----------|---------|----------|
| MIN_MAX | (x-min)/(max-min) | Default |
| Z_SCORE | (x-μ)/σ → sigmoid | Statistical |
| SOFTMAX | exp(x/T)/Σexp | Probability |
| PERCENTILE | rank/count | Rank-based |

### Aggregation Methods
- mean, max, min, median, weighted_mean
```

**Acceptance Criteria:**
- All 6 reranking/grading implementations documented.
- Architecture diagrams included.
- Configuration examples for each component.
- Performance comparison tables.
- Integration patterns explained.

**File to Create:** `docs/guides/advanced-retrieval-deep-dive.md`

---

### Story 18-21: Create Troubleshooting Guide

**Objective:** Create a comprehensive troubleshooting guide for common issues.

**Why Needed:** Users need a central resource for debugging issues. Currently, troubleshooting is scattered or missing.

**Priority:** HIGH

**Document Structure:**

```markdown
# Troubleshooting Guide

## Quick Diagnostic Commands

```bash
# Check system health
rag-cli doctor

# Check service status
docker compose ps

# View backend logs
docker compose logs backend --tail=100

# View Neo4j logs
docker compose logs neo4j --tail=100
```

---

## Common Issues

### Installation & Setup

#### Docker not running
**Symptom:** `Cannot connect to Docker daemon`
**Fix:**
```bash
# macOS/Windows
open -a Docker  # or start Docker Desktop

# Linux
sudo systemctl start docker
```

#### Port already in use
**Symptom:** `Bind for 0.0.0.0:8000 failed: port already in use`
**Fix:**
```bash
# Find process using port
lsof -i :8000

# Kill process or change port
BACKEND_PORT=8001 docker compose up
```

#### Out of memory
**Symptom:** `Killed` or services crash
**Fix:**
- Increase Docker memory allocation
- Use `--profile minimal`
- Reduce Neo4j heap: `NEO4J_HEAP_SIZE=512M`

---

### Database Issues

#### PostgreSQL connection refused
**Symptom:** `Connection refused to localhost:5432`
**Diagnosis:**
```bash
docker compose logs postgres
psql -h localhost -U postgres -d agentic_rag -c "\conninfo"
```
**Fixes:**
- Check DATABASE_URL format
- Verify PostgreSQL is running
- Check pg_hba.conf for allowed connections

#### Neo4j authentication failed
**Symptom:** `Neo.ClientError.Security.Unauthorized`
**Fixes:**
- Verify NEO4J_PASSWORD matches docker-compose.yml
- Reset password via Neo4j browser (http://localhost:7474)

#### Redis connection timeout
**Symptom:** `Redis connection timeout`
**Fixes:**
- Check REDIS_URL format
- Verify Redis is running
- Check Redis memory limits

---

### API Errors

#### 401 Unauthorized
**Symptom:** API returns 401
**Diagnosis:**
```bash
# Test with curl
curl -H "X-API-Key: your-key" http://localhost:8000/health
```
**Fixes:**
- Verify API key is set
- Check rate limiting (may be temporarily blocked)

#### 429 Too Many Requests
**Symptom:** Rate limited
**Response includes:** `Retry-After` header
**Fixes:**
- Wait for Retry-After period
- Increase rate limits: `RATE_LIMIT_PER_MINUTE=120`
- Use Redis backend for distributed limiting

#### 500 Internal Server Error
**Diagnosis:**
```bash
docker compose logs backend --tail=200 | grep ERROR
```
**Common causes:**
- Database connection issues
- Invalid API keys for providers
- Configuration errors

---

### LLM Provider Issues

#### OpenAI rate limited
**Symptom:** `Rate limit exceeded`
**Fixes:**
- Implement backoff (SDK handles this)
- Upgrade OpenAI tier
- Switch to OpenRouter for failover

#### Anthropic context length exceeded
**Symptom:** `max_tokens exceeds context window`
**Fix:** Reduce CHUNK_SIZE or use different model

#### Ollama model not found
**Symptom:** `model not found`
**Fix:**
```bash
ollama pull llama3.2
```

---

### Retrieval Issues

#### Empty search results
**Diagnosis:**
```bash
# Check ingestion status
curl http://localhost:8000/ingest/jobs

# Check vector count
curl http://localhost:8000/knowledge/stats
```
**Fixes:**
- Verify content was ingested
- Check tenant_id matches
- Try different retrieval strategy

#### Slow retrieval
**Diagnosis:**
```bash
# Check trajectory logs
curl http://localhost:8000/trajectories?limit=10
```
**Fixes:**
- Enable reranker caching
- Check database indexes
- Reduce TOP_K values

---

### Ingestion Issues

#### Crawl failures
**Symptom:** URLs fail to crawl
**Diagnosis:**
```bash
curl http://localhost:8000/ingest/jobs/{job_id}
```
**Fixes:**
- Try different crawl profile (thorough, stealth)
- Enable fallback providers
- Check URL accessibility

#### PDF parsing errors
**Symptom:** PDF fails to parse
**Fixes:**
- Check PDF is not encrypted
- Try different DOCLING_TABLE_MODE
- Verify MAX_UPLOAD_SIZE_MB

---

## Debug Mode

### Enable verbose logging
```bash
LOG_LEVEL=DEBUG docker compose up backend
```

### Enable trajectory debugging
```bash
TRAJECTORY_DEBUG_ENABLED=true
```

### View trajectory details
```bash
curl http://localhost:8000/trajectories/{id}
```

---

## Getting Help

1. Check this guide
2. Search GitHub issues
3. Enable debug logging and capture logs
4. Open new issue with:
   - Error message
   - Relevant logs
   - Configuration (redact secrets)
   - Steps to reproduce
```

**Acceptance Criteria:**
- Common issues categorized and documented.
- Diagnostic commands provided.
- Step-by-step fixes for each issue.
- Debug mode instructions.
- Clear escalation path.

**File to Create:** `docs/guides/troubleshooting.md`

---

### Story 18-22: Create Quickstart Tutorial

**Objective:** Create a step-by-step tutorial for users to achieve their first successful query in 10 minutes.

**Why Needed:** README provides setup commands but no guided tutorial. Users need a hands-on walkthrough.

**Priority:** HIGH

**Document Structure:**

```markdown
# Quickstart Tutorial

**Goal:** Get your first RAG query working in 10 minutes.

## Prerequisites

- Docker Desktop installed
- OpenAI API key (or alternative provider)
- Terminal/command line access

---

## Step 1: Clone and Configure (2 minutes)

```bash
# Clone the repository
git clone https://github.com/example/agentic-rag
cd agentic-rag

# Copy environment template
cp .env.example .env
```

Edit `.env` and set your API key:
```bash
OPENAI_API_KEY=sk-your-key-here
```

---

## Step 2: Start Services (3 minutes)

```bash
# Start all services
docker compose up -d

# Wait for services to be ready (about 30 seconds)
docker compose ps  # All should show "healthy"
```

**Expected output:**
```
NAME                STATUS
agentic-rag-backend   running (healthy)
agentic-rag-frontend  running (healthy)
agentic-rag-postgres  running (healthy)
agentic-rag-neo4j     running (healthy)
agentic-rag-redis     running (healthy)
```

---

## Step 3: Ingest Your First Document (3 minutes)

Let's ingest a sample URL to create some knowledge:

```bash
curl -X POST http://localhost:8000/api/v1/ingest/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://graphiti.dev/",
    "tenant_id": "demo",
    "max_depth": 1
  }'
```

**Expected response:**
```json
{
  "job_id": "abc123...",
  "status": "processing"
}
```

Check progress:
```bash
curl http://localhost:8000/api/v1/ingest/jobs/abc123
```

Wait until `status` is `completed`.

---

## Step 4: Run Your First Query (2 minutes)

Now query the knowledge base:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Graphiti and how does it work?",
    "tenant_id": "demo"
  }'
```

**Expected response:**
```json
{
  "answer": "Graphiti is a temporal knowledge graph...",
  "sources": [...],
  "trajectory_id": "..."
}
```

🎉 **Congratulations!** You've completed your first RAG query!

---

## Step 5: Try the UI (Optional)

Open http://localhost:3000 in your browser.

1. Click on the chat sidebar
2. Type your question
3. See the response with source citations

---

## Next Steps

- [Configure providers](./provider-configuration.md)
- [Enable advanced retrieval](./advanced-retrieval-configuration.md)
- [Ingest more content](./ingestion-pipeline.md)
- [Connect external agents](./protocol-integration/overview.md)

---

## Troubleshooting

**Services not starting?**
```bash
docker compose logs
```

**Query returns empty results?**
- Ensure ingestion completed successfully
- Check tenant_id matches

**See the full [Troubleshooting Guide](./troubleshooting.md)**
```

**Acceptance Criteria:**
- Can be completed in 10 minutes or less.
- Every step has copy-pasteable commands.
- Expected outputs shown for validation.
- Clear success criteria ("You've completed...").
- Links to next steps.

**File to Create:** `docs/tutorials/quickstart.md`

---

### Story 18-23: Create Architecture Decision Records

**Objective:** Document key architectural decisions with rationale for future reference.

**Why Needed:** Decisions like "Graphiti over custom graph" and "Profile-based config" need documented rationale.

**Priority:** MEDIUM

**Document Structure:**

```markdown
# Architecture Decision Records

## ADR Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| ADR-001 | Use Graphiti for Knowledge Graph | Accepted | 2025-12-29 |
| ADR-002 | Use Crawl4AI for Web Crawling | Accepted | 2026-01-04 |
| ADR-003 | Vision A: RAG as Service | Accepted | 2026-01-05 |
| ADR-004 | Profile-Based Configuration | Accepted | 2026-01-11 |

---

## ADR-001: Use Graphiti for Knowledge Graph

### Status
Accepted

### Context
We need a temporal knowledge graph for episodic memory and entity relationships.
Options considered:
1. Custom implementation with Neo4j
2. LangChain graph features
3. Graphiti library

### Decision
Use Graphiti library for knowledge graph.

### Rationale
- **Temporal awareness:** Built-in episode tracking and time-based queries
- **Entity extraction:** Automatic entity and relationship extraction
- **Maintenance:** Community-maintained, reducing our maintenance burden
- **Integration:** Native Neo4j support matches our stack

### Consequences
- Positive: Faster development, temporal features out-of-box
- Negative: Dependency on external library, less control
- Neutral: Must track Graphiti versions for compatibility

---

## ADR-002: Use Crawl4AI for Web Crawling

### Status
Accepted

### Context
Need to crawl JavaScript-heavy documentation sites.
Options considered:
1. Custom httpx implementation (existing)
2. Playwright direct
3. Crawl4AI library
4. Scrapy + Splash

### Decision
Migrate to Crawl4AI library.

### Rationale
- **JavaScript rendering:** Built-in browser automation
- **Parallel crawling:** Efficient concurrent crawling
- **Profiles:** Pre-configured stealth/fast/thorough profiles
- **Community:** Active development and anti-bot updates

### Consequences
- Positive: JS rendering, better success rate on modern sites
- Negative: Larger dependency, browser overhead
- Neutral: Must manage browser installation

---

## ADR-003: Vision A - RAG as Service

### Status
Accepted

### Context
How should external frameworks (PydanticAI, CrewAI, LangGraph) integrate?
Options considered:
- Vision A: RAG as Service (connect via A2A/MCP)
- Vision B: Internal adapters for each framework

### Decision
Adopt Vision A: RAG as Service.

### Rationale
- **Native support:** PydanticAI, CrewAI, LangGraph all have native A2A/MCP
- **Maintenance:** No adapters to maintain per framework
- **Standards:** Leverages open protocols
- **Flexibility:** Works with any A2A/MCP client

### Consequences
- Positive: Reduced maintenance, standards-based
- Negative: No deep framework integration
- Neutral: Templates replace adapters

---

## ADR-004: Profile-Based Configuration

### Status
Accepted

### Context
System has 200+ environment variables. Manual configuration is error-prone.

### Decision
Implement profile-based configuration (minimal/standard/enterprise).

### Rationale
- **Simplicity:** Single profile selection vs 200 variables
- **Validation:** Pre-tested configurations
- **Override:** Environment variables still override profiles
- **Migration:** Backward compatible with existing .env

### Consequences
- Positive: Easier setup, fewer misconfigurations
- Negative: Additional abstraction layer
- Neutral: Must document profile contents clearly
```

**Acceptance Criteria:**
- All major architectural decisions documented.
- Context, decision, rationale, and consequences for each.
- ADR index for navigation.
- Template for future ADRs.

**File to Create:** `docs/architecture/decisions.md`

---

### Story 18-24: Create Glossary and Terminology Reference

**Objective:** Create a glossary defining all domain-specific terms used in the system.

**Why Needed:** Terms like "episode", "community", "LazyRAG", "CRAG" need clear definitions.

**Priority:** MEDIUM

**Document Structure:**

```markdown
# Glossary

## A

### A2A (Agent-to-Agent)
Protocol for agent-to-agent communication and task delegation.
See: [A2A Protocol Guide](./guides/protocol-integration/a2a-protocol.md)

### A2UI (Agent-to-UI)
Protocol for agents to emit UI widgets directly to the frontend.
See: [A2UI Widgets Guide](./guides/protocol-integration/a2ui-widgets.md)

### AG-UI
CopilotKit's protocol for frontend state synchronization via Server-Sent Events.

## C

### Chunk
A segment of text created during document processing. Default size: 512 characters.

### ColBERT
A reranking model using token-level late interaction scoring.

### Community
A cluster of related entities detected via graph algorithms (Louvain/Leiden).

### Contextual Retrieval
Technique that enriches chunks with surrounding context during ingestion.

### CRAG (Corrective RAG)
Pattern that grades retrieval quality and falls back to web search if needed.

### Cross-Encoder
A model that jointly encodes query and document for relevance scoring.

## D

### Dual-Level Retrieval
Retrieval that combines low-level (entity) and high-level (theme) results.

## E

### Embedding
A vector representation of text, typically 1536 dimensions.

### Entity
A named concept extracted from documents (person, concept, API, etc.).

### Episode
A document or conversation turn stored in the temporal knowledge graph.

## G

### Graph Reranker
Reranking using graph signals (episode recency, node distance).

### Graphiti
Library for temporal knowledge graphs with episode-based storage.

## H

### HITL (Human-in-the-Loop)
User validation step before accepting agent actions or sources.

### Hybrid Retrieval
Combining vector similarity search with graph traversal.

## L

### LazyRAG
Pattern that defers entity summarization to query time (99% cost reduction).

## M

### MCP (Model Context Protocol)
Protocol for AI tool execution, developed by Anthropic.

### Memory Scope
Persistence level: session, user, agent, or global.

## N

### NDCG (Normalized Discounted Cumulative Gain)
Metric for ranking quality, accounts for position relevance.

## O

### Open-JSON-UI
Protocol for declarative UI components (text, code, table, etc.).

## P

### pgvector
PostgreSQL extension for vector similarity search.

### Profile
Pre-configured settings bundle (minimal, standard, enterprise).

## Q

### Query Routing
Automatic selection of global vs local retrieval strategy.

## R

### Reranking
Second-pass scoring of retrieved results for improved relevance.

### RRF (Reciprocal Rank Fusion)
Algorithm combining multiple ranked lists.

## S

### Sparse Vector
Lexical matching representation (BM42 encoding).

## T

### Tenant
Isolated data namespace for multi-tenant deployments.

### Trajectory
Log of agent decision-making steps for debugging.

## V

### Vector Search
Similarity search using embedding vectors.
```

**Acceptance Criteria:**
- All domain-specific terms defined.
- Terms linked to relevant documentation.
- Alphabetically organized.
- Easy to search/navigate.

**File to Create:** `docs/reference/glossary.md`

---

### Story 18-25: Create Feature Discovery Matrix

**Objective:** Create a visual matrix showing all features, their requirements, and when to use them.

**Why Needed:** Users don't know ColBERT reranking exists. Need feature discovery and comparison.

**Priority:** MEDIUM

**Document Structure:**

```markdown
# Feature Discovery Matrix

## Overview

This matrix helps you discover and compare features across the platform.

## Retrieval Features

| Feature | Profile | Config | Cost Impact | Latency Impact | Best For |
|---------|---------|--------|-------------|----------------|----------|
| **Vector Search** | All | Default | Base | Base | All queries |
| **Graph Traversal** | Standard+ | Default | Base | +10% | Relationship queries |
| **Hybrid Search** | Standard+ | Default | Base | +15% | Best overall |
| **FlashRank Reranking** | Standard+ | RERANKER_ENABLED | Free | +50ms | Cost-sensitive |
| **Cohere Reranking** | Enterprise | RERANKER_PROVIDER=cohere | $1/1K | +100ms | Production |
| **ColBERT Reranking** | Enterprise | COLBERT_ENABLED | Free (local) | +200ms | Precision |
| **CRAG Grading** | Standard+ | GRADER_ENABLED | Free-$$ | +100ms | Quality control |
| **Contextual Retrieval** | Enterprise | CONTEXTUAL_ENABLED | +90% ingestion | None | High relevance |
| **Sparse Vectors (BM42)** | Enterprise | SPARSE_ENABLED | Free | +30ms | Keyword matching |
| **Cross-Language** | Enterprise | CROSS_LANG_ENABLED | Free-$ | +50ms | Multilingual |

## Memory Features

| Feature | Profile | Config | Impact | Best For |
|---------|---------|--------|--------|----------|
| **Session Memory** | All | Default | Low | Conversation context |
| **User Memory** | Standard+ | MEMORY_SCOPE=user | Medium | Personalization |
| **Agent Memory** | Standard+ | MEMORY_SCOPE=agent | Medium | Agent specialization |
| **Global Memory** | Enterprise | MEMORY_SCOPE=global | High | Organization knowledge |
| **Consolidation** | Enterprise | CONSOLIDATION_ENABLED | Low | Memory cleanup |
| **Decay** | Enterprise | DECAY_ENABLED | Low | Relevance management |

## Graph Intelligence

| Feature | Profile | Config | Impact | Best For |
|---------|---------|--------|--------|----------|
| **Community Detection** | Enterprise | COMMUNITY_ENABLED | Medium | Theme discovery |
| **LazyRAG** | Enterprise | LAZY_RAG_ENABLED | Low query, -99% index | Cost reduction |
| **Query Routing** | Standard+ | ROUTING_ENABLED | Low | Query optimization |
| **Graph Rerankers** | Enterprise | GRAPH_RERANKER | Medium | Relationship context |

## Ingestion Sources

| Source | Profile | Requires | Best For |
|--------|---------|----------|----------|
| **URL Crawling** | All | Default | Documentation |
| **PDF** | All | Default | Documents |
| **YouTube** | All | Default | Video content |
| **Codebase** | Enterprise | CODEBASE_ENABLED | Developer context |
| **S3 Sync** | Enterprise | AWS credentials | Cloud documents |
| **Confluence** | Enterprise | Atlassian token | Wiki content |
| **Notion** | Enterprise | Notion API key | Team docs |

## Feature Combinations

### Development Setup
- Vector search only
- FlashRank reranking
- Session memory
- Fast crawl profile

### Production Setup
- Hybrid search
- Cohere reranking + CRAG
- User memory + consolidation
- Query routing
- Thorough crawl profile

### Maximum Capability
- All retrieval features
- Enterprise memory
- Community detection + LazyRAG
- All ingestion sources
- Voice I/O
- Codebase intelligence
```

**Acceptance Criteria:**
- All features listed with profile requirements.
- Cost and latency impact documented.
- Use case guidance provided.
- Common combinations suggested.

**File to Create:** `docs/reference/feature-matrix.md`

---

### Story 18-26: Create Testing Framework Documentation

**Objective:** Document the testing infrastructure for contributors and operators.

**Why Needed:** 129 backend test files + 42 frontend test files exist but lack documentation on usage and patterns.

**Priority:** MEDIUM

**Document Structure:**

```markdown
# Testing Framework Documentation

## Overview

The testing infrastructure includes:
- **Unit tests:** Fast, isolated component tests
- **Integration tests:** Real service tests (PostgreSQL, Neo4j, Redis)
- **Compliance tests:** Protocol conformance validation
- **Security tests:** Tenant isolation and attack simulation
- **Benchmarks:** Performance measurement and CI gating

---

## Running Tests

### Backend Tests

```bash
cd backend

# Run all unit tests
uv run pytest

# Run with coverage
uv run pytest --cov=agentic_rag_backend --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_retrieval.py

# Run tests matching pattern
uv run pytest -k "test_reranker"
```

### Integration Tests

```bash
# Requires running services
export INTEGRATION_TESTS=1
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/test
export NEO4J_URI=bolt://localhost:7687
export REDIS_URL=redis://localhost:6379

uv run pytest tests/integration/
```

### Frontend Tests

```bash
cd frontend

# Run all tests
pnpm test

# Run with coverage
pnpm test --coverage

# Run in watch mode
pnpm test --watch
```

---

## Test Categories

### Unit Tests (`tests/unit/`)
- Fast, no external dependencies
- Mock all external services
- Target: 80% coverage

### Integration Tests (`tests/integration/`)
- Require real PostgreSQL, Neo4j, Redis
- Gated by `INTEGRATION_TESTS=1`
- Test full retrieval pipelines

### Compliance Tests (`tests/compliance/`)
- Protocol conformance (MCP, A2A, AG-UI)
- Marked with `@pytest.mark.compliance`

### Security Tests (`tests/security/`)
- Tenant isolation validation
- Attack simulation (SQL injection, etc.)
- Marked with `@pytest.mark.security`

### Benchmarks (`tests/benchmarks/`)
- Performance measurement
- Run with `benchmark-retrieval` CLI
- Output to JSONL for CI

---

## Writing Tests

### Backend Test Pattern

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_client():
    """Create mock database client."""
    client = AsyncMock()
    client.search.return_value = [{"id": "1", "content": "test"}]
    return client

async def test_retrieval_returns_results(mock_client):
    """Test that retrieval returns expected results."""
    # Arrange
    service = RetrievalService(mock_client)

    # Act
    results = await service.search("test query")

    # Assert
    assert len(results) == 1
    mock_client.search.assert_called_once()
```

### Frontend Test Pattern

```typescript
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

describe('SearchInput', () => {
  it('calls onSearch when submitted', async () => {
    const onSearch = jest.fn();
    render(<SearchInput onSearch={onSearch} />);

    await userEvent.type(screen.getByRole('textbox'), 'test query');
    await userEvent.click(screen.getByRole('button', { name: /search/i }));

    expect(onSearch).toHaveBeenCalledWith('test query');
  });
});
```

---

## CI Integration

Tests run automatically on:
- Pull requests
- Push to main/epic branches

### Coverage Requirements
- Backend: 80% minimum
- Frontend: 80% minimum

### Benchmark Gating
- Ingestion: >10 docs/minute
- Query latency: p95 <3s

---

## Fixtures Reference

### Backend Fixtures (`tests/conftest.py`)
- `sample_tenant_id` - UUID fixture
- `mock_redis` - AsyncMock Redis
- `mock_postgres_client` - Mock PostgreSQL
- `mock_neo4j_client` - Mock Neo4j
- `mock_graphiti_search_result` - Graphiti mock
- `client` - FastAPI TestClient

### Security Fixtures (`tests/security/conftest.py`)
- `sql_injection_payloads` - 7 OWASP patterns
- `cypher_injection_payloads` - Neo4j patterns
- `tenant_a_id`, `tenant_b_id` - Cross-tenant testing
```

**Acceptance Criteria:**
- Test execution commands documented.
- Test categories explained.
- Fixture reference included.
- CI integration described.
- Test patterns with examples.

**File to Create:** `docs/testing/framework.md`

---

### Story 18-27: Create Security Best Practices Guide

**Objective:** Document security best practices for deployment and operation.

**Why Needed:** Frontend security checklist exists but backend security guidance is missing.

**Priority:** MEDIUM

**Document Structure:**

```markdown
# Security Best Practices Guide

## Overview

This guide covers security best practices for deploying and operating the RAG platform.

---

## Authentication & Authorization

### API Key Management
```bash
# Generate secure API key
openssl rand -hex 32

# Store in environment (not .env in production)
export API_KEY=your-generated-key
```

### Tenant Isolation
- **CRITICAL:** Every query MUST include tenant_id
- Database queries filter by tenant
- Neo4j queries scope by tenant
- Cross-tenant access blocked

**Testing Tenant Isolation:**
```bash
# This should fail
curl -X POST /query \
  -d '{"query": "test", "tenant_id": "other-tenant"}'
```

---

## Secrets Management

### Required Production Secrets
| Secret | Purpose | Generation |
|--------|---------|------------|
| `TRACE_ENCRYPTION_KEY` | Encrypt trajectories | `openssl rand -hex 32` |
| `SHARE_SECRET` | Share link signing | `openssl rand -hex 16` |
| Database passwords | DB access | Strong random |
| API keys | Provider access | From provider |

### DO NOT
- Commit secrets to git
- Use .env files in production
- Share API keys
- Use default passwords

### DO
- Use secrets manager (Vault, AWS Secrets Manager)
- Rotate keys periodically
- Use different keys per environment
- Audit secret access

---

## Network Security

### Backend Access
```nginx
# Reverse proxy only - no direct access
location /api/ {
    proxy_pass http://backend:8000;
    proxy_set_header X-Real-IP $remote_addr;
}
```

### CORS Configuration
```bash
# Restrict to known origins
CORS_ORIGINS=["https://app.example.com"]
```

### Rate Limiting
```bash
RATE_LIMIT_BACKEND=redis  # Not memory in production
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=10
```

---

## Input Validation

### SQL Injection Prevention
- All queries use parameterized statements
- Never interpolate user input into SQL

### Cypher Injection Prevention
- Neo4j queries use parameters
- User input never in query strings

### File Upload Security
```bash
MAX_UPLOAD_SIZE_MB=100
ALLOWED_FILE_TYPES=["pdf", "txt", "md"]
```

---

## Data Protection

### Encryption at Rest
- PostgreSQL: Enable TDE or disk encryption
- Neo4j: Enable encryption
- Redis: Enable encryption if persisting

### Encryption in Transit
- TLS everywhere
- Minimum TLS 1.2
- Valid certificates

### PII Handling
- Telemetry sanitizes PII before storage
- User data scoped to tenant
- Implement data retention policies

---

## Trajectory Security

### Encryption
```bash
# Required in production
TRACE_ENCRYPTION_KEY=<64-char-hex>
```

### Access Control
- Trajectories encrypted with AES-256-GCM
- Only decryptable with key
- Key rotation requires re-encryption

---

## Security Checklist

### Pre-Deployment
- [ ] All secrets in secrets manager
- [ ] TLS configured
- [ ] CORS restricted
- [ ] Rate limiting enabled
- [ ] Default passwords changed
- [ ] Debug mode disabled

### Ongoing
- [ ] Dependencies updated (Dependabot)
- [ ] Security scans passing (CodeQL)
- [ ] Access logs reviewed
- [ ] Keys rotated quarterly
- [ ] Penetration testing annually

---

## Incident Response

### If API Key Compromised
1. Revoke compromised key immediately
2. Generate new key
3. Update all clients
4. Review access logs
5. Report to provider if third-party

### If Data Breach Suspected
1. Isolate affected systems
2. Preserve logs
3. Assess scope
4. Notify affected parties
5. Document and remediate
```

**Acceptance Criteria:**
- Authentication and authorization documented.
- Secrets management best practices.
- Network security guidance.
- Input validation patterns.
- Security checklist provided.
- Incident response procedures.

**File to Create:** `docs/security/best-practices.md`

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
