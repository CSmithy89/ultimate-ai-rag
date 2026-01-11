# Epic 17 Tech Spec: Developer Experience, CLI & Framework Integration

**Date:** 2025-12-31
**Updated:** 2026-01-12 (CLI UX Design - Party Mode Consensus)
**Status:** Backlog
**Epic Owner:** Product and Engineering

---

## Overview

Epic 17 delivers the complete developer experience: an interactive CLI for guided setup, framework starter templates for external agent development, Agent Skills for the Anthropic ecosystem, and deployment verification.

### Key Decisions

**CLI is LAST because it must know all available options** (2026-01-03)

The CLI offers choices for:
- LLM providers (Epic 11)
- Embedding providers (Epic 11)
- Ingestion sources (Epic 13)
- Retrieval features (Epic 12)
- Framework templates for external development

**Vision A: RAG as Service** (2026-01-05)

Following party mode analysis, we adopted "Vision A" where:
- Agno remains the internal orchestrator
- External developers connect via A2A/MCP using their preferred framework
- Framework support = starter templates + documentation, not internal adapters
- PydanticAI, CrewAI, LangGraph all have native A2A + MCP support

**Merged from Epic 16:** Agent Skills (Anthropic) and framework templates.

**Decision Document:** `docs/roadmap-decisions-2026-01-03.md`

### Goals

- Provide a `rag-install` interactive CLI for guided setup.
- Detect hardware capabilities and recommend appropriate defaults.
- Auto-generate validated `.env` and verify docker compose startup.
- **Enable first response in under 15 minutes.**
- Provide framework starter templates for PydanticAI, CrewAI, LangGraph, Anthropic SDK.
- Expose Agent Skills for Claude ecosystem integration.

### Scope

**In scope**
- Interactive CLI with prompts for all configurable options.
- Hardware detection (CPU, GPU, RAM).
- Environment file generation with validation.
- Docker Compose startup verification with health checks.
- Profile-based configuration (minimal, standard, enterprise).
- Framework starter templates with A2A/MCP connection examples.
- Agent Skills for Anthropic ecosystem (`.skills/` folder).
- Protocol integration documentation.

**Out of scope**
- Production orchestration (Kubernetes, Helm charts).
- Cloud-specific deployment (AWS, GCP, Azure).
- Internal framework adapters (frameworks connect externally via protocols).

---

## CLI UX Design (Party Mode Consensus - 2026-01-12)

### Design Philosophy: Progressive Disclosure

The CLI uses a **funnel model** to minimize cognitive load while preserving power-user access:

```
                    ┌─────────────────┐
  STAGE 1: Fast     │  4-5 questions  │  ← Profile + Provider + API Key + Framework
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
  STAGE 2: Expand   │  [c] Customize  │  ← Profile-specific options only
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
  STAGE 3: Deep     │  rag-cli setup  │  ← Full configuration wizard
                    └─────────────────┘
```

**Key Principles:**
- **Auto-detect and recommend** - Don't ask if we can infer from hardware
- **Smart defaults that just work** - Every profile is valid out of the box
- **Escape hatch for power users** - `--customize` or `rag-cli setup` for deep config
- **15-minute target** - Fast path must complete in under 3 minutes of user time

### The 5-Question Fast Path

Default `rag-install` flow (no flags):

```
╔══════════════════════════════════════════════════════════════════════════╗
║  RAG SYSTEM INSTALLER                                                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  ✓ System detected: 16GB RAM, Apple M2, Metal GPU                         ║
║  ✓ Recommended: Standard Profile                                          ║
║                                                                           ║
║  ─────────────────────────────────────────────────────────────────────── ║
║                                                                           ║
║  ? [1/4] Accept recommended profile? (Standard)                           ║
║    ❯ Yes, use Standard profile                                            ║
║      No, let me choose (Minimal / Enterprise / Custom)                    ║
║                                                                           ║
║  ? [2/4] LLM Provider:                                                    ║
║    ❯ OpenAI (recommended for most users)                                  ║
║      Anthropic Claude                                                     ║
║      Ollama (local, requires GPU)                                         ║
║      More options...                                                      ║
║                                                                           ║
║  ? [3/4] Enter OpenAI API Key: sk-****************************XXXX        ║
║                                                                           ║
║  ? [4/4] Generate framework starter? (for external agents)                ║
║    ❯ No starter needed                                                    ║
║      PydanticAI                                                           ║
║      CrewAI                                                               ║
║      LangGraph                                                            ║
║      Anthropic SDK                                                        ║
║                                                                           ║
║  ─────────────────────────────────────────────────────────────────────── ║
║                                                                           ║
║  Ready to install! This will:                                             ║
║  • Generate .env with your configuration                                  ║
║  • Start Docker services (PostgreSQL, Neo4j, Redis, Backend, Frontend)   ║
║                                                                           ║
║  [Enter] Install now   [c] Customize more   [Esc] Cancel                  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### Profile Feature Matrix

Each profile is a **complete, working configuration**. Features are pre-configured based on target use case:

| Category | Feature | Minimal | Standard | Enterprise |
|----------|---------|:-------:|:--------:|:----------:|
| **LLM** | Default Provider | OpenAI | OpenAI | OpenRouter |
| | Model | gpt-4o-mini | gpt-4o | claude-3.5-sonnet |
| **Embedding** | Provider | OpenAI | OpenAI | Voyage AI |
| | Model | text-embedding-3-small | text-embedding-3-small | voyage-code-3 |
| **Retrieval** | Reranking | ❌ | ❌ (opt-in) | ✅ Cohere |
| | Contextual Retrieval | ❌ | ❌ | ✅ |
| | CRAG Grader | ❌ | ❌ | ✅ |
| | Sparse Vectors (BM42) | ❌ | ❌ | ✅ |
| | ColBERT | ❌ | ❌ | ✅ |
| | Hierarchical Chunks | ❌ | ✅ | ✅ |
| **Memory** | Scopes Enabled | ❌ | ✅ | ✅ |
| | Default Scope | session | session | user |
| | Consolidation | ❌ | ❌ | ✅ |
| **Graph** | Community Detection | ❌ | ❌ | ✅ Louvain |
| | LazyRAG | ❌ | ❌ | ✅ |
| | Query Routing | ✅ (rule) | ✅ (rule) | ✅ (LLM) |
| | Graph Reranker | ❌ | ❌ | ✅ Hybrid |
| **Ingestion** | URL Crawling | ✅ Fast | ✅ Thorough | ✅ Stealth |
| | Crawl Fallback | ❌ | ❌ | ✅ Apify/BrightData |
| | PDF Processing | ✅ Fast | ✅ Accurate | ✅ Enhanced |
| | YouTube | ✅ | ✅ | ✅ |
| | External Sync (S3, etc) | ❌ | ❌ | ✅ |
| | Codebase RAG | ❌ | ❌ | ✅ |
| **Voice** | STT (Whisper) | ❌ | ❌ (opt-in) | ✅ base |
| | TTS | ❌ | ❌ (opt-in) | ✅ OpenAI |
| **Observability** | Prometheus | ✅ | ✅ | ✅ |
| | Cost Tracking | ✅ | ✅ | ✅ |
| | Tenant Label Mode | global | global | hash |
| | Trace Encryption | ❌ | ❌ | ✅ |
| **Protocols** | A2A Enabled | ✅ | ✅ | ✅ |
| | A2A Sessions/Tenant | 50 | 100 | 500 |
| | A2A Messages/Session | 500 | 1000 | 5000 |
| | MCP Enabled | ✅ | ✅ | ✅ |
| | MCP-UI Rendering | ❌ | ✅ | ✅ |

**Legend:** ✅ = Enabled by default | ❌ = Disabled | (opt-in) = Shown in customize mode

### CLI Command Structure

| Command | Purpose | Questions Asked |
|---------|---------|-----------------|
| `rag-install` | Fast path installation | 4-5 (profile, LLM, API key, framework) |
| `rag-install --customize` | Installation with all options | Profile-appropriate subset |
| `rag-install --profile <name> --yes` | Non-interactive installation | 0 (uses profile defaults) |
| `rag-cli setup` | Deep configuration wizard | All options for current profile |
| `rag-cli doctor` | Validate configuration | 0 (diagnostic only) |
| `rag-cli doctor --fix` | Auto-fix issues | Prompts only for fixes |

### Time Budget (15-Minute Goal)

| Activity | Target | Fast Path Design |
|----------|--------|------------------|
| Download/clone | 2 min | — |
| `rag-install` prompts | **3 min** | 4-5 questions only |
| Docker image pull | 5 min | Background during prompts |
| Services startup | 3 min | ~13s actual (Story 17-4) |
| First query | 2 min | — |
| **TOTAL** | **15 min** | ✅ Achievable |

---

## Stories

### Story 17-1: Create rag-install CLI Tool

**Objective:** Build a guided CLI that walks users through setup with intelligent defaults using the **5-Question Fast Path** (see CLI UX Design section above).

**Technology:** Python with `rich` for TUI and `typer` for CLI framework.

**CLI Flow (Fast Path - Default):**

The default `rag-install` asks only 4-5 questions to achieve the 15-minute first-response goal:

```
$ rag-install

╔══════════════════════════════════════════════════════════════╗
║                    RAG SYSTEM INSTALLER                       ║
╚══════════════════════════════════════════════════════════════╝

Detecting hardware...
  ✓ CPU: 8 cores (Apple M2)
  ✓ RAM: 16 GB
  ✓ GPU: Apple Metal (MPS)
  ✓ Recommended: Standard Profile

? [1/4] Accept recommended profile? (Standard)
  ❯ Yes, use Standard profile
    No, let me choose

? [2/4] LLM Provider:
  ❯ OpenAI (recommended)
    Anthropic Claude
    Ollama (local)
    More options...

? [3/4] Enter OpenAI API Key: sk-********************************XXXX

? [4/4] Generate framework starter?
  ❯ No starter needed
    PydanticAI
    CrewAI
    LangGraph
    Anthropic SDK

Ready to install!
  • Profile: Standard
  • LLM: OpenAI (gpt-4o)
  • Embedding: OpenAI (text-embedding-3-small)

[Enter] Install now   [c] Customize more   [Esc] Cancel

Starting docker compose...
  ✓ PostgreSQL (pgvector) - healthy (2.1s)
  ✓ Neo4j - healthy (4.3s)
  ✓ Redis - healthy (0.8s)
  ✓ Backend - healthy (3.2s)
  ✓ Frontend - healthy (2.5s)

╔══════════════════════════════════════════════════════════════╗
║  SUCCESS! Your RAG system is running.                         ║
║                                                                ║
║  Frontend: http://localhost:3000                               ║
║  API Docs: http://localhost:8000/docs                          ║
║                                                                ║
║  Next steps:                                                   ║
║  1. Open http://localhost:3000 in your browser                 ║
║  2. Try: "What can you help me with?"                          ║
║  3. Ingest your first document via the UI                      ║
║                                                                ║
║  Run 'rag-cli setup' to customize advanced features.           ║
╚══════════════════════════════════════════════════════════════╝
```

**CLI Flow (Customize Mode - `--customize` or [c] key):**

When user presses [c] or runs `rag-install --customize`, show profile-appropriate additional options:

```
──────────────────────────────────────────────────────────────
CUSTOMIZE: Standard Profile Options
──────────────────────────────────────────────────────────────

? Embedding provider:
  ❯ OpenAI text-embedding-3-small (default)
    Voyage AI (best for code)
    Google Gemini

? Enable optional retrieval features?
  ☐ Cross-encoder reranking (+latency, +precision)
  ☐ Contextual retrieval (+cost during ingestion)

? Enable voice I/O?
  ☐ Speech-to-text (Whisper)
  ☐ Text-to-speech (OpenAI)

[Enter] Continue with selections   [Esc] Back to fast path
```

**Configuration Options:**

| Category | Options | Default |
|----------|---------|---------|
| **LLM Provider** | openai, anthropic, gemini, openrouter, ollama | openai |
| **Embedding Provider** | openai, voyage, gemini, ollama | openai |
| **Framework Template** | none, pydanticai, crewai, langgraph, anthropic | none |
| **Reranking** | cohere, flashrank, disabled | disabled (profile-based) |
| **Database** | postgresql+neo4j (fixed) | - |
| **Profile** | minimal, standard, enterprise | standard |

**Acceptance Criteria**
- Running `rag-install` starts the **fast path** (4-5 questions max) with `rich` TUI.
- Hardware detection runs automatically and recommends appropriate profile.
- Users can accept recommended profile or choose manually.
- [c] key or `--customize` flag shows profile-appropriate additional options.
- CLI validates API keys format before proceeding.
- CLI writes configuration into `.env` with comments explaining each setting.
- Non-interactive mode supported: `rag-install --profile standard --llm openai --yes`
- All profile configurations are valid out of the box (per Profile Feature Matrix).
- Fast path completes in under 3 minutes of user interaction time.

**Files to Create:**
- `cli/main.py` - Main CLI entry point with typer
- `cli/commands/install.py` - rag-install command implementation
- `cli/prompts/fast_path.py` - 5-question fast path prompt flow
- `cli/prompts/customize.py` - Profile-appropriate customize prompts
- `cli/ui/panels.py` - Rich TUI panel components
- `tests/cli/test_install.py` - CLI installation tests

### Story 17-2: Implement Auto Hardware Detection

**Objective:** Detect CPU, GPU, and memory to recommend appropriate defaults.

**Detection Capabilities:**

| Hardware | Detection | Impact on Defaults |
|----------|-----------|-------------------|
| **GPU (NVIDIA)** | `nvidia-smi` | Enable Ollama, recommend local models |
| **GPU (Apple MPS)** | `torch.backends.mps` | Enable Ollama with MPS |
| **GPU (None)** | Fallback | Recommend cloud providers |
| **RAM >= 32GB** | `psutil` | Enable larger local models |
| **RAM >= 16GB** | `psutil` | Standard profile |
| **RAM < 16GB** | `psutil` | Minimal profile, warn user |
| **CPU Cores >= 8** | `os.cpu_count()` | Higher concurrency settings |

**Profile Recommendations:**

| Profile | RAM | GPU | LLM | Embeddings | Features |
|---------|-----|-----|-----|------------|----------|
| Minimal | <16GB | None | OpenAI (mini) | OpenAI (small) | None |
| Standard | 16GB+ | Any | OpenAI/Anthropic | OpenAI | Optional |
| Enterprise | 32GB+ | NVIDIA | OpenRouter | Voyage AI | All enabled |

**Acceptance Criteria**
- CLI detects available GPU type (NVIDIA, Apple MPS, None).
- CLI detects RAM and recommends appropriate profile.
- Detection results are shown before applying configuration.
- User can override detected recommendations.
- Detection works on Linux, macOS, and Windows (WSL2).

### Story 17-3: Implement Env Generation Logic

**Objective:** Generate a validated `.env` from user selections.

**Validation Rules:**

| Variable | Validation | Error Message |
|----------|------------|---------------|
| `OPENAI_API_KEY` | Starts with `sk-` | "OpenAI keys start with 'sk-'" |
| `ANTHROPIC_API_KEY` | Starts with `sk-ant-` | "Anthropic keys start with 'sk-ant-'" |
| `DATABASE_URL` | Valid PostgreSQL URI | "Invalid PostgreSQL connection string" |
| `NEO4J_URI` | Valid bolt:// URI | "Neo4j URI must start with 'bolt://'" |

**Generated .env Structure:**

```bash
# ═══════════════════════════════════════════════════════════════
# AGENTIC RAG CONFIGURATION
# Generated by rag-install on 2026-01-05
# Profile: standard
# ═══════════════════════════════════════════════════════════════

# ─── LLM Provider ───────────────────────────────────────────────
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
# Available: openai, anthropic, gemini, openrouter, ollama

# ─── Embedding Provider ─────────────────────────────────────────
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
# Available: openai, voyage, gemini, ollama

# ─── Database ───────────────────────────────────────────────────
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/agentic_rag
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379

# ─── Advanced Retrieval (Optional) ──────────────────────────────
RERANKER_ENABLED=false
# RERANKER_PROVIDER=flashrank
CONTEXTUAL_RETRIEVAL_ENABLED=false
GRADER_ENABLED=false

# ─── Ingestion ──────────────────────────────────────────────────
CRAWL4AI_MAX_CONCURRENCY=10
CRAWL4AI_HEADLESS=true
```

**Acceptance Criteria**
- All required variables are populated with defaults or user input.
- CLI validates API key formats before writing.
- A backup `.env.bak` is created if existing file present.
- Generated `.env` includes helpful comments for each section.
- Sensitive values are masked in CLI output (show last 4 chars only).

### Story 17-4: Verify Docker Compose Startup

**Objective:** Validate that the stack boots successfully with health checks.

**Health Check Sequence:**

```
Starting services...
  [1/5] PostgreSQL... ✓ (2.1s)
  [2/5] Neo4j...      ✓ (4.3s)
  [3/5] Redis...      ✓ (0.8s)
  [4/5] Backend...    ✓ (3.2s) → http://localhost:8000/health
  [5/5] Frontend...   ✓ (2.5s) → http://localhost:3000

All services healthy! Total startup time: 12.9s
```

**Error Handling:**

| Error | Detection | Suggested Fix |
|-------|-----------|---------------|
| Port in use | Connection refused | "Port 8000 in use. Stop existing service or change BACKEND_PORT" |
| Docker not running | Docker socket error | "Docker daemon not running. Start Docker Desktop" |
| Out of memory | OOM killer | "Insufficient memory. Try 'rag-install --profile minimal'" |
| API key invalid | 401 from provider | "Invalid API key for OpenAI. Check OPENAI_API_KEY" |
| Database connection | Connection timeout | "Cannot connect to PostgreSQL. Check DATABASE_URL" |

**Acceptance Criteria**
- CLI runs `docker compose up -d` and monitors health endpoints.
- Each service shows status with timing.
- Failures produce actionable error messages with suggested fixes.
- Successful startup reports service URLs and next steps.
- `--dry-run` option shows what would happen without executing.

### Story 17-5: Create Framework Starter Templates

**Objective:** Generate ready-to-use starter code for connecting to the RAG from each framework.

**Origin:** Merged from Epic 16 (Framework Agnosticism) per Vision A decision.

**Template Structure:**

```
examples/
├── pydanticai/
│   ├── README.md           # Quick start guide
│   ├── pyproject.toml      # Dependencies
│   ├── agent.py            # Sample agent connecting via A2A
│   └── mcp_client.py       # Sample MCP tool consumer
├── crewai/
│   ├── README.md
│   ├── pyproject.toml
│   ├── crew.py             # Sample crew with A2A delegation
│   └── tasks.py            # Task definitions using RAG tools
├── langgraph/
│   ├── README.md
│   ├── pyproject.toml
│   ├── graph.py            # Sample graph with MCP tools
│   └── nodes.py            # Node definitions
└── anthropic/
    ├── README.md
    ├── pyproject.toml
    └── agent.py            # Sample agent with MCP tools
```

**PydanticAI Example (`examples/pydanticai/agent.py`):**

```python
"""PydanticAI agent that connects to RAG via A2A protocol."""
from pydantic_ai import Agent

# Your RAG exposes A2A at http://localhost:8000/a2a
rag_agent = Agent(
    'openai:gpt-4o',
    tools=[],  # Add your custom tools
)

# Connect to RAG via A2A
from fasta2a import A2AClient

rag_client = A2AClient("http://localhost:8000/a2a")

@rag_agent.tool
async def search_knowledge(query: str) -> str:
    """Search the RAG knowledge base."""
    result = await rag_client.send_message(query)
    return result.content
```

**CrewAI Example (`examples/crewai/crew.py`):**

```python
"""CrewAI crew that delegates to RAG via A2A protocol."""
from crewai import Agent, Crew, Task

# Install: pip install 'crewai[a2a]'
researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert at finding information",
    a2a_agents=[{
        "url": "http://localhost:8000/a2a",
        "name": "rag_knowledge_base",
        "description": "Search the knowledge graph and vector store"
    }]
)
```

**Acceptance Criteria**
- Each framework has a complete, runnable starter template.
- Templates include README with setup instructions.
- Templates demonstrate both A2A and MCP connection patterns.
- `rag-install --framework <name>` copies template to `examples/<name>/`.
- Templates are tested and verified working with current RAG version.

### Story 17-6: Implement Agent Skills for Anthropic Ecosystem

**Objective:** Expose RAG capabilities as Agent Skills for Claude Desktop, Claude Code, and API.

**Origin:** Story 16-4 from Epic 16 (kept due to unique value).

**Background:** Agent Skills is an [open standard adopted by Microsoft/VS Code, Cursor, and others](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills). It provides organized folders of instructions, scripts, and resources that agents can discover and load dynamically.

**Skills Structure:**

```
.skills/
├── rag-search/
│   ├── skill.yaml
│   ├── instructions.md
│   └── examples/
│       ├── basic-query.md
│       └── multi-hop-query.md
├── ingest-url/
│   ├── skill.yaml
│   ├── instructions.md
│   └── examples/
│       └── crawl-docs.md
├── ingest-pdf/
│   ├── skill.yaml
│   └── instructions.md
├── ingest-youtube/
│   ├── skill.yaml
│   └── instructions.md
└── explain-answer/
    ├── skill.yaml
    ├── instructions.md
    └── examples/
        └── trajectory-debug.md
```

**Sample skill.yaml (rag-search):**

```yaml
name: rag-search
version: 1.0.0
description: Search the knowledge graph and vector store for relevant information
author: Agentic RAG
tags: [rag, search, knowledge, retrieval]

# MCP tool this skill exposes
mcp_tool: knowledge.query

# When to use this skill
triggers:
  - "search for"
  - "find information about"
  - "what do you know about"
  - "look up"

# Parameters
parameters:
  query:
    type: string
    description: The search query
    required: true
  tenant_id:
    type: string
    description: Tenant identifier for multi-tenancy
    default: "default"

# Example invocations
examples:
  - input: "Search for information about GraphRAG"
    output: "Found 5 relevant documents about GraphRAG..."
```

**Sample instructions.md:**

```markdown
# RAG Search Skill

This skill searches the knowledge graph (Neo4j + Graphiti) and vector store
(PostgreSQL + pgvector) to find relevant information.

## Usage

Ask questions naturally. The skill will:
1. Analyze your query to select the best retrieval strategy
2. Search both vector embeddings and graph relationships
3. Rerank results for relevance (if enabled)
4. Return synthesized answer with source citations

## Examples

- "What is the relationship between X and Y?"
- "Summarize everything about topic Z"
- "Find documents mentioning keyword W"

## Configuration

The skill connects to your RAG backend at `http://localhost:8000`.
Ensure the backend is running before using this skill.
```

**Acceptance Criteria**
- `.skills/` folder is generated with all RAG capabilities exposed.
- Each skill has `skill.yaml` with proper metadata and MCP tool mapping.
- Each skill has `instructions.md` explaining usage.
- Skills are discoverable by Claude Desktop when RAG is running.
- Skills work with Claude Code for codebase-aware RAG queries.
- `rag-install --with-skills` generates the `.skills/` folder.
- Skills are validated against the Agent Skills schema.

### Story 17-7: Document Protocol Integration Guide

**Status:** ✅ COMPLETED via Epic 22-D1 (2026-01-11)

**Objective:** Create comprehensive documentation for connecting external agents via A2A, MCP, and AG-UI.

**Origin:** Replaces Epic 16 adapter stories with documentation-first approach.

**Completion Note:** Epic 22 Story 22-D1 delivered comprehensive protocol integration documentation covering all protocols. The documentation structure differs slightly from the original plan but provides superior coverage.

**Actual Documentation Created (by 22-D1):**

```
docs/guides/protocol-integration/
├── overview.md                      # High-level architecture + protocol summary
├── ag-ui-protocol.md                # AG-UI events, metrics, errors
├── a2a-protocol.md                  # A2A middleware, delegation, resource limits
├── mcp-integration.md               # MCP tool registration and invocation
├── a2ui-widgets.md                  # A2UI widget types and rendering
├── mcp-ui-rendering.md              # MCP-UI iframe security + postMessage
└── open-json-ui.md                  # Open-JSON-UI component types
```

**protocol-integration.md Content:**

```markdown
# Connecting to Agentic RAG

This guide explains how to connect your agents to the RAG platform
using standard protocols.

## Architecture Overview

```
Your Agent (Any Framework) → A2A/MCP → RAG Platform → Knowledge
```

## Protocols

| Protocol | Purpose | Endpoint |
|----------|---------|----------|
| A2A | Agent-to-agent collaboration | `POST /a2a/*` |
| MCP | Tool execution | `POST /mcp/call` |
| AG-UI | Frontend state sync | `POST /copilot` (SSE) |

## Quick Start by Framework

### PydanticAI
```python
from fasta2a import A2AClient
client = A2AClient("http://localhost:8000/a2a")
```

### CrewAI
```python
agent = Agent(a2a_agents=[{"url": "http://localhost:8000/a2a"}])
```

### LangGraph
```python
from langchain_mcp_adapters import MCPToolkit
toolkit = MCPToolkit("http://localhost:8000/mcp")
```

### Anthropic SDK
```python
from anthropic import Client
# Use MCP tools directly
```

## Available MCP Tools

- `knowledge.query` - Search knowledge base
- `knowledge.graph_stats` - Get graph statistics
- `vector_search` - Direct vector search
- `ingest_url` - Crawl and ingest URL
- `ingest_pdf` - Process PDF document
- `ingest_youtube` - Extract YouTube transcript
```

**Acceptance Criteria**
- Documentation covers A2A, MCP, and AG-UI protocols.
- Each framework has specific connection examples.
- All MCP tools are documented with input/output schemas.
- AG-UI event format is documented for custom UI implementations.
- Documentation is linked from README and CLI output.

---

## NEW STORIES (Added 2026-01-11 - Comprehensive Audit)

The following stories were added after a comprehensive system audit revealed 200+ environment variables across 10 major feature categories that the CLI must support.

### Story 17-8: Implement Profile-Based Configuration Architecture

**Objective:** Refactor configuration system from flat .env to profile-based architecture for simplified CLI experience.

**Problem:** Current system has 200+ environment variables, making manual configuration error-prone and overwhelming.

**Solution:** Profile-based configuration with three tiers:

```
config/
├── profiles/
│   ├── minimal.yaml      # CPU-only, basic features, low resource
│   ├── standard.yaml     # Cloud LLM, core features, balanced
│   ├── enterprise.yaml   # All features enabled, full capabilities
│   └── custom.yaml       # User overrides (gitignored)
├── schema.json           # JSON Schema for validation
└── README.md             # Profile documentation
```

**Profile Definitions:**

**minimal.yaml:**
```yaml
# Minimal Profile - Low resource, basic RAG
# Target: Development, testing, resource-constrained environments

llm:
  provider: openai
  model: gpt-4o-mini

embedding:
  provider: openai
  model: text-embedding-3-small
  dimension: 1536

retrieval:
  strategy: vector  # Vector-only, no graph
  reranker:
    enabled: false
  contextual_retrieval:
    enabled: false
  grader:
    enabled: false

memory:
  scopes_enabled: false
  consolidation_enabled: false

community:
  detection_enabled: false

ingestion:
  crawl_profile: fast
  fallback_enabled: false
  youtube_enabled: true
  pdf_enabled: true
  codebase_enabled: false
  external_sync_enabled: false

voice:
  enabled: false

graph_intelligence:
  lazy_rag_enabled: false
  query_routing_enabled: false
  graph_reranker_enabled: false

observability:
  prometheus_enabled: false
  cost_tracking_enabled: true

protocols:
  a2a:
    enabled: true
    max_sessions_per_tenant: 10
    max_messages_per_session: 100
  mcp:
    enabled: true
```

**standard.yaml:**
```yaml
# Standard Profile - Balanced features, cloud LLM
# Target: Production deployments, small-medium teams

llm:
  provider: openai  # or anthropic
  model: gpt-4o

embedding:
  provider: openai
  model: text-embedding-3-small
  dimension: 1536

retrieval:
  strategy: hybrid
  reranker:
    enabled: true
    provider: flashrank
    top_k: 10
  contextual_retrieval:
    enabled: false  # Cost consideration
  grader:
    enabled: true
    model: heuristic
    threshold: 0.5
    fallback_enabled: true
    fallback_strategy: web_search

memory:
  scopes_enabled: true
  default_scope: session
  consolidation_enabled: false

community:
  detection_enabled: false

ingestion:
  crawl_profile: thorough
  fallback_enabled: false
  youtube_enabled: true
  pdf_enabled: true
  codebase_enabled: false
  external_sync_enabled: false

voice:
  enabled: false

graph_intelligence:
  lazy_rag_enabled: false
  query_routing_enabled: true
  query_routing_use_llm: false
  graph_reranker_enabled: false

observability:
  prometheus_enabled: true
  cost_tracking_enabled: true

protocols:
  a2a:
    enabled: true
    max_sessions_per_tenant: 100
    max_messages_per_session: 1000
  mcp:
    enabled: true
```

**enterprise.yaml:**
```yaml
# Enterprise Profile - All features enabled
# Target: Large deployments, maximum capabilities

llm:
  provider: openrouter  # Access to 100+ models
  model: anthropic/claude-3.5-sonnet

embedding:
  provider: voyage  # Best for code
  model: voyage-code-3
  dimension: 1536

retrieval:
  strategy: hybrid
  reranker:
    enabled: true
    provider: cohere
    top_k: 10
    cache_enabled: true
    cache_ttl_seconds: 300
  contextual_retrieval:
    enabled: true
    model: claude-3-haiku-20240307
    prompt_caching: true
  grader:
    enabled: true
    model: cross-encoder/ms-marco-MiniLM-L-12-v2
    threshold: 0.5
    fallback_enabled: true
    fallback_strategy: web_search
    preload_model: true
  sparse_vectors:
    enabled: true
    model: Qdrant/bm42-all-minilm-l6-v2-attentions
  colbert:
    enabled: true
    model: colbert-ir/colbertv2.0
  hierarchical_chunks:
    enabled: true
    levels: [256, 512, 1024, 2048]
  dual_level:
    enabled: true

memory:
  scopes_enabled: true
  default_scope: user
  include_parent_scopes: true
  consolidation_enabled: true
  consolidation_schedule: "0 2 * * *"
  similarity_threshold: 0.9
  decay_half_life_days: 30

community:
  detection_enabled: true
  algorithm: louvain
  min_size: 3
  max_levels: 3
  refresh_schedule: "0 3 * * 0"

ingestion:
  crawl_profile: stealth
  fallback_enabled: true
  fallback_providers: [apify, brightdata]
  youtube_enabled: true
  pdf_enabled: true
  enhanced_docling: true
  codebase_enabled: true
  codebase_languages: [python, typescript, javascript]
  external_sync_enabled: true

voice:
  enabled: true
  whisper_model: base
  tts_provider: openai
  tts_voice: alloy

graph_intelligence:
  lazy_rag_enabled: true
  lazy_rag_max_entities: 50
  lazy_rag_max_hops: 2
  query_routing_enabled: true
  query_routing_use_llm: true
  graph_reranker_enabled: true
  graph_reranker_type: hybrid

observability:
  prometheus_enabled: true
  prometheus_path: /metrics
  cost_tracking_enabled: true
  metrics_tenant_label_mode: hash

protocols:
  a2a:
    enabled: true
    max_sessions_per_tenant: 500
    max_messages_per_session: 5000
    session_ttl_hours: 48
    message_rate_limit: 120
  mcp:
    enabled: true
    tool_timeout_seconds: 60
```

**Implementation Requirements:**

1. **Config Loader Module:**
```python
# backend/src/agentic_rag_backend/core/config_loader.py
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import yaml
from pathlib import Path
from typing import Any

class ConfigLoader:
    """Load configuration from profile + environment overrides."""

    PROFILE_DIR = Path("config/profiles")

    def __init__(self, profile: str = "standard"):
        self.profile = profile
        self._config: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Load profile and merge with env overrides."""
        profile_path = self.PROFILE_DIR / f"{self.profile}.yaml"
        if not profile_path.exists():
            raise ValueError(f"Profile not found: {self.profile}")

        with open(profile_path) as f:
            self._config = yaml.safe_load(f)

        # Apply environment variable overrides
        self._apply_env_overrides()
        return self._config

    def _apply_env_overrides(self) -> None:
        """Environment variables override profile defaults."""
        # Pattern: LLM_PROVIDER overrides config.llm.provider
        import os
        env_mappings = {
            "LLM_PROVIDER": ("llm", "provider"),
            "LLM_MODEL_ID": ("llm", "model"),
            "EMBEDDING_PROVIDER": ("embedding", "provider"),
            "RERANKER_ENABLED": ("retrieval", "reranker", "enabled"),
            # ... complete mapping
        }
        for env_var, path in env_mappings.items():
            if value := os.getenv(env_var):
                self._set_nested(path, value)
```

2. **Settings Integration:**
```python
# Update backend/src/agentic_rag_backend/core/settings.py
class Settings(BaseSettings):
    # Profile selection
    config_profile: str = "standard"

    # Core secrets (always from env)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    database_url: str
    neo4j_uri: str
    redis_url: str

    # All other settings loaded from profile
    # with env override capability

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        loader = ConfigLoader(self.config_profile)
        profile_config = loader.load()
        # Merge profile into settings
        self._apply_profile(profile_config)
```

3. **Schema Validation:**
```json
// config/schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["llm", "embedding", "retrieval"],
  "properties": {
    "llm": {
      "type": "object",
      "properties": {
        "provider": {"enum": ["openai", "anthropic", "gemini", "openrouter", "ollama"]},
        "model": {"type": "string"}
      }
    }
    // ... complete schema
  }
}
```

**Acceptance Criteria:**
- Profile loader reads YAML configuration files.
- Environment variables override profile defaults.
- Schema validation rejects invalid configurations.
- CLI `--profile` flag selects configuration profile.
- Custom profile support for advanced users.
- Migration guide from flat .env to profile-based config.
- Backward compatibility: existing .env files still work.

**Files to Create:**
- `backend/src/agentic_rag_backend/core/config_loader.py`
- `config/profiles/minimal.yaml`
- `config/profiles/standard.yaml`
- `config/profiles/enterprise.yaml`
- `config/profiles/custom.yaml.template`
- `config/schema.json`
- `config/README.md`
- `docs/guides/configuration-profiles.md`

---

### Story 17-9: CLI Ingestion Source Configuration

**Objective:** Add CLI prompts for configuring all ingestion sources with profile-aware defaults.

**Ingestion Sources to Configure:**

| Source | Variables | Profile Default |
|--------|-----------|-----------------|
| **URL Crawling** | CRAWL4AI_PROFILE, CRAWL4AI_HEADLESS, CRAWL4AI_MAX_CONCURRENT | Profile-based |
| **Crawl Fallback** | CRAWL_FALLBACK_ENABLED, APIFY_API_TOKEN, BRIGHTDATA_* | Enterprise only |
| **PDF Processing** | DOCLING_TABLE_MODE, ENHANCED_DOCLING_ENABLED | All profiles |
| **YouTube** | YOUTUBE_PREFERRED_LANGUAGES, YOUTUBE_CHUNK_DURATION_SECONDS | All profiles |
| **External Sync** | EXTERNAL_SYNC_ENABLED, S3_*, CONFLUENCE_*, NOTION_* | Enterprise only |
| **Codebase** | CODEBASE_RAG_ENABLED, CODEBASE_LANGUAGES | Enterprise only |

**CLI Flow Addition:**

```
? Configure ingestion sources? [Y/n]

URL Crawling (enabled by default):
  ? Select crawl profile:
    ❯ Fast (quick crawls, minimal JS wait)
      Thorough (full rendering, moderate wait)
      Stealth (anti-bot evasion, proxy support)

  ? Enable crawl fallback providers? (Apify/BrightData) [y/N]
  (If yes, prompt for API tokens)

Document Processing:
  ? PDF processing mode:
    ❯ Accurate (better tables, slower)
      Fast (quick processing)

  ? Enable enhanced table/layout extraction? [Y/n]

YouTube:
  ? Preferred transcript languages (comma-separated) [en,en-US]:
  ? Chunk duration (seconds) [120]:

External Data Sources (Enterprise profile only):
  ? Enable external data source sync? [y/N]
  (If yes, show sub-menu for S3, Confluence, Notion)

Codebase Intelligence (Enterprise profile only):
  ? Enable codebase RAG indexing? [y/N]
  ? Supported languages [python,typescript,javascript]:
  ? Enable hallucination detection? [Y/n]
```

**Acceptance Criteria:**
- CLI prompts for all ingestion sources based on selected profile.
- Minimal profile skips advanced ingestion prompts.
- Enterprise profile shows all options.
- API keys validated before proceeding.
- Generated config includes all ingestion settings.

**Files to Create:**
- `cli/commands/ingestion_config.py` - Ingestion source configuration prompts
- `cli/prompts/ingestion_prompts.py` - Prompt definitions for ingestion options
- `tests/cli/test_ingestion_config.py` - Tests for ingestion configuration flow

---

### Story 17-10: CLI Memory & Graph Intelligence Configuration

**Objective:** Add CLI prompts for memory platform and graph intelligence features.

**Features to Configure:**

| Feature | Variables | Description |
|---------|-----------|-------------|
| **Memory Scopes** | MEMORY_SCOPES_ENABLED, MEMORY_DEFAULT_SCOPE | Session/User/Agent/Global scopes |
| **Memory Consolidation** | MEMORY_CONSOLIDATION_ENABLED, schedule, thresholds | Dedup, decay, cleanup |
| **Community Detection** | COMMUNITY_DETECTION_ENABLED, COMMUNITY_ALGORITHM | Graph clustering |
| **LazyRAG** | LAZY_RAG_ENABLED, max_entities, max_hops | Deferred summarization |
| **Query Routing** | QUERY_ROUTING_ENABLED, QUERY_ROUTING_USE_LLM | Global vs local queries |
| **Graph Rerankers** | GRAPH_RERANKER_ENABLED, GRAPH_RERANKER_TYPE | Episode/distance/hybrid |

**CLI Flow Addition:**

```
? Configure memory & graph intelligence? [Y/n]

Memory Platform:
  ? Enable memory scopes? [Y/n]
  ? Default memory scope:
    ❯ Session (per conversation)
      User (per user, across sessions)
      Agent (per agent type)
      Global (shared across all)

  ? Enable memory consolidation? [y/N]
  (If yes, show schedule and threshold options)

Graph Intelligence:
  ? Enable community detection? [y/N]
  ? Detection algorithm:
    ❯ Louvain (fast, good quality)
      Leiden (slower, better quality)

  ? Enable LazyRAG pattern? [y/N]
  (Defers summarization to query time - 99% indexing cost reduction)

  ? Enable query routing? [Y/n]
  ? Use LLM for query classification? [y/N]

  ? Enable graph-based rerankers? [y/N]
  ? Reranker type:
    ❯ Hybrid (recommended)
      Episode recency
      Graph distance
```

**Acceptance Criteria:**
- CLI prompts for memory and graph features based on profile.
- Minimal profile disables all advanced features.
- Standard profile enables basic memory scopes.
- Enterprise profile shows all options.
- Generated config reflects all selections.

**Files to Create:**
- `cli/commands/memory_config.py` - Memory & graph configuration prompts
- `cli/prompts/memory_prompts.py` - Prompt definitions for memory/graph options
- `tests/cli/test_memory_config.py` - Tests for memory configuration flow

---

### Story 17-11: CLI Voice I/O Configuration

**Objective:** Add CLI prompts for voice input/output capabilities.

**Features to Configure:**

| Feature | Variables | Options |
|---------|-----------|---------|
| **Voice Enable** | VOICE_IO_ENABLED | true/false |
| **STT Model** | WHISPER_MODEL | tiny, base, small, medium, large |
| **TTS Provider** | TTS_PROVIDER | openai, elevenlabs, pyttsx3 |
| **TTS Voice** | TTS_VOICE | Provider-specific voice IDs |
| **TTS Speed** | TTS_SPEED | 0.25 to 4.0 |

**CLI Flow Addition:**

```
? Enable voice input/output? [y/N]

Speech-to-Text (STT):
  ? Whisper model size:
    ❯ Base (recommended, balanced)
      Tiny (fastest, lowest accuracy)
      Small (good accuracy)
      Medium (high accuracy)
      Large (best accuracy, slowest)

Text-to-Speech (TTS):
  ? TTS provider:
    ❯ OpenAI (high quality, cloud)
      ElevenLabs (premium voices, cloud)
      pyttsx3 (free, local, basic)

  ? Voice selection:
    ❯ Alloy (neutral)
      Echo (male)
      Fable (British)
      Onyx (deep)
      Nova (female)
      Shimmer (soft)

  ? Speech speed [1.0]:
```

**Acceptance Criteria:**
- Voice features only shown if user opts in.
- Model size recommendations based on hardware detection.
- ElevenLabs requires API key prompt.
- pyttsx3 noted as offline-capable.
- Generated config includes voice settings.

**Files to Create:**
- `cli/commands/voice_config.py` - Voice I/O configuration prompts
- `cli/prompts/voice_prompts.py` - Prompt definitions for voice options
- `tests/cli/test_voice_config.py` - Tests for voice configuration flow

---

### Story 17-12: CLI Observability Configuration

**Objective:** Add CLI prompts for monitoring and observability features.

**Features to Configure:**

| Feature | Variables | Description |
|---------|-----------|-------------|
| **Prometheus** | PROMETHEUS_ENABLED, PROMETHEUS_PATH | Metrics endpoint |
| **Metrics Labeling** | METRICS_TENANT_LABEL_MODE | Tenant label strategy |
| **Cost Tracking** | MODEL_PRICING_JSON | Per-model cost tracking |
| **Model Routing** | ROUTING_*_MODEL, thresholds | Cost-optimized routing |
| **Trace Encryption** | TRACE_ENCRYPTION_KEY | Encrypted trajectory storage |

**CLI Flow Addition:**

```
? Configure observability & monitoring? [Y/n]

Prometheus Metrics:
  ? Enable Prometheus metrics endpoint? [Y/n]
  ? Metrics path [/metrics]:
  ? Tenant label mode:
    ❯ Global (no tenant labels, lowest cardinality)
      Hash (hashed tenant IDs, medium cardinality)
      Full (full tenant IDs, highest cardinality)

Cost Tracking:
  ? Enable LLM cost tracking? [Y/n]
  ? Configure cost-optimized model routing? [y/N]
  (If yes, show routing model selection)

Trajectory Logging:
  ? Enable encrypted trajectory storage? [Y/n]
  (If yes, generate or prompt for encryption key)

  ⚠️  TRACE_ENCRYPTION_KEY will be auto-generated.
  Store this key securely - existing traces cannot be
  decrypted if the key is lost.
```

**Acceptance Criteria:**
- Observability prompts shown based on profile.
- Prometheus enabled by default in standard/enterprise.
- Encryption key auto-generated with secure random bytes.
- Warning displayed about key storage.
- Generated config includes observability settings.

**Files to Create:**
- `cli/commands/observability_config.py` - Observability configuration prompts
- `cli/prompts/observability_prompts.py` - Prompt definitions for monitoring options
- `cli/utils/key_generator.py` - Secure encryption key generation utility
- `tests/cli/test_observability_config.py` - Tests for observability configuration flow

---

### Story 17-13: CLI Codebase Intelligence Configuration

**Objective:** Add CLI prompts for codebase RAG and hallucination detection.

**Features to Configure:**

| Feature | Variables | Description |
|---------|-----------|-------------|
| **Codebase RAG** | CODEBASE_RAG_ENABLED | Index codebase as knowledge |
| **Languages** | CODEBASE_LANGUAGES | Supported programming languages |
| **Hallucination Detection** | CODEBASE_HALLUCINATION_THRESHOLD | Validation threshold |
| **Detector Mode** | CODEBASE_DETECTOR_MODE | warn or block |
| **Indexing** | CODEBASE_INCREMENTAL_INDEXING | Incremental vs full |

**CLI Flow Addition:**

```
? Enable codebase intelligence? [y/N]

Codebase RAG Indexing:
  ? Supported languages (comma-separated) [python,typescript,javascript]:
  ? Enable incremental indexing? [Y/n]
  ? Include class/function context? [Y/n]
  ? Maximum chunk size [1000]:

Hallucination Detection:
  ? Enable code hallucination detection? [Y/n]
  ? Detection threshold (0.0-1.0) [0.3]:
  ? Detection mode:
    ❯ Warn (log warnings, don't block)
      Block (reject invalid code references)

Rate Limiting:
  ? Max indexing requests per hour [10]:
  ? Index cache TTL (seconds) [86400]:
```

**Acceptance Criteria:**
- Codebase features only shown in enterprise profile or if opted in.
- Language selection with common defaults.
- Threshold explanation provided.
- Block mode warning displayed.
- Generated config includes codebase settings.

**Files to Create:**
- `cli/commands/codebase_config.py` - Codebase intelligence configuration prompts
- `cli/prompts/codebase_prompts.py` - Prompt definitions for codebase options
- `tests/cli/test_codebase_config.py` - Tests for codebase configuration flow

---

### Story 17-14: CLI Protocol Configuration

**Objective:** Add CLI prompts for A2A and MCP protocol settings.

**Features to Configure:**

| Feature | Variables | Description |
|---------|-----------|-------------|
| **A2A Enable** | A2A_ENABLED | Agent-to-agent protocol |
| **A2A Limits** | A2A_MAX_SESSIONS_*, A2A_MAX_MESSAGES_* | Resource limits |
| **A2A Persistence** | A2A_LIMITS_BACKEND | memory, redis, postgres |
| **MCP Enable** | MCP_TOOL_TIMEOUT_SECONDS | Tool execution timeout |
| **MCP-UI** | MCP_UI_ENABLED, MCP_UI_ALLOWED_ORIGINS | Iframe rendering |

**CLI Flow Addition:**

```
? Configure protocol settings? [Y/n]

A2A (Agent-to-Agent) Protocol:
  ? Enable A2A collaboration? [Y/n]
  ? Max sessions per tenant [100]:
  ? Max messages per session [1000]:
  ? Session TTL (hours) [24]:
  ? Message rate limit (per minute) [60]:
  ? Persistence backend:
    ❯ Memory (fast, non-persistent)
      Redis (persistent, recommended)
      Postgres (fully persistent)

MCP (Model Context Protocol):
  ? Default tool timeout (seconds) [30]:
  ? Max tool timeout (seconds) [300]:
  ? Enable MCP-UI iframe rendering? [y/N]
  (If yes, prompt for allowed origins)
```

**Acceptance Criteria:**
- Protocol settings shown based on profile.
- Resource limits have sensible defaults.
- Persistence backend options explained.
- MCP-UI security implications noted.
- Generated config includes protocol settings.

**Files to Create:**
- `cli/commands/protocol_config.py` - Protocol configuration prompts
- `cli/prompts/protocol_prompts.py` - Prompt definitions for A2A/MCP options
- `tests/cli/test_protocol_config.py` - Tests for protocol configuration flow

---

### Story 17-15: CLI Doctor/Validate Command

**Objective:** Provide a diagnostic command that validates configuration and checks system health.

**Problem:** Users will misconfigure settings. Without a validation tool, debugging is difficult and support requests increase.

**Command Interface:**

```bash
# Validate configuration and system health
$ rag-cli doctor

╔══════════════════════════════════════════════════════════════╗
║                    RAG SYSTEM DIAGNOSTICS                     ║
╚══════════════════════════════════════════════════════════════╝

Configuration Validation:
  ✓ Profile: standard (valid)
  ✓ LLM Provider: openai (configured)
  ✓ Embedding Provider: openai (configured)
  ✓ Database URL: valid PostgreSQL connection string
  ✓ Neo4j URI: valid bolt:// connection string
  ✓ Redis URL: valid redis:// connection string
  ⚠ RERANKER_ENABLED=true but RERANKER_PROVIDER not set (defaulting to flashrank)
  ✗ TAVILY_API_KEY not set but GRADER_FALLBACK_ENABLED=true

Service Connectivity:
  ✓ PostgreSQL: connected (pgvector extension installed)
  ✓ Neo4j: connected (v5.15.0)
  ✓ Redis: connected (v7.2.4)
  ✗ Backend: not running (expected at http://localhost:8000)
  ✗ Frontend: not running (expected at http://localhost:3000)

API Key Validation:
  ✓ OPENAI_API_KEY: valid format (sk-...)
  ✓ OPENAI_API_KEY: rate limit check passed
  ⚠ ANTHROPIC_API_KEY: not configured (optional)

Feature Availability:
  ✓ Reranking: ready (flashrank)
  ✓ Contextual Retrieval: disabled
  ✗ CRAG Grader: missing Tavily API key for web fallback
  ✓ Memory Scopes: enabled
  ✗ Voice I/O: Whisper model not downloaded

Recommendations:
  1. Set TAVILY_API_KEY for CRAG web fallback
  2. Run 'docker compose up -d' to start services
  3. Run 'rag-cli download-models' to install Whisper

╔══════════════════════════════════════════════════════════════╗
║  STATUS: 2 errors, 2 warnings                                 ║
║  Run 'rag-cli doctor --fix' to auto-fix correctable issues    ║
╚══════════════════════════════════════════════════════════════╝
```

**Validation Categories:**

| Category | Checks | Auto-Fix |
|----------|--------|----------|
| **Config Syntax** | YAML/env file parsing, schema validation | No |
| **Required Variables** | API keys, database URLs, required settings | Prompt user |
| **Variable Formats** | API key prefixes, URL formats, value ranges | Suggest correction |
| **Service Connectivity** | PostgreSQL, Neo4j, Redis, Backend, Frontend | No |
| **API Key Validity** | Format check, optional rate limit test | No |
| **Feature Dependencies** | Cross-feature requirements (CRAG needs Tavily) | Warn only |
| **Model Availability** | Local models downloaded (Whisper, FlashRank) | Offer download |

**Command Options:**

```bash
rag-cli doctor                    # Full diagnostics
rag-cli doctor --quick            # Config validation only (no service checks)
rag-cli doctor --fix              # Attempt to auto-fix issues
rag-cli doctor --json             # Output as JSON for CI/CD
rag-cli doctor --service backend  # Check specific service only
```

**Acceptance Criteria:**
- `rag-cli doctor` runs full system diagnostics.
- Clear status indicators (✓ ✗ ⚠) for each check.
- Actionable recommendations for each issue.
- `--quick` mode completes in under 5 seconds.
- `--json` output parseable by CI/CD systems.
- `--fix` mode prompts before making changes.
- Exit code 0 for healthy, 1 for warnings, 2 for errors.

---

### Story 17-16: CLI Interactive Setup Mode

**Objective:** Provide a guided, interactive setup experience beyond the basic `rag-install` flow.

**Problem:** Story 17-1 defines `rag-install` but advanced users need deeper configuration options without editing files manually.

**Command Interface:**

```bash
# Start interactive setup wizard
$ rag-cli setup

╔══════════════════════════════════════════════════════════════╗
║                    RAG SETUP WIZARD                           ║
╚══════════════════════════════════════════════════════════════╝

Welcome! This wizard will guide you through complete configuration.

? What would you like to configure?
  ❯ Full Setup (all options)
    Quick Start (profile-based defaults)
    Provider Configuration
    Retrieval Features
    Ingestion Sources
    Memory & Graph
    Observability
    Protocols
    Voice I/O

? Select configuration profile:
  ❯ Minimal - Development/testing, low resources
    Standard - Production, balanced (Recommended)
    Enterprise - All features, maximum capabilities
    Custom - Start from scratch

[Proceeds through category-specific prompts based on selection]

? Review configuration changes?

╔══════════════════════════════════════════════════════════════╗
║  Configuration Changes                                        ║
╠══════════════════════════════════════════════════════════════╣
║  + LLM_PROVIDER=openai                                        ║
║  + EMBEDDING_PROVIDER=openai                                  ║
║  ~ RERANKER_ENABLED=true (was: false)                         ║
║  + TAVILY_API_KEY=tvly-...                                    ║
╚══════════════════════════════════════════════════════════════╝

? Apply these changes? [Y/n]
? Backup existing configuration? [Y/n]

✓ Configuration saved to .env
✓ Backup saved to .env.bak.2026-01-11T12-30-00

? Start services now? [Y/n]
```

**Setup Modes:**

| Mode | Description | Target User |
|------|-------------|-------------|
| **Full Setup** | All configuration categories | First-time users |
| **Quick Start** | Profile-based with minimal prompts | Experienced users |
| **Category-Specific** | Configure single area | Updating specific feature |
| **Reconfigure** | Modify existing configuration | Changing settings |

**Acceptance Criteria:**
- `rag-cli setup` starts interactive wizard.
- Category selection allows targeted configuration.
- Configuration diff shown before applying.
- Backup offered before overwriting.
- Non-interactive: `rag-cli setup --category providers --llm anthropic --yes`
- Navigation: Can go back, skip, or cancel at any prompt.

---

### Story 17-17: Profile Migration Script

**Objective:** Migrate existing flat .env configurations to profile-based YAML format.

**Problem:** Users with existing .env files need a clear migration path to the new profile-based system.

**Command Interface:**

```bash
# Analyze current configuration
$ rag-cli migrate analyze

╔══════════════════════════════════════════════════════════════╗
║                    CONFIGURATION ANALYSIS                     ║
╚══════════════════════════════════════════════════════════════╝

Analyzing .env file...

Detected Settings:
  LLM Provider:        openai (gpt-4o)
  Embedding Provider:  openai (text-embedding-3-small)
  Retrieval:           hybrid + reranking (flashrank)
  Memory:              scopes enabled (session)
  Graph:               community detection disabled
  Ingestion:           crawl4ai (thorough profile)
  Voice:               disabled
  Observability:       prometheus enabled
  Protocols:           a2a + mcp enabled

Closest Profile Match: STANDARD (87% match)

Differences from Standard Profile:
  + RERANKER_ENABLED=true (enterprise feature)
  + MEMORY_SCOPES_ENABLED=true (enterprise feature)
  - CONTEXTUAL_RETRIEVAL_ENABLED=false (standard default)

Recommendation: Start with 'standard' profile and add overrides.

$ rag-cli migrate execute

? Select base profile:
  ❯ Standard (87% match - recommended)
    Enterprise (92% match)
    Minimal (45% match)
    Custom (keep all as overrides)

Generating configuration...
  ✓ Created config/profiles/custom.yaml with 12 overrides
  ✓ Updated .env with CONFIG_PROFILE=standard
  ✓ Backed up original .env to .env.pre-migration

Migration complete! Your custom settings are preserved in:
  config/profiles/custom.yaml

The system will now use:
  1. Standard profile as base
  2. Custom profile overrides
  3. Environment variable overrides (highest priority)
```

**Migration Strategy:**

```yaml
# Generated config/profiles/custom.yaml
# Migrated from .env on 2026-01-11
# Base profile: standard
# Override count: 12

# --- Overrides from original .env ---

retrieval:
  reranker:
    enabled: true       # Was: RERANKER_ENABLED=true
    provider: flashrank # Was: RERANKER_PROVIDER=flashrank

memory:
  scopes_enabled: true  # Was: MEMORY_SCOPES_ENABLED=true

# Note: These settings override the standard profile defaults.
# To remove an override, delete the key (inherit from standard).
```

**Acceptance Criteria:**
- `rag-cli migrate analyze` shows current config analysis.
- Profile matching algorithm suggests best base profile.
- `rag-cli migrate execute` creates profile with overrides.
- Original .env backed up before modification.
- Generated YAML includes comments explaining origin.
- Backward compatibility: system works with both .env-only and profile modes.

---

### Story 17-18: CLI Upgrade/Self-Update Command

**Objective:** Provide a mechanism for users to upgrade their RAG installation when new versions are released.

**Problem:** Users need to know when updates are available and how to apply them safely.

**Command Interface:**

```bash
# Check for updates
$ rag-cli update check

╔══════════════════════════════════════════════════════════════╗
║                    UPDATE AVAILABLE                           ║
╚══════════════════════════════════════════════════════════════╝

Current Version: 1.2.0
Latest Version:  1.3.0

Changes in 1.3.0:
  NEW: ColBERT reranking support
  NEW: Cross-language query translation
  FIX: Memory scope inheritance bug
  BREAKING: GRADER_MODEL renamed to GRADER_PROVIDER

Configuration Impact:
  ⚠ GRADER_MODEL detected in your config
    → Will be renamed to GRADER_PROVIDER automatically

? View full changelog? [y/N]
? Proceed with update? [Y/n]

# Apply update
$ rag-cli update apply

Updating RAG installation...
  ✓ Backed up current configuration
  ✓ Downloaded update (45MB)
  ✓ Applied configuration migration
  ✓ Pulled updated Docker images
  ✓ Ran database migrations

Update complete! 1.2.0 → 1.3.0

Post-Update Actions:
  1. Review migrated config: cat .env.migrated.diff
  2. Restart services: docker compose up -d
  3. Verify health: rag-cli doctor
```

**Update Flow:**

1. **Version Check:** Compare local version with latest release.
2. **Impact Analysis:** Identify breaking changes affecting user's config.
3. **Configuration Migration:** Apply automatic renames/transforms.
4. **Image Update:** Pull updated Docker images.
5. **Database Migration:** Run Alembic migrations if needed.
6. **Verification:** Run doctor command to validate.

**Command Options:**

```bash
rag-cli update check              # Check for updates without applying
rag-cli update apply              # Apply available update
rag-cli update apply --version 1.3.0  # Update to specific version
rag-cli update rollback           # Rollback to previous version
rag-cli update --skip-docker      # Update config only, skip Docker pulls
rag-cli update --dry-run          # Show what would change without applying
```

**Acceptance Criteria:**
- `rag-cli update check` shows available updates.
- Breaking changes highlighted with migration path.
- Configuration automatically migrated when possible.
- Backup created before any changes.
- Rollback available if update fails.
- `--dry-run` shows changes without applying.
- Works with offline installations (manual download mode).

---

### Story 17-19: Framework Template Details and Testing

**Objective:** Ensure framework starter templates (Story 17-5) are comprehensive, tested, and well-documented.

**Problem:** Story 17-5 defined template structure but lacks specific implementation details for each framework.

**Framework-Specific Requirements:**

**PydanticAI Template (`examples/pydanticai/`):**

```python
# examples/pydanticai/agent.py
"""PydanticAI agent with RAG integration via A2A and MCP."""

from pydantic_ai import Agent
from pydantic import BaseModel
from fasta2a import A2AClient
from mcp import Client as MCPClient

# Configuration
RAG_BASE_URL = "http://localhost:8000"
LLM_MODEL = "openai:gpt-4o"

# Data models for structured outputs
class SearchResult(BaseModel):
    content: str
    sources: list[str]
    confidence: float

class IngestResult(BaseModel):
    document_id: str
    chunks_created: int
    entities_extracted: int

# Initialize clients
a2a_client = A2AClient(f"{RAG_BASE_URL}/api/v1/a2a")
mcp_client = MCPClient(f"{RAG_BASE_URL}/api/v1/mcp")

# Create PydanticAI agent
rag_agent = Agent(
    LLM_MODEL,
    result_type=SearchResult,
    system_prompt="""You are a research assistant with access to a knowledge base.
    Use the search_knowledge tool to find relevant information."""
)

@rag_agent.tool
async def search_knowledge(query: str, max_results: int = 5) -> str:
    """Search the RAG knowledge base via MCP."""
    result = await mcp_client.call_tool(
        "knowledge.query",
        {"query": query, "tenant_id": "default", "limit": max_results}
    )
    return result.content

@rag_agent.tool
async def delegate_to_rag(task: str) -> str:
    """Delegate complex tasks to RAG orchestrator via A2A."""
    session = await a2a_client.create_session("default")
    response = await a2a_client.send_message(session.id, task)
    return response.content

# Example usage
async def main():
    result = await rag_agent.run("What is GraphRAG and how does it work?")
    print(f"Answer: {result.data.content}")
    print(f"Sources: {result.data.sources}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**CrewAI Template (`examples/crewai/`):**

```python
# examples/crewai/crew.py
"""CrewAI crew with RAG knowledge base integration."""

from crewai import Agent, Crew, Task, Process
from crewai.tools import tool

# RAG Configuration
RAG_BASE_URL = "http://localhost:8000"

# Define RAG-connected agent
researcher = Agent(
    role="Senior Researcher",
    goal="Find accurate, comprehensive information from the knowledge base",
    backstory="""You are an expert researcher with access to a powerful
    knowledge graph and vector database. You excel at finding relevant
    information and synthesizing it into clear insights.""",
    verbose=True,
    allow_delegation=True,
    a2a_agents=[{
        "url": f"{RAG_BASE_URL}/api/v1/a2a",
        "name": "rag_knowledge_base",
        "description": "Search and query the knowledge graph and vector store"
    }]
)

synthesizer = Agent(
    role="Research Synthesizer",
    goal="Combine findings into coherent, actionable summaries",
    backstory="""You are skilled at taking research findings and distilling
    them into clear, actionable insights.""",
    verbose=True
)

# Define tasks
research_task = Task(
    description="""Research the following topic using the RAG knowledge base:
    {topic}

    Find at least 5 relevant sources and summarize key findings.""",
    expected_output="Research findings with source citations",
    agent=researcher
)

synthesis_task = Task(
    description="""Synthesize the research findings into a comprehensive report.
    Include key insights, recommendations, and areas for further research.""",
    expected_output="Synthesized report with recommendations",
    agent=synthesizer,
    context=[research_task]
)

# Create crew
rag_crew = Crew(
    agents=[researcher, synthesizer],
    tasks=[research_task, synthesis_task],
    process=Process.sequential,
    verbose=True
)

# Example usage
def main():
    result = rag_crew.kickoff(inputs={"topic": "temporal knowledge graphs"})
    print(result)

if __name__ == "__main__":
    main()
```

**LangGraph Template (`examples/langgraph/`):**

```python
# examples/langgraph/graph.py
"""LangGraph workflow with RAG tool integration via MCP."""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters import MCPToolkit
from typing import TypedDict, Annotated, Sequence
import operator

# RAG Configuration
RAG_BASE_URL = "http://localhost:8000"

# Initialize MCP toolkit
mcp_toolkit = MCPToolkit(
    server_url=f"{RAG_BASE_URL}/api/v1/mcp",
    tool_names=["knowledge.query", "knowledge.graph_stats", "ingest_url"]
)

# State definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    context: str
    sources: list[str]

# LLM with tools
llm = ChatOpenAI(model="gpt-4o").bind_tools(mcp_toolkit.get_tools())

# Node definitions
async def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve relevant context from RAG."""
    query = state["messages"][-1].content
    result = await mcp_toolkit.call_tool(
        "knowledge.query",
        {"query": query, "tenant_id": "default"}
    )
    return {
        "context": result["content"],
        "sources": result.get("sources", [])
    }

async def generate_response(state: AgentState) -> AgentState:
    """Generate response using retrieved context."""
    messages = state["messages"]
    context = state["context"]

    system_message = f"""Answer based on this context:
    {context}

    Always cite sources when available."""

    response = await llm.ainvoke([
        {"role": "system", "content": system_message},
        *messages
    ])

    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Determine if we need more retrieval."""
    last_message = state["messages"][-1]
    if "need more information" in last_message.content.lower():
        return "retrieve"
    return END

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("generate", generate_response)
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges("generate", should_continue, {
    "retrieve": "retrieve",
    END: END
})
workflow.set_entry_point("retrieve")

# Compile
app = workflow.compile()

# Example usage
async def main():
    result = await app.ainvoke({
        "messages": [HumanMessage(content="Explain the Graphiti architecture")],
        "context": "",
        "sources": []
    })
    print(result["messages"][-1].content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Anthropic SDK Template (`examples/anthropic/`):**

```python
# examples/anthropic/agent.py
"""Anthropic Claude agent with RAG MCP tools."""

from anthropic import Anthropic
import httpx
import json

# RAG Configuration
RAG_BASE_URL = "http://localhost:8000"

# Initialize Anthropic client
client = Anthropic()

# Define MCP tools for Claude
tools = [
    {
        "name": "search_knowledge",
        "description": "Search the RAG knowledge base for relevant information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "ingest_url",
        "description": "Crawl and ingest a URL into the knowledge base",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to ingest"
                }
            },
            "required": ["url"]
        }
    }
]

async def call_mcp_tool(tool_name: str, params: dict) -> str:
    """Execute MCP tool on RAG backend."""
    async with httpx.AsyncClient() as http:
        response = await http.post(
            f"{RAG_BASE_URL}/api/v1/mcp/call",
            json={"tool": tool_name, "params": params}
        )
        return response.json()["result"]

async def run_agent(user_message: str) -> str:
    """Run Claude agent with RAG tool access."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        # Check if Claude wants to use a tool
        if response.stop_reason == "tool_use":
            tool_use = next(
                block for block in response.content
                if block.type == "tool_use"
            )

            # Execute tool via MCP
            tool_result = await call_mcp_tool(
                tool_use.name,
                tool_use.input
            )

            # Add assistant response and tool result
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": tool_result
                }]
            })
        else:
            # No more tool calls, return final response
            return response.content[0].text

# Example usage
async def main():
    result = await run_agent(
        "Search the knowledge base for information about RAG architectures"
    )
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Template Testing Requirements:**

| Template | Test Scope | CI Integration |
|----------|------------|----------------|
| PydanticAI | Unit tests + integration with mock RAG | GitHub Actions |
| CrewAI | Unit tests + integration with mock RAG | GitHub Actions |
| LangGraph | Unit tests + integration with mock RAG | GitHub Actions |
| Anthropic | Unit tests + integration with mock RAG | GitHub Actions |

**Acceptance Criteria:**
- Each template is a complete, runnable example.
- Templates include comprehensive README.md with setup instructions.
- Templates demonstrate both A2A and MCP connection patterns.
- Templates include unit tests that mock RAG backend.
- Templates tested against actual RAG in CI (weekly job).
- Templates versioned and updated with each RAG release.
- `rag-cli generate-example --framework <name>` creates local copy.

---

## Technical Notes

### CLI Technology Stack

- **Framework:** Python + Typer (CLI) + Rich (TUI)
- **Packaging:** Distributed via `pipx install rag-install` or included in repo
- **Config Schema:** Pydantic models for validation

### Installation Methods

```bash
# Option 1: pipx (recommended)
pipx install rag-install

# Option 2: From source
cd backend && uv run rag-install

# Option 3: Docker
docker run -it agentic-rag/installer
```

### Non-Interactive Mode

```bash
# Full automation for CI/CD
rag-install \
  --profile enterprise \
  --llm openrouter \
  --embedding voyage \
  --framework pydanticai \
  --with-skills \
  --enable-reranking \
  --yes

# Validate existing .env
rag-install validate

# Upgrade configuration
rag-install upgrade --from 1.0 --to 2.0
```

### Framework Integration Architecture (Vision A)

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG PLATFORM (Agno inside)                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Agno Orchestrator                        │   │
│  │    - Multi-step planning                                 │   │
│  │    - Tool selection                                      │   │
│  │    - Trajectory logging                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│              ┌───────────────┼───────────────┐                 │
│              ▼               ▼               ▼                 │
│         ┌────────┐      ┌────────┐      ┌────────┐            │
│         │  A2A   │      │  MCP   │      │ AG-UI  │            │
│         └────────┘      └────────┘      └────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
       ════════════════════════════════════════════
       DEVELOPER'S AGENTS (Connect via protocols)
       ════════════════════════════════════════════
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  PydanticAI   │     │    CrewAI     │     │   LangGraph   │
│  (Native A2A) │     │ (Native A2A)  │     │ (Native A2A)  │
│  (Native MCP) │     │ (Native MCP)  │     │ (Native MCP)  │
└───────────────┘     └───────────────┘     └───────────────┘
```

## Dependencies

- Provider selection options from Epic 11 (multi-provider).
- Retrieval features from Epic 12 (advanced retrieval).
- Ingestion options from Epic 13 (enterprise ingestion).
- A2A/MCP protocols from Epic 14 (connectivity).
- Docker Compose definitions in repository.

## Risks

- Hardware detection variability across OSes.
  - *Mitigation:* Test on Linux, macOS (Intel + Apple Silicon), Windows WSL2.
- CLI paths may diverge from manual setup.
  - *Mitigation:* CLI generates same .env as documentation describes.
- API key validation may hit rate limits.
  - *Mitigation:* Only validate format, not actual API call.
- Agent Skills schema may evolve.
  - *Mitigation:* Version-pin skills schema, provide migration path.
- Framework protocol support may change.
  - *Mitigation:* Link to official docs, test quarterly.

## Success Metrics

- New users reach a running stack in under 15 minutes.
- CLI setup success rate above 90% in internal testing.
- Zero manual .env editing required for standard profile.
- Error messages lead to successful resolution 80%+ of the time.
- Framework starter templates work out-of-the-box.
- Agent Skills discoverable from Claude Desktop within 5 minutes.

## References

- `docs/roadmap-decisions-2026-01-03.md` - CLI-last decision rationale
- [Anthropic Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [PydanticAI A2A Documentation](https://ai.pydantic.dev/a2a/)
- [CrewAI A2A Documentation](https://docs.crewai.com/en/learn/a2a-agent-delegation)
- [LangGraph MCP Documentation](https://docs.langchain.com/oss/python/langchain/mcp)
- `_bmad-output/prd.md`
- `_bmad-output/architecture.md`
- `_bmad-output/project-planning-artifacts/epics.md`
- `docs/recommendations_2025.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
