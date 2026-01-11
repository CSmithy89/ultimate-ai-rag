# Epic 17 Tech Spec: Developer Experience, CLI & Framework Integration

**Date:** 2025-12-31
**Updated:** 2026-01-05 (Merged Epic 16 Framework Integration)
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

## Stories

### Story 17-1: Create rag-install CLI Tool

**Objective:** Build a guided CLI that walks users through setup with intelligent defaults.

**Technology:** Python with `rich` for TUI and `typer` for CLI framework.

**CLI Flow:**

```
$ rag-install

╔══════════════════════════════════════════════════════════════╗
║                    RAG SYSTEM INSTALLER                       ║
╚══════════════════════════════════════════════════════════════╝

Detecting hardware...
  ✓ CPU: 8 cores (Apple M2)
  ✓ RAM: 16 GB
  ✓ GPU: Apple Metal (MPS)

Recommended profile: STANDARD

? Select installation profile:
  ❯ Minimal   - CPU only, local models, minimal resources
    Standard  - Balanced, cloud LLMs, moderate resources (Recommended)
    Enterprise - Full features, all providers, maximum resources

? Select LLM provider:
  ❯ OpenAI (Recommended for Standard)
    Anthropic
    Google Gemini
    OpenRouter (access to 100+ models)
    Ollama (local, requires GPU)

? Select embedding provider:
  ❯ OpenAI text-embedding-3-small (Recommended)
    Voyage AI (best for code)
    Google Gemini
    Ollama (local)

? Which framework will you build your agents in?
  ❯ None (use built-in orchestrator only)
    PydanticAI (type-safe agents)
    CrewAI (multi-agent crews)
    LangGraph (stateful workflows)
    Anthropic SDK (Claude-native + Agent Skills)

? Enable advanced retrieval features?
  ☐ Cross-encoder reranking (+latency, +precision)
  ☐ Contextual retrieval (+cost during ingestion, +relevance)
  ☐ Corrective RAG grader (+fallback capability)

? Enter your API keys:
  OpenAI API Key: sk-...
  (Keys are stored in .env, never transmitted)

Generating configuration...
  ✓ Created .env
  ✓ Backed up existing .env to .env.bak
  ✓ Generated framework starter template (if selected)

? Start services now? [Y/n]

Starting docker compose...
  ✓ PostgreSQL (pgvector) - healthy
  ✓ Neo4j - healthy
  ✓ Redis - healthy
  ✓ Backend - healthy (http://localhost:8000)
  ✓ Frontend - healthy (http://localhost:3000)

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
║  4. See examples/ for framework integration                    ║
╚══════════════════════════════════════════════════════════════╝
```

**Configuration Options:**

| Category | Options | Default |
|----------|---------|---------|
| **LLM Provider** | openai, anthropic, gemini, openrouter, ollama | openai |
| **Embedding Provider** | openai, voyage, gemini, ollama | openai |
| **Framework Template** | none, pydanticai, crewai, langgraph, anthropic | none |
| **Reranking** | cohere, flashrank, disabled | disabled |
| **Database** | postgresql+neo4j (fixed) | - |
| **Profile** | minimal, standard, enterprise | standard |

**Acceptance Criteria**
- Running `rag-install` starts an interactive setup wizard with `rich` TUI.
- Users can select providers, frameworks, and optional features.
- CLI validates API keys format before proceeding.
- CLI writes configuration into `.env` with comments explaining each setting.
- Non-interactive mode supported: `rag-install --profile standard --llm openai --yes`

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
