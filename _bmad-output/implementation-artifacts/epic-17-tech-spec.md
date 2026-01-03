# Epic 17 Tech Spec: Deployment & CLI (Final Polish)

**Date:** 2025-12-31
**Updated:** 2026-01-04 (Comprehensive CLI Design)
**Status:** Backlog
**Epic Owner:** Product and Engineering

---

## Overview

Epic 17 delivers a polished installation and deployment experience with an interactive CLI. The CLI guides users through provider selection, hardware detection, environment configuration, and startup verification.

### Key Decision (2026-01-03)

**CLI is LAST because it must know all available options.**

The CLI offers choices for:
- LLM providers (Epic 11)
- Embedding providers (Epic 11)
- Agent frameworks (Epic 16)
- Ingestion sources (Epic 13)
- Retrieval features (Epic 12)

**Decision Document:** `docs/roadmap-decisions-2026-01-03.md`

### Goals

- Provide a `rag-install` interactive CLI for guided setup.
- Detect hardware capabilities and recommend appropriate defaults.
- Auto-generate validated `.env` and verify docker compose startup.
- **Enable first response in under 15 minutes.**

### Scope

**In scope**
- Interactive CLI with prompts for all configurable options.
- Hardware detection (CPU, GPU, RAM).
- Environment file generation with validation.
- Docker Compose startup verification with health checks.
- Profile-based configuration (minimal, standard, enterprise).

**Out of scope**
- Production orchestration (Kubernetes, Helm charts).
- Cloud-specific deployment (AWS, GCP, Azure).

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

? Select agent framework:
  ❯ Agno (Default, battle-tested)
    PydanticAI (type-safe)
    CrewAI (multi-agent)
    LangGraph (stateful workflows)
    Anthropic Agent SDK (Claude-native)

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
╚══════════════════════════════════════════════════════════════╝
```

**Configuration Options:**

| Category | Options | Default |
|----------|---------|---------|
| **LLM Provider** | openai, anthropic, gemini, openrouter, ollama | openai |
| **Embedding Provider** | openai, voyage, gemini, ollama | openai |
| **Agent Framework** | agno, pydanticai, crewai, langgraph, anthropic | agno |
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
# Generated by rag-install on 2026-01-04
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

# ─── Agent Framework ────────────────────────────────────────────
AGENT_FRAMEWORK=agno
# Available: agno, pydanticai, crewai, langgraph, anthropic

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
  --framework agno \
  --enable-reranking \
  --yes

# Validate existing .env
rag-install validate

# Upgrade configuration
rag-install upgrade --from 1.0 --to 2.0
```

## Dependencies

- Provider selection options from Epic 11 (multi-provider).
- Framework options from Epic 16 (framework agnosticism).
- Retrieval features from Epic 12 (advanced retrieval).
- Ingestion options from Epic 13 (enterprise ingestion).
- Docker Compose definitions in repository.

## Risks

- Hardware detection variability across OSes.
  - *Mitigation:* Test on Linux, macOS (Intel + Apple Silicon), Windows WSL2.
- CLI paths may diverge from manual setup.
  - *Mitigation:* CLI generates same .env as documentation describes.
- API key validation may hit rate limits.
  - *Mitigation:* Only validate format, not actual API call.

## Success Metrics

- New users reach a running stack in under 15 minutes.
- CLI setup success rate above 90% in internal testing.
- Zero manual .env editing required for standard profile.
- Error messages lead to successful resolution 80%+ of the time.

## References

- `docs/roadmap-decisions-2026-01-03.md` - CLI-last decision rationale
- `_bmad-output/prd.md`
- `_bmad-output/architecture.md`
- `_bmad-output/project-planning-artifacts/epics.md`
- `docs/recommendations_2025.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
