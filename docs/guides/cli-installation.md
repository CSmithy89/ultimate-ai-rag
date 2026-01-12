# CLI Installation Manual

This guide provides comprehensive documentation for the Agentic RAG + GraphRAG CLI tools, including installation methods, command references, and troubleshooting.

## Table of Contents

- [Installation](#installation)
  - [Using pipx (Recommended)](#using-pipx-recommended)
  - [From Source](#from-source)
  - [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Commands Reference](#commands-reference)
  - [rag-install](#rag-install)
  - [rag-cli setup](#rag-cli-setup)
  - [rag-cli doctor](#rag-cli-doctor)
  - [rag-cli migrate](#rag-cli-migrate)
  - [rag-cli update](#rag-cli-update)
- [Configuration Options](#configuration-options)
  - [Profiles](#profiles)
  - [LLM Providers](#llm-providers)
  - [Frameworks](#frameworks)
  - [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

Before installing the CLI, ensure you have the following:

- **Python 3.11+** - Required for the backend and CLI tools
- **Docker Desktop** or Docker Engine - Required for running services
- **Git** - For cloning the repository and updates
- At least **8GB RAM** (16GB+ recommended for standard profile)

### Using pipx (Recommended)

The recommended way to install the CLI is using `pipx`, which installs Python applications in isolated environments:

```bash
# Install pipx if not already installed
python -m pip install --user pipx
python -m pipx ensurepath

# Install the Agentic RAG CLI
pipx install agentic-rag-backend
```

After installation, the following commands will be available:
- `rag-install` - Main installation wizard
- `rag-cli` - CLI utility commands

### From Source

To install from source for development or customization:

```bash
# Clone the repository
git clone https://github.com/your-org/agentic-rag-graphrag.git
cd agentic-rag-graphrag

# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd backend && uv sync

# Run CLI commands via uv
uv run python -m cli.main rag-install
```

Alternatively, you can install in development mode:

```bash
cd backend
uv pip install -e .
```

---

## Quick Start

Get up and running in under 5 minutes:

```bash
# 1. Run the installation wizard (interactive mode)
rag-install

# 2. The wizard will:
#    - Detect your hardware (CPU, RAM, GPU)
#    - Recommend an appropriate profile
#    - Prompt for LLM provider and API key
#    - Generate .env configuration
#    - Start Docker services
#    - Verify service health

# 3. Access your RAG system
#    Frontend: http://localhost:3000
#    API Docs: http://localhost:8000/docs
```

For non-interactive installation (CI/CD pipelines):

```bash
rag-install --profile standard --llm openai --api-key sk-xxx --yes
```

---

## Commands Reference

### rag-install

The main installation wizard that configures and deploys the RAG system.

**Usage:**
```bash
rag-install [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--profile <name>` | Deployment profile: `minimal`, `standard`, `enterprise` | Auto-detected |
| `--llm <provider>` | LLM provider: `openai`, `anthropic`, `openrouter`, `ollama`, `gemini` | Interactive |
| `--api-key <key>` | API key for the selected LLM provider | Interactive |
| `--framework <name>` | Generate framework starter: `none`, `pydanticai`, `crewai`, `langgraph`, `anthropic` | `none` |
| `--customize` | Enable advanced customization prompts | `false` |
| `--yes` | Non-interactive mode (requires `--profile` and `--llm`) | `false` |
| `--dry-run` | Preview configuration without starting services | `false` |
| `--with-skills` | Generate skill templates in `.skills/` directory | `false` |

**Examples:**

```bash
# Interactive installation with hardware detection
rag-install

# Quick install with OpenAI
rag-install --profile standard --llm openai --api-key sk-xxx --yes

# Install with Anthropic and PydanticAI framework starter
rag-install --llm anthropic --api-key sk-ant-xxx --framework pydanticai

# Dry run to preview configuration
rag-install --profile enterprise --dry-run

# Full customization with advanced options
rag-install --customize

# Non-interactive with skill templates
rag-install --profile standard --llm openai --api-key sk-xxx --yes --with-skills
```

**Interactive Flow:**

1. **Hardware Detection** - Automatically detects CPU cores, RAM, and GPU
2. **Profile Recommendation** - Suggests profile based on hardware:
   - `minimal` - RAM < 16GB
   - `standard` - RAM 16-32GB
   - `enterprise` - RAM >= 32GB
3. **LLM Provider Selection** - Choose your AI provider
4. **API Key Entry** - Secure password input for API keys
5. **Framework Starter** - Optional agent framework template
6. **Proceed Options**:
   - `y` - Proceed with installation
   - `n` - Cancel installation
   - `c` - Customize advanced options

**Customization Options** (when using `--customize` or selecting `c`):

- Embedding provider selection (openai, voyage, gemini, ollama)
- Cross-encoder reranking toggle
- Contextual retrieval toggle
- Speech-to-text (Whisper) toggle
- Text-to-speech toggle

---

### rag-cli setup

Configure advanced features after initial installation.

**Usage:**
```bash
rag-cli setup [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--category <name>` | Setup category to configure | Interactive |
| `--profile <name>` | Base profile to use | `standard` |
| `--yes` | Accept all defaults non-interactively | `false` |

**Categories:**

- `all` - Configure all categories
- `ingestion` - Web crawling and document ingestion settings
- `memory-graph` - Memory scopes, consolidation, and graph intelligence
- `voice` - Voice I/O configuration (Whisper, TTS)
- `observability` - Prometheus metrics and debugging
- `codebase` - Codebase indexing and hallucination detection
- `protocols` - A2A and MCP protocol settings

**Examples:**

```bash
# Interactive setup for all categories
rag-cli setup

# Configure only ingestion settings
rag-cli setup --category ingestion

# Setup voice features with enterprise base profile
rag-cli setup --category voice --profile enterprise

# Non-interactive setup with defaults
rag-cli setup --category all --yes
```

**Category Options:**

**Ingestion:**
- Crawl profile: `fast`, `thorough`, `stealth`
- Crawl fallback toggle
- PDF ingestion toggle
- YouTube ingestion toggle
- Codebase ingestion (enterprise only)
- External sync (enterprise only)

**Memory & Graph:**
- Memory scopes toggle
- Default scope: `session`, `user`, `agent`
- Memory consolidation toggle
- Community detection toggle
- LazyRAG toggle
- Query routing toggle
- Graph reranker toggle

**Voice:**
- Voice I/O toggle
- Whisper model: `tiny`, `base`, `small`, `medium`, `large`
- TTS provider: `openai`, `elevenlabs`, `pyttsx3`
- OpenAI TTS voice: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

**Observability:**
- Prometheus metrics toggle
- Cost tracking toggle
- Trajectory debugging toggle

**Protocols:**
- A2A protocol toggle
- Max sessions per tenant
- Max messages per session
- MCP protocol toggle

---

### rag-cli doctor

Diagnostic command to validate configuration and service health.

**Usage:**
```bash
rag-cli doctor [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--quick` | Skip service health checks | `false` |
| `--json` | Output results as JSON | `false` |
| `--service <name>` | Check specific service: `backend`, `frontend` | All services |
| `--fix` | Attempt to fix common issues | `false` |

**Examples:**

```bash
# Full diagnostic check
rag-cli doctor

# Quick configuration check (no service health)
rag-cli doctor --quick

# JSON output for scripting
rag-cli doctor --json

# Check only backend service
rag-cli doctor --service backend

# Attempt automatic fixes
rag-cli doctor --fix
```

**Checks Performed:**

1. **Environment File** - Validates `.env` exists and is readable
2. **Profile Configuration** - Validates profile YAML exists
3. **Backend Health** - HTTP check to `http://localhost:8000/health`
4. **Frontend Health** - HTTP check to `http://localhost:3000`

**Auto-Fix Capabilities:**

When `--fix` is specified:
- Creates `.env` from `.env.example` if missing

**JSON Output Format:**

```json
{
  "status": "ok",
  "checks": [
    {"check": "env", "status": "ok"},
    {"check": "profile", "status": "ok", "profile": "standard"},
    {"check": "backend", "status": "ok"},
    {"check": "frontend", "status": "ok"}
  ]
}
```

**Exit Codes:**
- `0` - All checks passed
- `1` - One or more checks failed

---

### rag-cli migrate

Analyze and migrate environment configuration to profile format.

**Usage:**
```bash
# Analyze current configuration
rag-cli migrate analyze [OPTIONS]

# Execute migration
rag-cli migrate execute [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--profile <name>` | Base profile for comparison | `standard` |

**Examples:**

```bash
# Analyze differences from standard profile
rag-cli migrate analyze

# Analyze against enterprise profile
rag-cli migrate analyze --profile enterprise

# Create custom.yaml with detected overrides
rag-cli migrate execute

# Migrate based on minimal profile
rag-cli migrate execute --profile minimal
```

**How It Works:**

1. **Analyze** - Compares `.env` values against the base profile YAML
2. **Detect Overrides** - Identifies environment variables that differ from profile defaults
3. **Execute** - Writes detected overrides to `config/profiles/custom.yaml`

**Mapped Environment Variables:**

| Environment Variable | Profile Path |
|---------------------|--------------|
| `LLM_PROVIDER` | `llm.provider` |
| `LLM_MODEL_ID` | `llm.model` |
| `EMBEDDING_PROVIDER` | `embedding.provider` |
| `EMBEDDING_MODEL` | `embedding.model` |
| `RERANKER_ENABLED` | `retrieval.reranker.enabled` |
| `CONTEXTUAL_RETRIEVAL_ENABLED` | `retrieval.contextual_retrieval.enabled` |
| `MEMORY_SCOPES_ENABLED` | `memory.scopes_enabled` |
| `VOICE_IO_ENABLED` | `voice.enabled` |
| `PROMETHEUS_ENABLED` | `observability.prometheus_enabled` |
| `A2A_ENABLED` | `protocols.a2a.enabled` |

---

### rag-cli update

Check for and apply updates from the Git repository.

**Usage:**
```bash
# Check for available updates
rag-cli update check

# Apply available updates
rag-cli update apply
```

**Examples:**

```bash
# Check if updates are available
rag-cli update check

# Apply updates (requires clean working tree)
rag-cli update apply
```

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_CLI_UPDATE_NO_FETCH` | Skip git fetch (use local refs) | `false` |
| `RAG_CLI_UPDATE_DRY_RUN` | Preview update without applying | `false` |

**Update Process:**

1. **Check** - Fetches from origin and compares commit counts
2. **Apply** - Performs `git pull --rebase origin main`

**Requirements:**
- Must be run from a git checkout (not pip-installed)
- Working tree must be clean (no uncommitted changes)

---

## Configuration Options

### Profiles

Three pre-configured profiles optimize settings for different deployment scenarios:

#### Minimal Profile

**Target:** Development, testing, resource-constrained environments

| Setting | Value |
|---------|-------|
| LLM Model | gpt-4o-mini |
| Embedding | text-embedding-3-small |
| Retrieval Strategy | vector |
| Reranking | Disabled |
| Memory Scopes | Disabled |
| Voice I/O | Disabled |
| Prometheus | Disabled |
| A2A Max Sessions | 10 |

#### Standard Profile

**Target:** Production deployments, small-medium teams

| Setting | Value |
|---------|-------|
| LLM Model | gpt-4o |
| Embedding | text-embedding-3-small |
| Retrieval Strategy | hybrid |
| Reranking | FlashRank |
| Memory Scopes | Enabled |
| Query Routing | Enabled |
| Prometheus | Enabled |
| A2A Max Sessions | 100 |

#### Enterprise Profile

**Target:** Large teams, advanced retrieval, enterprise ingestion

| Setting | Value |
|---------|-------|
| LLM Model | claude-3.5-sonnet (via OpenRouter) |
| Embedding | voyage-code-3 |
| Retrieval Strategy | hybrid |
| Reranking | Cohere |
| Contextual Retrieval | Enabled |
| Memory Consolidation | Enabled |
| Community Detection | Enabled |
| LazyRAG | Enabled |
| Codebase Indexing | Enabled |
| Voice I/O | Enabled |
| A2A Max Sessions | 500 |

### LLM Providers

| Provider | API Key Prefix | Notes |
|----------|---------------|-------|
| `openai` | `sk-` | OpenAI API |
| `anthropic` | `sk-ant-` | Anthropic Claude API |
| `openrouter` | `sk-or-` | OpenRouter multi-provider gateway |
| `gemini` | (any) | Google Gemini API |
| `ollama` | (none) | Local Ollama (no API key required) |

### Frameworks

Optional framework starters generated in `examples/<framework>/`:

| Framework | Description |
|-----------|-------------|
| `none` | No framework template |
| `pydanticai` | PydanticAI agent with MCP client |
| `crewai` | CrewAI multi-agent crew |
| `langgraph` | LangGraph state machine |
| `anthropic` | Anthropic Claude agent |

### Environment Variables

**Timeout Configuration:**

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_CLI_SUBPROCESS_TIMEOUT` | Timeout for shell commands (seconds) | `5.0` |
| `RAG_CLI_DOCKER_TIMEOUT` | Timeout for Docker operations (seconds) | `300.0` |

**Update Configuration:**

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_CLI_UPDATE_NO_FETCH` | Skip git fetch during update check | `false` |
| `RAG_CLI_UPDATE_DRY_RUN` | Dry-run mode for update apply | `false` |

---

## Troubleshooting

### Docker Not Running

**Symptom:**
```
Docker daemon not running. Start Docker Desktop
```

**Solutions:**

1. **macOS/Windows:** Start Docker Desktop application
2. **Linux:** Start the Docker service:
   ```bash
   sudo systemctl start docker
   ```
3. **Verify Docker is running:**
   ```bash
   docker info
   ```

### Port Already in Use

**Symptom:**
```
Error: Port 8000 is already in use
```

**Solutions:**

1. **Find the process using the port:**
   ```bash
   # Linux/macOS
   lsof -i :8000

   # Windows
   netstat -ano | findstr :8000
   ```

2. **Stop the conflicting process or change ports:**
   ```bash
   # Stop existing containers
   docker compose down

   # Or modify docker-compose.yml to use different ports
   ```

3. **Common port conflicts:**
   - Port 8000: Backend API (may conflict with other Python services)
   - Port 3000: Frontend (may conflict with Node.js apps)
   - Port 5432: PostgreSQL
   - Port 7687: Neo4j
   - Port 6379: Redis

### Out of Memory

**Symptom:**
```
Container killed: Out of memory
```

**Solutions:**

1. **Use the minimal profile:**
   ```bash
   rag-install --profile minimal
   ```

2. **Increase Docker memory allocation:**
   - Docker Desktop > Settings > Resources > Memory
   - Recommended: 8GB minimum, 16GB for standard profile

3. **Check current memory usage:**
   ```bash
   docker stats
   ```

4. **Reduce concurrent services:**
   ```bash
   # Edit docker-compose.yml to disable unused services
   ```

### API Key Validation Failed

**Symptom:**
```
Invalid key format. Please try again.
```

**Solutions:**

1. **Verify key prefix:**
   - OpenAI: Must start with `sk-`
   - Anthropic: Must start with `sk-ant-`
   - OpenRouter: Must start with `sk-or-`

2. **Check for whitespace:**
   ```bash
   # Keys should not have leading/trailing spaces
   ```

3. **For Ollama (no key required):**
   ```bash
   rag-install --llm ollama --profile minimal
   ```

### Services Not Starting

**Symptom:**
```
Backend failed to become healthy. Check for port conflicts or docker logs.
```

**Solutions:**

1. **Check Docker logs:**
   ```bash
   docker compose logs backend
   docker compose logs frontend
   ```

2. **Verify .env file exists:**
   ```bash
   rag-cli doctor
   ```

3. **Recreate containers:**
   ```bash
   docker compose down -v
   docker compose up -d
   ```

4. **Check disk space:**
   ```bash
   df -h
   ```

### Profile Not Found

**Symptom:**
```
Profile not found: custom
```

**Solutions:**

1. **List available profiles:**
   ```bash
   ls config/profiles/
   ```

2. **Use a standard profile:**
   ```bash
   rag-cli setup --profile standard
   ```

3. **Create custom profile via migration:**
   ```bash
   rag-cli migrate execute
   ```

### Git Update Fails

**Symptom:**
```
Working tree is dirty. Commit or stash changes first.
```

**Solutions:**

1. **Stash local changes:**
   ```bash
   git stash
   rag-cli update apply
   git stash pop
   ```

2. **Discard local changes (caution):**
   ```bash
   git checkout -- .
   rag-cli update apply
   ```

3. **Check git status:**
   ```bash
   git status
   ```

### Docker Compose Timeout

**Symptom:**
```
Docker compose timed out after 300s
```

**Solutions:**

1. **Increase timeout:**
   ```bash
   export RAG_CLI_DOCKER_TIMEOUT=600
   rag-install
   ```

2. **Check network connectivity:**
   ```bash
   docker pull postgres:16
   ```

3. **Use pre-pulled images:**
   ```bash
   docker compose pull
   rag-install
   ```

---

## Getting Help

- **Documentation:** `http://localhost:8000/docs` (when running)
- **Health Check:** `rag-cli doctor`
- **GitHub Issues:** Report bugs and feature requests on GitHub
- **Configuration:** Check `.env` and `config/profiles/` for current settings
