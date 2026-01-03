---
stepsCompleted: [1, 2, 3, 4]
status: complete
inputDocuments:
  - '_bmad-output/prd.md'
  - '_bmad-output/architecture.md'
  - '_bmad-output/project-planning-artifacts/ux-design-specification.md'
---

# Agentic Rag and Graphrag with copilot - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for Agentic Rag and Graphrag with copilot, decomposing the requirements from the PRD, UX Design, and Architecture requirements into implementable stories.

## Requirements Inventory

### Functional Requirements

**Integration & Installation (Developer Experience)**
- FR1: Developers can deploy the core system as a single Docker container with zero initial code.
- FR2: Developers can configure the system using standard environment variables (API keys, database URLs).
- FR3: React developers can integrate the "Copilot" UI into their applications using a dedicated npm package.
- FR4: Python developers can extend the core agent logic using a dedicated pip package.
- FR5: Systems using any programming language can interact with the core "Brain" via a standardized AG-UI protocol over HTTP/WebSockets.

**Agentic Orchestration & Reasoning**
- FR6: The system can autonomously plan a multi-step execution strategy based on complex user queries.
- FR7: The agent can dynamically select the best retrieval method (Vector vs. Graph) for a given query.
- FR8: The agent can use external "Tools" defined via the Model Context Protocol (MCP).
- FR9: Multiple agents can collaborate and delegate tasks to each other using the A2A protocol.
- FR10: The system maintains a persistent "Thought Trace" (trajectory) for every complex interaction.

**Hybrid Retrieval (GraphRAG + Vector)**
- FR11: The system can perform semantic similarity search across unstructured text chunks (Vector RAG).
- FR12: The system can perform relationship-based traversal across structured knowledge (GraphRAG).
- FR13: The system can synthesize a single coherent answer by combining results from both Vector and Graph databases.
- FR14: The system can explain how it arrived at an answer by referencing the specific nodes and edges used from the knowledge graph.

**Ingestion & Knowledge Construction**
- FR15: Users can trigger an autonomous crawl of documentation websites using the integrated Crawl4AI tool.
- FR16: The system can parse complex document layouts (tables, headers, footnotes) from PDFs using Docling.
- FR17: An "Agentic Indexer" can autonomously extract entities and relationships from parsed text to build the knowledge graph.
- FR18: Data Engineers can visualize the current state of the knowledge graph to identify gaps or orphan nodes.

**Interactive "Copilot" Interface**
- FR19: End-users can interact with the system through a pre-built chat sidebar.
- FR20: The UI can dynamically render specialized components (e.g., Graph Visualizers) sent from the agent (Generative UI).
- FR21: Users can review and approve/reject retrieved sources before the agent generates an answer (Human-in-the-Loop).
- FR22: The agent can take "Frontend Actions" within the host application (e.g., highlighting text, opening modals).

**Operations & Observability**
- FR23: Ops Engineers can monitor the real-time cost of LLM interactions.
- FR24: The system can intelligently route queries to different LLM models based on task complexity to optimize costs.
- FR25: Developers can review the reasoning "trajectory" of past queries for debugging purposes.

### NonFunctional Requirements

**Performance**
- NFR1: Total end-to-end response time for a complex agentic query should aim for < 10 seconds (including reasoning and tool calls).
- NFR2: The ingestion pipeline should process a standard 50-page documentation site in < 5 minutes.

**Security**
- NFR3: The system must ensure strict logical isolation between knowledge stores if used in a multi-tenant environment.
- NFR4: Reasoning traces and internal "thought traces" must be encrypted at rest.

**Scalability**
- NFR5: The system must handle knowledge graphs with up to 1 million nodes/edges without significant retrieval degradation.
- NFR6: The backend must support 50+ concurrent autonomous agent runs on standard production hardware.

**Integration & Reliability**
- NFR7: The system must adhere 100% to the MCP and AG-UI specifications for future-proof interoperability.
- NFR8: The system must support stateless recovery, allowing agents to resume their reasoning state after a container restart.

### Additional Requirements

**From Architecture Document:**
- Starter Template: Agno agent-api (backend) + Next.js + CopilotKit (frontend) - Epic 1 Story 1 MUST initialize using these
- Python 3.12+ with uv package manager
- Neo4j 5.x Community for graph storage
- PostgreSQL 16.x + pgvector for vector storage
- Redis 7.x for semantic caching and message queue
- Docling 2.66.0 for document processing
- Docker Compose for development environment
- MCP protocol implementation for tool execution
- A2A protocol implementation for agent delegation
- AG-UI protocol via CopilotKit for frontend state sync
- Trajectory logging via Agno built-in patterns
- Multi-tenancy via namespace isolation in all DB queries
- RFC 7807 error response format
- TanStack Query v5 for frontend server state
- shadcn/ui for UI components
- React Flow for graph visualization

**From UX Design Document:**
- "Thought Trace" Stepper: Vertical progress indicator streaming agent's current task with expandable raw logs
- HITL Source Approval: Side-panel "Gatekeeper" with Approve/Reject toggle cards, non-blocking pattern
- Generative Graph Visualizer: Force-directed graph component rendered within chat flow
- Color Palette: Indigo-600 (primary/intelligence), Emerald-500 (success/validated), Slate (neutrals), Amber-400 (HITL attention)
- Typography: Inter for headings/body, JetBrains Mono for traces and code
- Experience Principles: Radical Transparency, Frictionless Guardrails, Standardized Aesthetics

### FR Coverage Map

| FR | Epic | Description |
|----|------|-------------|
| FR1 | Epic 1 | Docker container deployment |
| FR2 | Epic 1 | Environment variable configuration |
| FR3 | Epic 6 | npm package for React integration |
| FR4 | Epic 7 | pip package for Python extension |
| FR5 | Epic 7 | AG-UI protocol for any language |
| FR6 | Epic 2 | Multi-step execution planning |
| FR7 | Epic 2 | Dynamic retrieval method selection |
| FR8 | Epic 7 | MCP external tools |
| FR9 | Epic 7 | A2A agent collaboration |
| FR10 | Epic 2 | Persistent trajectory/thought trace |
| FR11 | Epic 3 | Vector semantic search |
| FR12 | Epic 3 | Graph relationship traversal |
| FR13 | Epic 3 | Hybrid answer synthesis |
| FR14 | Epic 3 | Graph-based explainability |
| FR15 | Epic 4 | Crawl4AI documentation crawling |
| FR16 | Epic 4 | Docling PDF parsing |
| FR17 | Epic 4 | Agentic entity extraction |
| FR18 | Epic 4 | Knowledge graph visualization |
| FR19 | Epic 6 | Chat sidebar interface |
| FR20 | Epic 6 | Generative UI components |
| FR21 | Epic 6 | Human-in-the-Loop validation |
| FR22 | Epic 6 | Frontend actions |
| FR23 | Epic 8 | LLM cost monitoring |
| FR24 | Epic 8 | Intelligent model routing |
| FR25 | Epic 8 | Trajectory debugging |

## Epic List

### Epic 1: Foundation & Developer Quick Start
Developers can deploy and configure the complete system infrastructure, achieving first response in under 15 minutes using Docker Compose with zero initial code.

**FRs covered:** FR1, FR2
**NFRs addressed:** NFR7 (protocol compliance foundation), NFR8 (stateless architecture)

### Epic 2: Agentic Query & Reasoning
Users can ask complex questions and receive intelligent multi-step reasoning with visible thought traces, dynamic method selection, and persistent trajectory logging.

**FRs covered:** FR6, FR7, FR10
**NFRs addressed:** NFR1 (<10s response), NFR6 (50+ concurrent agents)

### Epic 3: Hybrid Knowledge Retrieval
The system delivers accurate, explainable answers by combining vector semantic search with graph relationship traversal, synthesizing results from both databases.

**FRs covered:** FR11, FR12, FR13, FR14
**NFRs addressed:** NFR1 (<10s response), NFR5 (1M+ nodes)

### Epic 4: Knowledge Ingestion Pipeline
Users can ingest documents from URLs or PDFs and autonomously build knowledge graphs with entity extraction, relationship mapping, and graph visualization.

**FRs covered:** FR15, FR16, FR17, FR18
**NFRs addressed:** NFR2 (<5 min ingestion), NFR5 (1M+ nodes)

### Epic 5: Graphiti Temporal Knowledge Graph Integration
Replace the custom knowledge graph pipeline with Graphiti to enable temporal ingestion and querying while reducing maintenance overhead.

**FRs covered:** (architecture addendum; no new FRs)
**NFRs addressed:** NFR5 (1M+ nodes)

### Epic 6: Interactive Copilot Experience
End-users have a polished chat interface with Human-in-the-Loop source validation, generative UI components, and frontend action capabilities.

**FRs covered:** FR3, FR19, FR20, FR21, FR22
**NFRs addressed:** NFR1 (<10s response)

### Epic 7: Protocol Integration & Extensibility
The system integrates with external tools via MCP, enables agent collaboration via A2A, and provides SDKs for Python extension and universal AG-UI access.

**FRs covered:** FR4, FR5, FR8, FR9
**NFRs addressed:** NFR7 (100% MCP/AG-UI compliance)

### Epic 8: Operations & Observability
Ops engineers can monitor real-time LLM costs, configure intelligent model routing for cost optimization, and debug agent trajectories.

**FRs covered:** FR23, FR24, FR25
**NFRs addressed:** NFR3 (multi-tenant isolation), NFR4 (encrypted traces)

### Epic 9: Process & Quality Foundation (Tech Debt)
Strengthen planning and review rigor with templates, checklists, and status synchronization.

**FRs covered:** (roadmap enhancements; no new FRs)
**NFRs addressed:** (quality gates and process reliability)

### Epic 10: Testing Infrastructure (Tech Debt)
Establish integration test coverage across retrieval, ingestion, and protocol compliance.

**FRs covered:** (roadmap enhancements; no new FRs)
**NFRs addressed:** (quality and reliability validation)

### Epic 11: Code Cleanup & Migration (Revised)
Complete Graphiti migration and harden platform fundamentals including HITL, persistence, and multi-provider support.

**FRs covered:** (roadmap enhancements; no new FRs)
**NFRs addressed:** NFR3 (isolation), NFR4 (trace security), NFR8 (recovery)

### Epic 12: Advanced Retrieval (Archon Upgrade)
Improve retrieval quality with reranking, contextual embeddings, and corrective grading.

**Key Decision (2026-01-03):** Reranker + Graphiti are COMPLEMENTARY (Stage 1 + Stage 2 retrieval). All features are OPT-IN via configuration flags. See `docs/guides/advanced-retrieval-configuration.md`.

**FRs covered:** (roadmap enhancements; no new FRs)
**NFRs addressed:** NFR1 (response quality), NFR2 (retrieval precision)

### Epic 13: Enterprise Ingestion
Add resilient ingestion adapters, transcript-first YouTube processing, and **migrate to actual Crawl4AI library**.

**Key Decision (2026-01-04):** Current `crawler.py` does NOT use the installed Crawl4AI library. Must migrate from custom httpx crawler to AsyncWebCrawler for JS rendering, parallel crawling, and 10x throughput. See `epic-13-tech-spec.md`.

**FRs covered:** (roadmap enhancements; no new FRs)
**NFRs addressed:** NFR2 (ingestion performance)

### Epic 14: Connectivity (MCP Wrapper Architecture)
Expose tools via MCP server by wrapping Graphiti MCP and extending with RAG-specific tools. Strengthen A2A collaboration.

**Key Decision (2026-01-03):** WRAP Graphiti MCP, don't duplicate. Extend with RAG tools (vector_search, ingest_*, query_with_reranking). See `docs/guides/mcp-wrapper-architecture.md`.

**FRs covered:** (roadmap enhancements; no new FRs)
**NFRs addressed:** NFR7 (protocol compliance)

### Epic 15: Codebase Intelligence
Focus on codebase-specific features that differentiate a developer-focused RAG platform.

**Key Decision (2026-01-03):** REMOVED multimodal video/image processing (YouTube transcript covers 90%+ of RAG needs). FOCUSED on codebase hallucination detection as unique differentiator.

**FRs covered:** (roadmap enhancements; no new FRs)
**NFRs addressed:** NFR1 (response quality)

### Epic 16: Framework Agnosticism (Developer Extension Points)
Define headless agent protocol and framework adapters for developers to build on.

**Key Decision (2026-01-03):** These are DEVELOPER EXTENSION POINTS, not runtime switching. CLI asks "Which framework?" and configures the project. Includes Agent Skills integration for Anthropic adapter.

**FRs covered:** (roadmap enhancements; no new FRs)
**NFRs addressed:** NFR7 (interoperability)

### Epic 17: Deployment & CLI (Final Polish)
Deliver an interactive CLI install flow with hardware detection and startup verification.

**FRs covered:** (roadmap enhancements; no new FRs)
**NFRs addressed:** (installation reliability)

### Epic 18: Enhanced Documentation & DevOps (Manual)
Add documentation and DevOps automation for long-term maintainability.

**FRs covered:** (roadmap enhancements; no new FRs)
**NFRs addressed:** (security and maintainability)

---

## Epic 1: Foundation & Developer Quick Start

Developers can deploy and configure the complete system infrastructure, achieving first response in under 15 minutes using Docker Compose with zero initial code.

### Story 1.1: Backend Project Initialization

As a **developer**,
I want **to initialize the backend using the Agno agent-api starter template**,
So that **I have a working Python/FastAPI foundation with agent scaffolding**.

**Acceptance Criteria:**

**Given** a developer has cloned the repository
**When** they run `uv sync` in the backend directory
**Then** all Python dependencies are installed
**And** the pyproject.toml includes Agno v2.3.21, FastAPI, and required packages
**And** the project structure matches the architecture specification

### Story 1.2: Frontend Project Initialization

As a **developer**,
I want **to initialize the frontend using Next.js with CopilotKit integration**,
So that **I have a working React/TypeScript foundation with AI copilot capabilities**.

**Acceptance Criteria:**

**Given** a developer has cloned the repository
**When** they run `pnpm install` in the frontend directory
**Then** all npm dependencies are installed
**And** CopilotKit React components are available
**And** the project uses Next.js 15+ with App Router

### Story 1.3: Docker Compose Development Environment

As a **developer**,
I want **a Docker Compose configuration that orchestrates all services**,
So that **I can start the entire stack with a single command**.

**Acceptance Criteria:**

**Given** Docker and Docker Compose are installed
**When** the developer runs `docker compose up -d`
**Then** the following services start successfully:
- Backend (FastAPI on port 8000)
- Frontend (Next.js on port 3000)
- PostgreSQL with pgvector (port 5432)
- Neo4j Community (ports 7474, 7687)
- Redis (port 6379)
**And** health checks pass for all services
**And** hot reload is enabled for backend and frontend

### Story 1.4: Environment Configuration System

As a **developer**,
I want **to configure the system using environment variables**,
So that **I can customize API keys and database URLs without code changes**.

**Acceptance Criteria:**

**Given** the developer has created a `.env` file from `.env.example`
**When** they set `OPENAI_API_KEY` and other required variables
**Then** the backend reads configuration from environment
**And** database connection strings are configurable
**And** the system validates required variables on startup
**And** missing required variables produce clear error messages

---

## Epic 2: Agentic Query & Reasoning

Users can ask complex questions and receive intelligent multi-step reasoning with visible thought traces, dynamic method selection, and persistent trajectory logging.

### Story 2.1: Orchestrator Agent Foundation

As a **user**,
I want **an orchestrator agent that can receive and process my queries**,
So that **I get intelligent responses from the system**.

**Acceptance Criteria:**

**Given** the backend is running
**When** a user submits a query via the API
**Then** the orchestrator agent receives the query
**And** returns a response within the NFR1 target (<10s)
**And** the agent uses Agno's built-in patterns for execution

### Story 2.2: Multi-Step Query Planning

As a **user**,
I want **the agent to autonomously plan multi-step strategies for complex queries**,
So that **sophisticated questions are broken down and answered systematically**.

**Acceptance Criteria:**

**Given** a user submits a complex query requiring multiple steps
**When** the orchestrator agent processes the query
**Then** it generates a visible execution plan
**And** each step is logged as a "thought"
**And** steps execute in logical sequence
**And** the plan adapts if intermediate steps reveal new information

### Story 2.3: Dynamic Retrieval Method Selection

As a **user**,
I want **the agent to automatically choose the best retrieval method**,
So that **my queries use vector search, graph traversal, or both as appropriate**.

**Acceptance Criteria:**

**Given** the agent is processing a query
**When** it needs to retrieve information
**Then** it analyzes the query type (semantic vs. relational)
**And** selects Vector RAG for semantic similarity queries
**And** selects GraphRAG for relationship-based queries
**And** selects hybrid for complex multi-hop queries
**And** logs the selection decision in the trajectory

### Story 2.4: Persistent Trajectory Logging

As a **developer**,
I want **every agent interaction to maintain a persistent thought trace**,
So that **I can debug and understand the agent's reasoning process**.

**Acceptance Criteria:**

**Given** an agent is processing a query
**When** it makes decisions, calls tools, or generates responses
**Then** each thought is logged using `agent.log_thought()`
**And** each action is logged using `agent.log_action()`
**And** each observation is logged using `agent.log_observation()`
**And** trajectories are persisted to the database
**And** trajectories survive container restarts (NFR8)

---

## Epic 3: Hybrid Knowledge Retrieval

The system delivers accurate, explainable answers by combining vector semantic search with graph relationship traversal, synthesizing results from both databases.

### Story 3.1: Vector Semantic Search

As a **user**,
I want **to search for information using semantic similarity**,
So that **I can find relevant content even when exact keywords don't match**.

**Acceptance Criteria:**

**Given** documents have been indexed with embeddings in pgvector
**When** a user submits a semantic query
**Then** the system generates an embedding for the query
**And** performs cosine similarity search against stored vectors
**And** returns the top-k most relevant chunks
**And** each result includes similarity score and source reference

### Story 3.2: Graph Relationship Traversal

As a **user**,
I want **to query relationships between entities in the knowledge graph**,
So that **I can discover connections that semantic search alone would miss**.

**Acceptance Criteria:**

**Given** entities and relationships exist in Neo4j
**When** a user submits a relationship-based query
**Then** the system identifies relevant starting entities
**And** traverses relationships using Cypher queries
**And** returns connected entities and relationship paths
**And** respects tenant isolation via property filters (NFR3)

### Story 3.3: Hybrid Answer Synthesis

As a **user**,
I want **the system to combine vector and graph results into a coherent answer**,
So that **I get comprehensive responses leveraging both retrieval methods**.

**Acceptance Criteria:**

**Given** both vector search and graph traversal have returned results
**When** the retriever agent synthesizes the answer
**Then** it merges results from both sources
**And** ranks combined results by relevance
**And** generates a unified response using the LLM
**And** includes citations from both vector chunks and graph entities

### Story 3.4: Graph-Based Explainability

As a **user**,
I want **to see how the system arrived at its answer using graph connections**,
So that **I can verify the reasoning and trust the response**.

**Acceptance Criteria:**

**Given** an answer was generated using graph traversal
**When** the response is returned to the user
**Then** it includes the specific nodes referenced
**And** shows the relationship edges that connected them
**And** provides a human-readable explanation of the path
**And** allows the user to explore the subgraph visually

---

## Epic 4: Knowledge Ingestion Pipeline

Users can ingest documents from URLs or PDFs and autonomously build knowledge graphs with entity extraction, relationship mapping, and graph visualization.

### Story 4.1: URL Documentation Crawling

As a **data engineer**,
I want **to trigger autonomous crawling of documentation websites**,
So that **I can ingest external knowledge sources without manual downloads**.

**Acceptance Criteria:**

**Given** a valid documentation URL is provided
**When** the user triggers crawling via the ingestion API
**Then** Crawl4AI crawls the documentation site
**And** respects robots.txt and rate limits
**And** extracts content from all linked pages
**And** queues extracted content for processing
**And** reports crawl progress and statistics

### Story 4.2: PDF Document Parsing

As a **data engineer**,
I want **to parse complex PDF documents with tables and structured layouts**,
So that **information is accurately extracted regardless of document format**.

**Acceptance Criteria:**

**Given** a PDF file is uploaded via the ingestion API
**When** Docling processes the document
**Then** it extracts text preserving structure
**And** parses tables into structured data
**And** identifies headers, sections, and footnotes
**And** processes a 50-page document in <5 minutes (NFR2)
**And** outputs standardized chunks for indexing

### Story 4.3: Agentic Entity Extraction

As a **data engineer**,
I want **an agent to autonomously extract entities and relationships from text**,
So that **the knowledge graph is built without manual schema mapping**.

**Acceptance Criteria:**

**Given** document chunks are ready for indexing
**When** the Agentic Indexer processes them
**Then** it identifies named entities (people, organizations, concepts)
**And** extracts relationships between entities
**And** creates nodes in Neo4j with appropriate labels
**And** creates edges with relationship types
**And** stores chunk embeddings in pgvector
**And** logs extraction decisions in the trajectory

### Story 4.4: Knowledge Graph Visualization

As a **data engineer**,
I want **to visualize the current state of the knowledge graph**,
So that **I can identify gaps, orphan nodes, and data quality issues**.

**Acceptance Criteria:**

**Given** entities and relationships exist in Neo4j
**When** the user opens the graph visualization view
**Then** a React Flow component renders the graph
**And** nodes are displayed with labels and types
**And** edges show relationship types
**And** users can zoom, pan, and select nodes
**And** orphan nodes are highlighted for attention
**And** users can filter by entity type or relationship

---

## Epic 5: Graphiti Temporal Knowledge Graph Integration

Replace the custom knowledge graph pipeline with Graphiti to enable temporal ingestion and querying while reducing maintenance overhead.

### Story 5.1: Graphiti Installation & Setup

As a **developer**,
I want **to install and configure Graphiti**,
So that **temporal knowledge graph capabilities are available in the backend**.

**Acceptance Criteria:**

**Given** the backend codebase
**When** Graphiti is installed and configured
**Then** the dependency is pinned and documented
**And** the Graphiti client connects to Neo4j successfully
**And** a basic smoke test verifies ingestion can run

### Story 5.2: Episode Ingestion Pipeline

As a **developer**,
I want **to ingest documents as Graphiti episodes**,
So that **entities and relationships are tracked with temporal context**.

**Acceptance Criteria:**

**Given** a parsed document
**When** it is ingested as a Graphiti episode
**Then** entities and relationships are created with temporal metadata
**And** tenant isolation is preserved
**And** ingestion errors are logged in the trajectory

### Story 5.3: Hybrid Retrieval Integration

As a **developer**,
I want **hybrid retrieval to use Graphiti for graph lookups**,
So that **answers include temporal context alongside vector results**.

**Acceptance Criteria:**

**Given** a user query
**When** hybrid retrieval runs
**Then** Graphiti graph results are included in the response
**And** vector and graph results are synthesized
**And** tenant isolation is preserved

### Story 5.4: Temporal Query Capabilities

As a **developer**,
I want **to query the knowledge graph at a specific point in time**,
So that **historical answers can be generated when needed**.

**Acceptance Criteria:**

**Given** a query with a time filter
**When** it is executed
**Then** the graph results reflect that point in time
**And** the system defaults to latest data when no time is provided

### Story 5.5: Legacy Code Removal & Migration

As a **developer**,
I want **to remove legacy graph ingestion code and migrate existing data**,
So that **maintenance is simplified and the pipeline is consistent**.

**Acceptance Criteria:**

**Given** the Graphiti pipeline is working
**When** legacy ingestion code is removed
**Then** the codebase no longer uses the old extraction pipeline
**And** existing data is available in the new schema
**And** documentation reflects the migration

### Story 5.6: Test Suite Adaptation

As a **developer**,
I want **tests updated for Graphiti workflows**,
So that **regressions are caught in ingestion and retrieval**.

**Acceptance Criteria:**

**Given** Graphiti integration
**When** the test suite runs
**Then** ingestion and retrieval tests pass
**And** coverage is maintained for graph-related components

---

## Epic 6: Interactive Copilot Experience

End-users have a polished chat interface with Human-in-the-Loop source validation, generative UI components, and frontend action capabilities.

### Story 6.1: CopilotKit React Integration

As a **React developer**,
I want **to integrate the Copilot UI using an npm package**,
So that **I can add AI chat capabilities to my app with minimal code**.

**Acceptance Criteria:**

**Given** a React/Next.js application exists
**When** the developer installs `@copilotkit/react-core` and `@copilotkit/react-ui`
**Then** they can wrap their app in a CopilotProvider
**And** the provider connects to the backend AG-UI endpoint
**And** the integration requires only environment configuration
**And** TypeScript types are available for all components

### Story 6.2: Chat Sidebar Interface

As an **end-user**,
I want **to interact with the AI through a pre-built chat sidebar**,
So that **I can ask questions and receive responses naturally**.

**Acceptance Criteria:**

**Given** the CopilotKit integration is configured
**When** the user opens the chat sidebar
**Then** they see a polished chat interface (shadcn/ui styling)
**And** can type messages and submit queries
**And** see streaming responses as they generate
**And** view the "Thought Trace" stepper showing agent progress
**And** the sidebar uses the design system colors (Indigo-600, Slate)

### Story 6.3: Generative UI Components

As an **end-user**,
I want **the AI to render specialized UI components within the chat**,
So that **I can see interactive visualizations like graph explorers**.

**Acceptance Criteria:**

**Given** the agent determines a visualization would help
**When** it sends a Generative UI payload
**Then** the frontend dynamically renders the appropriate component
**And** graph visualizers show entity relationships (React Flow)
**And** data tables render structured information
**And** components are interactive (clickable, zoomable)
**And** the UX follows the "Professional Forge" design direction

### Story 6.4: Human-in-the-Loop Source Validation

As a **researcher**,
I want **to review and approve/reject sources before the answer is generated**,
So that **I can ensure the AI uses only trusted, relevant information**.

**Acceptance Criteria:**

**Given** the agent has retrieved sources for a query
**When** HITL validation is triggered
**Then** a side-panel displays retrieved source cards
**And** each card shows source title, snippet, and metadata
**And** users can Approve or Reject each source
**And** rejected sources are excluded from synthesis
**And** the UI uses Amber-400 for attention items
**And** validation is non-blocking if user continues typing

### Story 6.5: Frontend Actions

As an **end-user**,
I want **the AI to take actions within my application**,
So that **it can highlight text, open modals, or trigger UI changes**.

**Acceptance Criteria:**

**Given** the agent determines a frontend action is needed
**When** it sends an action payload via AG-UI
**Then** the frontend executes the registered action
**And** supported actions include: highlight text, open modal, scroll to section
**And** actions are defined by the host application
**And** the agent confirms action completion in the trajectory

---

## Epic 7: Protocol Integration & Extensibility

The system integrates with external tools via MCP, enables agent collaboration via A2A, and provides SDKs for Python extension and universal AG-UI access.

### Story 7.1: MCP Tool Server Implementation

As a **developer**,
I want **agents to use external tools via the Model Context Protocol**,
So that **they can access databases, APIs, and services through a standard interface**.

**Acceptance Criteria:**

**Given** an MCP tool is registered in the system
**When** the agent needs to use external functionality
**Then** it discovers available tools via `/mcp/v1/tools`
**And** calls tools using JSON-RPC 2.0 format
**And** receives structured responses
**And** tool calls are logged in the trajectory
**And** the implementation adheres 100% to MCP specification (NFR7)

### Story 7.2: A2A Agent Collaboration

As a **developer**,
I want **multiple agents to collaborate and delegate tasks**,
So that **complex workflows can be distributed across specialized agents**.

**Acceptance Criteria:**

**Given** multiple agents are registered in the system
**When** an orchestrator agent needs specialized processing
**Then** it can delegate tasks to other agents via A2A protocol
**And** the delegated agent receives full context
**And** results are returned to the orchestrator
**And** delegation is tracked in the trajectory
**And** the implementation adheres to A2A specification

### Story 7.3: Python Extension SDK

As a **Python developer**,
I want **to extend the core agent logic using a pip package**,
So that **I can customize agent behavior without forking the codebase**.

**Acceptance Criteria:**

**Given** a Python developer wants to extend the system
**When** they install the extension package
**Then** they can create custom agents inheriting from base classes
**And** register custom tools with the MCP server
**And** add custom retrieval strategies
**And** extend the indexing pipeline
**And** documentation covers all extension points

### Story 7.4: Universal AG-UI Protocol Access

As a **developer using any language**,
I want **to interact with the system via standardized AG-UI protocol**,
So that **I can build custom frontends in any technology stack**.

**Acceptance Criteria:**

**Given** the backend is running
**When** a client connects via HTTP/WebSocket
**Then** it can send queries following AG-UI specification
**And** receives streaming responses via SSE
**And** can send frontend state updates
**And** receives agent state changes
**And** the protocol is language-agnostic (works with Go, Rust, etc.)
**And** OpenAPI documentation is auto-generated

---

## Epic 8: Operations & Observability

Ops engineers can monitor real-time LLM costs, configure intelligent model routing for cost optimization, and debug agent trajectories.

### Story 8.1: LLM Cost Monitoring

As an **ops engineer**,
I want **to monitor real-time LLM interaction costs**,
So that **I can track spending and identify optimization opportunities**.

**Acceptance Criteria:**

**Given** the system is processing LLM requests
**When** an ops engineer views the cost dashboard
**Then** they see real-time token usage per request
**And** costs are calculated based on model pricing
**And** usage is aggregated by tenant (NFR3)
**And** historical cost trends are displayed
**And** alerts can be configured for spending thresholds

### Story 8.2: Intelligent Model Routing

As an **ops engineer**,
I want **the system to route queries to different LLM models based on complexity**,
So that **simple queries use cheaper models while complex ones get premium models**.

**Acceptance Criteria:**

**Given** a query is submitted to the system
**When** the routing logic evaluates the query
**Then** it classifies query complexity (simple, medium, complex)
**And** routes simple queries to cost-effective models (e.g., GPT-4o-mini)
**And** routes complex queries to premium models (e.g., GPT-4o, Claude)
**And** routing decisions are configurable via settings
**And** routing decisions are logged in the trajectory
**And** cost savings are tracked and reported

### Story 8.3: Trajectory Debugging Interface

As a **developer**,
I want **to review the reasoning trajectory of past queries**,
So that **I can debug agent behavior and identify issues**.

**Acceptance Criteria:**

**Given** an agent has processed queries with trajectory logging
**When** a developer opens the trajectory viewer
**Then** they see a list of past agent sessions
**And** can drill into individual trajectories
**And** see thoughts, actions, and observations in sequence
**And** view tool calls and their results
**And** see timing information for each step
**And** can filter by error status or agent type

### Story 8.4: Encrypted Trace Storage

As a **security engineer**,
I want **reasoning traces to be encrypted at rest**,
So that **sensitive query content is protected from unauthorized access**.

**Acceptance Criteria:**

**Given** trajectory data is being persisted
**When** it is written to the database
**Then** sensitive fields are encrypted using AES-256
**And** encryption keys are managed securely
**And** decryption only occurs for authorized access
**And** multi-tenant data remains isolated (NFR3)
**And** encryption does not significantly impact query latency

---

## Epic 9: Process & Quality Foundation (Tech Debt)

Improve delivery quality with shared standards, checklists, and consistent status tracking.

### Story 9.1: Retrospective Action Item Tracking

As a **scrum master**,
I want **to track retrospective action items with owners and due dates**,
So that **continuous improvement items are implemented**.

**Acceptance Criteria:**

**Given** a retrospective is completed
**When** action items are recorded
**Then** each item includes owner, due date, and status
**And** items are visible in sprint tracking

### Story 9.2: Story Template Standards Section

As a **scrum master**,
I want **story templates to include standards and expectations**,
So that **teams apply consistent quality criteria**.

**Acceptance Criteria:**

**Given** a new story is created
**When** the template is applied
**Then** a standards section is included
**And** reviewers use it during story review

### Story 9.3: Pre-Review Quality Checklist

As a **developer**,
I want **a pre-review checklist**,
So that **code meets quality expectations before review**.

**Acceptance Criteria:**

**Given** a story is ready for review
**When** the developer completes the checklist
**Then** required items are validated and recorded
**And** missing items block review submission

### Story 9.4: Protocol Compliance Checklist

As a **developer**,
I want **a protocol compliance checklist**,
So that **MCP, A2A, and AG-UI requirements are not missed**.

**Acceptance Criteria:**

**Given** a protocol-related story is completed
**When** compliance is reviewed
**Then** checklist items confirm spec adherence
**And** non-compliance is logged for remediation

### Story 9.5: Story File Status Synchronization

As a **scrum master**,
I want **story file status to stay in sync with sprint status**,
So that **tracking is accurate across artifacts**.

**Acceptance Criteria:**

**Given** a story status changes
**When** the change is recorded
**Then** sprint-status.yaml and the story file match
**And** discrepancies are reported

### Story 9.6: Story Completion Dev Notes

As a **developer**,
I want **to capture completion notes**,
So that **future work has context and decisions are recorded**.

**Acceptance Criteria:**

**Given** a story is completed
**When** it is marked done
**Then** dev notes include decisions and tradeoffs
**And** notes are stored in the story file

---

## Epic 10: Testing Infrastructure (Tech Debt)

Establish integration and benchmark testing coverage across core workflows.

### Story 10.1: Integration Test Framework

As a **test engineer**,
I want **an integration test framework**,
So that **cross-service workflows are validated in CI**.

**Acceptance Criteria:**

**Given** the test suite runs in CI
**When** integration tests execute
**Then** required services are provisioned
**And** tests report deterministic results

### Story 10.2: Hybrid Retrieval Integration Tests

As a **test engineer**,
I want **hybrid retrieval integration tests**,
So that **vector and graph retrieval work together correctly**.

**Acceptance Criteria:**

**Given** seeded data exists
**When** a hybrid query executes
**Then** results include vector and graph sources
**And** synthesis output is validated

### Story 10.3: Ingestion Pipeline Integration Tests

As a **test engineer**,
I want **ingestion pipeline integration tests**,
So that **document ingestion is reliable across formats**.

**Acceptance Criteria:**

**Given** a URL or PDF input
**When** ingestion runs
**Then** chunks are stored with metadata
**And** errors are captured with actionable logs

### Story 10.4: Graphiti Integration Tests

As a **test engineer**,
I want **Graphiti integration tests**,
So that **episode ingestion and temporal queries are validated**.

**Acceptance Criteria:**

**Given** Graphiti is configured
**When** ingestion and queries run
**Then** temporal results match expected ranges
**And** migrations are validated

### Story 10.5: Protocol Integration Tests

As a **test engineer**,
I want **protocol integration tests**,
So that **MCP, A2A, and AG-UI endpoints are compliant**.

**Acceptance Criteria:**

**Given** protocol endpoints are available
**When** compliance tests run
**Then** JSON-RPC and streaming formats are validated
**And** failures report spec gaps

### Story 10.6: Skipped Test Resolution

As a **test engineer**,
I want **skipped tests resolved**,
So that **coverage reflects actual quality**.

**Acceptance Criteria:**

**Given** skipped tests exist
**When** the suite runs
**Then** skips are removed or justified in docs
**And** CI enforces the policy

### Story 10.7: NFR Benchmark Validation

As a **test engineer**,
I want **NFR benchmarks**,
So that **performance and scale targets are validated**.

**Acceptance Criteria:**

**Given** benchmark inputs are defined
**When** benchmarks execute
**Then** results are recorded against NFR targets
**And** regressions are flagged

---

## Epic 11: Code Cleanup & Migration (Revised)

Complete Graphiti migration and harden platform fundamentals.

### Story 11.1: Fix Datetime Deprecation

As a **developer**,
I want **to replace deprecated datetime usage**,
So that **runtime warnings are eliminated**.

**Acceptance Criteria:**

**Given** runtime warnings exist
**When** deprecated APIs are replaced
**Then** warnings are removed
**And** tests pass

### Story 11.2: Delete Legacy Modules

As a **developer**,
I want **to remove legacy modules**,
So that **the codebase is simplified**.

**Acceptance Criteria:**

**Given** legacy modules are identified
**When** removal is complete
**Then** no references remain
**And** documentation is updated

### Story 11.3: Complete Graphiti Migration

As a **developer**,
I want **Graphiti to be the default graph pipeline**,
So that **legacy ingestion is fully retired**.

**Acceptance Criteria:**

**Given** Graphiti is configured
**When** ingestion and retrieval run
**Then** Graphiti handles graph operations
**And** legacy pipelines are removed

### Story 11.4: Wire HITL Validation

As a **developer**,
I want **HITL validation wired end-to-end**,
So that **users can approve sources before synthesis**.

**Acceptance Criteria:**

**Given** sources are retrieved
**When** HITL is triggered
**Then** UI approval flows operate
**And** decisions are logged

### Story 11.5: Implement Workspace Persistence

As a **developer**,
I want **workspace persistence**,
So that **agent state survives restarts**.

**Acceptance Criteria:**

**Given** a session is active
**When** the service restarts
**Then** workspace state is restored
**And** tenant isolation is preserved

### Story 11.6: Replace HTML/Markdown Regex

As a **developer**,
I want **structured parsing instead of brittle regex**,
So that **content extraction is more reliable**.

**Acceptance Criteria:**

**Given** HTML/Markdown inputs exist
**When** parsing runs
**Then** structured parsing is used
**And** extraction accuracy is improved

### Story 11.7: Configure Neo4j Pooling

As a **developer**,
I want **Neo4j connection pooling configured**,
So that **high concurrency is supported**.

**Acceptance Criteria:**

**Given** production traffic
**When** the pool is configured
**Then** connections are reused efficiently
**And** pool metrics are monitored

### Story 11.8: Add LLM Token Monitoring

As an **ops engineer**,
I want **LLM token monitoring**,
So that **cost and usage are observable**.

**Acceptance Criteria:**

**Given** LLM calls execute
**When** usage is tracked
**Then** token metrics are logged
**And** costs are computed per request

### Story 11.9: A2A Session Persistence

As a **developer**,
I want **A2A sessions persisted**,
So that **agent handoffs survive interruptions**.

**Acceptance Criteria:**

**Given** agents collaborate
**When** a session is resumed
**Then** context is preserved
**And** timeouts are handled cleanly

### Story 11.10: Refactor Config for Multi-Provider

As a **developer**,
I want **configuration support for multiple LLM providers**,
So that **the system is not locked to a single vendor**.

**Acceptance Criteria:**

**Given** provider settings are configured
**When** the app starts
**Then** provider selection is validated
**And** defaults are documented

### Story 11.11: Implement Provider Adapter Pattern

As a **developer**,
I want **a provider adapter pattern**,
So that **new LLM providers can be added consistently**.

**Acceptance Criteria:**

**Given** adapters are implemented
**When** providers are selected
**Then** calls route through the adapter interface
**And** providers include OpenAI, Anthropic, Gemini, Ollama, and OpenRouter

### Story 11.12: Implement Multi-Provider Embeddings

As a **developer**,
I want **multi-provider embedding support**,
So that **embeddings can use Gemini, Voyage AI, or OpenAI based on configuration**.

**Acceptance Criteria:**

**Given** embedding provider is configured
**When** embeddings are generated
**Then** the configured provider is used for both Graphiti and pgvector
**And** supported providers include OpenAI, Gemini, Voyage AI, OpenRouter, and Ollama
**And** smart defaults match embedding provider to LLM provider when supported

---

## Epic 12: Advanced Retrieval (Archon Upgrade)

Improve retrieval quality with reranking, contextual embeddings, and corrective grading.

### Story 12.1: Implement Cross-Encoder Reranking

As a **developer**,
I want **cross-encoder reranking**,
So that **top results are more relevant**.

**Acceptance Criteria:**

**Given** retrieval returns candidates
**When** reranking is enabled
**Then** results are rescored and reordered
**And** reranker selection is configurable

### Story 12.2: Implement Contextual Retrieval Chunking

As a **developer**,
I want **contextual chunk enrichment**,
So that **embeddings include title and summary context**.

**Acceptance Criteria:**

**Given** a document is ingested
**When** chunks are embedded
**Then** title and summary context are prepended
**And** reindexing is supported for existing content

### Story 12.3: Implement Corrective RAG Grader Agent

As a **developer**,
I want **a grader that triggers fallback retrieval**,
So that **low-quality retrieval can be corrected**.

**Acceptance Criteria:**

**Given** retrieval results are scored
**When** scores fall below threshold
**Then** fallback retrieval is triggered
**And** decisions are logged in the trajectory

---

## Epic 13: Enterprise Ingestion

Add resilient ingestion adapters, transcript-first YouTube processing, and migrate to actual Crawl4AI library.

**Decision (2026-01-04):** Current `crawler.py` uses httpx, NOT the installed Crawl4AI library. Must migrate. See `epic-13-tech-spec.md`.

### Story 13.1: Integrate Apify BrightData Fallback

As a **developer**,
I want **Apify and BrightData fallbacks**,
So that **ingestion succeeds when Crawl4AI is blocked**.

**Acceptance Criteria:**

**Given** a crawl fails or is blocked
**When** fallback providers are configured
**Then** ingestion routes to Apify or BrightData
**And** provider selection and fallback reason are logged
**And** credentials are configured via environment variables

### Story 13.2: Implement YouTube Transcript API Ingestion

As a **developer**,
I want **transcript-first YouTube ingestion**,
So that **videos can be processed quickly without full video download**.

**Acceptance Criteria:**

**Given** a YouTube URL
**When** ingestion runs
**Then** transcripts are fetched using youtube-transcript-api
**And** missing transcripts are reported with clear error
**And** transcript chunks include source metadata
**And** ingestion completes in under 30 seconds for typical videos

### Story 13.3: Migrate to Crawl4AI Library

As a **developer**,
I want **to migrate from custom httpx crawler to actual Crawl4AI library**,
So that **JavaScript-rendered content is captured and crawling is parallelized**.

**Why This Matters:** Current `crawler.py` uses httpx which cannot render JavaScript, crawls sequentially, has no caching, and has no proxy support.

**Acceptance Criteria:**

**Given** a crawl request
**When** using Crawl4AI AsyncWebCrawler
**Then** JavaScript-rendered content is captured (SPAs, React sites work)
**And** multiple URLs crawl in parallel via `arun_many()` with MemoryAdaptiveDispatcher
**And** caching is enabled (unchanged pages not re-fetched)
**And** proxy support is configurable for blocked sites
**And** throughput improves 10x (50 pages in <30 seconds)
**And** legacy httpx implementation is deprecated/removed

### Story 13.4: Implement Crawl Configuration Profiles

As a **developer**,
I want **pre-configured crawl profiles for common use cases**,
So that **I can optimize crawl settings without manual tuning**.

**Profiles:**
- `fast`: Documentation sites (headless, high concurrency)
- `thorough`: Complex SPAs (JS wait, screenshots, lower concurrency)
- `stealth`: Bot-protected sites (proxy, random delays, fingerprint evasion)

**Acceptance Criteria:**

**Given** a profile name (fast/thorough/stealth)
**When** crawl is triggered
**Then** appropriate settings are applied automatically
**And** profiles are selectable via `CRAWL4AI_PROFILE` or API parameter

---

## Epic 14: Connectivity (MCP Wrapper Architecture)

Expose tools via MCP server by wrapping Graphiti MCP and extending with RAG-specific tools.

**Decision:** WRAP Graphiti MCP, don't duplicate. See `docs/guides/mcp-wrapper-architecture.md`.

### Story 14.1: Expose RAG Engine via MCP Server

As a **developer**,
I want **an MCP server that wraps Graphiti MCP and extends with RAG tools**,
So that **external clients get both graph and vector capabilities**.

**Acceptance Criteria:**

**Given** the MCP server is running
**When** tools are requested
**Then** Graphiti tools (add_memory, search_nodes, search_facts) are proxied
**And** RAG extension tools (vector_search, hybrid_retrieve, ingest_url, ingest_pdf, ingest_youtube, query_with_reranking, explain_answer) are available
**And** all tools enforce tenant isolation
**And** authentication and rate limiting are configurable

### Story 14.2: Implement Robust A2A Protocol

As a **developer**,
I want **a robust A2A protocol**,
So that **agent collaboration is reliable**.

**Acceptance Criteria:**

**Given** A2A requests are sent
**When** failures occur
**Then** errors use RFC 7807 format
**And** retries and timeouts are supported

---

## Epic 15: Codebase Intelligence

Focus on codebase-specific features that differentiate a developer-focused RAG platform.

**Decision (2026-01-03):** REMOVED multimodal video/image processing (YouTube transcript covers 90%+ of RAG needs). See `docs/roadmap-decisions-2026-01-03.md`.

### Story 15.1: Implement Codebase Hallucination Detector

As a **developer**,
I want **a hallucination detector that validates LLM responses against actual codebase symbols**,
So that **responses referencing non-existent code are flagged before users see them**.

**Why This Matters:** LLMs frequently hallucinate non-existent functions, classes, file paths, and API endpoints. For a developer platform, catching these is critical for trust.

**Acceptance Criteria:**

**Given** an LLM response references code elements
**When** validation runs
**Then** AST parsing detects unknown classes, functions, and files
**And** warnings are recorded with a list of missing symbols
**And** detection is configurable (warn or block mode)
**And** Python, TypeScript, and JavaScript are supported initially
**And** feature is opt-in via `HALLUCINATION_DETECTOR_ENABLED` flag

### Story 15.2: Implement Codebase RAG Context

As a **developer**,
I want **to index a code repository as a knowledge source for RAG queries**,
So that **users can ask questions about the codebase and get accurate answers**.

**Use Cases:**
- "How does authentication work in this codebase?"
- "What functions call the UserService?"
- "Explain the data flow from API to database"

**Acceptance Criteria:**

**Given** a code repository
**When** indexing runs
**Then** symbols are extracted (functions, classes, methods) with AST parsing
**And** symbol relationships are captured in the knowledge graph (e.g., "function A calls function B")
**And** queries about the codebase return relevant code context
**And** indexing respects .gitignore and configurable exclusion patterns
**And** incremental indexing is supported for changed files

---

## Epic 16: Framework Agnosticism (Developer Extension Points)

Define a headless agent protocol and framework adapters for developers to build on.

**Decision (2026-01-03):** These are DEVELOPER EXTENSION POINTS, not runtime switching. CLI asks "Which framework?" and configures the project.

### Story 16.1: Define Headless Agent Protocol Interface

As a **developer**,
I want **a headless agent protocol**,
So that **frameworks can plug in consistently**.

**Acceptance Criteria:**

**Given** the protocol definition
**When** the orchestrator uses it
**Then** run and stream behaviors are standardized
**And** the interface is documented

### Story 16.2: Implement Pydantic AI Adapter

As a **developer**,
I want **a PydanticAI adapter**,
So that **the system can run on PydanticAI**.

**Acceptance Criteria:**

**Given** AGENT_FRAMEWORK is set to pydanticai
**When** a request is processed
**Then** the adapter handles execution and streaming
**And** tests cover basic flows

### Story 16.3: Implement CrewAI Adapter

As a **developer**,
I want **a CrewAI adapter**,
So that **multi-agent orchestration is supported**.

**Acceptance Criteria:**

**Given** AGENT_FRAMEWORK is set to crewai
**When** a request is processed
**Then** the adapter executes a crew workflow
**And** errors use RFC 7807 format

### Story 16.4: Implement Anthropic Agent Adapter

As a **developer**,
I want **an Anthropic agent adapter with Agent Skills integration**,
So that **Claude-native workflows and skills are supported**.

**Agent Skills Integration (2026-01-03):** Agent Skills is an [open standard adopted by Microsoft/VS Code, Cursor, and others](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills). Skills provide organized folders of instructions that agents can discover.

**Acceptance Criteria:**

**Given** AGENT_FRAMEWORK is set to anthropic
**When** requests stream responses
**Then** outputs match existing front-end expectations
**And** model selection is configurable
**And** Agent Skills are exposed for RAG capabilities (search, ingest, explain)
**And** Skills work with Claude.ai, Claude Code, and API

### Story 16.5: Implement LangGraph Adapter

As a **developer**,
I want **a LangGraph adapter**,
So that **graph-based workflows are supported**.

**Acceptance Criteria:**

**Given** AGENT_FRAMEWORK is set to langgraph
**When** a request is processed
**Then** state transitions are logged
**And** integration tests cover basic flows

---

## Epic 17: Deployment & CLI (Final Polish)

Deliver an interactive CLI install flow with hardware detection and startup verification.

### Story 17.1: Create rag-install CLI Tool

As a **developer**,
I want **an interactive rag-install CLI**,
So that **setup is guided and repeatable**.

**Acceptance Criteria:**

**Given** rag-install is executed
**When** prompts are completed
**Then** a valid configuration is produced
**And** next steps are displayed

### Story 17.2: Implement Auto Hardware Detection

As a **developer**,
I want **automatic hardware detection**,
So that **defaults match the local machine**.

**Acceptance Criteria:**

**Given** the CLI runs
**When** hardware is detected
**Then** CPU/GPU/RAM info is recorded
**And** defaults are adjusted accordingly

### Story 17.3: Implement Env Generation Logic

As a **developer**,
I want **.env generation and validation**,
So that **configuration is correct on first run**.

**Acceptance Criteria:**

**Given** user selections
**When** the CLI writes .env
**Then** required values are populated
**And** an existing file is backed up

### Story 17.4: Verify Docker Compose Startup

As a **developer**,
I want **startup verification**,
So that **the stack boots successfully**.

**Acceptance Criteria:**

**Given** docker compose is invoked
**When** services start
**Then** health checks are validated
**And** failures provide actionable errors

---

## Epic 18: Enhanced Documentation & DevOps (Manual)

Add documentation and DevOps automation for long-term maintainability.

### Story 18.1: Document Observability Metrics

As a **technical writer**,
I want **observability documentation**,
So that **operators know what to monitor**.

**Acceptance Criteria:**

**Given** observability features exist
**When** documentation is published
**Then** metrics, dashboards, and alert thresholds are described

### Story 18.2: Configure Dependabot Security Updates

As a **devops engineer**,
I want **Dependabot configured**,
So that **dependency updates are automated**.

**Acceptance Criteria:**

**Given** repository automation is enabled
**When** Dependabot runs
**Then** update PRs are created on schedule
**And** labels and scopes are defined

### Story 18.3: Configure CodeQL Analysis

As a **devops engineer**,
I want **CodeQL analysis**,
So that **security issues are detected in CI**.

**Acceptance Criteria:**

**Given** the CodeQL workflow is configured
**When** PRs are opened
**Then** CodeQL scans run and report findings

### Story 18.4: Document Provider Configuration Guide

As a **technical writer**,
I want **a provider configuration guide**,
So that **users can configure OpenAI, Anthropic, Gemini, OpenRouter, and Ollama**.

**Acceptance Criteria:**

**Given** provider setup is supported
**When** the guide is published
**Then** .env examples and pitfalls are covered

### Story 18.5: Create Advanced Retrieval Tuning Guide

As a **technical writer**,
I want **an advanced retrieval tuning guide**,
So that **teams can optimize reranking and grading**.

**Acceptance Criteria:**

**Given** retrieval features exist
**When** documentation is published
**Then** tuning knobs and benchmarking tips are described

### Story 18.6: Document Headless Agent Protocol

As a **technical writer**,
I want **headless agent protocol documentation**,
So that **framework adapters are consistent**.

**Acceptance Criteria:**

**Given** the protocol interface is defined
**When** documentation is published
**Then** interface and usage examples are included

### Story 18.7: Create MCP Server Usage Guide

As a **technical writer**,
I want **an MCP server usage guide**,
So that **external clients can integrate quickly**.

**Acceptance Criteria:**

**Given** MCP tools are available
**When** the guide is published
**Then** tool lists and examples are provided

### Story 18.8: Create CLI Installation Manual

As a **technical writer**,
I want **a CLI installation manual**,
So that **users can onboard without assistance**.

**Acceptance Criteria:**

**Given** the CLI exists
**When** the manual is published
**Then** setup and troubleshooting steps are included
