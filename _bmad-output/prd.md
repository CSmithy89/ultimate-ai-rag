---
stepsCompleted: [1, 2, 3, 4, 6, 7, 8, 9, 10]
inputDocuments:
  - '_bmad-output/project-planning-artifacts/research/technical-Agentic-RAG-and-GraphRAG-System-research-2025-12-24.md'
  - 'docs/recommendations_2025.md'
documentCounts:
  briefs: 0
  research: 1
  brainstorming: 0
  projectDocs: 1
workflowType: 'prd'
lastStep: 11
project_name: 'Agentic Rag and Graphrag with copilot'
user_name: 'Chris'
date: '2025-12-24'
---

# Product Requirements Document - Agentic Rag and Graphrag with copilot

**Author:** Chris
**Date:** 2025-12-24

## Executive Summary

**Agentic Rag and Graphrag with copilot** is an advanced, reusable AI infrastructure tool designed to accelerate the development of sophisticated RAG-enabled applications. It provides a pre-built, "drop-in" system that integrates **Agentic RAG** (autonomous planning and reasoning) with **GraphRAG** (structured knowledge retrieval) and a **CopilotKit** frontend. This solution solves the complexity of setting up advanced retrieval strategies from scratch, allowing developers to plug any agent system into it and attach it to any project. It serves as a foundational layer for building intelligent, context-aware applications across multiple domains.

### What Makes This Special

*   **Pre-Packaged Sophistication:** Unlike standard RAG tutorials, this is a production-ready "box" containing advanced strategies (Agentic + Graph) ready to be populated with data.
*   **Universal "Plug-and-Play" Design:** Designed to be agnostic to the specific end-user application, allowing developers to "attach" it to any project to instantly gain advanced AI capabilities.
*   **Accelerated Development:** Drastically reduces the time to market for future builds by providing a pre-done, high-quality RAG implementation, saving developers from reinventing the wheel for every new app.
*   **Human-in-the-Loop Integration:** Built-in support for transparency and control (inspired by reference architectures like `human-in-the-loop-rag-agent`), enabling users to validate sources and guide the AI's reasoning.
*   **Extensible Integration Layer:** Planned architecture for easy addition of future integrations and connections, ensuring the system grows with the user's needs.

## Project Classification

**Technical Type:** developer_tool
**Domain:** general
**Complexity:** high
**Project Context:** Greenfield - new project

**Classification Signals:**
*   **Developer Tool:** "plug any agent system into it", "attach it to any project", "speed up development of future builds".
*   **General Domain:** "multiple applications", "any project".
*   **High Complexity:** "extremely advance", "sophisticated rag system", "agentic rag and graphrag", "human-in-the-loop".

## Success Criteria

### User Success (Developer & End-User)
*   **Time-to-Value:** A developer can deploy the full stack (Agno + Graph + CopilotKit) and get a first response in **< 15 minutes**.
*   **Zero-Friction Configuration:** "Plug-and-play" setup using standard environment variables; no deep knowledge of Graph databases required to start.
*   **Intuitive Validation:** End-users can intuitively validate sources via the Human-in-the-Loop UI without training.

### Business Success
*   **Reusability:** The system successfully powers **multiple future applications** without code modification, only configuration changes.
*   **Development Speed:** Reduces the setup time for advanced RAG strategies in new projects from weeks to minutes.

### Technical Success
*   **Integration:** Seamless interoperability between Agno (Orchestrator), GraphRAG (Knowledge), and CopilotKit (UI).
*   **Retrieval Quality:** High hit rate and answer faithfulness scores, validated by "LLM-as-a-Judge".
*   **Scalability:** The architecture handles both simple vector lookups and complex graph traversals efficiently.

## Product Scope

### MVP - Minimum Viable Product (The Full "Sophisticated" Box)
*   **Core Stack:** Agno (Orchestrator), Docling (Ingestion), CopilotKit (Frontend), Neo4j/Memgraph (Graph), pgvector (Vector).
*   **Strategies:** Hybrid RAG (Vector + Graph) enabled out-of-the-box.
*   **Integration Layer:** Pre-built MCP integration for **BrightData/Apify** and open-source **Firecrawl**.
*   **UI Features:** Generative UI for graph visualization and Human-in-the-Loop source validation.
*   **Agentic Capabilities:** Autonomous planning agent capable of multi-step reasoning.

### Growth Features (Post-MVP)
*   **Advanced Analytics:** Dashboards for trajectory analysis and cost monitoring.
*   **Multi-Modal Support:** Processing video and audio inputs via Docling extensions.

### Vision (Future)
*   **Self-Evolving Knowledge:** Agents that autonomously identify gaps in the knowledge graph and initiate crawling/research to fill them.

## User Journeys

**Journey 1: Alex - The "Speed-Driven" Developer (Success Path)**
Alex is a senior developer at a fast-growing startup. They need to add a "Knowledge Hub" to their product by next week but don't have the time to learn the intricacies of Graph databases or Agent orchestration from scratch.

*   **Opening Scene:** Alex is staring at a blank repository, dreading the manual setup of Vector stores, Graph schemas, and coordination logic.
*   **Rising Action:** Alex finds your system, installs it, and configures the environment variables. They use the pre-built Agno orchestrator and drop the CopilotKit sidebar into their React frontend.
*   **Climax:** Within 10 minutes, Alex uploads a complex PDF manual. They ask a question that requires connecting two different sections of the document. The system uses GraphRAG to find the relationship and Agentic RAG to synthesize a perfect answer.
*   **Resolution:** Alex demos the feature to the CEO the same afternoon. Instead of weeks of infrastructure work, they spend the rest of the sprint polishing the UI.

**Journey 2: Sarah - The "Safety-First" End User (Edge Case/Human-in-the-Loop)**
Sarah is a researcher using an application powered by your system. She is working on a high-stakes medical report and cannot afford a hallucination.

*   **Opening Scene:** Sarah asks the AI a complex question about drug interactions.
*   **Rising Action:** The system doesn't just give an answer. It pauses and shows Sarah a "Source Validation" panel (Human-in-the-Loop).
*   **Climax:** Sarah sees that the AI has retrieved 5 sources. She notices one source is from an outdated study. She clicks "Reject" on that source and "Approve" on the others.
*   **Resolution:** The AI re-synthesizes the answer using only the approved, current data. Sarah feels a deep sense of trust in the system because she is in control of the knowledge.

**Journey 3: Jordan - The "Reliability" Ops Engineer (Management/Monitoring)**
Jordan is responsible for maintaining the production deployment of the RAG system across multiple company apps.

*   **Opening Scene:** It's Monday morning, and Jordan needs to ensure the knowledge graphs are healthy and the LLM costs are under control.
*   **Rising Action:** Jordan logs into the admin dashboard. They see a "Trajectory Trace" for a complex query that failed over the weekend.
*   **Climax:** Jordan identifies that the agent got stuck in a loop because a tool was timing out. They also notice that 80% of queries are being routed to the "Expensive" model unnecessarily.
*   **Resolution:** Jordan adjusts the "Intelligent Model Routing" settings to send simple queries to GPT-4o-mini and fixes the tool timeout. The system becomes 40% cheaper and more reliable by lunch.

**Journey 4: Maya - The "Quality" Data Engineer (Knowledge Maintenance)**
Maya's job is to ensure the Knowledge Graph is the ultimate source of truth for the company.

*   **Opening Scene:** Maya is notified that a new set of API documentation has been released and needs to be indexed.
*   **Rising Action:** Maya uses the system's integrated **Firecrawl** MCP to crawl the new docs. She watches as **Docling** parses the complex tables.
*   **Climax:** She triggers the **Graph Indexing Agent**. She uses the Generative UI to visualize the new nodes being added to the Neo4j graph, checking for "orphan nodes."
*   **Resolution:** The knowledge base is updated automatically. Any agent plugged into the system now has instant, structured access to the new API docs.

### Journey Requirements Summary

These journeys reveal we need:
1.  **Fast Ingestion CLI/API:** For Alex's < 15 min setup.
2.  **HITL UI Components:** For Sarah's source validation.
3.  **Observability Dashboard:** For Jordan's trajectory and cost tracking.
4.  **Graph Visualization & Scraping MCPs:** For Maya's quality control.

## Innovation & Novel Patterns

### Detected Innovation Areas

*   **Democratic Sophistication:** You are challenging the assumption that "sophisticated" Agentic and GraphRAG systems require massive engineering teams or expensive SaaS subscriptions. By packaging Agno, GraphRAG, and Crawl4AI into a "drop-in" tool, you are democratizing high-end AI infrastructure.
*   **Fully Autonomous Open-Source Pipeline:** The combination of **Crawl4AI** (crawling) + **Docling** (structured parsing) + **Agno/GraphRAG** (indexing/reasoning) creates a first-of-its-kind, completely free and open-source autonomous knowledge loop.
*   **Standardized "Trust" Protocol:** Integrating Human-in-the-Loop (HITL) directly into the retrieval flow (via CopilotKit) creates a new paradigm where AI transparency is a feature, not a byproduct. You are making "verifiable AI" a plug-and-play component.
*   **Universal Orchestration Bridge:** By using the **AG-UI protocol**, you are creating a system that is agnostic to the underlying agent framework, allowing developers to switch from Agno to LangGraph or others without losing their UI or knowledge base integration.

### Market Context & Competitive Landscape

*   **Current State:** Most RAG solutions are either too simple (basic vector search) or too "locked-in" (proprietary enterprise platforms).
*   **Your Innovation:** You are filling the "Advanced + Open + Reusable" gap. Developers currently have to spend weeks stitching these components together; your project provides the "glue" as a pre-built infrastructure.

### Validation Approach

*   **Developer "Speed Test":** Validate by timing how quickly a new developer can go from a fresh project to a working GraphRAG answer using the Crawl4AI ingestion.
*   **Factual Accuracy Benchmarking:** Use the "LLM-as-a-Judge" pattern to compare the hybrid system's performance against traditional RAG across a set of complex, multi-hop questions.

### Risk Mitigation

*   **Scraping Complexity:** Since Crawl4AI is open-source, we will provide fallback configurations for common documentation platforms (GitBook, Docusaurus) to ensure high reliability.
*   **Graph Complexity:** We'll use "Agentic Indexing" to automate the extraction of entities, mitigating the risk of users needing to write complex Cypher queries manually.

## Developer Tool Specific Requirements

### Project-Type Overview
**Agentic Rag and Graphrag with copilot** functions as an "AI Infrastructure-in-a-Box." It separates the complex "Brain" (Agents + Graph) from the user's application, connecting them via standard protocols. This architecture ensures maximum compatibility across different tech stacks.

### Technical Architecture Considerations

**Core Architecture (The "Brain"):**
*   **Delivery Mechanism:** Docker Container (`bmad/agentic-rag-copilot`).
*   **Internal Stack:** Agno (Orchestration), Docling (Parsing), Neo4j/Memgraph (Graph), pgvector (Vector), FastAPI (Interface).
*   **Exposure:** Exposes endpoints via HTTP/WebSockets using AG-UI and MCP protocols.

**The "Bridge" (SDKs):**
*   **Frontend SDK:** `npm` package (`@bmad/copilot-rag`) for React/Next.js. Provides pre-built CopilotKit hooks that auto-connect to the local or remote Docker container.
*   **Backend SDK:** `pip` package (`bmad-agent-rag`) for Python. Allows deep customization of the agent orchestration logic if the user wants to extend the default behavior.

### Language & Platform Support Matrix

| Ecosystem | Support Level | Implementation |
| :--- | :--- | :--- |
| **Python** | **First-Class** | Core container + `pip` SDK for extension. |
| **TypeScript / React** | **First-Class** | `npm` SDK for UI integration + CopilotKit hooks. |
| **Docker / Container** | **Universal** | The primary deployment method; runs on Linux, Mac, Windows, Cloud. |
| **Other (Go, Rust, etc.)** | **Protocol-Level** | Supported via raw HTTP/WebSocket calls to the AG-UI/MCP endpoints. |

### Installation Methods

**1. The "Zero-Config" Start (Docker Compose):**
```yaml
services:
  rag-copilot:
    image: bmad/agentic-rag-copilot:latest
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "8000:8000"
```

**2. The Frontend Integration (React):**
```bash
npm install @bmad/copilot-rag
```
```jsx
import { RagCopilotProvider } from "@bmad/copilot-rag";

export default function App() {
  return (
    <RagCopilotProvider endpoint="http://localhost:8000">
      <YourApp />
    </RagCopilotProvider>
  );
}
```

### API Surface & Code Examples

**Standardized Protocols:**
*   **AG-UI Endpoint:** `/copilot/v1/stream` (Standard interaction stream).
*   **MCP Endpoint:** `/mcp/v1/tools` (Tool discovery for external agents).
*   **Ingestion API:** `POST /api/v1/ingest` (Accepts PDF/URL, triggers Docling + Graph Indexing).

### Migration Guide
*   **From "No AI":** Simply spin up the container and wrap the app in the Provider.
*   **From "Basic RAG":** Point the ingestion API to existing document sources. The system builds the Graph automatically.

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP Approach:** **Platform MVP** (The "Full Box"). We are delivering the complete foundational infrastructure—Agentic Reasoning, Graph Retrieval, and Interactive UI—as a single production-ready package.
**Resource Requirements:** High-skill build requiring expertise in Agentic Orchestration (Agno), Graph Data Modeling (Neo4j), and Full-Stack Integration (React/FastAPI).

### MVP Feature Set (Phase 1 - The "Full Box")

**Core User Journeys Supported:**
*   **The Developer:** < 15 min setup via Docker.
*   **The Researcher:** Human-in-the-Loop source validation.
*   **The Data Engineer:** Fully autonomous ingestion loop.
*   **The Ops Engineer:** Trajectory and cost tracking.

**Must-Have Capabilities:**
*   **Backend:** Agno Orchestrator + FastAPI + Docker packaging.
*   **Retrieval:** Hybrid GraphRAG (Neo4j/Memgraph) + Vector (pgvector).
*   **Ingestion:** Crawl4AI (Crawler) + Docling (Parser) + Agentic Indexer.
*   **Frontend:** CopilotKit integration with Generative UI and HITL components.
*   **Protocol:** Full support for MCP (tools) and A2A (collaboration).

### Post-MVP Features (Growth & Expansion)

**Phase 2 (Growth):**
*   **Multi-Modal Ingestion:** Direct processing of audio/video sources via Docling extensions.
*   **Cross-Cloud Orchestration:** Native support for distributed agent deployments across AWS/Azure/GCP.

**Phase 3 (Expansion):**
*   **Self-Healing Knowledge:** Agents that autonomously identify missing documentation and initiate their own crawling/indexing tasks.
*   **Natural Language Infrastructure:** A CLI that allows developers to configure the entire RAG stack using plain English.

### Revised Roadmap 2025 (Recommendations)

Aligned to `docs/recommendations_2025.md` and reflected in Epics 11-18.

*   **Epic 11: Code Cleanup & Migration** - Multi-provider config, adapter pattern, persistence hardening.
*   **Epic 12: Advanced Retrieval** - Cross-encoder reranking, contextual chunking, corrective grading.
*   **Epic 13: Enterprise Ingestion** - Apify/BrightData adapters, YouTube transcript ingestion, Crawl4AI tuning.
*   **Epic 14: Connectivity** - MCP server exposure and robust A2A protocol.
*   **Epic 15: Multimodal & Codebase** - Video/image ingestion and hallucination detection.
*   **Epic 16: Framework Agnosticism** - Headless agent protocol and adapters.
*   **Epic 17: Deployment & CLI** - Interactive install CLI, hardware detection, startup validation.
*   **Epic 18: Documentation & DevOps** - Provider guides, retrieval tuning, MCP usage, security automation.

### Risk Mitigation Strategy

**Technical Risks:** Complexity of syncing Graph and Vector states.
*   *Mitigation:* Use Agno's built-in session storage and standardized Pydantic models for state synchronization.
**Market Risks:** Competition from "black-box" enterprise RAG.
*   *Mitigation:* Double down on the "100% Open Source and Free" differentiator.
**Resource Risks:** Over-engineering.
*   *Mitigation:* Focus on the "Drop-in" experience first—ensure the default configurations work perfectly before adding custom toggles.

## Functional Requirements

### Integration & Installation (Developer Experience)
*   **FR1:** Developers can deploy the core system as a single Docker container with zero initial code.
*   **FR2:** Developers can configure the system using standard environment variables (API keys, database URLs).
*   **FR3:** React developers can integrate the "Copilot" UI into their applications using a dedicated npm package.
*   **FR4:** Python developers can extend the core agent logic using a dedicated pip package.
*   **FR5:** Systems using any programming language can interact with the core "Brain" via a standardized AG-UI protocol over HTTP/WebSockets.

### Agentic Orchestration & Reasoning
*   **FR6:** The system can autonomously plan a multi-step execution strategy based on complex user queries.
*   **FR7:** The agent can dynamically select the best retrieval method (Vector vs. Graph) for a given query.
*   **FR8:** The agent can use external "Tools" defined via the Model Context Protocol (MCP).
*   **FR9:** Multiple agents can collaborate and delegate tasks to each other using the A2A protocol.
*   **FR10:** The system maintains a persistent "Thought Trace" (trajectory) for every complex interaction.

### Hybrid Retrieval (GraphRAG + Vector)
*   **FR11:** The system can perform semantic similarity search across unstructured text chunks (Vector RAG).
*   **FR12:** The system can perform relationship-based traversal across structured knowledge (GraphRAG).
*   **FR13:** The system can synthesize a single coherent answer by combining results from both Vector and Graph databases.
*   **FR14:** The system can explain how it arrived at an answer by referencing the specific nodes and edges used from the knowledge graph.

### Ingestion & Knowledge Construction
*   **FR15:** Users can trigger an autonomous crawl of documentation websites using the integrated Crawl4AI tool.
*   **FR16:** The system can parse complex document layouts (tables, headers, footnotes) from PDFs using Docling.
*   **FR17:** An "Agentic Indexer" can autonomously extract entities and relationships from parsed text to build the knowledge graph.
*   **FR18:** Data Engineers can visualize the current state of the knowledge graph to identify gaps or orphan nodes.

### Interactive "Copilot" Interface
*   **FR19:** End-users can interact with the system through a pre-built chat sidebar.
*   **FR20:** The UI can dynamically render specialized components (e.g., Graph Visualizers) sent from the agent (Generative UI).
*   **FR21:** Users can review and approve/reject retrieved sources before the agent generates an answer (Human-in-the-Loop).
*   **FR22:** The agent can take "Frontend Actions" within the host application (e.g., highlighting text, opening modals).

### Operations & Observability
*   **FR23:** Ops Engineers can monitor the real-time cost of LLM interactions.
*   **FR24:** The system can intelligently route queries to different LLM models based on task complexity to optimize costs.
*   **FR25:** Developers can review the reasoning "trajectory" of past queries for debugging purposes.

## Non-Functional Requirements

### Performance
*   **NFR1:** Total end-to-end response time for a complex agentic query should aim for **< 10 seconds** (including reasoning and tool calls).
*   **NFR2:** The ingestion pipeline should process a standard 50-page documentation site in **< 5 minutes**.

### Security
*   **NFR3:** The system must ensure **strict logical isolation** between knowledge stores if used in a multi-tenant environment.
*   **NFR4:** Reasoning traces and internal "thought traces" must be encrypted at rest.

### Scalability
*   **NFR5:** The system must handle knowledge graphs with **up to 1 million nodes/edges** without significant retrieval degradation.
*   **NFR6:** The backend must support **50+ concurrent autonomous agent runs** on standard production hardware.

### Integration & Reliability
*   **NFR7:** The system must adhere 100% to the **MCP** and **AG-UI** specifications for future-proof interoperability.
*   **NFR8:** The system must support **stateless recovery**, allowing agents to resume their reasoning state after a container restart.
