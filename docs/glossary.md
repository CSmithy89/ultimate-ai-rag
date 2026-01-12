# Glossary and Terminology Reference

**Version:** 1.0
**Last Updated:** 2026-01-13
**Related Story:** 18-24

---

This glossary provides definitions for key terms and concepts used throughout the Agentic RAG + GraphRAG platform. Terms are organized alphabetically for easy reference.

---

## A

### A2A (Agent-to-Agent Protocol)
A protocol enabling communication and task delegation between AI agents. In this platform, A2A allows the orchestrator agent to delegate specialized tasks to sub-agents (retrieval, indexing) with context preservation and session management.

**Related terms:** [Agent](#agent), [Orchestrator Agent](#orchestrator-agent), [MCP](#mcp-model-context-protocol)

### A2UI (Agent-to-UI)
A widget rendering protocol that allows backend agents to emit declarative UI components (cards, tables, forms, charts) that are rendered in the frontend. Part of the AG-UI event stream.

**Related terms:** [AG-UI](#ag-ui-agent-ui-protocol), [Open-JSON-UI](#open-json-ui)

### Access Count
The number of times a memory has been retrieved or accessed. Used in importance scoring to boost frequently accessed memories during consolidation.

**Related terms:** [Memory Consolidation](#memory-consolidation), [Importance Scoring](#importance-scoring)

### ADR (Architecture Decision Record)
A document that captures an important architectural decision made along with its context and consequences. The platform maintains ADRs for GraphRAG architecture, protocol selection, Graphiti integration, and CopilotKit frontend choices.

**Related terms:** [Architecture](#architecture)

### AG-UI (Agent-UI Protocol)
The event streaming protocol used by CopilotKit for real-time communication between backend agents and the frontend UI. Events include `RUN_STARTED`, `TEXT_DELTA`, `STATE_DELTA`, `TOOL_CALL_START`, `TOOL_CALL_END`, and `RUN_FINISHED`.

**Related terms:** [CopilotKit](#copilotkit), [SSE](#sse-server-sent-events), [A2UI](#a2ui-agent-to-ui)

### Agent
An autonomous AI system that can perceive its environment, make decisions, and take actions to achieve goals. In this platform, agents are built using the Agno framework and include orchestrator, retrieval, and indexing agents.

**Related terms:** [Agno](#agno), [Orchestrator Agent](#orchestrator-agent), [Multi-Agent System](#multi-agent-system)

### Agent Skills
Pre-packaged capabilities that can be loaded into AI assistants like Claude Desktop or Claude Code. The platform provides skills for RAG queries, document ingestion, and knowledge graph operations.

**Related terms:** [MCP](#mcp-model-context-protocol), [Agent](#agent)

### Agno
The Python agent orchestration framework (v2.3.21) used for building and coordinating AI agents in this platform. Provides built-in support for trajectory logging, tool execution, and multi-agent coordination.

**Related terms:** [Agent](#agent), [Trajectory Logging](#trajectory-logging)

### APOC
A Neo4j plugin providing "Awesome Procedures on Cypher" - a library of useful procedures and functions for graph operations, data transformation, and integration tasks.

**Related terms:** [Neo4j](#neo4j), [Cypher](#cypher)

### Architecture
The high-level structure of the system including component organization, data flow patterns, and integration points. This platform uses a polyglot persistence architecture with separate stores for graph, vector, relational, and cache data.

**Related terms:** [Monorepo](#monorepo), [Microservice](#microservice)

### AST (Abstract Syntax Tree)
A tree representation of the syntactic structure of source code. Used by the codebase hallucination detector for symbol validation and API endpoint matching.

**Related terms:** [Codebase RAG](#codebase-rag), [Hallucination Detection](#hallucination-detection)

### Augmentation
In RAG, the process of enhancing an LLM prompt with retrieved context before generation. This includes formatting, filtering, and structuring retrieved documents for optimal model consumption.

**Related terms:** [RAG](#rag-retrieval-augmented-generation), [Context Window](#context-window)

---

## B

### BM25
A probabilistic ranking function used for lexical/keyword-based search. Part of hybrid retrieval, complementing semantic vector search with exact term matching.

**Related terms:** [Hybrid Retrieval](#hybrid-retrieval), [Sparse Vector](#sparse-vector), [BM42](#bm42)

### BM42
An enhanced sparse vector search algorithm implemented via fastembed. Provides improved lexical matching compared to traditional BM25 while maintaining efficiency.

**Related terms:** [BM25](#bm25), [Sparse Vector](#sparse-vector), [Fastembed](#fastembed)

### Bloom Filter
A probabilistic data structure for testing set membership. Used to optimize URL deduplication in large crawls (>10,000 pages) with minimal memory overhead.

**Related terms:** [Crawl4AI](#crawl4ai), [Ingestion Pipeline](#ingestion-pipeline)

---

## C

### Chain of Thought
A prompting technique where the model is encouraged to break down complex reasoning into step-by-step explanations. Improves accuracy on multi-step problems.

**Related terms:** [Prompt](#prompt), [LLM](#llm-large-language-model)

### Chunk
A segment of a document created during ingestion for embedding and retrieval. Chunk size balances context preservation with embedding quality and retrieval precision.

**Related terms:** [Chunking](#chunking), [Contextual Retrieval](#contextual-retrieval)

### Chunking
The process of splitting documents into smaller segments for embedding and retrieval. The platform supports various strategies including fixed-size, semantic, and hierarchical chunking.

**Related terms:** [Chunk](#chunk), [Parent-Child Chunking](#parent-child-chunking)

### Circuit Breaker
A design pattern that prevents cascading failures by detecting failures and encapsulating logic to prevent repeated failure during maintenance or outages. Used in Redis cache and external API integrations.

**Related terms:** [Resilience](#resilience), [Retry Pattern](#retry-pattern)

### Codebase RAG
RAG capabilities specialized for software repositories. Includes indexing code symbols, extracting function signatures, and providing code-aware context for developer questions.

**Related terms:** [RAG](#rag-retrieval-augmented-generation), [AST](#ast-abstract-syntax-tree)

### Cohere
An AI company providing embedding and reranking models. The platform integrates Cohere's Rerank v3.5 for cross-encoder reranking with support for 100+ languages.

**Related terms:** [Reranking](#reranking), [Cross-Encoder](#cross-encoder)

### ColBERT
"Contextualized Late Interaction over BERT" - a neural ranking model that uses token-level interactions for efficient and effective retrieval. Provides late-interaction reranking capability.

**Related terms:** [Reranking](#reranking), [BERT](#bert)

### Community Detection
Graph algorithms (Louvain, Leiden) that identify densely connected groups of nodes. Used for creating community summaries and enabling global query routing in GraphRAG.

**Related terms:** [Knowledge Graph](#knowledge-graph), [Louvain Algorithm](#louvain-algorithm), [Leiden Algorithm](#leiden-algorithm)

### Configuration Profile
Predefined sets of feature configurations (minimal, standard, enterprise) that simplify deployment setup. Each profile enables appropriate features for different use cases.

**Related terms:** [Environment Variable](#environment-variable)

### Context Window
The maximum number of tokens an LLM can process in a single request, including both input (prompt + context) and output (generated response). Models have varying context window sizes.

**Related terms:** [Token](#token), [LLM](#llm-large-language-model)

### Contextual Retrieval
A technique that enriches chunks with surrounding document context before embedding. Improves retrieval accuracy by 67% compared to naive chunking, with prompt caching reducing costs by ~90%.

**Related terms:** [Chunk](#chunk), [Embedding](#embedding), [Prompt Caching](#prompt-caching)

### CopilotKit
A React framework for building AI copilot interfaces. Provides chat UI components, state synchronization, and AG-UI protocol support for real-time agent interaction.

**Related terms:** [AG-UI](#ag-ui-agent-ui-protocol), [React](#react), [Frontend](#frontend)

### Corrective RAG (CRAG)
A retrieval pattern that evaluates retrieved document quality and triggers fallback strategies (web search, query expansion) when relevance scores fall below thresholds.

**Related terms:** [Grader](#grader), [Fallback Strategy](#fallback-strategy), [RAG](#rag-retrieval-augmented-generation)

### Cosine Similarity
A metric measuring the cosine of the angle between two vectors, used to determine semantic similarity between embeddings. Values range from -1 (opposite) to 1 (identical).

**Related terms:** [Embedding](#embedding), [Vector Search](#vector-search)

### Crawl4AI
The web crawling library used for URL ingestion. Supports JavaScript rendering, parallel crawling, stealth mode, and configurable rate limiting through crawl profiles.

**Related terms:** [Ingestion Pipeline](#ingestion-pipeline), [Crawl Profile](#crawl-profile)

### Crawl Profile
Configuration presets for web crawling (fast, thorough, stealth) that define rate limits, JavaScript handling, and user agent rotation for different site types.

**Related terms:** [Crawl4AI](#crawl4ai)

### Cross-Encoder
A neural model architecture that jointly encodes query and document pairs for relevance scoring. More accurate than bi-encoders but slower, typically used for reranking small candidate sets.

**Related terms:** [Reranking](#reranking), [Bi-Encoder](#bi-encoder)

### Cypher
The query language for Neo4j graph databases. Used for creating, reading, updating, and traversing graph data structures.

**Related terms:** [Neo4j](#neo4j), [Knowledge Graph](#knowledge-graph)

---

## D

### Decay
In memory systems, the gradual reduction of importance scores over time. Memories that are not accessed decay, eventually falling below cleanup thresholds.

**Related terms:** [Memory Consolidation](#memory-consolidation), [Importance Scoring](#importance-scoring)

### Delegation
In multi-agent systems, the process of an orchestrator agent assigning tasks to specialized sub-agents. Managed through the A2A protocol with context preservation.

**Related terms:** [A2A](#a2a-agent-to-agent-protocol), [Orchestrator Agent](#orchestrator-agent)

### Docker Compose
A tool for defining and running multi-container Docker applications. The platform uses Docker Compose for local development with PostgreSQL, Neo4j, Redis, and application services.

**Related terms:** [Container](#container), [Development Environment](#development-environment)

### Docling
A document processing service for parsing PDFs and extracting structured content including text, tables, and layout information. Version 2.66.0 is used in the platform.

**Related terms:** [Ingestion Pipeline](#ingestion-pipeline), [PDF Parsing](#pdf-parsing)

### DOMPurify
A JavaScript library for sanitizing HTML to prevent XSS attacks. Used in Open-JSON-UI rendering to ensure safe content display.

**Related terms:** [XSS](#xss-cross-site-scripting), [Security](#security)

### Dual-Level Retrieval
A LightRAG-inspired pattern that retrieves both low-level (specific entities) and high-level (themes, summaries) information for comprehensive query coverage.

**Related terms:** [GraphRAG](#graphrag), [Entity](#entity)

---

## E

### Edge
In graph terminology, a connection between two nodes representing a relationship. In Neo4j, edges have types (MENTIONS, AUTHORED_BY) and can have properties.

**Related terms:** [Node](#node), [Relationship](#relationship), [Knowledge Graph](#knowledge-graph)

### Embedding
A dense vector representation of text that captures semantic meaning. Generated by embedding models (OpenAI, Anthropic, local models) and stored in pgvector for similarity search.

**Related terms:** [Vector](#vector), [Semantic Search](#semantic-search), [pgvector](#pgvector)

### Entity
A distinct object, concept, or thing identified during knowledge graph construction. Entities become nodes in the graph with properties describing their characteristics.

**Related terms:** [Entity Extraction](#entity-extraction), [Node](#node), [Knowledge Graph](#knowledge-graph)

### Entity Extraction
The NLP process of identifying and classifying named entities (people, organizations, concepts, APIs) from text. Graphiti handles this automatically during episode ingestion.

**Related terms:** [Entity](#entity), [Graphiti](#graphiti), [NER](#ner-named-entity-recognition)

### Environment Variable
Configuration values set outside the application code. The platform uses environment variables for API keys, feature flags, database connections, and service configuration.

**Related terms:** [Configuration Profile](#configuration-profile)

### Episode
In Graphiti, a unit of information (document, conversation turn, event) that is processed into the knowledge graph. Episodes have timestamps enabling temporal queries.

**Related terms:** [Graphiti](#graphiti), [Temporal Query](#temporal-query)

---

## F

### Fallback Strategy
Alternative retrieval or generation approaches used when primary methods fail or return low-quality results. Includes web search (Tavily), query expansion, and alternative data sources.

**Related terms:** [Corrective RAG](#corrective-rag-crag), [Resilience](#resilience)

### Fastembed
A lightweight embedding library supporting BM42 sparse vectors and local embedding models. Provides CPU-optimized inference without GPU requirements.

**Related terms:** [Embedding](#embedding), [BM42](#bm42), [Sparse Vector](#sparse-vector)

### FastAPI
A modern Python web framework for building APIs. The platform backend is built on FastAPI, providing automatic OpenAPI documentation, async support, and Pydantic integration.

**Related terms:** [Backend](#backend), [OpenAPI](#openapi), [Pydantic](#pydantic)

### Feature Flag
A configuration mechanism that enables or disables features without code changes. Advanced retrieval features (reranking, CRAG, contextual retrieval) are controlled via feature flags.

**Related terms:** [Environment Variable](#environment-variable), [Configuration Profile](#configuration-profile)

### FlashRank
A CPU-optimized reranking library providing local cross-encoder models. Alternative to cloud-based Cohere reranking for cost-sensitive deployments.

**Related terms:** [Reranking](#reranking), [Cross-Encoder](#cross-encoder)

### Frontend
The client-side portion of the application built with Next.js 15, React, and CopilotKit. Handles user interaction, state management, and visualization.

**Related terms:** [CopilotKit](#copilotkit), [Next.js](#nextjs), [React](#react)

---

## G

### Generation
The final stage of RAG where the LLM produces a response based on the original query and retrieved context. Quality depends on both retrieval accuracy and prompt design.

**Related terms:** [RAG](#rag-retrieval-augmented-generation), [LLM](#llm-large-language-model)

### Global Query
A GraphRAG query pattern that seeks broad, thematic information across the entire knowledge graph. Uses community summaries for comprehensive coverage.

**Related terms:** [Local Query](#local-query), [Community Detection](#community-detection), [GraphRAG](#graphrag)

### Grader
A component in Corrective RAG that evaluates the relevance of retrieved documents to the query. Can use heuristics or cross-encoder models to score document quality.

**Related terms:** [Corrective RAG](#corrective-rag-crag), [Cross-Encoder](#cross-encoder)

### Grafana
An observability platform for visualizing metrics and creating dashboards. The platform provides Grafana dashboard templates for retrieval quality metrics.

**Related terms:** [Prometheus](#prometheus), [Observability](#observability)

### Graph Database
A database optimized for storing and querying graph structures (nodes and edges). Neo4j is used as the graph database in this platform.

**Related terms:** [Neo4j](#neo4j), [Knowledge Graph](#knowledge-graph)

### GraphRAG
Retrieval-Augmented Generation enhanced with knowledge graph capabilities. Combines semantic search with graph traversal to provide relationship-aware retrieval.

**Related terms:** [RAG](#rag-retrieval-augmented-generation), [Knowledge Graph](#knowledge-graph), [Hybrid Retrieval](#hybrid-retrieval)

### Graphiti
Zep's temporal knowledge graph framework integrated in Epic 5. Handles entity extraction, graph building, temporal edge management, and hybrid retrieval with bi-temporal tracking.

**Related terms:** [Knowledge Graph](#knowledge-graph), [Temporal Query](#temporal-query), [Episode](#episode)

---

## H

### Hallucination
When an LLM generates information that is factually incorrect, inconsistent with provided context, or fabricated. RAG reduces hallucination by grounding responses in retrieved documents.

**Related terms:** [Hallucination Detection](#hallucination-detection), [Grounding](#grounding)

### Hallucination Detection
Automated validation that generated content is supported by retrieved sources. The codebase intelligence feature includes AST-based validation for code-related claims.

**Related terms:** [Hallucination](#hallucination), [Codebase RAG](#codebase-rag)

### Health Check
Endpoints (`/health`, `/ready`) that report service status for container orchestration. Used by Kubernetes and Docker for liveness and readiness probes.

**Related terms:** [Kubernetes](#kubernetes), [Docker Compose](#docker-compose)

### HITL (Human-in-the-Loop)
A workflow pattern where human approval is required for certain agent actions. The platform supports source validation through CopilotKit's useHumanInTheLoop hook.

**Related terms:** [Source Validation](#source-validation), [CopilotKit](#copilotkit)

### Hybrid Retrieval
A retrieval approach combining multiple methods (vector semantic search, keyword/BM25, graph traversal) for more comprehensive results than any single method alone.

**Related terms:** [Vector Search](#vector-search), [BM25](#bm25), [GraphRAG](#graphrag)

### Hybrid RAG
RAG systems that combine multiple retrieval strategies, typically vector-based semantic search with graph-based relationship traversal.

**Related terms:** [RAG](#rag-retrieval-augmented-generation), [Hybrid Retrieval](#hybrid-retrieval), [GraphRAG](#graphrag)

---

## I

### Iframe Sandbox
A security feature that restricts capabilities of embedded content. MCP-UI uses sandboxed iframes with origin validation for secure external tool rendering.

**Related terms:** [MCP-UI](#mcp-ui), [Security](#security)

### Importance Scoring
A numeric score (0.0-1.0) assigned to memories indicating their relative significance. Influenced by content relevance, access frequency, and time since creation.

**Related terms:** [Memory Consolidation](#memory-consolidation), [Decay](#decay)

### Indexing
The process of processing documents and storing them in searchable formats. Includes chunking, embedding generation, entity extraction, and graph construction.

**Related terms:** [Ingestion Pipeline](#ingestion-pipeline), [Embedding](#embedding)

### Ingestion Pipeline
The document processing workflow from source (URL, PDF, YouTube) through parsing, chunking, embedding, entity extraction, and graph construction.

**Related terms:** [Indexing](#indexing), [Crawl4AI](#crawl4ai), [Docling](#docling)

---

## J

### JSON-RPC 2.0
A remote procedure call protocol encoded in JSON. MCP uses JSON-RPC 2.0 for tool invocation over HTTP/SSE transport.

**Related terms:** [MCP](#mcp-model-context-protocol), [RPC](#rpc-remote-procedure-call)

### JSON Patch
A format (RFC 6902) for describing changes to JSON documents. Used in AG-UI `STATE_DELTA` events for efficient incremental state updates.

**Related terms:** [AG-UI](#ag-ui-agent-ui-protocol), [State Management](#state-management)

---

## K

### Knowledge Graph
A graph structure representing entities (nodes) and their relationships (edges) extracted from documents. Enables relationship-aware queries and explainable retrieval.

**Related terms:** [Neo4j](#neo4j), [Entity](#entity), [Relationship](#relationship), [Graphiti](#graphiti)

### Kubernetes
A container orchestration platform for automating deployment, scaling, and management. The platform provides Helm charts for Kubernetes deployment.

**Related terms:** [Docker Compose](#docker-compose), [Helm](#helm)

---

## L

### LangSmith
LangChain's observability platform for monitoring and debugging LLM applications. Integrated via Agno for trajectory visualization.

**Related terms:** [Observability](#observability), [Trajectory Logging](#trajectory-logging)

### LazyRAG
A query-time summarization pattern that defers expensive summarization until needed, reducing indexing costs by 99% while maintaining query-time performance.

**Related terms:** [GraphRAG](#graphrag), [Community Detection](#community-detection)

### Leiden Algorithm
A community detection algorithm that improves on Louvain by guaranteeing well-connected communities. Alternative algorithm for graph clustering.

**Related terms:** [Community Detection](#community-detection), [Louvain Algorithm](#louvain-algorithm)

### LLM (Large Language Model)
A neural network trained on large text corpora for natural language understanding and generation. Examples: GPT-4, Claude, Gemini. The platform supports multiple LLM providers.

**Related terms:** [Token](#token), [Context Window](#context-window), [Provider](#provider)

### Local Query
A GraphRAG query pattern focused on specific entities and their immediate relationships. Provides detailed, targeted information for entity-specific questions.

**Related terms:** [Global Query](#global-query), [GraphRAG](#graphrag)

### Louvain Algorithm
A community detection algorithm that optimizes modularity through iterative node movement. Used for identifying dense clusters in the knowledge graph.

**Related terms:** [Community Detection](#community-detection), [Leiden Algorithm](#leiden-algorithm)

---

## M

### MCP (Model Context Protocol)
A protocol for tools and context sharing with LLMs. Enables standardized tool invocation via JSON-RPC 2.0 over HTTP/SSE. The platform exposes RAG capabilities as MCP tools.

**Related terms:** [JSON-RPC 2.0](#json-rpc-20), [Tool](#tool), [A2A](#a2a-agent-to-agent-protocol)

### MCP-UI
A protocol for rendering external tool interfaces via secure iframe embedding. Includes origin validation, signed URLs, and postMessage communication.

**Related terms:** [MCP](#mcp-model-context-protocol), [Iframe Sandbox](#iframe-sandbox)

### Memory Consolidation
An automated process that cleans up the memory store by merging duplicates, applying decay, and removing low-importance memories. Runs on configurable schedules.

**Related terms:** [Memory Scope](#memory-scope), [Importance Scoring](#importance-scoring), [Decay](#decay)

### Memory Scope
A hierarchical level determining memory visibility and persistence. Scopes: session (ephemeral), user (persistent per-user), agent (per-agent), global (tenant-wide).

**Related terms:** [Memory Consolidation](#memory-consolidation), [Multi-Tenancy](#multi-tenancy)

### Metrics
Quantitative measurements of system behavior and performance. The platform exports Prometheus metrics for retrieval quality, API latency, and resource usage.

**Related terms:** [Prometheus](#prometheus), [Observability](#observability)

### Middleware
Code that runs between request receipt and route handling. The platform uses middleware for authentication, tenant isolation, rate limiting, and request tracing.

**Related terms:** [FastAPI](#fastapi), [Multi-Tenancy](#multi-tenancy)

### Model Routing
Intelligent selection of LLM models based on query complexity, cost constraints, or specialized capabilities. Optimizes cost-performance tradeoffs.

**Related terms:** [LLM](#llm-large-language-model), [Provider](#provider)

### Monorepo
A version control strategy where multiple projects are stored in a single repository. This platform uses a monorepo with `backend/` and `frontend/` directories.

**Related terms:** [Turborepo](#turborepo)

### MRR (Mean Reciprocal Rank)
A retrieval quality metric measuring the average of reciprocal ranks of the first relevant result. Higher MRR indicates better ranking of relevant documents.

**Related terms:** [NDCG](#ndcg-normalized-discounted-cumulative-gain), [Retrieval Quality](#retrieval-quality)

### Multi-Agent System
An architecture where multiple specialized agents collaborate to complete tasks. The platform uses orchestrator, retrieval, and indexing agents coordinated via A2A.

**Related terms:** [Agent](#agent), [A2A](#a2a-agent-to-agent-protocol), [Delegation](#delegation)

### Multi-Tenancy
An architecture where a single application instance serves multiple tenants (customers) with data isolation. Every database query must include `tenant_id` filtering.

**Related terms:** [Tenant Isolation](#tenant-isolation), [Namespace](#namespace)

---

## N

### Namespace
A logical partition for isolating resources. Used for multi-tenant data separation in databases and cache systems.

**Related terms:** [Multi-Tenancy](#multi-tenancy), [Tenant Isolation](#tenant-isolation)

### NDCG (Normalized Discounted Cumulative Gain)
A retrieval quality metric that measures ranking quality with position-based discounting. Accounts for graded relevance unlike binary metrics.

**Related terms:** [MRR](#mrr-mean-reciprocal-rank), [Retrieval Quality](#retrieval-quality)

### Neo4j
The graph database (version 5.x Community) used for storing knowledge graphs. Supports Cypher queries, full-text search, and vector indexes.

**Related terms:** [Knowledge Graph](#knowledge-graph), [Cypher](#cypher), [Graph Database](#graph-database)

### Next.js
A React framework for server-side rendering and static site generation. Version 15+ with App Router is used for the frontend.

**Related terms:** [React](#react), [Frontend](#frontend), [App Router](#app-router)

### Node
In graph terminology, a vertex representing an entity. Nodes have labels (Document, Entity, Chunk) and properties (name, createdAt).

**Related terms:** [Edge](#edge), [Entity](#entity), [Knowledge Graph](#knowledge-graph)

---

## O

### Observability
The ability to understand system internal state through external outputs. Includes logging, metrics, and tracing. The platform provides comprehensive observability hooks.

**Related terms:** [Prometheus](#prometheus), [Trajectory Logging](#trajectory-logging), [Metrics](#metrics)

### Ontology
A formal representation of knowledge as concepts and relationships within a domain. The platform supports OWL ontologies via owlready2 for structured domain modeling.

**Related terms:** [Knowledge Graph](#knowledge-graph), [OWL](#owl)

### Open-JSON-UI
A declarative UI protocol for rendering structured components (text, headings, code, tables, buttons, cards) from JSON specifications. Alternative to A2UI for simpler use cases.

**Related terms:** [A2UI](#a2ui-agent-to-ui), [AG-UI](#ag-ui-agent-ui-protocol)

### OpenAPI
A specification for describing REST APIs. FastAPI automatically generates OpenAPI 3.1 documentation for all endpoints.

**Related terms:** [FastAPI](#fastapi), [REST API](#rest-api)

### Orchestrator Agent
The primary agent that receives user queries, plans multi-step reasoning, and delegates to specialized sub-agents. Coordinates the overall query resolution process.

**Related terms:** [Agent](#agent), [Delegation](#delegation), [A2A](#a2a-agent-to-agent-protocol)

### OWL (Web Ontology Language)
A semantic web language for representing ontologies. Used for formal knowledge representation and reasoning.

**Related terms:** [Ontology](#ontology)

---

## P

### Parent-Child Chunking
A hierarchical chunking strategy where documents are split into large parent chunks and smaller child chunks. Small chunks are retrieved, but parent context is provided for generation.

**Related terms:** [Chunking](#chunking), [Chunk](#chunk)

### PDF Parsing
Extraction of text, tables, and structure from PDF documents. Handled by Docling with support for complex layouts and embedded images.

**Related terms:** [Docling](#docling), [Ingestion Pipeline](#ingestion-pipeline)

### pgvector
A PostgreSQL extension for vector similarity search. Stores embeddings and enables efficient nearest-neighbor queries using IVFFlat or HNSW indexes.

**Related terms:** [Vector](#vector), [Embedding](#embedding), [PostgreSQL](#postgresql)

### Point-in-Time Query
A temporal query that retrieves knowledge graph state as it existed at a specific historical date. Enabled by Graphiti's bi-temporal tracking.

**Related terms:** [Temporal Query](#temporal-query), [Graphiti](#graphiti)

### PostgreSQL
The relational database (version 16.x) used for structured data, sessions, and vector storage via pgvector extension.

**Related terms:** [pgvector](#pgvector), [Relational Database](#relational-database)

### Precision
A retrieval metric measuring the proportion of retrieved documents that are relevant. Precision@K measures precision in the top K results.

**Related terms:** [Recall](#recall), [Retrieval Quality](#retrieval-quality)

### Prometheus
An open-source monitoring system for collecting and querying metrics. The platform exports retrieval quality, API performance, and resource utilization metrics.

**Related terms:** [Metrics](#metrics), [Grafana](#grafana), [Observability](#observability)

### Prompt
The input text sent to an LLM, including system instructions, user query, and retrieved context. Prompt engineering optimizes this input for better responses.

**Related terms:** [LLM](#llm-large-language-model), [Context Window](#context-window)

### Prompt Caching
A cost optimization technique where repeated prompt prefixes are cached to reduce token costs. Especially effective for contextual retrieval (90% cost reduction).

**Related terms:** [Token](#token), [Contextual Retrieval](#contextual-retrieval)

### Protocol
A standardized specification for communication between components. The platform implements MCP, A2A, AG-UI, A2UI, MCP-UI, and Open-JSON-UI protocols.

**Related terms:** [MCP](#mcp-model-context-protocol), [A2A](#a2a-agent-to-agent-protocol), [AG-UI](#ag-ui-agent-ui-protocol)

### Provider
An external service supplying LLM or embedding capabilities. Supported providers: OpenAI, Anthropic, Google (Gemini), Ollama, OpenRouter.

**Related terms:** [LLM](#llm-large-language-model), [Embedding](#embedding)

### Pydantic
A Python library for data validation using type annotations. All backend data models are defined as Pydantic models.

**Related terms:** [Validation](#validation), [FastAPI](#fastapi)

---

## Q

### Query Expansion
A technique that reformulates or expands queries to improve retrieval coverage. Used as a fallback strategy in Corrective RAG.

**Related terms:** [Corrective RAG](#corrective-rag-crag), [Fallback Strategy](#fallback-strategy)

### Query Router
A component that determines whether a query should use global (thematic) or local (entity-specific) retrieval strategies.

**Related terms:** [Global Query](#global-query), [Local Query](#local-query)

---

## R

### RAG (Retrieval-Augmented Generation)
A technique that enhances LLM responses by retrieving relevant documents and including them in the prompt context. Reduces hallucination and enables knowledge updates without retraining.

**Related terms:** [Retrieval](#retrieval), [Augmentation](#augmentation), [Generation](#generation)

### Rate Limiting
Restricting the number of requests a client can make within a time window. Implemented via Redis for API protection and fair resource usage.

**Related terms:** [Redis](#redis), [Middleware](#middleware)

### React
A JavaScript library for building user interfaces. The frontend is built with React 18+ via Next.js.

**Related terms:** [Next.js](#nextjs), [Frontend](#frontend), [CopilotKit](#copilotkit)

### React Flow
A library for building node-based interactive diagrams. Used for knowledge graph visualization in the frontend.

**Related terms:** [Knowledge Graph](#knowledge-graph), [Frontend](#frontend)

### Recall
A retrieval metric measuring the proportion of relevant documents that were retrieved. High recall ensures comprehensive coverage.

**Related terms:** [Precision](#precision), [Retrieval Quality](#retrieval-quality)

### Redis
An in-memory data store (version 7.x) used for caching, message queues (Redis Streams), rate limiting, and session storage.

**Related terms:** [Cache](#cache), [Rate Limiting](#rate-limiting)

### Relationship
A connection between entities in a knowledge graph, representing how they relate (MENTIONS, AUTHORED_BY, CONTAINS). Stored as edges in Neo4j.

**Related terms:** [Edge](#edge), [Entity](#entity), [Knowledge Graph](#knowledge-graph)

### Reranking
A second-stage retrieval process that re-scores initial candidates using more expensive but accurate models (cross-encoders). Improves precision on top-K results.

**Related terms:** [Cross-Encoder](#cross-encoder), [Cohere](#cohere), [FlashRank](#flashrank)

### REST API
An architectural style for web APIs using HTTP methods (GET, POST, PUT, DELETE) and resource-based URLs. The platform follows REST conventions for all endpoints.

**Related terms:** [OpenAPI](#openapi), [FastAPI](#fastapi)

### Retrieval
The process of finding relevant documents from a corpus based on a query. Can use vector similarity, keyword matching, or graph traversal.

**Related terms:** [RAG](#rag-retrieval-augmented-generation), [Vector Search](#vector-search), [Hybrid Retrieval](#hybrid-retrieval)

### Retrieval Quality
Metrics measuring how well retrieval systems find relevant documents. Includes MRR, NDCG, Precision@K, and Recall@K.

**Related terms:** [MRR](#mrr-mean-reciprocal-rank), [NDCG](#ndcg-normalized-discounted-cumulative-gain), [Precision](#precision)

### Retry Pattern
A resilience pattern that automatically retries failed operations with exponential backoff. Used for LLM API calls and external service interactions.

**Related terms:** [Circuit Breaker](#circuit-breaker), [Resilience](#resilience)

### RFC 7807
"Problem Details for HTTP APIs" - a specification for error response format. All API errors follow this standard with type, title, status, detail, and instance fields.

**Related terms:** [REST API](#rest-api), [Error Handling](#error-handling)

---

## S

### Semantic Search
Retrieval based on meaning rather than exact keyword matching. Uses embeddings to find conceptually similar documents even with different terminology.

**Related terms:** [Embedding](#embedding), [Vector Search](#vector-search), [Cosine Similarity](#cosine-similarity)

### Session
A bounded interaction period between a user and the system. Sessions have IDs for tracking, can store ephemeral memories, and have configurable TTLs.

**Related terms:** [Memory Scope](#memory-scope), [State Management](#state-management)

### shadcn/ui
A collection of reusable React components built with Radix UI and Tailwind CSS. Provides the UI component foundation for the frontend.

**Related terms:** [React](#react), [Tailwind CSS](#tailwind-css)

### Source Validation
A HITL workflow where users approve or reject retrieved sources before they're used in generation. Implements human oversight for critical applications.

**Related terms:** [HITL](#hitl-human-in-the-loop), [Retrieval](#retrieval)

### Sparse Vector
A vector representation where most values are zero, capturing keyword/term presence. BM42 creates sparse vectors complementing dense semantic embeddings.

**Related terms:** [BM42](#bm42), [BM25](#bm25), [Dense Vector](#dense-vector)

### SSE (Server-Sent Events)
A web technology for servers to push updates to clients over HTTP. AG-UI uses SSE for real-time event streaming to the frontend.

**Related terms:** [AG-UI](#ag-ui-agent-ui-protocol), [WebSocket](#websocket)

### State Management
Handling application state across components and sessions. CopilotKit provides AG-UI state synchronization; TanStack Query manages server state.

**Related terms:** [TanStack Query](#tanstack-query), [CopilotKit](#copilotkit)

---

## T

### Tailwind CSS
A utility-first CSS framework for building custom designs. Used for styling throughout the frontend.

**Related terms:** [shadcn/ui](#shadcnui), [Frontend](#frontend)

### TanStack Query
A data fetching and caching library for React (formerly React Query). All frontend data fetching uses TanStack Query, never raw fetch().

**Related terms:** [React](#react), [State Management](#state-management)

### Tavily
A search API optimized for AI applications. Used as a web search fallback in Corrective RAG when local retrieval quality is insufficient.

**Related terms:** [Corrective RAG](#corrective-rag-crag), [Fallback Strategy](#fallback-strategy)

### Telemetry
Automated collection of usage and performance data. The platform supports telemetry hooks for analytics and debugging.

**Related terms:** [Observability](#observability), [Metrics](#metrics)

### Temporal Query
Queries that incorporate time dimensions, asking about knowledge state at specific dates or changes over time periods.

**Related terms:** [Point-in-Time Query](#point-in-time-query), [Graphiti](#graphiti)

### Tenant Isolation
Security controls ensuring one tenant cannot access another tenant's data. Enforced via tenant_id filtering on all database queries.

**Related terms:** [Multi-Tenancy](#multi-tenancy), [Security](#security)

### Token
The basic unit of text processing for LLMs. Text is tokenized into subword units; pricing and context limits are measured in tokens.

**Related terms:** [Context Window](#context-window), [LLM](#llm-large-language-model)

### Tool
A capability that an agent can invoke to perform actions or retrieve information. MCP standardizes tool definition and invocation.

**Related terms:** [MCP](#mcp-model-context-protocol), [Agent](#agent)

### Trajectory Logging
Recording the sequence of agent thoughts, actions, and observations during query processing. Essential for debugging and observability.

**Related terms:** [Agno](#agno), [Observability](#observability)

### TTS (Text-to-Speech)
Converting text to spoken audio. The platform supports TTS via OpenAI, ElevenLabs, or local pyttsx3.

**Related terms:** [STT](#stt-speech-to-text), [Voice I/O](#voice-io)

### Turborepo
A build system for JavaScript/TypeScript monorepos. Orchestrates frontend build tasks with caching and parallelization.

**Related terms:** [Monorepo](#monorepo), [Frontend](#frontend)

---

## U

### useCopilotReadable
A CopilotKit hook that exposes application state to the AI agent, enabling context-aware responses.

**Related terms:** [CopilotKit](#copilotkit), [React Hook](#react-hook)

### useFrontendTool
A modern CopilotKit hook for defining frontend-executed tools. Replaces deprecated useCopilotAction patterns.

**Related terms:** [CopilotKit](#copilotkit), [Tool](#tool)

### useHumanInTheLoop
A CopilotKit hook for implementing approval workflows where human confirmation is required for agent actions.

**Related terms:** [HITL](#hitl-human-in-the-loop), [CopilotKit](#copilotkit)

### uv
A fast Python package manager written in Rust. The platform uses uv for dependency management, replacing pip for 10-100x faster installs.

**Related terms:** [Python](#python), [Package Manager](#package-manager)

---

## V

### Validation
Verifying that data conforms to expected schemas and constraints. Backend uses Pydantic; frontend uses Zod.

**Related terms:** [Pydantic](#pydantic), [Zod](#zod)

### Vector
A numerical array representing data in a high-dimensional space. Text embeddings are vectors enabling mathematical similarity comparisons.

**Related terms:** [Embedding](#embedding), [pgvector](#pgvector)

### Vector Database
A database optimized for storing and querying vector embeddings. pgvector extends PostgreSQL with vector capabilities.

**Related terms:** [pgvector](#pgvector), [Embedding](#embedding)

### Vector Search
Finding similar items by comparing vector embeddings, typically using cosine similarity or Euclidean distance.

**Related terms:** [Semantic Search](#semantic-search), [Cosine Similarity](#cosine-similarity)

### Voice I/O
Input/output via speech. Includes STT (Whisper) for transcription and TTS for synthesis.

**Related terms:** [STT](#stt-speech-to-text), [TTS](#tts-text-to-speech), [Whisper](#whisper)

---

## W

### WebSocket
A protocol for full-duplex communication over TCP. Alternative to SSE for real-time updates.

**Related terms:** [SSE](#sse-server-sent-events)

### Whisper
OpenAI's speech recognition model for STT transcription. Used for voice input in the platform.

**Related terms:** [STT](#stt-speech-to-text), [Voice I/O](#voice-io)

---

## X

### XSS (Cross-Site Scripting)
A security vulnerability where malicious scripts are injected into web pages. Prevented via DOMPurify sanitization in Open-JSON-UI.

**Related terms:** [Security](#security), [DOMPurify](#dompurify)

---

## Y

### YAML
A human-readable data serialization format. Used for configuration files, crawl profiles, and sprint status tracking.

**Related terms:** [Configuration Profile](#configuration-profile)

---

## Z

### Zod
A TypeScript-first schema validation library. All frontend data validation uses Zod schemas.

**Related terms:** [Validation](#validation), [TypeScript](#typescript)

---

## Additional Concepts

### Backend
The server-side portion of the application built with Python, FastAPI, and Agno. Handles API requests, agent orchestration, and database operations.

**Related terms:** [FastAPI](#fastapi), [Agno](#agno)

### BERT
Bidirectional Encoder Representations from Transformers - a foundational language model architecture. Cross-encoders and embedding models often build on BERT.

**Related terms:** [Cross-Encoder](#cross-encoder), [Embedding](#embedding)

### Bi-Encoder
A model architecture that encodes query and document independently, enabling fast retrieval from large corpora. Contrast with cross-encoders.

**Related terms:** [Cross-Encoder](#cross-encoder), [Embedding](#embedding)

### Cache
Temporary storage for frequently accessed data to reduce latency and load. Redis provides the caching layer with circuit breaker patterns.

**Related terms:** [Redis](#redis), [Circuit Breaker](#circuit-breaker)

### Container
A lightweight, standalone executable package including code and dependencies. The platform uses Docker containers for deployment.

**Related terms:** [Docker Compose](#docker-compose), [Kubernetes](#kubernetes)

### Dense Vector
A vector where most or all values are non-zero, capturing semantic meaning. Contrast with sparse vectors.

**Related terms:** [Embedding](#embedding), [Sparse Vector](#sparse-vector)

### Development Environment
The local setup for building and testing the application. Docker Compose orchestrates all required services.

**Related terms:** [Docker Compose](#docker-compose)

### Error Handling
Strategies for gracefully managing failures. The platform uses RFC 7807 error responses and AppError classes.

**Related terms:** [RFC 7807](#rfc-7807), [Circuit Breaker](#circuit-breaker)

### Grounding
Anchoring LLM responses in retrieved facts to reduce hallucination. RAG provides grounding by including source documents in context.

**Related terms:** [Hallucination](#hallucination), [RAG](#rag-retrieval-augmented-generation)

### Helm
A package manager for Kubernetes. The platform provides Helm charts for production deployment.

**Related terms:** [Kubernetes](#kubernetes)

### Microservice
An architectural style where applications are composed of small, independent services. The platform is primarily monolithic but supports A2A for distributed agents.

**Related terms:** [Architecture](#architecture), [A2A](#a2a-agent-to-agent-protocol)

### NER (Named Entity Recognition)
An NLP task identifying and classifying named entities in text. Graphiti handles NER during entity extraction.

**Related terms:** [Entity Extraction](#entity-extraction)

### Package Manager
A tool for installing and managing software dependencies. The platform uses uv (Python) and pnpm (JavaScript).

**Related terms:** [uv](#uv), [pnpm](#pnpm)

### pnpm
A fast, disk-efficient JavaScript package manager. Used for frontend dependency management.

**Related terms:** [Package Manager](#package-manager), [Frontend](#frontend)

### Python
The programming language (3.12+) used for the backend, agents, and tools.

**Related terms:** [Backend](#backend), [FastAPI](#fastapi)

### React Hook
A function for adding state and lifecycle features to React components. CopilotKit provides hooks for AI integration.

**Related terms:** [React](#react), [CopilotKit](#copilotkit)

### Relational Database
A database organized in tables with rows and columns. PostgreSQL stores structured data, sessions, and vectors.

**Related terms:** [PostgreSQL](#postgresql)

### Resilience
The ability of a system to handle and recover from failures. Implemented via circuit breakers, retries, and fallback strategies.

**Related terms:** [Circuit Breaker](#circuit-breaker), [Retry Pattern](#retry-pattern)

### RPC (Remote Procedure Call)
A protocol for executing procedures on remote systems. MCP uses JSON-RPC 2.0.

**Related terms:** [JSON-RPC 2.0](#json-rpc-20), [MCP](#mcp-model-context-protocol)

### Security
Measures protecting the system from unauthorized access and attacks. Includes tenant isolation, input validation, and XSS prevention.

**Related terms:** [Tenant Isolation](#tenant-isolation), [XSS](#xss-cross-site-scripting)

### STT (Speech-to-Text)
Converting spoken audio to text. The platform uses Whisper for transcription.

**Related terms:** [Whisper](#whisper), [Voice I/O](#voice-io)

### TypeScript
A typed superset of JavaScript. Version 5.x is used for all frontend code.

**Related terms:** [Frontend](#frontend), [JavaScript](#javascript)

### App Router
Next.js 13+ routing system using file-system based routing in the `app/` directory.

**Related terms:** [Next.js](#nextjs)

### JavaScript
The programming language for web browsers. TypeScript compiles to JavaScript.

**Related terms:** [TypeScript](#typescript)

---

## Index by Category

### RAG Terms
[RAG](#rag-retrieval-augmented-generation), [GraphRAG](#graphrag), [Hybrid RAG](#hybrid-rag), [Retrieval](#retrieval), [Augmentation](#augmentation), [Generation](#generation), [Corrective RAG](#corrective-rag-crag), [Contextual Retrieval](#contextual-retrieval)

### Vector Terms
[Embedding](#embedding), [Vector](#vector), [pgvector](#pgvector), [Semantic Search](#semantic-search), [Cosine Similarity](#cosine-similarity), [Dense Vector](#dense-vector), [Sparse Vector](#sparse-vector), [BM42](#bm42)

### Graph Terms
[Knowledge Graph](#knowledge-graph), [Entity](#entity), [Relationship](#relationship), [Community Detection](#community-detection), [Graphiti](#graphiti), [Node](#node), [Edge](#edge), [Neo4j](#neo4j), [Cypher](#cypher)

### Protocol Terms
[MCP](#mcp-model-context-protocol), [A2A](#a2a-agent-to-agent-protocol), [AG-UI](#ag-ui-agent-ui-protocol), [A2UI](#a2ui-agent-to-ui), [MCP-UI](#mcp-ui), [Open-JSON-UI](#open-json-ui), [JSON-RPC 2.0](#json-rpc-20), [SSE](#sse-server-sent-events)

### AI Terms
[LLM](#llm-large-language-model), [Token](#token), [Prompt](#prompt), [Context Window](#context-window), [Hallucination](#hallucination), [Agent](#agent), [Chain of Thought](#chain-of-thought)

### Architecture Terms
[Multi-Tenancy](#multi-tenancy), [Trajectory Logging](#trajectory-logging), [Circuit Breaker](#circuit-breaker), [Rate Limiting](#rate-limiting), [Monorepo](#monorepo), [Microservice](#microservice)

### Retrieval Terms
[Reranking](#reranking), [Cross-Encoder](#cross-encoder), [BM25](#bm25), [Hybrid Retrieval](#hybrid-retrieval), [Grader](#grader), [Global Query](#global-query), [Local Query](#local-query)

### Platform Terms
[CopilotKit](#copilotkit), [FastAPI](#fastapi), [Neo4j](#neo4j), [Redis](#redis), [PostgreSQL](#postgresql), [Agno](#agno), [Docling](#docling), [Crawl4AI](#crawl4ai)

---

**Document Version:** 1.0
**Created:** 2026-01-13
**Maintainer:** Project Team
