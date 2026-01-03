# Epic 16 Tech Spec: Framework Agnosticism (Developer Extension Points)

**Date:** 2025-12-31
**Updated:** 2026-01-03 (Party Mode Analysis)
**Status:** Backlog
**Epic Owner:** Product and Engineering

---

## Overview

Epic 16 introduces a headless agent protocol to make the system framework-agnostic. It adds adapters for major agent frameworks so developers can **build on** their preferred framework.

### Key Decision (2026-01-03)

**These are DEVELOPER EXTENSION POINTS, not runtime switching.**

The CLI asks "Which framework?" and configures the project for that framework. Developers then build their application on top.

**Decision Document:** `docs/roadmap-decisions-2026-01-03.md`

### Framework Audience

| Framework | Target Developer | Key Benefit |
|-----------|-----------------|-------------|
| Agno | General-purpose | Default, battle-tested |
| PydanticAI | Type-safety enthusiasts | Type-safe agents, Pydantic validation |
| CrewAI | Multi-agent builders | Hierarchical process, crew orchestration |
| LangGraph | Workflow engineers | Stateful graphs, checkpointing |
| Anthropic Agent SDK | Claude-native developers | Agent Skills, computer use |

### Goals

- Define a stable agent protocol interface for core operations.
- Provide adapters for PydanticAI, CrewAI, Anthropic Agent SDK, and LangGraph.
- **Enable developers to build on their preferred framework.**

### Scope

**In scope**
- Agent protocol interface definition and reference implementation.
- Adapter implementations for target frameworks.
- Configuration setting to select active framework.
- **Agent Skills integration for Anthropic adapter (Story 16-4).**

**Out of scope**
- Changes to front-end UX.

---

## Stories

### Story 16-1: Define Headless Agent Protocol Interface

**Objective:** Create a framework-agnostic interface for agent execution.

**Acceptance Criteria**
- A protocol interface defines `run` and `stream` methods with typed inputs and outputs.
- The core orchestrator can target the protocol interface without framework-specific code.
- Protocol definitions are documented for external contributors.

### Story 16-2: Implement PydanticAI Adapter

**Objective:** Add adapter support for PydanticAI.

**Acceptance Criteria**
- Given `AGENT_FRAMEWORK=pydanticai`, the system runs via the PydanticAI adapter.
- Adapter maps system prompts, tools, and output formats correctly.
- Adapter includes unit tests for basic run and stream flows.

### Story 16-3: Implement CrewAI Adapter

**Objective:** Add adapter support for CrewAI.

**Acceptance Criteria**
- Given `AGENT_FRAMEWORK=crewai`, the system executes via CrewAI.
- Multi-agent orchestration works for at least one sample workflow.
- Adapter errors are surfaced using RFC 7807 format.

### Story 16-4: Implement Anthropic Agent Adapter

**Objective:** Add adapter support for Anthropic Agent SDK with Agent Skills integration.

**Key Addition (2026-01-03):** Include Agent Skills integration.

Agent Skills is an [open standard adopted by Microsoft/VS Code, Cursor, and others](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills). It provides organized folders of instructions, scripts, and resources that agents can discover and load dynamically.

**Agent Skills Integration:**
```
.skills/
├── rag-search/
│   ├── skill.yaml
│   ├── instructions.md
│   └── examples/
├── ingest-url/
│   ├── skill.yaml
│   └── instructions.md
└── explain-answer/
    ├── skill.yaml
    └── instructions.md
```

**Acceptance Criteria**
- Given `AGENT_FRAMEWORK=anthropic`, the system runs using Anthropic SDK tools.
- Tool invocation and streaming output are compatible with existing front-end expectations.
- Adapter supports configurable model selection.
- **Agent Skills are exposed for RAG capabilities (search, ingest, explain).**
- **Skills work with Claude.ai, Claude Code, and API.**

### Story 16-5: Implement LangGraph Adapter

**Objective:** Add adapter support for LangGraph.

**Acceptance Criteria**
- Given `AGENT_FRAMEWORK=langgraph`, the system executes a graph-based workflow.
- State transitions are logged and traceable in the trajectory store.
- Adapter includes basic integration tests.

---

## Technical Notes

- Use a factory pattern in orchestrator initialization to select adapters.
- Maintain a consistent `AgentResponse` structure across frameworks.
- **Each adapter should leverage framework-specific strengths:**
  - PydanticAI: Type-safe outputs, structured validation
  - CrewAI: Crew/agent/task hierarchy
  - LangGraph: StateGraph, checkpointing
  - Anthropic: Agent Skills, computer use

## Framework Capabilities Matrix

| Capability | PydanticAI | CrewAI | LangGraph | Anthropic |
|------------|------------|--------|-----------|-----------|
| Type-safe outputs | ✅ Native | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual |
| Multi-agent | ⚠️ Basic | ✅ Native | ✅ Native | ⚠️ Basic |
| Stateful workflows | ⚠️ Manual | ⚠️ Manual | ✅ Native | ⚠️ Manual |
| A2A/MCP | ✅ Native | ⚠️ Plugin | ⚠️ Plugin | ✅ Native |
| Agent Skills | ❌ | ❌ | ❌ | ✅ Native |

## Dependencies

- Multi-provider config system (Epic 11).
- A2A compatibility considerations (Epic 14).
- MCP wrapper (Epic 14) for tool exposure.

## Risks

- Framework API churn may require frequent adapter updates.
- Consistency between streaming formats across frameworks.
- **Mitigation:** Abstract common interface, version-pin dependencies.

## Success Metrics

- Users can switch frameworks via environment config without code changes.
- At least two frameworks pass the baseline integration test suite.
- **Agent Skills are discoverable from Claude Desktop/Cursor for Anthropic adapter.**

## References

- `docs/roadmap-decisions-2026-01-03.md` - Decision rationale
- [Anthropic Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [PydanticAI Documentation](https://context7.com/pydantic/pydantic-ai)
- [CrewAI Documentation](https://github.com/crewaiinc/crewai)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph)
- `_bmad-output/prd.md`
- `_bmad-output/architecture.md`
- `_bmad-output/project-planning-artifacts/epics.md`
- `docs/recommendations_2025.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
