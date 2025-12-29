from __future__ import annotations

from dataclasses import dataclass
import asyncio
import logging
import re
from uuid import UUID

from ..retrieval_router import RetrievalStrategy, select_retrieval_strategy
from ..db.neo4j import Neo4jClient
from ..db.postgres import PostgresClient
from ..indexing.embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingGenerator
from ..retrieval import GraphTraversalService, VectorSearchService
from ..retrieval.hybrid_synthesis import build_hybrid_prompt
from ..retrieval.types import GraphTraversalResult, VectorHit
from ..schemas import (
    GraphEdgeEvidence,
    GraphEvidence,
    GraphNodeEvidence,
    GraphPathEvidence,
    PlanStep,
    RetrievalEvidence,
    VectorCitation,
)
from ..trajectory import EventType, TrajectoryLogger

try:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
except ImportError:  # pragma: no cover - optional dependency at runtime
    Agent = None
    OpenAIChat = None

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Result payload returned by the orchestrator."""
    answer: str
    plan: list[PlanStep]
    thoughts: list[str]
    retrieval_strategy: RetrievalStrategy
    trajectory_id: UUID | None
    evidence: RetrievalEvidence | None = None


class OrchestratorAgent:
    """Run orchestration flow for user queries."""

    def __init__(
        self,
        api_key: str,
        model_id: str = "gpt-4o-mini",
        logger: TrajectoryLogger | None = None,
        postgres: PostgresClient | None = None,
        neo4j: Neo4jClient | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._agent = None
        self._logger = logger
        self._vector_search: VectorSearchService | None = None
        self._graph_traversal: GraphTraversalService | None = None
        if Agent and OpenAIChat:
            self._agent = Agent(model=OpenAIChat(api_key=api_key, id=model_id))
        if postgres:
            embedding_generator = EmbeddingGenerator(
                api_key=api_key,
                model=embedding_model,
            )
            self._vector_search = VectorSearchService(
                postgres=postgres,
                embedding_generator=embedding_generator,
            )
        if neo4j:
            self._graph_traversal = GraphTraversalService(neo4j=neo4j)

    async def run(
        self, query: str, tenant_id: str, session_id: str | None = None
    ) -> OrchestratorResult:
        """Run the orchestrator for a query and return the response payload."""
        trajectory_id = (
            await self._logger.start_trajectory(tenant_id, session_id)
            if self._logger
            else None
        )
        plan = self._build_plan(query)
        logger.debug("Generated plan with %s steps", len(plan))
        completed_plan, thoughts, events = self._execute_plan(plan)
        strategy = select_retrieval_strategy(query)
        strategy_note = f"Selected retrieval strategy: {strategy.value}"
        thoughts.append(strategy_note)
        events.append((EventType.ACTION, strategy_note))
        logger.debug("Retrieval strategy selected: %s", strategy.value)

        vector_hits: list[VectorHit] = []
        graph_result: GraphTraversalResult | None = None
        if strategy in {RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID}:
            vector_hits = await self._run_vector_search(query, tenant_id, events, thoughts)
        if strategy in {RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID}:
            graph_result = await self._run_graph_traversal(query, tenant_id, events, thoughts)

        evidence = self._build_evidence(vector_hits, graph_result)
        prompt = self._build_prompt(query, vector_hits, graph_result)

        if self._agent:
            response = await asyncio.to_thread(self._agent.run, prompt)
            content = getattr(response, "content", response)
            answer = str(content)
        else:
            logger.warning("Agno agent unavailable; returning placeholder response.")
            answer = f"Received query: {query}"

        events.append(
            (EventType.OBSERVATION, f"Generated response ({len(answer)} chars)")
        )

        if self._logger and trajectory_id:
            await self._logger.log_events(tenant_id, trajectory_id, events)

        return OrchestratorResult(
            answer=answer,
            plan=completed_plan,
            thoughts=thoughts,
            retrieval_strategy=strategy,
            trajectory_id=trajectory_id,
            evidence=evidence,
        )

    def _build_plan(self, query: str) -> list[PlanStep]:
        base_steps = [
            "Understand the question intent",
            "Select retrieval strategy",
            "Gather evidence",
            "Synthesize response",
        ]
        if self._has_token(query, "compare") or self._has_token(query, "versus"):
            base_steps = self._insert_after(
                base_steps,
                "Understand the question intent",
                "Break down into sub-questions",
            )

        if self._has_token(query, "if") or self._has_token(query, "depending"):
            base_steps = self._insert_after(
                base_steps,
                "Gather evidence",
                "Refine plan based on intermediate signals",
            )

        return [PlanStep(step=step, status="pending") for step in base_steps]

    def _has_token(self, query: str, token: str) -> bool:
        """Return True when a token appears as a whole word in the query."""
        return re.search(rf"\b{re.escape(token)}\b", query.lower()) is not None

    def _insert_after(self, steps: list[str], anchor: str, new_step: str) -> list[str]:
        """Insert a step after the anchor label when present."""
        if anchor not in steps:
            return steps + [new_step]
        index = steps.index(anchor)
        return steps[: index + 1] + [new_step] + steps[index + 1 :]

    def _execute_plan(
        self, plan: list[PlanStep]
    ) -> tuple[list[PlanStep], list[str], list[tuple[EventType, str]]]:
        """Execute a plan and return updated steps, thoughts, and log events."""
        thoughts: list[str] = []
        events: list[tuple[EventType, str]] = []
        completed_plan: list[PlanStep] = []
        for step in plan:
            thought = f"Plan step: {step.step}"
            thoughts.append(thought)
            events.append((EventType.THOUGHT, thought))
            completed_plan.append(PlanStep(step=step.step, status="completed"))
        return completed_plan, thoughts, events

    async def _run_vector_search(
        self,
        query: str,
        tenant_id: str,
        events: list[tuple[EventType, str]],
        thoughts: list[str],
    ) -> list[VectorHit]:
        if not self._vector_search:
            note = "Vector search unavailable; missing postgres or embedding generator."
            thoughts.append(note)
            events.append((EventType.OBSERVATION, note))
            return []
        events.append((EventType.ACTION, "Run vector semantic search"))
        hits = await self._vector_search.search(query, tenant_id)
        events.append((EventType.OBSERVATION, f"Vector hits: {len(hits)}"))
        return hits

    def _build_evidence(
        self,
        vector_hits: list[VectorHit],
        graph_result: GraphTraversalResult | None,
    ) -> RetrievalEvidence | None:
        if not vector_hits and not graph_result:
            return None
        return RetrievalEvidence(
            vector=self._build_vector_citations(vector_hits) if vector_hits else [],
            graph=self._build_graph_evidence(graph_result),
        )

    def _build_vector_citations(self, vector_hits: list[VectorHit]) -> list[VectorCitation]:
        citations: list[VectorCitation] = []
        for hit in vector_hits:
            source = None
            metadata = hit.metadata or {}
            if metadata:
                source = (
                    metadata.get("source_url")
                    or metadata.get("filename")
                    or metadata.get("source_type")
                )
            if not source:
                source = hit.document_id
            preview = hit.content.replace("\n", " ").strip()
            if len(preview) > 240:
                preview = preview[:240].rstrip() + "..."
            citations.append(
                VectorCitation(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    similarity=hit.similarity,
                    source=source,
                    content_preview=preview,
                    metadata=metadata or None,
                )
            )
        return citations

    def _build_prompt(
        self,
        query: str,
        vector_hits: list[VectorHit],
        graph_result: GraphTraversalResult | None,
    ) -> str:
        if not vector_hits and not graph_result:
            return query
        return build_hybrid_prompt(query, vector_hits, graph_result)

    async def _run_graph_traversal(
        self,
        query: str,
        tenant_id: str,
        events: list[tuple[EventType, str]],
        thoughts: list[str],
    ) -> GraphTraversalResult | None:
        if not self._graph_traversal:
            note = "Graph traversal unavailable; missing Neo4j client."
            thoughts.append(note)
            events.append((EventType.OBSERVATION, note))
            return None
        events.append((EventType.ACTION, "Run graph relationship traversal"))
        result = await self._graph_traversal.traverse(query, tenant_id)
        events.append((EventType.OBSERVATION, f"Graph paths: {len(result.paths)}"))
        return result

    def _build_graph_evidence(
        self,
        graph_result: GraphTraversalResult | None,
    ) -> GraphEvidence | None:
        if not graph_result:
            return None
        nodes = [
            GraphNodeEvidence(
                id=node.id,
                name=node.name,
                type=node.type,
                description=node.description,
                source_chunks=node.source_chunks or [],
            )
            for node in graph_result.nodes
        ]
        edges = [
            GraphEdgeEvidence(
                source_id=edge.source_id,
                target_id=edge.target_id,
                type=edge.type,
                confidence=edge.confidence,
                source_chunk=edge.source_chunk,
            )
            for edge in graph_result.edges
        ]
        paths = [
            GraphPathEvidence(node_ids=path.node_ids, edge_types=path.edge_types)
            for path in graph_result.paths
        ]
        explanation = self._build_graph_explanation(graph_result)
        if explanation is None:
            explanation = "No traversal paths found for the current query."
        return GraphEvidence(
            nodes=nodes,
            edges=edges,
            paths=paths,
            explanation=explanation,
        )

    def _build_graph_explanation(self, graph_result: GraphTraversalResult) -> str | None:
        if not graph_result.paths or not graph_result.nodes:
            return None
        name_map = {node.id: node.name or node.id for node in graph_result.nodes}
        explanations = []
        for path in graph_result.paths:
            segments = []
            for idx, node_id in enumerate(path.node_ids[:-1]):
                next_id = path.node_ids[idx + 1]
                edge_type = (
                    path.edge_types[idx]
                    if idx < len(path.edge_types)
                    else "RELATED_TO"
                )
                segments.append(
                    f"{name_map.get(node_id, node_id)} -[{edge_type}]-> {name_map.get(next_id, next_id)}"
                )
            if segments:
                explanations.append(" ".join(segments))
        if not explanations:
            return None
        return "; ".join(explanations)
