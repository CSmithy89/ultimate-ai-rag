from __future__ import annotations

from dataclasses import dataclass
import inspect
import asyncio
import math
import re
from typing import Any, TYPE_CHECKING
from uuid import UUID

import structlog

from ..retrieval_router import RetrievalStrategy, select_retrieval_strategy
from ..db.neo4j import Neo4jClient
from ..db.postgres import PostgresClient
from ..embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingGenerator
from ..llm.providers import EmbeddingProviderAdapter, EmbeddingProviderType
from ..retrieval import (
    GraphTraversalService,
    VectorSearchService,
    RerankerClient,
    RetrievalPipeline,
    VectorSearchResult,
    SmallToBigAdapter,
    GraphReranker,
)
from ..retrieval.dual_level import DualLevelRetriever
from ..retrieval.lazy_rag import LazyRAGRetriever
from ..retrieval.query_router import QueryRouter
from ..retrieval.query_router_models import QueryType
from ..retrieval.grader import RetrievalGrader, RetrievalHit
from ..retrieval.constants import (
    DEFAULT_ENTITY_LIMIT,
    DEFAULT_MAX_HOPS,
    DEFAULT_PATH_LIMIT,
    DEFAULT_RETRIEVAL_CACHE_TTL_SECONDS,
    DEFAULT_RETRIEVAL_TIMEOUT_SECONDS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VECTOR_LIMIT,
)
from ..retrieval.hybrid_synthesis import build_hybrid_prompt
from ..retrieval.types import GraphNode, GraphPath, GraphTraversalResult, VectorHit
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
from ..ops import CostTracker, ModelRouter

if TYPE_CHECKING:
    from agno.agent import Agent as AgnoAgentType
else:  # pragma: no cover - typing only
    AgnoAgentType = Any

AgnoAgentImpl: type[Any] | None = None
AgnoOpenAIChatImpl: type[Any] | None = None
AgnoClaudeImpl: type[Any] | None = None
AgnoGeminiImpl: type[Any] | None = None

try:  # pragma: no cover - optional dependency at runtime
    from agno.agent import Agent as AgnoAgentImpl
except ImportError:  # pragma: no cover - optional dependency at runtime
    AgnoAgentImpl = None

try:  # pragma: no cover - optional dependency at runtime
    from agno.models.openai import OpenAIChat as AgnoOpenAIChatImpl
except ImportError:  # pragma: no cover - optional dependency at runtime
    AgnoOpenAIChatImpl = None

try:  # pragma: no cover - optional dependency at runtime
    from agno.models.anthropic import Claude as AgnoClaudeImpl
except ImportError:  # pragma: no cover - optional dependency at runtime
    AgnoClaudeImpl = None

try:  # pragma: no cover - optional dependency at runtime
    from agno.models.google import Gemini as AgnoGeminiImpl
except ImportError:  # pragma: no cover - optional dependency at runtime
    AgnoGeminiImpl = None

logger = structlog.get_logger(__name__)


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
        provider: str = "openai",
        model_id: str = "gpt-4o-mini",
        base_url: str | None = None,
        embedding_provider: str = "openai",
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        logger: TrajectoryLogger | None = None,
        cost_tracker: CostTracker | None = None,
        model_router: ModelRouter | None = None,
        postgres: PostgresClient | None = None,
        neo4j: Neo4jClient | None = None,
        graphiti_client: Any | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        vector_limit: int | None = None,
        vector_similarity_threshold: float | None = None,
        graph_max_hops: int | None = None,
        graph_path_limit: int | None = None,
        graph_entity_limit: int | None = None,
        retrieval_timeout_seconds: float | None = None,
        retrieval_cache_ttl_seconds: float | None = None,
        reranker: RerankerClient | None = None,
        reranker_top_k: int = 10,
        grader: RetrievalGrader | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
        small_to_big_adapter: SmallToBigAdapter | None = None,
        graph_reranker: GraphReranker | None = None,
        query_router: QueryRouter | None = None,
        lazy_rag_retriever: LazyRAGRetriever | None = None,
        dual_level_retriever: DualLevelRetriever | None = None,
    ) -> None:
        self._api_key = api_key
        self._provider = provider
        self._logger = logger
        self._cost_tracker = cost_tracker
        self._model_router = model_router
        self._model_id = model_id
        self._base_url = base_url
        self._embedding_provider = embedding_provider
        self._embedding_api_key = embedding_api_key or api_key
        self._embedding_base_url = embedding_base_url or base_url
        self._agents: dict[str, Any] = {}
        self._vector_search: VectorSearchService | None = None
        self._graph_traversal: GraphTraversalService | None = None
        self._reranker = reranker
        self._reranker_top_k = reranker_top_k
        self._grader = grader
        self._retrieval_pipeline: RetrievalPipeline | None = retrieval_pipeline
        self._query_router = query_router
        self._lazy_rag_retriever = lazy_rag_retriever
        self._dual_level_retriever = dual_level_retriever
        if postgres:
            # Create embedding generator using provider adapter
            embedding_adapter = EmbeddingProviderAdapter(
                provider=EmbeddingProviderType(embedding_provider),
                api_key=self._embedding_api_key,
                base_url=self._embedding_base_url,
                model=embedding_model,
            )
            embedding_generator = EmbeddingGenerator.from_adapter(
                adapter=embedding_adapter,
                cost_tracker=cost_tracker,
            )
            self._vector_search = VectorSearchService(
                postgres=postgres,
                embedding_generator=embedding_generator,
                limit=vector_limit if vector_limit is not None else DEFAULT_VECTOR_LIMIT,
                similarity_threshold=(
                    vector_similarity_threshold
                    if vector_similarity_threshold is not None
                    else DEFAULT_SIMILARITY_THRESHOLD
                ),
                timeout_seconds=(
                    retrieval_timeout_seconds
                    if retrieval_timeout_seconds is not None
                    else DEFAULT_RETRIEVAL_TIMEOUT_SECONDS
                ),
                cache_ttl_seconds=(
                    retrieval_cache_ttl_seconds
                    if retrieval_cache_ttl_seconds is not None
                    else DEFAULT_RETRIEVAL_CACHE_TTL_SECONDS
                ),
            )
        if neo4j:
            self._graph_traversal = GraphTraversalService(
                neo4j=neo4j,
                max_hops=graph_max_hops if graph_max_hops is not None else DEFAULT_MAX_HOPS,
                path_limit=(
                    graph_path_limit
                    if graph_path_limit is not None
                    else DEFAULT_PATH_LIMIT
                ),
                entity_limit=(
                    graph_entity_limit
                    if graph_entity_limit is not None
                    else DEFAULT_ENTITY_LIMIT
                ),
                timeout_seconds=(
                    retrieval_timeout_seconds
                    if retrieval_timeout_seconds is not None
                    else DEFAULT_RETRIEVAL_TIMEOUT_SECONDS
                ),
                cache_ttl_seconds=(
                    retrieval_cache_ttl_seconds
                    if retrieval_cache_ttl_seconds is not None
                    else DEFAULT_RETRIEVAL_CACHE_TTL_SECONDS
                ),
            )

        if self._retrieval_pipeline is None:
            self._retrieval_pipeline = RetrievalPipeline(
                vector_search=self._vector_search,
                graph_traversal=self._graph_traversal,
                graphiti_client=graphiti_client,
                reranker=self._reranker,
                reranker_top_k=self._reranker_top_k,
                small_to_big=small_to_big_adapter,
                graph_reranker=graph_reranker,
            )

    @property
    def vector_search_service(self) -> "VectorSearchService | None":
        """Public access to the vector search service for A2A capabilities."""
        return self._vector_search

    @property
    def retrieval_pipeline(self) -> RetrievalPipeline:
        """Public access to the unified retrieval pipeline."""
        if self._retrieval_pipeline is None:
            raise RuntimeError("Retrieval pipeline not initialized")
        return self._retrieval_pipeline

    def _build_chat_model(self, model_id: str) -> Any:
        if self._provider in {"openai", "openrouter", "ollama"}:
            model_cls = AgnoOpenAIChatImpl
        elif self._provider == "anthropic":
            model_cls = AgnoClaudeImpl
        elif self._provider == "gemini":
            model_cls = AgnoGeminiImpl
        else:
            raise RuntimeError(f"Unsupported LLM provider: {self._provider}")

        if model_cls is None:
            raise RuntimeError(
                f"Provider {self._provider!r} requires optional agno dependencies."
            )

        try:
            params = dict(inspect.signature(model_cls).parameters)
        except (TypeError, ValueError):  # pragma: no cover - fallback path
            params = {}

        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
        kwargs: dict[str, Any] = {}
        if "api_key" in params or accepts_kwargs:
            kwargs["api_key"] = self._api_key
        if "id" in params:
            kwargs["id"] = model_id
        elif "model" in params:
            kwargs["model"] = model_id
        elif "model_id" in params:
            kwargs["model_id"] = model_id
        else:
            kwargs["id"] = model_id

        if self._provider in {"openai", "openrouter", "ollama"} and self._base_url:
            if "base_url" in params or accepts_kwargs:
                kwargs["base_url"] = self._base_url
            elif "api_base" in params or accepts_kwargs:
                kwargs["api_base"] = self._base_url

        return model_cls(**kwargs)

    def _get_agent(self, model_id: str) -> AgnoAgentType | None:
        if AgnoAgentImpl is None:
            return None
        cached = self._agents.get(model_id)
        if cached:
            return cached
        try:
            agent = AgnoAgentImpl(model=self._build_chat_model(model_id))
        except RuntimeError as exc:
            logger.warning("agno_agent_unavailable", error=str(exc))
            return None
        self._agents[model_id] = agent
        return agent

    async def run(
        self, query: str, tenant_id: str, session_id: str | None = None
    ) -> OrchestratorResult:
        """Run the orchestrator for a query and return the response payload."""
        trajectory_id = (
            await self._logger.start_trajectory(tenant_id, session_id, "orchestrator")
            if self._logger
            else None
        )
        routing_decision = None
        selected_model_id = self._model_id
        baseline_model_id = None
        if self._model_router:
            routing_decision = self._model_router.route(query)
            selected_model_id = routing_decision.model_id
            baseline_model_id = routing_decision.baseline_model_id
            routing_note = (
                "Routing decision: "
                f"{routing_decision.complexity} "
                f"(score={routing_decision.score}) "
                f"model={routing_decision.model_id}"
            )
            if routing_decision.reason:
                routing_note += f" reason={','.join(routing_decision.reason)}"
        else:
            routing_note = f"Routing decision: default model={self._model_id}"
        plan = self._build_plan(query)
        logger.debug("orchestrator_plan_generated", steps=len(plan))
        completed_plan, thoughts, events = self._execute_plan(plan)
        thoughts.append(routing_note)
        events.append((EventType.ACTION, routing_note))
        strategy = select_retrieval_strategy(query)
        strategy_note = f"Selected retrieval strategy: {strategy.value}"
        thoughts.append(strategy_note)
        events.append((EventType.ACTION, strategy_note))
        logger.debug("retrieval_strategy_selected", strategy=strategy.value)

        vector_hits: list[VectorHit] = []
        graph_result: GraphTraversalResult | None = None
        routed = False
        if self._query_router:
            try:
                decision = await self._query_router.route(query, tenant_id)
                routing_note = (
                    f"Query routing: {decision.query_type.value} "
                    f"(confidence={decision.confidence:.2f}, "
                    f"method={decision.classification_method})"
                )
                thoughts.append(routing_note)
                events.append((EventType.ACTION, routing_note))

                if decision.query_type == QueryType.GLOBAL and self._lazy_rag_retriever:
                    lazy_result = await self._lazy_rag_retriever.query(
                        query=query,
                        tenant_id=tenant_id,
                        include_summary=True,
                    )
                    vector_hits = self._build_lazy_rag_hits(lazy_result)
                    routed = True
                elif (
                    decision.query_type == QueryType.HYBRID
                    and self._dual_level_retriever
                ):
                    dual_result = await self._dual_level_retriever.retrieve(
                        query=query,
                        tenant_id=tenant_id,
                        include_synthesis=True,
                    )
                    vector_hits = self._build_dual_level_hits(dual_result)
                    routed = True
            except Exception as exc:
                logger.warning("query_routing_failed", error=str(exc))
                thoughts.append(
                    f"Query routing failed; falling back to standard retrieval ({exc})"
                )

        if not routed and strategy == RetrievalStrategy.HYBRID:
            vector_task = self._run_vector_search(
                query,
                tenant_id,
                events,
                thoughts,
                trajectory_id,
                strategy=strategy.value,
            )
            graph_task = self._run_graph_traversal(
                query,
                tenant_id,
                events,
                thoughts,
                trajectory_id,
            )
            vector_hits, graph_result = await asyncio.gather(vector_task, graph_task)
        elif not routed:
            if strategy == RetrievalStrategy.VECTOR:
                vector_hits = await self._run_vector_search(
                    query,
                    tenant_id,
                    events,
                    thoughts,
                    trajectory_id,
                    strategy=strategy.value,
                )
            if strategy == RetrievalStrategy.GRAPH:
                graph_result = await self._run_graph_traversal(
                    query,
                    tenant_id,
                    events,
                    thoughts,
                    trajectory_id,
                )

        evidence = self._build_evidence(vector_hits, graph_result)
        prompt = self._build_prompt(query, vector_hits, graph_result)

        agent = self._get_agent(selected_model_id)
        if agent:
            response = await asyncio.to_thread(agent.run, prompt)
            content = getattr(response, "content", response)
            answer = str(content)
        else:
            logger.warning("agno_agent_unavailable")
            answer = f"Received query: {query}"

        events.append(
            (EventType.OBSERVATION, f"Generated response ({len(answer)} chars)")
        )

        if self._logger and trajectory_id:
            await self._logger.log_events(tenant_id, trajectory_id, events)

        if self._cost_tracker:
            try:
                await self._cost_tracker.record_usage(
                    tenant_id=tenant_id,
                    model_id=selected_model_id,
                    prompt=prompt,
                    completion=answer,
                    trajectory_id=trajectory_id,
                    complexity=routing_decision.complexity if routing_decision else None,
                    baseline_model_id=baseline_model_id,
                )
            except Exception as exc:  # pragma: no cover - non-critical telemetry
                logger.warning("cost_tracking_failed", error=str(exc))

        return OrchestratorResult(
            answer=answer,
            plan=completed_plan,
            thoughts=thoughts,
            retrieval_strategy=strategy,
            trajectory_id=trajectory_id,
            evidence=evidence,
        )

    def _build_lazy_rag_hits(self, result: Any) -> list[VectorHit]:
        hits: list[VectorHit] = []
        summary = getattr(result, "summary", None)
        confidence = float(getattr(result, "confidence", 0.8) or 0.8)

        if summary:
            hits.append(
                VectorHit(
                    chunk_id="lazy-rag-summary",
                    document_id="lazy-rag",
                    content=summary,
                    similarity=confidence,
                    metadata={
                        "source_type": "lazy_rag",
                        "seed_entity_count": getattr(result, "seed_entity_count", 0),
                        "expanded_entity_count": getattr(result, "expanded_entity_count", 0),
                        "confidence": confidence,
                    },
                )
            )

        entities = getattr(result, "entities", []) or []
        for entity in entities[:5]:
            content = getattr(entity, "description", None) or getattr(entity, "summary", None)
            if not content:
                content = getattr(entity, "name", "")
            hits.append(
                VectorHit(
                    chunk_id=f"lazy-rag-entity:{getattr(entity, 'id', '')}",
                    document_id="lazy-rag",
                    content=content,
                    similarity=0.5,
                    metadata={
                        "source_type": "lazy_rag",
                        "entity_id": getattr(entity, "id", ""),
                        "entity_name": getattr(entity, "name", ""),
                    },
                )
            )
        return hits

    def _build_dual_level_hits(self, result: Any) -> list[VectorHit]:
        hits: list[VectorHit] = []
        synthesis = getattr(result, "synthesis", None)
        confidence = float(getattr(result, "confidence", 0.7) or 0.7)

        if synthesis:
            hits.append(
                VectorHit(
                    chunk_id="dual-level-synthesis",
                    document_id="dual-level",
                    content=synthesis,
                    similarity=confidence,
                    metadata={
                        "source_type": "dual_level",
                        "confidence": confidence,
                    },
                )
            )

        low_level = getattr(result, "low_level_results", []) or []
        high_level = getattr(result, "high_level_results", []) or []

        for item in low_level[:5]:
            content = getattr(item, "content", None) or getattr(item, "name", "")
            hits.append(
                VectorHit(
                    chunk_id=f"dual-level-low:{getattr(item, 'id', '')}",
                    document_id="dual-level",
                    content=content,
                    similarity=float(getattr(item, "score", 0.0) or 0.0),
                    metadata={
                        "source_type": "dual_level",
                        "level": "low",
                        "entity_name": getattr(item, "name", ""),
                    },
                )
            )

        for item in high_level[:5]:
            content = getattr(item, "summary", None) or getattr(item, "name", "")
            hits.append(
                VectorHit(
                    chunk_id=f"dual-level-high:{getattr(item, 'id', '')}",
                    document_id="dual-level",
                    content=content,
                    similarity=float(getattr(item, "score", 0.0) or 0.0),
                    metadata={
                        "source_type": "dual_level",
                        "level": "high",
                        "theme_name": getattr(item, "name", ""),
                    },
                )
            )

        return hits

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
        trajectory_id: UUID | None,
        strategy: str = "vector",
    ) -> list[VectorHit]:
        """Run vector search and log retrieval events."""
        if not self._vector_search:
            note = "Vector search unavailable; missing postgres or embedding generator."
            thoughts.append(note)
            if self._logger and trajectory_id:
                await self._logger.log_observation(tenant_id, trajectory_id, note)
            else:
                events.append((EventType.OBSERVATION, note))
            return []
        thought_note = "Select vector semantic search"
        thoughts.append(thought_note)
        if self._logger and trajectory_id:
            await self._logger.log_thought(tenant_id, trajectory_id, thought_note)
        else:
            events.append((EventType.THOUGHT, thought_note))
        action_note = "Run vector semantic search"
        if self._logger and trajectory_id:
            await self._logger.log_action(tenant_id, trajectory_id, action_note)
        else:
            events.append((EventType.ACTION, action_note))
        try:
            vector_result: VectorSearchResult = await self._retrieval_pipeline.vector_search(
                query=query,
                tenant_id=tenant_id,
                use_reranking=self._reranker is not None,
                top_k=self._reranker_top_k,
                strategy=strategy,
            )
            hits = vector_result.hits
        except Exception as exc:
            error_note = f"Vector search failed: {exc}"
            if self._logger and trajectory_id:
                await self._logger.log_observation(tenant_id, trajectory_id, error_note)
            else:
                events.append((EventType.OBSERVATION, error_note))
            logger.warning("vector_search_failed", error=str(exc))
            return []

        observation = f"Vector hits: {len(hits)}"
        if self._logger and trajectory_id:
            await self._logger.log_observation(tenant_id, trajectory_id, observation)
        else:
            events.append((EventType.OBSERVATION, observation))

        # Epic 12: Apply reranking if enabled
        if vector_result.reranking_applied and vector_result.reranked:
            rerank_thought = f"Reranking {len(hits)} results with {self._reranker.get_model()}"
            thoughts.append(rerank_thought)
            if self._logger and trajectory_id:
                await self._logger.log_thought(tenant_id, trajectory_id, rerank_thought)
            else:
                events.append((EventType.THOUGHT, rerank_thought))

            rerank_action = "Run cross-encoder reranking"
            if self._logger and trajectory_id:
                await self._logger.log_action(tenant_id, trajectory_id, rerank_action)
            else:
                events.append((EventType.ACTION, rerank_action))

            # Log pre-rerank scores
            pre_rerank_scores = [
                f"{hit.content[:50]}... ({hit.similarity:.3f})"
                for hit in vector_result.original_hits[:5]
            ]
            pre_rerank_note = f"Pre-rerank top-5: {pre_rerank_scores}"
            if self._logger and trajectory_id:
                await self._logger.log_observation(tenant_id, trajectory_id, pre_rerank_note)
            else:
                events.append((EventType.OBSERVATION, pre_rerank_note))

            post_rerank_scores = [
                f"{r.hit.content[:50]}... ({r.rerank_score:.3f}, was rank {r.original_rank})"
                for r in vector_result.reranked[:5]
            ]
            post_rerank_note = f"Post-rerank top-5: {post_rerank_scores}"
            if self._logger and trajectory_id:
                await self._logger.log_observation(tenant_id, trajectory_id, post_rerank_note)
            else:
                events.append((EventType.OBSERVATION, post_rerank_note))

            rerank_observation = f"Reranked to {len(hits)} results"
            if self._logger and trajectory_id:
                await self._logger.log_observation(tenant_id, trajectory_id, rerank_observation)
            else:
                events.append((EventType.OBSERVATION, rerank_observation))

        # Epic 12: Apply CRAG grading if enabled
        if self._grader and hits:
            grader_thought = f"Grading retrieval quality with {self._grader.get_model()}"
            thoughts.append(grader_thought)
            if self._logger and trajectory_id:
                await self._logger.log_thought(tenant_id, trajectory_id, grader_thought)
            else:
                events.append((EventType.THOUGHT, grader_thought))

            grader_action = "Run CRAG grader evaluation"
            if self._logger and trajectory_id:
                await self._logger.log_action(tenant_id, trajectory_id, grader_action)
            else:
                events.append((EventType.ACTION, grader_action))

            try:
                # Convert VectorHit to RetrievalHit for grader
                grader_hits = [
                    RetrievalHit(
                        content=hit.content,
                        score=hit.similarity,
                        metadata=hit.metadata,
                    )
                    for hit in hits
                ]

                grader_result, fallback_hits = await self._grader.grade_and_fallback(
                    query=query,
                    hits=grader_hits,
                    tenant_id=tenant_id,
                )

                grade_observation = (
                    f"Grade: {grader_result.score:.3f} "
                    f"(threshold: {grader_result.threshold}, "
                    f"passed: {grader_result.passed}, "
                    f"grading_time: {grader_result.grading_time_ms}ms)"
                )
                if self._logger and trajectory_id:
                    await self._logger.log_observation(tenant_id, trajectory_id, grade_observation)
                else:
                    events.append((EventType.OBSERVATION, grade_observation))

                # If fallback was triggered, log it and append fallback hits
                if grader_result.fallback_triggered and fallback_hits:
                    strategy = (
                        grader_result.fallback_strategy.value
                        if grader_result.fallback_strategy
                        else "unknown"
                    )
                    fallback_note = (
                        f"Fallback triggered ({strategy}): "
                        f"added {len(fallback_hits)} additional results"
                    )
                    thoughts.append(fallback_note)
                    if self._logger and trajectory_id:
                        await self._logger.log_observation(tenant_id, trajectory_id, fallback_note)
                    else:
                        events.append((EventType.OBSERVATION, fallback_note))

                    # Convert fallback RetrievalHit back to VectorHit format
                    for fh in fallback_hits:
                        fallback_vector_hit = VectorHit(
                            chunk_id="fallback:0",
                            document_id=fh.metadata.get("url", "fallback") if fh.metadata else "fallback",
                            content=fh.content,
                            similarity=fh.score or 0.0,
                            metadata=fh.metadata,
                        )
                        hits.append(fallback_vector_hit)

            except Exception as exc:
                grader_error = f"Grading failed, continuing with ungraded results: {exc}"
                thoughts.append(grader_error)
                if self._logger and trajectory_id:
                    await self._logger.log_observation(tenant_id, trajectory_id, grader_error)
                else:
                    events.append((EventType.OBSERVATION, grader_error))
                logger.warning("grading_failed", error=str(exc))
                # Continue with original hits on grading failure

        return hits

    def _build_evidence(
        self,
        vector_hits: list[VectorHit],
        graph_result: GraphTraversalResult | None,
    ) -> RetrievalEvidence | None:
        """Build retrieval evidence payload for the API response."""
        if not vector_hits and not graph_result:
            return None
        return RetrievalEvidence(
            vector=self._build_vector_citations(vector_hits) if vector_hits else [],
            graph=self._build_graph_evidence(graph_result),
        )

    def _build_vector_citations(self, vector_hits: list[VectorHit]) -> list[VectorCitation]:
        """Build vector citations for API responses."""
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
            similarity = hit.similarity
            if not math.isfinite(similarity):
                logger.warning(
                    "vector_similarity_non_finite",
                    chunk_id=hit.chunk_id,
                    similarity=similarity,
                )
                similarity = 0.0
            elif similarity < 0 or similarity > 1:
                logger.warning(
                    "vector_similarity_out_of_range",
                    chunk_id=hit.chunk_id,
                    similarity=similarity,
                )
                similarity = min(max(similarity, 0.0), 1.0)
            citations.append(
                VectorCitation(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    similarity=similarity,
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
        trajectory_id: UUID | None,
    ) -> GraphTraversalResult | None:
        """Run graph traversal and log retrieval events."""
        if not self._graph_traversal:
            note = "Graph traversal unavailable; missing Neo4j client."
            thoughts.append(note)
            if self._logger and trajectory_id:
                await self._logger.log_observation(tenant_id, trajectory_id, note)
            else:
                events.append((EventType.OBSERVATION, note))
            return None
        thought_note = "Select graph relationship traversal"
        thoughts.append(thought_note)
        if self._logger and trajectory_id:
            await self._logger.log_thought(tenant_id, trajectory_id, thought_note)
        else:
            events.append((EventType.THOUGHT, thought_note))
        action_note = "Run graph relationship traversal"
        if self._logger and trajectory_id:
            await self._logger.log_action(tenant_id, trajectory_id, action_note)
        else:
            events.append((EventType.ACTION, action_note))
        try:
            result = await self._retrieval_pipeline.graph_traversal(query, tenant_id)
        except Exception as exc:
            error_note = f"Graph traversal failed: {exc}"
            if self._logger and trajectory_id:
                await self._logger.log_observation(tenant_id, trajectory_id, error_note)
            else:
                events.append((EventType.OBSERVATION, error_note))
            logger.warning("graph_traversal_failed", error=str(exc))
            return None
        observation = f"Graph paths: {len(result.paths)}"
        if self._logger and trajectory_id:
            await self._logger.log_observation(tenant_id, trajectory_id, observation)
        else:
            events.append((EventType.OBSERVATION, observation))
        return result

    def _build_graph_evidence(
        self,
        graph_result: GraphTraversalResult | None,
    ) -> GraphEvidence | None:
        """Build graph evidence payload for API responses."""
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
        valid_paths = self._filter_graph_paths(graph_result.paths)
        paths = [
            GraphPathEvidence(node_ids=path.node_ids, edge_types=path.edge_types)
            for path in valid_paths
        ]
        explanation = self._build_graph_explanation(graph_result.nodes, valid_paths)
        if explanation is None:
            explanation = "No traversal paths found for the current query."
        return GraphEvidence(
            nodes=nodes,
            edges=edges,
            paths=paths,
            explanation=explanation,
        )

    def _build_graph_explanation(
        self, nodes: list[GraphNode], paths: list[GraphPath]
    ) -> str | None:
        if not nodes and not paths:
            return "No graph nodes or paths found for the current query."
        if not nodes:
            return "No graph nodes found for the current query."
        if not paths:
            return "No graph paths found for the current query."
        name_map = {node.id: node.name or node.id for node in nodes}
        explanations = []
        for path in paths:
            segments = []
            for idx, node_id in enumerate(path.node_ids[:-1]):
                next_id = path.node_ids[idx + 1]
                edge_type = path.edge_types[idx]
                segments.append(
                    f"{name_map.get(node_id, node_id)} -[{edge_type}]-> {name_map.get(next_id, next_id)}"
                )
            if segments:
                explanations.append(" ".join(segments))
        if not explanations:
            return None
        return "; ".join(explanations)

    def _filter_graph_paths(self, paths: list[GraphPath]) -> list[GraphPath]:
        valid_paths: list[GraphPath] = []
        for path in paths:
            expected_edges = max(len(path.node_ids) - 1, 0)
            if len(path.edge_types) != expected_edges:
                logger.warning(
                    "graph_path_edge_mismatch",
                    node_count=len(path.node_ids),
                    edge_count=len(path.edge_types),
                    path=path.node_ids[:3],
                )
                continue
            valid_paths.append(path)
        return valid_paths
