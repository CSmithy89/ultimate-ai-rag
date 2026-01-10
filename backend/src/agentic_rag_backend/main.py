"""FastAPI application entry point."""

from contextlib import asynccontextmanager
import hashlib
from datetime import datetime, timezone
import logging
import os
from typing import AsyncGenerator, Awaitable, Callable, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
import psycopg
from pydantic import ValidationError
import redis.asyncio as redis
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse, Response
import structlog

from .agents.orchestrator import OrchestratorAgent
from .retrieval import (
    create_reranker_client,
    create_graph_reranker,
    get_reranker_adapter,
    get_graph_reranker_adapter,
    get_small_to_big_adapter,
    PostgresHierarchicalChunkStore,
)
from .retrieval.dual_level import DualLevelRetriever
from .retrieval.lazy_rag import LazyRAGRetriever
from .retrieval.query_router import QueryRouter
from .retrieval.grader import create_grader
from .retrieval.reranking import init_reranker_cache
from .api.routes import (
    ingest_router,
    knowledge_router,
    copilot_router,
    workspace_router,
    mcp_router,
    a2a_router,
    ag_ui_router,
    ops_router,
    codebase_router,
    memories_router,
    communities_router,
    lazy_rag_router,
    query_router,
    dual_level_router,
    telemetry_router,
)
from .mcp_server.routes import router as mcp_server_router
from .api.routes.ingest import limiter as slowapi_limiter
from .api.utils import rate_limit_exceeded
from .config import Settings, load_settings
from .llm import UnsupportedLLMProviderError, get_llm_adapter
from .core.errors import AppError, app_error_handler, http_exception_handler
from .protocols.a2a import A2ASessionManager
from .protocols.ag_ui_bridge import HITLManager
from .protocols.mcp import MCPToolRegistry
from .protocols.a2a_registry import A2AAgentRegistry, RegistryConfig
from .protocols.a2a_delegation import TaskDelegationManager, DelegationConfig
from .protocols.a2a_messages import get_implemented_rag_capabilities
from .memory.consolidation import MemoryConsolidator
from .memory.scheduler import create_consolidation_scheduler
from .rate_limit import InMemoryRateLimiter, RateLimiter, RedisRateLimiter, close_redis
from .schemas import QueryEnvelope, QueryRequest, QueryResponse, ResponseMeta
from .trajectory import TrajectoryLogger, close_pool, create_pool
from .ops import CostTracker, ModelRouter, TraceCrypto
from .observability import create_metrics_endpoint, MetricsConfig
from .voice import create_voice_adapter

logger = logging.getLogger(__name__)
struct_logger = structlog.get_logger(__name__)


def _should_skip_pool() -> bool:
    return os.getenv("SKIP_DB_POOL") == "1"


def _should_skip_graphiti() -> bool:
    return os.getenv("SKIP_GRAPHITI") == "1"


def _create_retrieval_enhancements(settings: Settings) -> tuple:
    """Create reranker and grader instances based on settings.

    Story 19-G1: Initializes the reranker cache if enabled.
    Story 19-G3: Model preloading is handled by the reranker/grader constructors.

    Returns:
        Tuple of (reranker, grader) - either may be None if disabled
    """
    reranker = None
    if settings.reranker_enabled:
        try:
            # Story 19-G1: Initialize reranker cache
            reranker_cache = init_reranker_cache(settings)
            struct_logger.info(
                "reranker_cache_initialized",
                enabled=settings.reranker_cache_enabled,
                ttl_seconds=settings.reranker_cache_ttl_seconds,
                max_size=settings.reranker_cache_max_size,
            )

            # Story 19-G3: Preloading is handled in get_reranker_adapter
            reranker_adapter = get_reranker_adapter(settings)
            reranker = create_reranker_client(reranker_adapter)
            struct_logger.info(
                "reranker_enabled",
                provider=settings.reranker_provider,
                model=settings.reranker_model,
                preload=settings.reranker_preload_model,
            )
        except Exception as e:
            struct_logger.warning("reranker_init_failed", error=str(e))

    grader = create_grader(settings)
    if grader:
        struct_logger.info(
            "grader_enabled",
            model=grader.get_model(),
            threshold=settings.grader_threshold,
            fallback_enabled=settings.grader_fallback_enabled,
            preload=settings.grader_preload_model,
            normalization_strategy=settings.grader_normalization_strategy,
        )

    return reranker, grader


def _create_small_to_big_adapter(settings: Settings, postgres_client) -> object:
    if postgres_client is None:
        return get_small_to_big_adapter(None, settings)
    chunk_store = PostgresHierarchicalChunkStore(postgres=postgres_client)
    return get_small_to_big_adapter(chunk_store, settings)


def _create_graph_reranker(settings: Settings, neo4j_client, graphiti_client) -> object:
    if neo4j_client is None:
        return None
    adapter = get_graph_reranker_adapter(settings)
    if not adapter.enabled:
        return None
    return create_graph_reranker(
        adapter=adapter,
        neo4j_client=neo4j_client,
        graphiti_client=graphiti_client,
    )


def _create_community_detector(settings: Settings, neo4j_client, rate_limiter) -> object:
    if not settings.community_detection_enabled or neo4j_client is None:
        return None

    try:
        from .graph.community import CommunityDetector, CommunityAlgorithm, NETWORKX_AVAILABLE
    except Exception as exc:  # pragma: no cover - optional dependency path
        struct_logger.warning("community_detector_import_failed", error=str(exc))
        return None

    if not NETWORKX_AVAILABLE:
        struct_logger.warning("community_detector_unavailable", reason="networkx not installed")
        return None

    algorithm = CommunityAlgorithm.LOUVAIN
    if settings.community_algorithm == "leiden":
        algorithm = CommunityAlgorithm.LEIDEN

    try:
        detector = CommunityDetector(
            neo4j_client=neo4j_client,
            llm_client=None,
            algorithm=algorithm,
            min_community_size=settings.community_min_size,
            max_hierarchy_levels=settings.community_max_levels,
            summary_model=settings.community_summary_model,
            rate_limiter=rate_limiter,
        )
        return detector
    except Exception as exc:
        struct_logger.warning("community_detector_init_failed", error=str(exc))
        return None


def _create_query_router(settings: Settings) -> object:
    if not settings.query_routing_enabled:
        return None
    return QueryRouter(settings)


def _create_lazy_rag_retriever(settings: Settings, graphiti_client, neo4j_client, community_detector, rate_limiter) -> object:
    if not settings.lazy_rag_enabled or neo4j_client is None:
        return None
    return LazyRAGRetriever(
        graphiti_client=graphiti_client,
        neo4j_client=neo4j_client,
        settings=settings,
        community_detector=community_detector,
        rate_limiter=rate_limiter,
    )


def _create_dual_level_retriever(settings: Settings, graphiti_client, neo4j_client, community_detector, rate_limiter) -> object:
    if not settings.dual_level_retrieval_enabled or neo4j_client is None:
        return None
    return DualLevelRetriever(
        graphiti_client=graphiti_client,
        neo4j_client=neo4j_client,
        settings=settings,
        community_detector=community_detector,
        rate_limiter=rate_limiter,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Initializes database connections on startup and closes them on shutdown.
    Stores clients in app.state for dependency injection.
    """
    settings = load_settings()
    app.state.settings = settings
    try:
        llm_adapter = get_llm_adapter(settings)
    except UnsupportedLLMProviderError as exc:
        struct_logger.error(
            "llm_provider_unsupported",
            provider=settings.llm_provider,
            error=str(exc),
        )
        raise RuntimeError(str(exc)) from exc
    key_bytes = bytes.fromhex(settings.trace_encryption_key)
    app.state.trace_key_fingerprint = hashlib.sha256(key_bytes).hexdigest()
    app.state.trace_key_source = (
        "env" if os.getenv("TRACE_ENCRYPTION_KEY") else "generated"
    )
    app.state.trace_crypto = TraceCrypto(settings.trace_encryption_key)

    # Epic 4: Initialize database clients for knowledge ingestion
    from .db.neo4j import Neo4jClient
    from .db.postgres import PostgresClient
    from .db.redis import RedisClient

    if _should_skip_pool():
        app.state.redis_client = None
        app.state.postgres = None
        app.state.neo4j = None
        app.state.cost_tracker = None
        app.state.model_router = ModelRouter(
            simple_model=settings.routing_simple_model,
            medium_model=settings.routing_medium_model,
            complex_model=settings.routing_complex_model,
            baseline_model=settings.routing_baseline_model,
            simple_max_score=settings.routing_simple_max_score,
            complex_min_score=settings.routing_complex_min_score,
        )
        struct_logger.info(
            "database_connections_skipped",
            reason="SKIP_DB_POOL",
        )
    else:
        try:
            # Initialize Redis (for knowledge ingestion)
            app.state.redis_client = RedisClient(settings.redis_url)
            await app.state.redis_client.connect()

            # Initialize PostgreSQL (for knowledge ingestion)
            app.state.postgres = PostgresClient(settings.database_url)
            await app.state.postgres.connect()
            await app.state.postgres.create_tables()
            app.state.cost_tracker = CostTracker(
                app.state.postgres.pool,
                pricing_json=settings.model_pricing_json,
            )
            app.state.model_router = ModelRouter(
                simple_model=settings.routing_simple_model,
                medium_model=settings.routing_medium_model,
                complex_model=settings.routing_complex_model,
                baseline_model=settings.routing_baseline_model,
                simple_max_score=settings.routing_simple_max_score,
                complex_min_score=settings.routing_complex_min_score,
            )

            # Initialize Neo4j
            app.state.neo4j = Neo4jClient(
                settings.neo4j_uri,
                settings.neo4j_user,
                settings.neo4j_password,
                pool_min_size=settings.neo4j_pool_min,
                pool_max_size=settings.neo4j_pool_max,
                pool_acquire_timeout=settings.neo4j_pool_acquire_timeout_seconds,
                connection_timeout=settings.neo4j_connection_timeout_seconds,
                max_connection_lifetime=settings.neo4j_max_connection_lifetime_seconds,
            )
            await app.state.neo4j.connect()
            await app.state.neo4j.create_indexes()

            struct_logger.info("database_connections_initialized")
        except Exception as e:
            struct_logger.warning("database_connection_failed", error=str(e))
            app.state.cost_tracker = None
            app.state.model_router = ModelRouter(
                simple_model=settings.routing_simple_model,
                medium_model=settings.routing_medium_model,
                complex_model=settings.routing_complex_model,
                baseline_model=settings.routing_baseline_model,
                simple_max_score=settings.routing_simple_max_score,
                complex_min_score=settings.routing_complex_min_score,
            )

    redis_client = getattr(app.state, "redis_client", None)
    app.state.hitl_manager = HITLManager(
        timeout=300.0,
        redis_client=redis_client,
    )
    struct_logger.info(
        "hitl_manager_initialized",
        storage="redis" if redis_client else "memory",
    )

    # Epic 5: Initialize Graphiti temporal knowledge graph
    app.state.graphiti = None
    if not _should_skip_graphiti():
        try:
            from .db.graphiti import GraphitiClient, GRAPHITI_AVAILABLE

            if GRAPHITI_AVAILABLE:
                graphiti_client = GraphitiClient(
                    uri=settings.neo4j_uri,
                    user=settings.neo4j_user,
                    password=settings.neo4j_password,
                    llm_provider=llm_adapter.provider,
                    llm_api_key=llm_adapter.api_key,
                    llm_base_url=llm_adapter.base_url,
                    embedding_provider=settings.embedding_provider,
                    embedding_api_key=settings.embedding_api_key,
                    embedding_base_url=settings.embedding_base_url,
                    embedding_model=settings.graphiti_embedding_model,
                    llm_model=settings.graphiti_llm_model,
                )
                await graphiti_client.connect()
                await graphiti_client.build_indices()
                app.state.graphiti = graphiti_client
                struct_logger.info("graphiti_initialized")
            else:
                struct_logger.warning("graphiti_not_available", reason="graphiti-core not installed")
        except Exception as e:
            struct_logger.warning("graphiti_initialization_failed", error=str(e))
            app.state.graphiti = None

    # Epic 2: Initialize query orchestrator components
    pool = None
    if _should_skip_pool():
        app.state.pool = None
        app.state.trajectory_logger = None
        app.state.rate_limiter = InMemoryRateLimiter(
            max_requests=settings.rate_limit_per_minute,
            window_seconds=60,
        )
        app.state.codebase_index_limiter = InMemoryRateLimiter(
            max_requests=settings.codebase_index_rate_limit_max,
            window_seconds=settings.codebase_index_rate_limit_window_seconds,
        )
        app.state.redis = None
        # Create retrieval enhancements (reranker, grader) if enabled
        reranker, grader = _create_retrieval_enhancements(settings)
        small_to_big_adapter = _create_small_to_big_adapter(
            settings,
            getattr(app.state, "postgres", None),
        )
        graph_reranker = _create_graph_reranker(
            settings,
            getattr(app.state, "neo4j", None),
            getattr(app.state, "graphiti", None),
        )
        community_detector = _create_community_detector(
            settings,
            getattr(app.state, "neo4j", None),
            app.state.rate_limiter,
        )
        if community_detector:
            app.state.community_detector = community_detector
        query_router = _create_query_router(settings)
        lazy_rag_retriever = _create_lazy_rag_retriever(
            settings,
            getattr(app.state, "graphiti", None),
            getattr(app.state, "neo4j", None),
            community_detector,
            app.state.rate_limiter,
        )
        dual_level_retriever = _create_dual_level_retriever(
            settings,
            getattr(app.state, "graphiti", None),
            getattr(app.state, "neo4j", None),
            community_detector,
            app.state.rate_limiter,
        )

        app.state.orchestrator = OrchestratorAgent(
            api_key=llm_adapter.api_key or "",
            provider=llm_adapter.provider,
            model_id=settings.llm_model_id,
            base_url=llm_adapter.base_url,
            embedding_provider=settings.embedding_provider,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            logger=None,
            postgres=getattr(app.state, "postgres", None),
            neo4j=getattr(app.state, "neo4j", None),
            graphiti_client=getattr(app.state, "graphiti", None),
            embedding_model=settings.embedding_model,
            cost_tracker=getattr(app.state, "cost_tracker", None),
            model_router=getattr(app.state, "model_router", None),
            reranker=reranker,
            reranker_top_k=settings.reranker_top_k,
            grader=grader,
            small_to_big_adapter=small_to_big_adapter,
            graph_reranker=graph_reranker,
            query_router=query_router,
            lazy_rag_retriever=lazy_rag_retriever,
            dual_level_retriever=dual_level_retriever,
        )
        app.state.reranker = reranker
    else:
        pool = create_pool(settings.database_url, settings.db_pool_min, settings.db_pool_max)
        try:
            await pool.open()
            app.state.pool = pool
        except Exception as e:
            struct_logger.warning("pool_open_failed", error=str(e))
            await pool.close()
            pool = None
            app.state.pool = None

        if pool:
            trajectory_logger = TrajectoryLogger(
                pool=pool,
                crypto=app.state.trace_crypto,
            )
            app.state.trajectory_logger = trajectory_logger
        else:
            app.state.trajectory_logger = None

        if settings.rate_limit_backend == "redis":
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            app.state.redis = redis_client
            app.state.rate_limiter = RedisRateLimiter(
                client=redis_client,
                max_requests=settings.rate_limit_per_minute,
                window_seconds=60,
                key_prefix=settings.rate_limit_redis_prefix,
            )
            app.state.codebase_index_limiter = RedisRateLimiter(
                client=redis_client,
                max_requests=settings.codebase_index_rate_limit_max,
                window_seconds=settings.codebase_index_rate_limit_window_seconds,
                key_prefix=f"{settings.rate_limit_redis_prefix}:codebase-index",
            )
            # LLM API rate limiter (100 requests per minute per tenant)
            app.state.llm_rate_limiter = RedisRateLimiter(
                client=redis_client,
                max_requests=100,
                window_seconds=60,
                key_prefix=f"{settings.rate_limit_redis_prefix}:llm",
            )
        else:
            app.state.redis = None
            app.state.rate_limiter = InMemoryRateLimiter(
                max_requests=settings.rate_limit_per_minute,
                window_seconds=60,
            )
            app.state.codebase_index_limiter = InMemoryRateLimiter(
                max_requests=settings.codebase_index_rate_limit_max,
                window_seconds=settings.codebase_index_rate_limit_window_seconds,
            )
            app.state.llm_rate_limiter = InMemoryRateLimiter(
                max_requests=100,
                window_seconds=60,
            )

        # Create retrieval enhancements (reranker, grader) if enabled
        reranker, grader = _create_retrieval_enhancements(settings)
        small_to_big_adapter = _create_small_to_big_adapter(
            settings,
            getattr(app.state, "postgres", None),
        )
        graph_reranker = _create_graph_reranker(
            settings,
            getattr(app.state, "neo4j", None),
            getattr(app.state, "graphiti", None),
        )
        community_detector = _create_community_detector(
            settings,
            getattr(app.state, "neo4j", None),
            app.state.llm_rate_limiter,
        )
        if community_detector:
            app.state.community_detector = community_detector
        query_router = _create_query_router(settings)
        lazy_rag_retriever = _create_lazy_rag_retriever(
            settings,
            getattr(app.state, "graphiti", None),
            getattr(app.state, "neo4j", None),
            community_detector,
            app.state.llm_rate_limiter,
        )
        dual_level_retriever = _create_dual_level_retriever(
            settings,
            getattr(app.state, "graphiti", None),
            getattr(app.state, "neo4j", None),
            community_detector,
            app.state.llm_rate_limiter,
        )

        app.state.orchestrator = OrchestratorAgent(
            api_key=llm_adapter.api_key or "",
            provider=llm_adapter.provider,
            model_id=settings.llm_model_id,
            base_url=llm_adapter.base_url,
            embedding_provider=settings.embedding_provider,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            logger=app.state.trajectory_logger,
            postgres=getattr(app.state, "postgres", None),
            neo4j=getattr(app.state, "neo4j", None),
            graphiti_client=getattr(app.state, "graphiti", None),
            embedding_model=settings.embedding_model,
            cost_tracker=getattr(app.state, "cost_tracker", None),
            model_router=getattr(app.state, "model_router", None),
            reranker=reranker,
            reranker_top_k=settings.reranker_top_k,
            grader=grader,
            small_to_big_adapter=small_to_big_adapter,
            graph_reranker=graph_reranker,
            query_router=query_router,
            lazy_rag_retriever=lazy_rag_retriever,
            dual_level_retriever=dual_level_retriever,
        )
        app.state.reranker = reranker

    app.state.a2a_manager = A2ASessionManager(
        session_ttl_seconds=settings.a2a_session_ttl_seconds,
        max_sessions_per_tenant=settings.a2a_max_sessions_per_tenant,
        max_sessions_total=settings.a2a_max_sessions_total,
        max_messages_per_session=settings.a2a_max_messages_per_session,
        redis_client=getattr(app.state, "redis_client", None),
    )
    await app.state.a2a_manager.start_cleanup_task(
        settings.a2a_cleanup_interval_seconds
    )

    # Epic 14: Initialize A2A agent registry and task delegation
    if settings.a2a_enabled:
        registry_config = RegistryConfig(
            heartbeat_interval_seconds=settings.a2a_heartbeat_interval_seconds,
            heartbeat_timeout_seconds=settings.a2a_heartbeat_timeout_seconds,
            cleanup_interval_seconds=settings.a2a_cleanup_interval_seconds,
        )
        app.state.a2a_registry = A2AAgentRegistry(
            config=registry_config,
            redis_client=getattr(app.state, "redis_client", None),
        )
        await app.state.a2a_registry.start_cleanup_task()

        delegation_config = DelegationConfig(
            default_timeout_seconds=settings.a2a_task_default_timeout_seconds,
            max_retries=settings.a2a_task_max_retries,
        )
        app.state.a2a_delegation_manager = TaskDelegationManager(
            registry=app.state.a2a_registry,
            config=delegation_config,
            redis_client=getattr(app.state, "redis_client", None),
        )

        # Self-register this agent's RAG capabilities in the registry
        # Use a default tenant for system-level registration
        default_tenant = "system"
        try:
            await app.state.a2a_registry.register_agent(
                agent_id=settings.a2a_agent_id,
                agent_type="rag_engine",
                endpoint_url=settings.a2a_endpoint_url,
                capabilities=get_implemented_rag_capabilities(),
                tenant_id=default_tenant,
                metadata={
                    "version": "0.1.0",
                    "self_registered": True,
                },
            )
            struct_logger.info(
                "a2a_agent_self_registered",
                agent_id=settings.a2a_agent_id,
                capabilities=[c.name for c in get_implemented_rag_capabilities()],
            )
        except Exception:
            # Use exception() to log full traceback for debugging startup issues
            struct_logger.exception(
                "a2a_self_registration_failed",
                agent_id=settings.a2a_agent_id,
            )

        struct_logger.info(
            "a2a_protocol_initialized",
            agent_id=settings.a2a_agent_id,
            endpoint_url=settings.a2a_endpoint_url,
            heartbeat_interval=settings.a2a_heartbeat_interval_seconds,
        )
    else:
        app.state.a2a_registry = None
        app.state.a2a_delegation_manager = None
        struct_logger.info("a2a_protocol_disabled")

    app.state.mcp_registry = MCPToolRegistry(
        orchestrator=app.state.orchestrator,
        neo4j=getattr(app.state, "neo4j", None),
        timeout_seconds=settings.mcp_tool_timeout_seconds,
        tool_timeouts=settings.mcp_tool_timeout_overrides,
        max_timeout_seconds=settings.mcp_tool_max_timeout_seconds,
    )

    # Epic 14: Initialize MCP server (dedicated MCP protocol endpoints)
    app.state.mcp_server = None
    try:
        from .mcp_server.server import MCPServerFactory
        from .mcp_server.tools import register_graphiti_tools, register_rag_tools

        mcp_server = MCPServerFactory.create_server(
            name="agentic-rag-mcp",
            version="1.0.0",
            enable_auth=True,
            rate_limit_requests=60,
            rate_limit_window=60,
            default_timeout=settings.mcp_tool_timeout_seconds,
        )

        graphiti_client = getattr(app.state, "graphiti", None)
        if graphiti_client and getattr(graphiti_client, "is_connected", False):
            register_graphiti_tools(mcp_server.registry, graphiti_client)

            vector_service = app.state.orchestrator.vector_search_service
            if vector_service:
                register_rag_tools(
                    registry=mcp_server.registry,
                    graphiti_client=graphiti_client,
                    vector_service=vector_service,
                    reranker=app.state.reranker,
                    retrieval_pipeline=app.state.orchestrator.retrieval_pipeline,
                )

        app.state.mcp_server = mcp_server
        struct_logger.info(
            "mcp_server_initialized",
            has_graphiti=graphiti_client is not None,
            has_vector_service=app.state.orchestrator.vector_search_service is not None,
        )
    except Exception as e:
        struct_logger.warning("mcp_server_init_failed", error=str(e))

    # Story 21-E1, 21-E2: Initialize Voice Adapter
    app.state.voice_adapter = None
    if settings.voice_io_enabled:
        # Validate OpenAI API key if using OpenAI TTS provider
        if settings.tts_provider == "openai" and not settings.openai_api_key:
            struct_logger.warning(
                "voice_adapter_skipped",
                reason="OpenAI TTS provider requires OPENAI_API_KEY",
            )
        else:
            try:
                app.state.voice_adapter = create_voice_adapter(
                    enabled=True,
                    whisper_model=settings.whisper_model,
                    tts_provider=settings.tts_provider,
                    tts_voice=settings.tts_voice,
                    tts_speed=settings.tts_speed,
                    openai_api_key=settings.openai_api_key,
                )
                struct_logger.info(
                    "voice_adapter_initialized",
                    whisper_model=settings.whisper_model,
                    tts_provider=settings.tts_provider,
                )
            except Exception as e:
                struct_logger.warning("voice_adapter_init_failed", error=str(e))

    # Story 20-A2: Initialize memory consolidation
    app.state.memory_consolidator = None
    app.state.memory_consolidation_scheduler = None

    if settings.memory_scopes_enabled and settings.memory_consolidation_enabled:
        try:
            # Get memory store (created lazily in memories.py, but we need it here)
            # Create it now if database clients are available
            postgres_client = getattr(app.state, "postgres", None)
            redis_client_for_memory = getattr(app.state, "redis_client", None)

            if postgres_client:
                from .memory import ScopedMemoryStore

                memory_store = ScopedMemoryStore(
                    postgres_client=postgres_client,
                    redis_client=redis_client_for_memory,
                    graphiti_client=getattr(app.state, "graphiti", None),
                    embedding_provider=settings.embedding_provider,
                    embedding_api_key=settings.embedding_api_key,
                    embedding_base_url=settings.embedding_base_url,
                    embedding_model=settings.embedding_model,
                    cache_ttl_seconds=settings.memory_cache_ttl_seconds,
                    max_per_scope=settings.memory_max_per_scope,
                    embedding_dimension=settings.embedding_dimension,
                )
                app.state.memory_store = memory_store

                # Create consolidator
                consolidator = MemoryConsolidator(
                    store=memory_store,
                    similarity_threshold=settings.memory_similarity_threshold,
                    decay_half_life_days=settings.memory_decay_half_life_days,
                    min_importance=settings.memory_min_importance,
                    consolidation_batch_size=settings.memory_consolidation_batch_size,
                )
                app.state.memory_consolidator = consolidator

                # Create and start scheduler
                scheduler = create_consolidation_scheduler(
                    consolidator=consolidator,
                    schedule=settings.memory_consolidation_schedule,
                    enabled=True,
                )
                await scheduler.start()
                app.state.memory_consolidation_scheduler = scheduler

                struct_logger.info(
                    "memory_consolidation_initialized",
                    schedule=settings.memory_consolidation_schedule,
                    similarity_threshold=settings.memory_similarity_threshold,
                    decay_half_life_days=settings.memory_decay_half_life_days,
                    min_importance=settings.memory_min_importance,
                )
            else:
                struct_logger.warning(
                    "memory_consolidation_skipped",
                    reason="PostgreSQL client not available",
                )
        except Exception as e:
            struct_logger.error(
                "memory_consolidation_init_failed",
                error=str(e),
                hint="Application will continue without memory consolidation",
            )
            # Ensure partial state is cleaned up
            app.state.memory_consolidator = None
            app.state.memory_consolidation_scheduler = None
    elif settings.memory_scopes_enabled:
        struct_logger.info(
            "memory_consolidation_disabled",
            hint="Set MEMORY_CONSOLIDATION_ENABLED=true to enable",
        )

    yield

    # Shutdown: Close database connections
    # Story 20-A2: Stop memory consolidation scheduler
    if hasattr(app.state, "memory_consolidation_scheduler") and app.state.memory_consolidation_scheduler:
        await app.state.memory_consolidation_scheduler.stop()
        struct_logger.info("memory_consolidation_scheduler_stopped")

    if hasattr(app.state, "a2a_manager") and app.state.a2a_manager:
        await app.state.a2a_manager.stop_cleanup_task()
    # Epic 14: Stop A2A registry cleanup task
    if hasattr(app.state, "a2a_registry") and app.state.a2a_registry:
        await app.state.a2a_registry.stop_cleanup_task()
    # Epic 14: Close A2A delegation manager HTTP client
    if hasattr(app.state, "a2a_delegation_manager") and app.state.a2a_delegation_manager:
        await app.state.a2a_delegation_manager.close()
    # Epic 5: Graphiti connection
    if hasattr(app.state, "graphiti") and app.state.graphiti:
        await app.state.graphiti.disconnect()

    # Story 21-E1, 21-E2: Close voice adapter
    if hasattr(app.state, "voice_adapter") and app.state.voice_adapter:
        await app.state.voice_adapter.close()
        struct_logger.info("voice_adapter_closed")

    # Epic 4 connections
    if hasattr(app.state, "neo4j") and app.state.neo4j:
        await app.state.neo4j.disconnect()
    if hasattr(app.state, "redis_client") and app.state.redis_client:
        await app.state.redis_client.disconnect()
    if hasattr(app.state, "postgres") and app.state.postgres:
        await app.state.postgres.disconnect()

    # Epic 2 connections
    if hasattr(app.state, "redis") and app.state.redis:
        await close_redis(app.state.redis)
    if hasattr(app.state, "pool") and app.state.pool:
        await close_pool(app.state.pool)

    struct_logger.info("database_connections_closed")


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Agentic RAG Backend",
        version="0.1.0",
        description="Backend API for the Agentic RAG + GraphRAG system",
        lifespan=lifespan,
    )
    install_middleware(app)

    # Register slowapi rate limiter for knowledge endpoints
    app.state.limiter = slowapi_limiter
    app.add_exception_handler(
        RateLimitExceeded,
        cast(Callable[[Request, Exception], Response], _rate_limit_exceeded_handler),
    )

    # Register exception handlers
    app.add_exception_handler(
        AppError,
        cast(Callable[[Request, Exception], Awaitable[Response]], app_error_handler),
    )
    app.add_exception_handler(
        HTTPException,
        cast(Callable[[Request, Exception], Awaitable[Response]], http_exception_handler),
    )

    # Register routers
    app.include_router(router)  # Query router
    app.include_router(ingest_router, prefix="/api/v1")  # Epic 4: Ingestion
    app.include_router(knowledge_router, prefix="/api/v1")  # Epic 4: Knowledge graph
    app.include_router(copilot_router, prefix="/api/v1")  # Epic 6: Copilot
    app.include_router(workspace_router, prefix="/api/v1")  # Epic 6: Workspace actions
    app.include_router(mcp_router, prefix="/api/v1")  # Epic 7: MCP tools
    app.include_router(a2a_router, prefix="/api/v1")  # Epic 7: A2A collaboration
    app.include_router(ag_ui_router, prefix="/api/v1")  # Epic 7: AG-UI universal
    app.include_router(ops_router, prefix="/api/v1")  # Epic 8: Ops dashboard
    app.include_router(codebase_router, prefix="/api/v1")  # Epic 15: Codebase intelligence
    app.include_router(memories_router, prefix="/api/v1")  # Epic 20: Memory Platform
    app.include_router(communities_router, prefix="/api/v1")  # Epic 20: Community Detection
    app.include_router(lazy_rag_router, prefix="/api/v1")  # Epic 20: LazyRAG Pattern
    app.include_router(query_router, prefix="/api/v1")  # Epic 20: Query Routing
    app.include_router(dual_level_router, prefix="/api/v1")  # Epic 20: Dual-Level Retrieval
    app.include_router(telemetry_router, prefix="/api/v1")  # Epic 21: Telemetry
    app.include_router(mcp_server_router)  # Epic 14: MCP Server (protocol endpoints)

    # Story 19-C5: Mount Prometheus metrics endpoint
    # Note: Settings are loaded fresh here since app.state.settings is set in lifespan
    # which runs after create_app. We load settings directly for endpoint registration.
    from .config import load_settings
    try:
        startup_settings = load_settings()
        metrics_config = MetricsConfig(
            enabled=startup_settings.prometheus_enabled,
            path=startup_settings.prometheus_path,
        )
        create_metrics_endpoint(app, metrics_config)
    except Exception as e:
        struct_logger.warning(
            "prometheus_metrics_init_failed",
            error=str(e),
            hint="Prometheus metrics disabled due to configuration error",
        )

    return app


def get_app_settings(request: Request) -> Settings:
    """Provide settings from application state."""
    return request.app.state.settings


def get_orchestrator(request: Request) -> OrchestratorAgent:
    """Provide the orchestrator from application state."""
    return request.app.state.orchestrator


def get_rate_limiter(request: Request) -> RateLimiter:
    """Provide the per-process rate limiter."""
    return request.app.state.rate_limiter


router = APIRouter()


def install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def enforce_request_size(request: Request, call_next):
        settings = request.app.state.settings
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.request_max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large"},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid content-length header"},
                )
        return await call_next(request)


@router.get("/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/query", response_model=QueryEnvelope)
async def run_query(
    payload: QueryRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> QueryEnvelope:
    try:
        if not await limiter.allow(payload.tenant_id):
            raise rate_limit_exceeded()
        result = await orchestrator.run(
            payload.query,
            payload.tenant_id,
            payload.session_id,
        )
        data = QueryResponse(
            answer=result.answer,
            plan=result.plan,
            thoughts=result.thoughts,
            retrieval_strategy=result.retrieval_strategy.value,
            trajectory_id=str(result.trajectory_id) if result.trajectory_id else None,
            evidence=result.evidence,
        )
        meta = ResponseMeta(
            requestId=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
        )
        return QueryEnvelope(data=data, meta=meta)
    except ValidationError as exc:
        logger.exception("Validation error")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except psycopg.OperationalError as exc:
        logger.exception("Database unavailable")
        raise HTTPException(
            status_code=503, detail="Service temporarily unavailable"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error processing query")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


def run() -> None:
    """Run the application with uvicorn."""
    import uvicorn

    settings = load_settings()
    uvicorn.run("agentic_rag_backend.main:app", host=settings.backend_host, port=settings.backend_port)


app = create_app()
