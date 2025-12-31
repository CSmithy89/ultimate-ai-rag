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
from .api.routes import (
    ingest_router,
    knowledge_router,
    copilot_router,
    workspace_router,
    mcp_router,
    a2a_router,
    ag_ui_router,
    ops_router,
)
from .api.routes.ingest import limiter as slowapi_limiter
from .api.utils import rate_limit_exceeded
from .config import Settings, load_settings
from .core.errors import AppError, app_error_handler, http_exception_handler
from .protocols.a2a import A2ASessionManager
from .protocols.ag_ui_bridge import HITLManager
from .protocols.mcp import MCPToolRegistry
from .rate_limit import InMemoryRateLimiter, RateLimiter, RedisRateLimiter, close_redis
from .schemas import QueryEnvelope, QueryRequest, QueryResponse, ResponseMeta
from .trajectory import TrajectoryLogger, close_pool, create_pool
from .ops import CostTracker, ModelRouter, TraceCrypto

logger = logging.getLogger(__name__)
struct_logger = structlog.get_logger(__name__)


def _should_skip_pool() -> bool:
    return os.getenv("SKIP_DB_POOL") == "1"


def _should_skip_graphiti() -> bool:
    return os.getenv("SKIP_GRAPHITI") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Initializes database connections on startup and closes them on shutdown.
    Stores clients in app.state for dependency injection.
    """
    settings = load_settings()
    app.state.settings = settings
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
                    openai_api_key=settings.openai_api_key,
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
        app.state.redis = None
        app.state.orchestrator = OrchestratorAgent(
            api_key=settings.openai_api_key,
            model_id=settings.openai_model_id,
            logger=None,
            postgres=getattr(app.state, "postgres", None),
            neo4j=getattr(app.state, "neo4j", None),
            embedding_model=settings.embedding_model,
            cost_tracker=getattr(app.state, "cost_tracker", None),
            model_router=getattr(app.state, "model_router", None),
        )
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
        else:
            app.state.redis = None
            app.state.rate_limiter = InMemoryRateLimiter(
                max_requests=settings.rate_limit_per_minute,
                window_seconds=60,
            )
        app.state.orchestrator = OrchestratorAgent(
            api_key=settings.openai_api_key,
            model_id=settings.openai_model_id,
            logger=app.state.trajectory_logger,
            postgres=getattr(app.state, "postgres", None),
            neo4j=getattr(app.state, "neo4j", None),
            embedding_model=settings.embedding_model,
            cost_tracker=getattr(app.state, "cost_tracker", None),
            model_router=getattr(app.state, "model_router", None),
        )

    app.state.a2a_manager = A2ASessionManager(
        session_ttl_seconds=settings.a2a_session_ttl_seconds,
        max_sessions_per_tenant=settings.a2a_max_sessions_per_tenant,
        max_sessions_total=settings.a2a_max_sessions_total,
        max_messages_per_session=settings.a2a_max_messages_per_session,
    )
    await app.state.a2a_manager.start_cleanup_task(
        settings.a2a_cleanup_interval_seconds
    )
    app.state.mcp_registry = MCPToolRegistry(
        orchestrator=app.state.orchestrator,
        neo4j=getattr(app.state, "neo4j", None),
        timeout_seconds=settings.mcp_tool_timeout_seconds,
        tool_timeouts=settings.mcp_tool_timeout_overrides,
        max_timeout_seconds=settings.mcp_tool_max_timeout_seconds,
    )

    yield

    # Shutdown: Close database connections
    if hasattr(app.state, "a2a_manager") and app.state.a2a_manager:
        await app.state.a2a_manager.stop_cleanup_task()
    # Epic 5: Graphiti connection
    if hasattr(app.state, "graphiti") and app.state.graphiti:
        await app.state.graphiti.disconnect()

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
