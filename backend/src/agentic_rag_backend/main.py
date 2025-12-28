from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
import psycopg
from starlette.responses import JSONResponse

from pydantic import ValidationError

from .config import Settings, load_settings
from .agents.orchestrator import OrchestratorAgent
from .rate_limit import RateLimiter
from .schemas import QueryEnvelope, QueryRequest, QueryResponse, ResponseMeta
from .trajectory import TrajectoryLogger, close_pool, create_pool

logger = logging.getLogger(__name__)


def _should_skip_pool() -> bool:
    return os.getenv("SKIP_DB_POOL") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings

    if _should_skip_pool():
        app.state.pool = None
        app.state.trajectory_logger = None
        app.state.rate_limiter = RateLimiter(
            max_requests=settings.rate_limit_per_minute,
            window_seconds=60,
        )
        app.state.orchestrator = OrchestratorAgent(
            api_key=settings.openai_api_key,
            model_id=settings.openai_model_id,
            logger=None,
        )
        yield
        return

    pool = create_pool(settings.database_url, settings.db_pool_min, settings.db_pool_max)
    app.state.pool = pool
    trajectory_logger = TrajectoryLogger(pool=pool)
    app.state.trajectory_logger = trajectory_logger
    app.state.rate_limiter = RateLimiter(
        max_requests=settings.rate_limit_per_minute,
        window_seconds=60,
    )
    app.state.orchestrator = OrchestratorAgent(
        api_key=settings.openai_api_key,
        model_id=settings.openai_model_id,
        logger=trajectory_logger,
    )
    yield
    await run_in_threadpool(close_pool, pool)


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Agentic RAG Backend", version="0.1.0", lifespan=lifespan)
    install_middleware(app)
    app.include_router(router)
    return app


def get_settings(request: Request) -> Settings:
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
    return {"status": "ok"}


@router.post("/query", response_model=QueryEnvelope)
async def run_query(
    payload: QueryRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> QueryEnvelope:
    try:
        if not limiter.allow(payload.tenant_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        result = await run_in_threadpool(
            orchestrator.run,
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
        )
        meta = ResponseMeta(
            request_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
        )
        return QueryEnvelope(data=data, meta=meta)
    except ValidationError as exc:
        logger.exception("Validation error")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except psycopg.OperationalError as exc:
        logger.exception("Database unavailable")
        raise HTTPException(
            status_code=503, detail="Service temporarily unavailable"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error processing query")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


def run() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run("agentic_rag_backend.main:app", host=settings.backend_host, port=settings.backend_port)


app = create_app()
