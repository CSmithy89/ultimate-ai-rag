from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
import psycopg

from .config import Settings, load_settings
from .agents.orchestrator import OrchestratorAgent
from .schemas import QueryEnvelope, QueryRequest, QueryResponse, ResponseMeta
from .trajectory import TrajectoryLogger, close_pool, create_pool, ensure_trajectory_schema

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
    app.state.orchestrator = OrchestratorAgent(
        api_key=settings.openai_api_key,
        model_id=settings.openai_model_id,
        logger=trajectory_logger,
    )
    await run_in_threadpool(ensure_trajectory_schema, pool)
    yield
    await run_in_threadpool(close_pool, pool)


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    return FastAPI(title="Agentic RAG Backend", version="0.1.0", lifespan=lifespan)


app = create_app()


def get_settings(request: Request) -> Settings:
    """Provide settings from application state."""
    return request.app.state.settings


def get_orchestrator(request: Request) -> OrchestratorAgent:
    """Provide the orchestrator from application state."""
    return request.app.state.orchestrator


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryEnvelope)
async def run_query(
    payload: QueryRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
) -> QueryEnvelope:
    try:
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
    except psycopg.OperationalError as exc:
        logger.exception("Database unavailable")
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    except Exception as exc:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail="Query processing failed") from exc


def run() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run("agentic_rag_backend.main:app", host=settings.backend_host, port=settings.backend_port)
