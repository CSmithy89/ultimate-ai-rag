from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
import psycopg

from .config import load_settings
from .orchestrator import OrchestratorAgent
from .schemas import QueryRequest, QueryResponse
from .trajectory import TrajectoryLogger, close_pool, ensure_trajectory_schema, get_pool

settings = load_settings()
pool = get_pool(settings.database_url)
trajectory_logger = TrajectoryLogger(pool=pool)
orchestrator = OrchestratorAgent(
    api_key=settings.openai_api_key,
    model_id=settings.openai_model_id,
    logger=trajectory_logger,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(ensure_trajectory_schema, settings.database_url)
    yield
    await run_in_threadpool(close_pool)


app = FastAPI(title="Agentic RAG Backend", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def run_query(payload: QueryRequest) -> QueryResponse:
    try:
        result = await run_in_threadpool(
            orchestrator.run,
            payload.query,
            payload.tenant_id,
            payload.session_id,
        )
        return QueryResponse(
            answer=result.answer,
            plan=result.plan,
            thoughts=result.thoughts,
            retrieval_strategy=result.retrieval_strategy.value,
            trajectory_id=str(result.trajectory_id) if result.trajectory_id else None,
        )
    except psycopg.OperationalError as exc:
        logger.exception("Database unavailable")
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    except Exception as exc:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail="Query processing failed") from exc


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.backend_host, port=settings.backend_port)
