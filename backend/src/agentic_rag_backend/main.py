from fastapi import FastAPI

from .config import load_settings
from .orchestrator import OrchestratorAgent
from .schemas import QueryRequest, QueryResponse
from .trajectory import TrajectoryLogger, ensure_trajectory_schema

settings = load_settings()
trajectory_logger = TrajectoryLogger(database_url=settings.database_url)
orchestrator = OrchestratorAgent(
    api_key=settings.openai_api_key,
    logger=trajectory_logger,
)

app = FastAPI(title="Agentic RAG Backend", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.on_event("startup")
def prepare_storage() -> None:
    ensure_trajectory_schema(settings.database_url)


@app.post("/query", response_model=QueryResponse)
def run_query(payload: QueryRequest) -> QueryResponse:
    result = orchestrator.run(payload.query, session_id=payload.session_id)
    return QueryResponse(
        answer=result.answer,
        plan=result.plan,
        thoughts=result.thoughts,
        retrieval_strategy=result.retrieval_strategy.value,
        trajectory_id=str(result.trajectory_id) if result.trajectory_id else None,
    )


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.backend_host, port=settings.backend_port)
