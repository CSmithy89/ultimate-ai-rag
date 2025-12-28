from fastapi import FastAPI

from .config import load_settings
from .orchestrator import OrchestratorAgent
from .schemas import QueryRequest, QueryResponse

settings = load_settings()
orchestrator = OrchestratorAgent(api_key=settings.openai_api_key)

app = FastAPI(title="Agentic RAG Backend", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def run_query(payload: QueryRequest) -> QueryResponse:
    result = orchestrator.run(payload.query)
    return QueryResponse(answer=result.answer)


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.backend_host, port=settings.backend_port)
