from fastapi import FastAPI

from .config import load_settings

settings = load_settings()

app = FastAPI(title="Agentic RAG Backend", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.backend_host, port=settings.backend_port)
