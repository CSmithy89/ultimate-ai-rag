from fastapi import FastAPI

app = FastAPI(title="Agentic RAG Backend", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
