"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from .api.routes import ingest_router
from .config import load_settings
from .core.errors import AppError, app_error_handler

settings = load_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Initializes database connections on startup and closes them on shutdown.
    """
    # Startup: Initialize database clients
    from .db.postgres import get_postgres_client
    from .db.redis import get_redis_client

    try:
        await get_redis_client(settings.redis_url)
        await get_postgres_client(settings.database_url)
    except Exception:
        # Allow app to start even if databases aren't available
        # (useful for development/testing)
        pass

    yield

    # Shutdown: Close database connections
    from .db.postgres import close_postgres_client
    from .db.redis import close_redis_client

    await close_redis_client()
    await close_postgres_client()


app = FastAPI(
    title="Agentic RAG Backend",
    version="0.1.0",
    description="Backend API for the Agentic RAG + GraphRAG system",
    lifespan=lifespan,
)

# Register exception handlers
app.add_exception_handler(AppError, app_error_handler)

# Register routers
app.include_router(ingest_router, prefix="/api/v1")


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


def run() -> None:
    """Run the application with uvicorn."""
    import uvicorn

    uvicorn.run(app, host=settings.backend_host, port=settings.backend_port)
