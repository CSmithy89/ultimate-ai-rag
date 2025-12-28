"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI

from .api.routes import ingest_router, knowledge_router
from .config import get_settings
from .core.errors import AppError, app_error_handler

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Initializes database connections on startup and closes them on shutdown.
    Stores clients in app.state for dependency injection.
    """
    # Startup: Initialize database clients
    from .db.neo4j import Neo4jClient
    from .db.postgres import PostgresClient
    from .db.redis import RedisClient

    settings = get_settings()
    app.state.settings = settings

    try:
        # Initialize Redis
        app.state.redis = RedisClient(settings.redis_url)
        await app.state.redis.connect()

        # Initialize PostgreSQL
        app.state.postgres = PostgresClient(settings.database_url)
        await app.state.postgres.connect()
        await app.state.postgres.create_tables()

        # Initialize Neo4j
        app.state.neo4j = Neo4jClient(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
        )
        await app.state.neo4j.connect()
        await app.state.neo4j.create_indexes()

        logger.info("database_connections_initialized")
    except Exception as e:
        logger.warning("database_connection_failed", error=str(e))

    yield

    # Shutdown: Close database connections
    if hasattr(app.state, "neo4j") and app.state.neo4j:
        await app.state.neo4j.disconnect()
    if hasattr(app.state, "redis") and app.state.redis:
        await app.state.redis.disconnect()
    if hasattr(app.state, "postgres") and app.state.postgres:
        await app.state.postgres.disconnect()

    logger.info("database_connections_closed")


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
app.include_router(knowledge_router, prefix="/api/v1")


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


def run() -> None:
    """Run the application with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.backend_host, port=settings.backend_port)
