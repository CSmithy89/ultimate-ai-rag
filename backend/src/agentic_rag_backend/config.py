"""Configuration management for the Agentic RAG Backend."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv


# Search configuration constants
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 100

@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str
    openai_model_id: str
    database_url: str
    db_pool_min: int
    db_pool_max: int
    request_max_bytes: int
    rate_limit_per_minute: int
    rate_limit_backend: str
    rate_limit_redis_prefix: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    redis_url: str
    backend_host: str
    backend_port: int
    frontend_url: str
    # Epic 4 - Crawl settings
    crawl4ai_rate_limit: float
    # Story 4.2 - PDF Document Parsing settings
    docling_table_mode: str
    max_upload_size_mb: int
    temp_upload_dir: str
    docling_service_url: Optional[str]
    # Story 4.3 - Agentic Entity Extraction settings
    entity_extraction_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    entity_similarity_threshold: float
    # Epic 5 - Graphiti settings
    graphiti_embedding_model: str
    graphiti_llm_model: str
    ingestion_backend: str  # "graphiti" or "legacy"
    retrieval_backend: str  # "graphiti" or "legacy"


def load_settings() -> Settings:
    """
    Load settings from environment variables.

    Returns:
        Settings instance with all configuration values

    Raises:
        RuntimeError: If required environment variables are missing
    """
    load_dotenv()

    min_pool_size = 1
    try:
        backend_port = int(os.getenv("BACKEND_PORT", "8000"))
    except ValueError as exc:
        raise RuntimeError(
            "BACKEND_PORT must be a valid integer. Check your .env file."
        ) from exc
    try:
        db_pool_min = int(os.getenv("DB_POOL_MIN", str(min_pool_size)))
        db_pool_max = int(os.getenv("DB_POOL_MAX", "50"))
        request_max_bytes = int(os.getenv("REQUEST_MAX_BYTES", "1048576"))
        rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        rate_limit_backend = os.getenv("RATE_LIMIT_BACKEND", "memory")
        rate_limit_redis_prefix = os.getenv("RATE_LIMIT_REDIS_PREFIX", "rate-limit")
    except ValueError as exc:
        raise RuntimeError(
            "DB_POOL_MIN, DB_POOL_MAX, REQUEST_MAX_BYTES, and RATE_LIMIT_PER_MINUTE "
            "must be valid integers. Check your .env file."
        ) from exc
    if db_pool_min < min_pool_size or db_pool_max < db_pool_min:
        raise RuntimeError(
            "DB_POOL_MIN must be >= 1 and DB_POOL_MAX must be >= DB_POOL_MIN."
        )
    if request_max_bytes < 1:
        raise RuntimeError("REQUEST_MAX_BYTES must be >= 1.")
    if rate_limit_per_minute < 1:
        raise RuntimeError("RATE_LIMIT_PER_MINUTE must be >= 1.")
    if rate_limit_backend not in {"memory", "redis"}:
        raise RuntimeError("RATE_LIMIT_BACKEND must be 'memory' or 'redis'.")

    try:
        crawl4ai_rate_limit = float(os.getenv("CRAWL4AI_RATE_LIMIT", "1.0"))
    except ValueError:
        crawl4ai_rate_limit = 1.0

    try:
        max_upload_size_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
    except ValueError:
        max_upload_size_mb = 100

    # Story 4.3 settings parsing
    try:
        chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    except ValueError:
        chunk_size = 512

    try:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))
    except ValueError:
        chunk_overlap = 64

    try:
        entity_similarity_threshold = float(os.getenv("ENTITY_SIMILARITY_THRESHOLD", "0.95"))
    except ValueError:
        entity_similarity_threshold = 0.95

    # Epic 5 - Validate backend selections
    ingestion_backend = os.getenv("INGESTION_BACKEND", "graphiti")
    if ingestion_backend not in {"graphiti", "legacy"}:
        raise RuntimeError(
            f"Invalid INGESTION_BACKEND: {ingestion_backend}. "
            "Must be 'graphiti' or 'legacy'.")

    retrieval_backend = os.getenv("RETRIEVAL_BACKEND", "graphiti")
    if retrieval_backend not in {"graphiti", "legacy"}:
        raise RuntimeError(
            f"Invalid RETRIEVAL_BACKEND: {retrieval_backend}. "
            "Must be 'graphiti' or 'legacy'.")

    required = [
        "OPENAI_API_KEY",
        "DATABASE_URL",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "REDIS_URL",
    ]
    values = {key: os.getenv(key) for key in required}
    missing = [key for key, value in values.items() if not value]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise RuntimeError(
            "Missing required environment variables: "
            f"{missing_list}. Copy .env.example to .env and fill values."
        )

    return Settings(
        openai_api_key=values["OPENAI_API_KEY"],
        openai_model_id=os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini"),
        database_url=values["DATABASE_URL"],
        db_pool_min=db_pool_min,
        db_pool_max=db_pool_max,
        request_max_bytes=request_max_bytes,
        rate_limit_per_minute=rate_limit_per_minute,
        rate_limit_backend=rate_limit_backend,
        rate_limit_redis_prefix=rate_limit_redis_prefix,
        neo4j_uri=values["NEO4J_URI"],
        neo4j_user=values["NEO4J_USER"],
        neo4j_password=values["NEO4J_PASSWORD"],
        redis_url=values["REDIS_URL"],
        backend_host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        backend_port=backend_port,
        frontend_url=os.getenv("FRONTEND_URL", "http://localhost:3000"),
        crawl4ai_rate_limit=crawl4ai_rate_limit,
        # Story 4.2 - PDF Document Parsing settings
        docling_table_mode=os.getenv("DOCLING_TABLE_MODE", "accurate"),
        max_upload_size_mb=max_upload_size_mb,
        temp_upload_dir=os.getenv("TEMP_UPLOAD_DIR", "/tmp/uploads"),
        docling_service_url=os.getenv("DOCLING_SERVICE_URL"),
        # Story 4.3 - Agentic Entity Extraction settings
        entity_extraction_model=os.getenv("ENTITY_EXTRACTION_MODEL", "gpt-4o"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        entity_similarity_threshold=entity_similarity_threshold,
        # Epic 5 - Graphiti settings
        graphiti_embedding_model=os.getenv("GRAPHITI_EMBEDDING_MODEL", "text-embedding-3-small"),
        graphiti_llm_model=os.getenv("GRAPHITI_LLM_MODEL", "gpt-4o-mini"),
        ingestion_backend=ingestion_backend,
        retrieval_backend=retrieval_backend,
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once
    from environment variables.

    Returns:
        Cached Settings instance
    """
    return load_settings()
