"""Configuration management for the Agentic RAG Backend."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str
    database_url: str
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


def load_settings() -> Settings:
    """
    Load settings from environment variables.

    Returns:
        Settings instance with all configuration values

    Raises:
        RuntimeError: If required environment variables are missing
    """
    load_dotenv()

    try:
        backend_port = int(os.getenv("BACKEND_PORT", "8000"))
    except ValueError as exc:
        raise RuntimeError(
            "BACKEND_PORT must be a valid integer. Check your .env file."
        ) from exc

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
        database_url=values["DATABASE_URL"],
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
    )
