import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model_id: str
    database_url: str
    db_pool_min: int
    db_pool_max: int
    request_max_bytes: int
    rate_limit_per_minute: int
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    redis_url: str
    backend_host: str
    backend_port: int
    frontend_url: str


def load_settings() -> Settings:
    load_dotenv()

    try:
        backend_port = int(os.getenv("BACKEND_PORT", "8000"))
    except ValueError as exc:
        raise RuntimeError(
            "BACKEND_PORT must be a valid integer. Check your .env file."
        ) from exc
    try:
        db_pool_min = int(os.getenv("DB_POOL_MIN", "1"))
        db_pool_max = int(os.getenv("DB_POOL_MAX", "50"))
        request_max_bytes = int(os.getenv("REQUEST_MAX_BYTES", "1048576"))
        rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    except ValueError as exc:
        raise RuntimeError(
            "DB_POOL_MIN, DB_POOL_MAX, REQUEST_MAX_BYTES, and RATE_LIMIT_PER_MINUTE "
            "must be valid integers. Check your .env file."
        ) from exc
    if db_pool_min < 1 or db_pool_max < db_pool_min:
        raise RuntimeError(
            "DB_POOL_MIN must be >= 1 and DB_POOL_MAX must be >= DB_POOL_MIN."
        )
    if request_max_bytes < 1:
        raise RuntimeError("REQUEST_MAX_BYTES must be >= 1.")
    if rate_limit_per_minute < 1:
        raise RuntimeError("RATE_LIMIT_PER_MINUTE must be >= 1.")

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
        neo4j_uri=values["NEO4J_URI"],
        neo4j_user=values["NEO4J_USER"],
        neo4j_password=values["NEO4J_PASSWORD"],
        redis_url=values["REDIS_URL"],
        backend_host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        backend_port=backend_port,
        frontend_url=os.getenv("FRONTEND_URL", "http://localhost:3000"),
    )
