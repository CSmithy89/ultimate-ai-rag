import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    database_url: str
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
    )
