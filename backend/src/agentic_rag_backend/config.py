"""Configuration management for the Agentic RAG Backend."""

import json
import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, cast

from dotenv import load_dotenv
import structlog


def get_bool_env(key: str, default: str = "false") -> bool:
    """Parse a boolean environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set (default: "false")

    Returns:
        True if value is "true", "1", or "yes" (case-insensitive), False otherwise
    """
    return os.getenv(key, default).strip().lower() in {"true", "1", "yes"}


def get_int_env(key: str, default: int, min_val: Optional[int] = None) -> int:
    """Parse an integer environment variable with optional minimum validation.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Optional minimum value; if current value < min_val, returns default

    Returns:
        Parsed integer value, or default if parsing fails or value < min_val
    """
    try:
        value = int(os.getenv(key, str(default)))
        if min_val is not None and value < min_val:
            return default
        return value
    except ValueError:
        return default


def get_float_env(key: str, default: float, min_val: Optional[float] = None) -> float:
    """Parse a float environment variable with optional minimum validation.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Optional minimum value; if current value < min_val, returns default

    Returns:
        Parsed float value, or default if parsing fails or value < min_val
    """
    try:
        value = float(os.getenv(key, str(default)))
        if min_val is not None and value < min_val:
            return default
        return value
    except ValueError:
        return default


# Search configuration constants
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 100
LLM_PROVIDERS = {"openai", "openrouter", "ollama", "anthropic", "gemini"}
EMBEDDING_PROVIDERS = {"openai", "openrouter", "ollama", "gemini", "voyage"}
RERANKER_PROVIDERS = {"cohere", "flashrank"}
FALLBACK_STRATEGIES = {"web_search", "expanded_query", "alternate_index"}
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

logger = structlog.get_logger(__name__)

@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_env: str
    llm_provider: str
    llm_api_key: Optional[str]
    llm_base_url: Optional[str]
    llm_model_id: str
    openai_api_key: str
    openai_model_id: str
    openai_base_url: Optional[str]
    openrouter_api_key: Optional[str]
    openrouter_base_url: str
    ollama_base_url: str
    anthropic_api_key: Optional[str]
    gemini_api_key: Optional[str]
    voyage_api_key: Optional[str]
    # Embedding provider settings (separate from LLM provider)
    embedding_provider: str
    embedding_api_key: Optional[str]
    embedding_base_url: Optional[str]
    database_url: str
    db_pool_min: int
    db_pool_max: int
    request_max_bytes: int
    rate_limit_per_minute: int
    rate_limit_backend: str
    rate_limit_redis_prefix: str
    rate_limit_retry_after_seconds: int
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_pool_min: int
    neo4j_pool_max: int
    neo4j_pool_acquire_timeout_seconds: float
    neo4j_connection_timeout_seconds: float
    neo4j_max_connection_lifetime_seconds: int
    redis_url: str
    backend_host: str
    backend_port: int
    frontend_url: str
    # Epic 4 / Story 13.3 - Crawl4AI settings
    crawl4ai_rate_limit: float  # Legacy - kept for backward compatibility
    crawl4ai_headless: bool
    crawl4ai_max_concurrent: int
    crawl4ai_cache_enabled: bool
    crawl4ai_proxy_url: Optional[str]
    crawl4ai_js_wait_seconds: float
    crawl4ai_page_timeout_ms: int
    # Story 13.4 - Crawl Configuration Profiles
    crawl4ai_profile: str
    crawl4ai_stealth_proxy: Optional[str]
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
    # Epic 6 - Workspace settings
    share_secret: str  # Secret for signing share links (set via SHARE_SECRET env var)
    # Epic 7 - A2A settings
    a2a_session_ttl_seconds: int
    a2a_cleanup_interval_seconds: int
    a2a_max_sessions_per_tenant: int
    a2a_max_sessions_total: int
    a2a_max_messages_per_session: int
    # Epic 7 - MCP settings
    mcp_tool_timeout_seconds: float
    mcp_tool_timeout_overrides: dict[str, float]
    mcp_tool_max_timeout_seconds: float
    # Epic 8 - Ops settings
    model_pricing_json: str
    routing_simple_model: str
    routing_medium_model: str
    routing_complex_model: str
    routing_baseline_model: str
    routing_simple_max_score: int
    routing_complex_min_score: int
    trace_encryption_key: str
    # Epic 12 - Advanced Retrieval (Reranking)
    reranker_enabled: bool
    reranker_provider: str
    reranker_model: str
    reranker_top_k: int
    cohere_api_key: Optional[str]
    # Epic 12 - Advanced Retrieval (Contextual Retrieval)
    contextual_retrieval_enabled: bool
    contextual_model: str
    contextual_prompt_caching: bool
    contextual_reindex_batch_size: int
    # Epic 12 - Advanced Retrieval (Corrective RAG Grader)
    grader_enabled: bool
    grader_threshold: float
    grader_fallback_enabled: bool
    grader_fallback_strategy: str
    tavily_api_key: Optional[str]
    # Epic 13 - Enterprise Ingestion (Fallback Providers)
    crawl_fallback_enabled: bool
    crawl_fallback_providers: list[str]
    apify_api_token: Optional[str]
    brightdata_username: Optional[str]
    brightdata_password: Optional[str]
    brightdata_zone: str
    # Epic 13 - Enterprise Ingestion (YouTube Transcripts)
    youtube_preferred_languages: list[str]
    youtube_chunk_duration_seconds: int


def load_settings() -> Settings:
    """
    Load settings from environment variables.

    Returns:
        Settings instance with all configuration values

    Raises:
        RuntimeError: If required environment variables are missing
    """
    load_dotenv()

    app_env = os.getenv("APP_ENV", "development").strip().lower()
    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if llm_provider not in LLM_PROVIDERS:
        raise ValueError(
            "LLM_PROVIDER must be one of: openai, openrouter, ollama, anthropic, gemini."
        )
    llm_model_id = os.getenv("LLM_MODEL_ID")
    openai_model_id = os.getenv("OPENAI_MODEL_ID")
    if not llm_model_id and not openai_model_id:
        openai_model_id = "gpt-4o-mini"
        llm_model_id = openai_model_id
    else:
        if not llm_model_id:
            llm_model_id = openai_model_id
        if not openai_model_id:
            openai_model_id = llm_model_id

    min_pool_size = 1
    try:
        backend_port = int(os.getenv("BACKEND_PORT", "8000"))
    except ValueError as exc:
        raise ValueError(
            "BACKEND_PORT must be a valid integer. Check your .env file."
        ) from exc
    try:
        db_pool_min = int(os.getenv("DB_POOL_MIN", str(min_pool_size)))
        db_pool_max = int(os.getenv("DB_POOL_MAX", "50"))
        request_max_bytes = int(os.getenv("REQUEST_MAX_BYTES", "1048576"))
        rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        rate_limit_backend = os.getenv("RATE_LIMIT_BACKEND", "memory")
        rate_limit_redis_prefix = os.getenv("RATE_LIMIT_REDIS_PREFIX", "rate-limit")
        rate_limit_retry_after_seconds = int(os.getenv("RATE_LIMIT_RETRY_AFTER_SECONDS", "60"))
    except ValueError as exc:
        raise ValueError(
            "DB_POOL_MIN, DB_POOL_MAX, REQUEST_MAX_BYTES, and RATE_LIMIT_PER_MINUTE "
            "must be valid integers. Check your .env file."
        ) from exc
    if db_pool_min < min_pool_size or db_pool_max < db_pool_min:
        raise ValueError(
            "DB_POOL_MIN must be >= 1 and DB_POOL_MAX must be >= DB_POOL_MIN."
        )
    if request_max_bytes < 1:
        raise ValueError("REQUEST_MAX_BYTES must be >= 1.")

    try:
        neo4j_pool_min = int(os.getenv("NEO4J_POOL_MIN", "1"))
        neo4j_pool_max = int(os.getenv("NEO4J_POOL_MAX", "50"))
        neo4j_pool_acquire_timeout_seconds = float(
            os.getenv("NEO4J_POOL_ACQUIRE_TIMEOUT_SECONDS", "30")
        )
        neo4j_connection_timeout_seconds = float(
            os.getenv("NEO4J_CONNECTION_TIMEOUT_SECONDS", "30")
        )
        neo4j_max_connection_lifetime_seconds = int(
            os.getenv("NEO4J_MAX_CONNECTION_LIFETIME_SECONDS", "3600")
        )
    except ValueError as exc:
        raise ValueError(
            "NEO4J_POOL_MIN, NEO4J_POOL_MAX, and Neo4j timeout settings must be numeric."
        ) from exc
    if neo4j_pool_min < 1:
        raise ValueError("NEO4J_POOL_MIN must be >= 1.")
    if neo4j_pool_max < max(1, neo4j_pool_min):
        raise ValueError(
            "NEO4J_POOL_MAX must be >= max(1, NEO4J_POOL_MIN)."
        )
    if neo4j_pool_acquire_timeout_seconds <= 0:
        raise ValueError("NEO4J_POOL_ACQUIRE_TIMEOUT_SECONDS must be > 0.")
    if neo4j_connection_timeout_seconds <= 0:
        raise ValueError("NEO4J_CONNECTION_TIMEOUT_SECONDS must be > 0.")
    if neo4j_max_connection_lifetime_seconds <= 0:
        raise ValueError("NEO4J_MAX_CONNECTION_LIFETIME_SECONDS must be > 0.")
    if rate_limit_per_minute < 1:
        raise ValueError("RATE_LIMIT_PER_MINUTE must be >= 1.")
    if rate_limit_retry_after_seconds < 1:
        raise ValueError("RATE_LIMIT_RETRY_AFTER_SECONDS must be >= 1.")
    if rate_limit_backend not in {"memory", "redis"}:
        raise ValueError("RATE_LIMIT_BACKEND must be 'memory' or 'redis'.")

    try:
        crawl4ai_rate_limit = float(os.getenv("CRAWL4AI_RATE_LIMIT", "1.0"))
    except ValueError:
        crawl4ai_rate_limit = 1.0

    # Story 13.3 - Crawl4AI configuration (using helper functions)
    crawl4ai_headless = get_bool_env("CRAWL4AI_HEADLESS", "true")
    crawl4ai_max_concurrent = get_int_env("CRAWL4AI_MAX_CONCURRENT", 10, min_val=1)
    crawl4ai_cache_enabled = get_bool_env("CRAWL4AI_CACHE_ENABLED", "true")
    crawl4ai_proxy_url = os.getenv("CRAWL4AI_PROXY_URL") or None
    crawl4ai_js_wait_seconds = get_float_env("CRAWL4AI_JS_WAIT_SECONDS", 2.0, min_val=0.0)
    crawl4ai_page_timeout_ms = get_int_env("CRAWL4AI_PAGE_TIMEOUT_MS", 60000, min_val=1000)

    # Story 13.4 - Crawl Configuration Profiles
    crawl4ai_profile = os.getenv("CRAWL4AI_PROFILE", "fast").strip().lower()
    valid_profiles = {"fast", "thorough", "stealth"}
    if crawl4ai_profile not in valid_profiles:
        logger.warning(
            "invalid_crawl_profile",
            profile=crawl4ai_profile,
            valid_profiles=list(valid_profiles),
            fallback="fast",
        )
        crawl4ai_profile = "fast"
    crawl4ai_stealth_proxy = os.getenv("CRAWL4AI_STEALTH_PROXY") or None

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

    try:
        a2a_session_ttl_seconds = int(os.getenv("A2A_SESSION_TTL_SECONDS", "21600"))
    except ValueError:
        a2a_session_ttl_seconds = 21600

    try:
        a2a_cleanup_interval_seconds = int(os.getenv("A2A_CLEANUP_INTERVAL_SECONDS", "3600"))
    except ValueError:
        a2a_cleanup_interval_seconds = 3600

    try:
        a2a_max_sessions_per_tenant = int(os.getenv("A2A_MAX_SESSIONS_PER_TENANT", "100"))
    except ValueError:
        a2a_max_sessions_per_tenant = 100

    try:
        a2a_max_sessions_total = int(os.getenv("A2A_MAX_SESSIONS_TOTAL", "1000"))
    except ValueError:
        a2a_max_sessions_total = 1000

    try:
        a2a_max_messages_per_session = int(os.getenv("A2A_MAX_MESSAGES_PER_SESSION", "1000"))
    except ValueError:
        a2a_max_messages_per_session = 1000

    try:
        mcp_tool_timeout_seconds = float(os.getenv("MCP_TOOL_TIMEOUT_SECONDS", "30"))
    except ValueError:
        mcp_tool_timeout_seconds = 30.0

    try:
        mcp_tool_max_timeout_seconds = float(os.getenv("MCP_TOOL_MAX_TIMEOUT_SECONDS", "300"))
    except ValueError:
        mcp_tool_max_timeout_seconds = 300.0
    if mcp_tool_max_timeout_seconds <= 0:
        mcp_tool_max_timeout_seconds = 300.0

    raw_mcp_timeouts = os.getenv("MCP_TOOL_TIMEOUT_OVERRIDES", "")
    mcp_tool_timeout_overrides: dict[str, float] = {}
    if raw_mcp_timeouts:
        try:
            parsed = json.loads(raw_mcp_timeouts)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "MCP_TOOL_TIMEOUT_OVERRIDES must be valid JSON (e.g. "
                '{\"knowledge.query\": 30, \"knowledge.graph_stats\": 10}).'
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("MCP_TOOL_TIMEOUT_OVERRIDES must be a JSON object.")
        for key, value in parsed.items():
            try:
                mcp_tool_timeout_overrides[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "MCP_TOOL_TIMEOUT_OVERRIDES values must be numeric."
                ) from exc

    required = [
        "DATABASE_URL",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "REDIS_URL",
    ]
    provider_required_keys = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    provider_key = provider_required_keys.get(llm_provider)
    if provider_key:
        required.append(provider_key)
    values = {key: os.getenv(key) for key in required}
    missing = [key for key, value in values.items() if not value]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            "Missing required environment variables: "
            f"{missing_list}. Copy .env.example to .env and fill values."
        )

    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url = os.getenv(
        "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
    )
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    voyage_api_key = os.getenv("VOYAGE_API_KEY")

    # Embedding provider configuration
    # Default to LLM_PROVIDER if it supports embeddings, otherwise openai
    embedding_provider_env = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if embedding_provider_env:
        embedding_provider = embedding_provider_env
    elif llm_provider in EMBEDDING_PROVIDERS:
        # Use LLM provider for embeddings if it supports them
        embedding_provider = llm_provider
    else:
        # Anthropic doesn't have native embeddings, default to openai
        embedding_provider = "openai"
        logger.warning(
            "embedding_provider_fallback",
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            hint="Set EMBEDDING_PROVIDER explicitly. Consider 'voyage' for Anthropic (recommended by Anthropic docs).",
        )

    if embedding_provider not in EMBEDDING_PROVIDERS:
        raise ValueError(
            "EMBEDDING_PROVIDER must be one of: openai, openrouter, ollama, gemini, voyage."
        )

    if llm_provider == "openai":
        llm_api_key = openai_api_key
        llm_base_url = openai_base_url or None
    elif llm_provider == "openrouter":
        llm_api_key = openrouter_api_key
        llm_base_url = openrouter_base_url
    elif llm_provider == "ollama":
        llm_api_key = os.getenv("OLLAMA_API_KEY") or None
        llm_base_url = ollama_base_url
    elif llm_provider == "anthropic":
        llm_api_key = anthropic_api_key
        llm_base_url = None
    else:
        llm_api_key = gemini_api_key
        llm_base_url = None

    # Derive embedding API credentials based on embedding_provider
    if embedding_provider == "openai":
        embedding_api_key = openai_api_key
        embedding_base_url = openai_base_url or None
    elif embedding_provider == "openrouter":
        embedding_api_key = openrouter_api_key
        embedding_base_url = openrouter_base_url
    elif embedding_provider == "ollama":
        embedding_api_key = os.getenv("OLLAMA_API_KEY") or None
        embedding_base_url = ollama_base_url
    elif embedding_provider == "gemini":
        embedding_api_key = gemini_api_key
        embedding_base_url = None  # Gemini uses its own client
    elif embedding_provider == "voyage":
        embedding_api_key = voyage_api_key
        embedding_base_url = None  # Voyage uses its own client
    else:
        embedding_api_key = openai_api_key  # Fallback to OpenAI
        embedding_base_url = openai_base_url or None

    # Validate embedding provider API key is set
    embedding_provider_keys = {
        "openai": ("OPENAI_API_KEY", openai_api_key),
        "openrouter": ("OPENROUTER_API_KEY", openrouter_api_key),
        "gemini": ("GEMINI_API_KEY", gemini_api_key),
        "voyage": ("VOYAGE_API_KEY", voyage_api_key),
    }
    if embedding_provider in embedding_provider_keys:
        key_name, key_value = embedding_provider_keys[embedding_provider]
        if not key_value:
            raise ValueError(
                f"{key_name} is required when EMBEDDING_PROVIDER={embedding_provider}."
            )

    database_url = cast(str, values["DATABASE_URL"])
    neo4j_uri = cast(str, values["NEO4J_URI"])
    neo4j_user = cast(str, values["NEO4J_USER"])
    neo4j_password = cast(str, values["NEO4J_PASSWORD"])
    redis_url = cast(str, values["REDIS_URL"])
    model_pricing_json = os.getenv("MODEL_PRICING_JSON", "")
    if model_pricing_json:
        try:
            pricing_data = json.loads(model_pricing_json)
            if not isinstance(pricing_data, dict):
                raise ValueError("MODEL_PRICING_JSON must be a JSON object (dict).")
            for model_name, model_pricing in pricing_data.items():
                if not isinstance(model_name, str):
                    raise ValueError(f"MODEL_PRICING_JSON keys must be strings, got {type(model_name).__name__}.")
                if not isinstance(model_pricing, dict):
                    raise ValueError(f"MODEL_PRICING_JSON['{model_name}'] must be a dict with pricing info.")
        except json.JSONDecodeError as exc:
            raise ValueError("MODEL_PRICING_JSON must be valid JSON.") from exc

    routing_simple_model = os.getenv("ROUTING_SIMPLE_MODEL", "gpt-4o-mini")
    routing_medium_model = os.getenv("ROUTING_MEDIUM_MODEL", "gpt-4o")
    routing_complex_model = os.getenv("ROUTING_COMPLEX_MODEL", "gpt-4o")
    routing_baseline_model = os.getenv(
        "ROUTING_BASELINE_MODEL", routing_complex_model
    )
    try:
        routing_simple_max_score = int(os.getenv("ROUTING_SIMPLE_MAX_SCORE", "2"))
    except ValueError as exc:
        raise ValueError("ROUTING_SIMPLE_MAX_SCORE must be an integer.") from exc
    try:
        routing_complex_min_score = int(os.getenv("ROUTING_COMPLEX_MIN_SCORE", "5"))
    except ValueError as exc:
        raise ValueError("ROUTING_COMPLEX_MIN_SCORE must be an integer.") from exc
    if routing_simple_max_score < 0:
        raise ValueError("ROUTING_SIMPLE_MAX_SCORE must be >= 0.")
    if routing_complex_min_score <= routing_simple_max_score:
        raise ValueError(
            f"ROUTING_COMPLEX_MIN_SCORE ({routing_complex_min_score}) must be greater "
            f"than ROUTING_SIMPLE_MAX_SCORE ({routing_simple_max_score})."
        )

    # Epic 12 - Reranker configuration
    reranker_enabled = get_bool_env("RERANKER_ENABLED", "false")
    reranker_provider = os.getenv("RERANKER_PROVIDER", "flashrank").strip().lower()
    if reranker_enabled and reranker_provider not in RERANKER_PROVIDERS:
        raise ValueError(
            f"RERANKER_PROVIDER must be one of: {', '.join(sorted(RERANKER_PROVIDERS))}. "
            f"Got {reranker_provider!r}."
        )
    # Default models per provider
    default_reranker_models = {
        "cohere": "rerank-v3.5",
        "flashrank": "ms-marco-MiniLM-L-12-v2",
    }
    reranker_model = os.getenv(
        "RERANKER_MODEL", default_reranker_models.get(reranker_provider, "")
    )
    try:
        reranker_top_k = int(os.getenv("RERANKER_TOP_K", "10"))
    except ValueError:
        reranker_top_k = 10
    if reranker_top_k < 1:
        raise ValueError("RERANKER_TOP_K must be >= 1.")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if reranker_enabled and reranker_provider == "cohere" and not cohere_api_key:
        raise ValueError(
            "COHERE_API_KEY is required when RERANKER_ENABLED=true and RERANKER_PROVIDER=cohere."
        )

    trace_key = os.getenv("TRACE_ENCRYPTION_KEY")
    if trace_key:
        try:
            key_bytes = bytes.fromhex(trace_key)
        except ValueError as exc:
            raise ValueError("TRACE_ENCRYPTION_KEY must be hex-encoded.") from exc
        if len(key_bytes) != 32:
            raise ValueError(
                "TRACE_ENCRYPTION_KEY must be 64 hex chars (32 bytes)."
            )
        trace_encryption_key = trace_key
    elif app_env in {"development", "dev", "test", "local"}:
        logger.warning("trace_encryption_key_autogenerated", env=app_env)
        trace_encryption_key = secrets.token_hex(32)
    else:
        raise ValueError(
            "TRACE_ENCRYPTION_KEY must be set for non-development environments."
        )

    # Epic 12 - Contextual Retrieval settings
    contextual_retrieval_enabled = get_bool_env("CONTEXTUAL_RETRIEVAL_ENABLED", "false")
    contextual_model = os.getenv("CONTEXTUAL_MODEL", "claude-3-haiku-20240307")
    contextual_prompt_caching = get_bool_env("CONTEXTUAL_PROMPT_CACHING", "true")
    try:
        contextual_reindex_batch_size = int(os.getenv("CONTEXTUAL_REINDEX_BATCH_SIZE", "100"))
    except ValueError:
        contextual_reindex_batch_size = 100

    # Epic 12 - Corrective RAG Grader settings
    grader_enabled = get_bool_env("GRADER_ENABLED", "false")
    try:
        grader_threshold = float(os.getenv("GRADER_THRESHOLD", "0.5"))
        grader_threshold = max(0.0, min(1.0, grader_threshold))  # Clamp to 0.0-1.0
    except ValueError:
        grader_threshold = 0.5
    grader_fallback_enabled = get_bool_env("GRADER_FALLBACK_ENABLED", "true")
    grader_fallback_strategy = os.getenv("GRADER_FALLBACK_STRATEGY", "web_search")
    if grader_fallback_strategy not in FALLBACK_STRATEGIES:
        grader_fallback_strategy = "web_search"
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    # Epic 13 - Crawl Fallback settings
    crawl_fallback_enabled = get_bool_env("CRAWL_FALLBACK_ENABLED", "true")
    crawl_fallback_providers_raw = os.getenv(
        "CRAWL_FALLBACK_PROVIDERS", '["apify", "brightdata"]'
    )
    try:
        crawl_fallback_providers = json.loads(crawl_fallback_providers_raw)
        if not isinstance(crawl_fallback_providers, list):
            crawl_fallback_providers = ["apify", "brightdata"]
    except (json.JSONDecodeError, ValueError):
        crawl_fallback_providers = ["apify", "brightdata"]
    apify_api_token = os.getenv("APIFY_API_TOKEN")
    brightdata_username = os.getenv("BRIGHTDATA_USERNAME")
    brightdata_password = os.getenv("BRIGHTDATA_PASSWORD")
    brightdata_zone = os.getenv("BRIGHTDATA_ZONE", "scraping_browser")

    # Epic 13 - YouTube Transcript settings
    youtube_preferred_languages_raw = os.getenv(
        "YOUTUBE_PREFERRED_LANGUAGES", '["en", "en-US"]'
    )
    try:
        youtube_preferred_languages = json.loads(youtube_preferred_languages_raw)
        if not isinstance(youtube_preferred_languages, list):
            youtube_preferred_languages = ["en", "en-US"]
    except (json.JSONDecodeError, ValueError):
        youtube_preferred_languages = ["en", "en-US"]
    try:
        youtube_chunk_duration_seconds = int(
            os.getenv("YOUTUBE_CHUNK_DURATION_SECONDS", "120")
        )
        if youtube_chunk_duration_seconds < 1:
            youtube_chunk_duration_seconds = 120
    except ValueError:
        youtube_chunk_duration_seconds = 120

    return Settings(
        app_env=app_env,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model_id=llm_model_id,
        openai_api_key=openai_api_key,
        openai_model_id=openai_model_id,
        openai_base_url=openai_base_url,
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=openrouter_base_url,
        ollama_base_url=ollama_base_url,
        anthropic_api_key=anthropic_api_key,
        gemini_api_key=gemini_api_key,
        voyage_api_key=voyage_api_key,
        embedding_provider=embedding_provider,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        database_url=database_url,
        db_pool_min=db_pool_min,
        db_pool_max=db_pool_max,
        request_max_bytes=request_max_bytes,
        rate_limit_per_minute=rate_limit_per_minute,
        rate_limit_backend=rate_limit_backend,
        rate_limit_redis_prefix=rate_limit_redis_prefix,
        rate_limit_retry_after_seconds=rate_limit_retry_after_seconds,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_pool_min=neo4j_pool_min,
        neo4j_pool_max=neo4j_pool_max,
        neo4j_pool_acquire_timeout_seconds=neo4j_pool_acquire_timeout_seconds,
        neo4j_connection_timeout_seconds=neo4j_connection_timeout_seconds,
        neo4j_max_connection_lifetime_seconds=neo4j_max_connection_lifetime_seconds,
        redis_url=redis_url,
        backend_host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        backend_port=backend_port,
        frontend_url=os.getenv("FRONTEND_URL", "http://localhost:3000"),
        crawl4ai_rate_limit=crawl4ai_rate_limit,
        # Story 13.3 - Crawl4AI settings
        crawl4ai_headless=crawl4ai_headless,
        crawl4ai_max_concurrent=crawl4ai_max_concurrent,
        crawl4ai_cache_enabled=crawl4ai_cache_enabled,
        crawl4ai_proxy_url=crawl4ai_proxy_url,
        crawl4ai_js_wait_seconds=crawl4ai_js_wait_seconds,
        crawl4ai_page_timeout_ms=crawl4ai_page_timeout_ms,
        # Story 13.4 - Crawl Configuration Profiles
        crawl4ai_profile=crawl4ai_profile,
        crawl4ai_stealth_proxy=crawl4ai_stealth_proxy,
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
        # Epic 6 - Workspace settings
        # Default generates random secret if not set (for dev), but production should set SHARE_SECRET
        share_secret=os.getenv("SHARE_SECRET", secrets.token_hex(32)),
        # Epic 7 - A2A settings
        a2a_session_ttl_seconds=a2a_session_ttl_seconds,
        a2a_cleanup_interval_seconds=a2a_cleanup_interval_seconds,
        a2a_max_sessions_per_tenant=a2a_max_sessions_per_tenant,
        a2a_max_sessions_total=a2a_max_sessions_total,
        a2a_max_messages_per_session=a2a_max_messages_per_session,
        # Epic 7 - MCP settings
        mcp_tool_timeout_seconds=mcp_tool_timeout_seconds,
        mcp_tool_timeout_overrides=mcp_tool_timeout_overrides,
        mcp_tool_max_timeout_seconds=mcp_tool_max_timeout_seconds,
        # Epic 8 - Ops settings
        model_pricing_json=model_pricing_json,
        routing_simple_model=routing_simple_model,
        routing_medium_model=routing_medium_model,
        routing_complex_model=routing_complex_model,
        routing_baseline_model=routing_baseline_model,
        routing_simple_max_score=routing_simple_max_score,
        routing_complex_min_score=routing_complex_min_score,
        trace_encryption_key=trace_encryption_key,
        # Epic 12 - Reranker settings
        reranker_enabled=reranker_enabled,
        reranker_provider=reranker_provider,
        reranker_model=reranker_model,
        reranker_top_k=reranker_top_k,
        cohere_api_key=cohere_api_key,
        # Epic 12 - Contextual Retrieval settings
        contextual_retrieval_enabled=contextual_retrieval_enabled,
        contextual_model=contextual_model,
        contextual_prompt_caching=contextual_prompt_caching,
        contextual_reindex_batch_size=contextual_reindex_batch_size,
        # Epic 12 - Corrective RAG Grader settings
        grader_enabled=grader_enabled,
        grader_threshold=grader_threshold,
        grader_fallback_enabled=grader_fallback_enabled,
        grader_fallback_strategy=grader_fallback_strategy,
        tavily_api_key=tavily_api_key,
        # Epic 13 - Crawl Fallback settings
        crawl_fallback_enabled=crawl_fallback_enabled,
        crawl_fallback_providers=crawl_fallback_providers,
        apify_api_token=apify_api_token,
        brightdata_username=brightdata_username,
        brightdata_password=brightdata_password,
        brightdata_zone=brightdata_zone,
        # Epic 13 - YouTube Transcript settings
        youtube_preferred_languages=youtube_preferred_languages,
        youtube_chunk_duration_seconds=youtube_chunk_duration_seconds,
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
