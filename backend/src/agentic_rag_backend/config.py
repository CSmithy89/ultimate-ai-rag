"""Configuration management for the Agentic RAG Backend."""

import json
import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, cast

from dotenv import load_dotenv
import structlog

# Initialize logger early for use in helper functions
_config_logger = structlog.get_logger("agentic_rag_backend.config")


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
    raw_value = os.getenv(key)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
        if min_val is not None and value < min_val:
            _config_logger.warning(
                "config_value_below_minimum",
                key=key,
                value=value,
                min_val=min_val,
                using_default=default,
            )
            return default
        return value
    except ValueError:
        _config_logger.warning(
            "config_parse_error",
            key=key,
            raw_value=raw_value,
            expected_type="int",
            using_default=default,
        )
        return default


def get_float_env(
    key: str,
    default: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """Parse a float environment variable with optional min/max validation.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Optional minimum value; if current value < min_val, returns default
        max_val: Optional maximum value; if current value > max_val, returns default

    Returns:
        Parsed float value, or default if parsing fails or value out of range
    """
    raw_value = os.getenv(key)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
        if min_val is not None and value < min_val:
            _config_logger.warning(
                "config_value_below_minimum",
                key=key,
                value=value,
                min_val=min_val,
                using_default=default,
            )
            return default
        if max_val is not None and value > max_val:
            _config_logger.warning(
                "config_value_above_maximum",
                key=key,
                value=value,
                max_val=max_val,
                using_default=default,
            )
            return default
        return value
    except ValueError:
        _config_logger.warning(
            "config_parse_error",
            key=key,
            raw_value=raw_value,
            expected_type="float",
            using_default=default,
        )
        return default


# Search configuration constants
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 100
LLM_PROVIDERS = {"openai", "openrouter", "ollama", "anthropic", "gemini"}
EMBEDDING_PROVIDERS = {"openai", "openrouter", "ollama", "gemini", "voyage"}
RERANKER_PROVIDERS = {"cohere", "flashrank"}
FALLBACK_STRATEGIES = {"web_search", "expanded_query", "alternate_index"}

PRODUCTION_ENVS = {"production", "prod"}
DEVELOPMENT_ENVS = {"development", "dev", "test", "local"}


def is_production_env(app_env: str) -> bool:
    return app_env in PRODUCTION_ENVS


def is_development_env(app_env: str) -> bool:
    return app_env in DEVELOPMENT_ENVS
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
    embedding_dimension: int
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
    neo4j_transaction_timeout_seconds: float  # Query timeout for expensive operations
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
    crawler_strict_validation: bool
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
    # Epic 14 - Enhanced A2A Protocol settings
    a2a_enabled: bool
    a2a_agent_id: str
    a2a_endpoint_url: str
    a2a_heartbeat_interval_seconds: int
    a2a_heartbeat_timeout_seconds: int
    a2a_task_default_timeout_seconds: int
    a2a_task_max_retries: int
    # Story 22-A2 - A2A Resource Limits settings
    a2a_limits_backend: str
    a2a_session_limit_per_tenant: int
    a2a_message_limit_per_session: int
    a2a_session_ttl_hours: int
    a2a_message_rate_limit: int
    a2a_limits_cleanup_interval_minutes: int
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
    grader_model: str
    grader_threshold: float
    grader_fallback_enabled: bool
    grader_fallback_strategy: str
    tavily_api_key: Optional[str]
    # Story 19-F4 - Heuristic grader length weight configuration
    grader_heuristic_length_weight: float
    grader_heuristic_min_length: int
    grader_heuristic_max_length: int
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
    # Epic 15 - Codebase Intelligence settings
    codebase_hallucination_threshold: float
    codebase_detector_mode: str
    codebase_cache_ttl_seconds: int
    codebase_symbol_table_max_symbols: int
    codebase_index_rate_limit_max: int
    codebase_index_rate_limit_window_seconds: int
    codebase_detection_slow_ms: int
    codebase_allowed_base_path: Optional[str]
    codebase_rag_enabled: bool
    codebase_languages: list[str]
    codebase_exclude_patterns: list[str]
    codebase_max_chunk_size: int
    codebase_include_class_context: bool
    codebase_incremental_indexing: bool
    codebase_index_cache_ttl_seconds: int
    # Story 19-C5 - Prometheus Observability settings
    prometheus_enabled: bool
    prometheus_path: str
    # Story 19-G1 - Reranking Cache settings
    reranker_cache_enabled: bool
    reranker_cache_ttl_seconds: int
    reranker_cache_max_size: int
    # Story 19-G2 - Contextual Retrieval Prompt Path
    contextual_retrieval_prompt_path: Optional[str]
    # Story 19-G3 - Model Preloading settings
    grader_preload_model: bool
    reranker_preload_model: bool
    # Story 19-G4 - Score Normalization settings
    grader_normalization_strategy: str
    # Epic 20 - Memory Platform settings (Story 20-A1)
    memory_scopes_enabled: bool
    memory_default_scope: str
    memory_include_parent_scopes: bool
    memory_cache_ttl_seconds: int
    memory_max_per_scope: int
    # Story 20-A2 - Memory Consolidation
    memory_consolidation_enabled: bool
    memory_consolidation_schedule: str
    memory_similarity_threshold: float
    memory_decay_half_life_days: int
    memory_min_importance: float
    memory_consolidation_batch_size: int
    # Story 20-B1 - Community Detection (Graph Intelligence)
    community_detection_enabled: bool
    community_algorithm: str
    community_min_size: int
    community_max_levels: int
    community_summary_model: str
    community_refresh_schedule: str
    # Story 20-B2 - LazyRAG Pattern (Query-Time Summarization)
    lazy_rag_enabled: bool
    lazy_rag_max_entities: int
    lazy_rag_max_hops: int
    lazy_rag_summary_model: str
    lazy_rag_use_communities: bool
    # Story 20-B3 - Query Routing (Global/Local)
    query_routing_enabled: bool
    query_routing_use_llm: bool
    query_routing_llm_model: str
    query_routing_confidence_threshold: float
    # Story 20-C1 - Graph-Based Rerankers
    graph_reranker_enabled: bool
    graph_reranker_type: str
    graph_reranker_episode_weight: float
    graph_reranker_distance_weight: float
    graph_reranker_original_weight: float
    graph_reranker_episode_window_days: int
    graph_reranker_max_distance: int
    # Story 20-C2 - Dual-Level Retrieval
    dual_level_retrieval_enabled: bool
    dual_level_low_weight: float
    dual_level_high_weight: float
    dual_level_low_limit: int
    dual_level_high_limit: int
    dual_level_synthesis_model: str
    dual_level_synthesis_temperature: float
    # Story 20-C3 - Parent-Child Chunk Hierarchy
    hierarchical_chunks_enabled: bool
    hierarchical_chunk_levels: list[int]
    hierarchical_overlap_ratio: float
    small_to_big_return_level: int
    hierarchical_embedding_level: int
    # Story 20-D1 - Enhanced Table/Layout Extraction
    enhanced_docling_enabled: bool
    docling_table_extraction: bool
    docling_preserve_layout: bool
    docling_table_as_markdown: bool
    # Story 20-D2 - Multimodal Ingestion
    multimodal_ingestion_enabled: bool
    office_docs_enabled: bool
    # Story 20-E1 - Ontology Support
    ontology_support_enabled: bool
    ontology_path: Optional[str]
    ontology_auto_type: bool
    # Story 20-E2 - Self-Improving Feedback Loop
    feedback_loop_enabled: bool
    feedback_min_samples: int
    feedback_decay_days: int
    feedback_boost_max: float
    feedback_boost_min: float
    # Story 20-H1 - Sparse Vector Search (BM42)
    sparse_vectors_enabled: bool
    sparse_model: str
    hybrid_dense_weight: float
    hybrid_sparse_weight: float
    # Story 20-H2 - Cross-Language Query
    cross_language_enabled: bool
    cross_language_embedding: str
    cross_language_translation: bool
    # Story 20-H3 - External Data Source Sync
    external_sync_enabled: bool
    sync_sources: str
    s3_sync_bucket: str
    s3_sync_prefix: str
    confluence_url: str
    confluence_api_token: str
    confluence_spaces: str
    notion_api_key: str
    notion_database_ids: str
    # Story 20-H4 - Voice I/O
    voice_io_enabled: bool
    whisper_model: str
    tts_provider: str
    tts_voice: str
    tts_speed: float
    # Story 20-H5 - ColBERT Reranking
    colbert_enabled: bool
    colbert_model: str
    colbert_max_length: int


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
    llm_model_id = llm_model_id or "gpt-4o-mini"
    openai_model_id = openai_model_id or llm_model_id

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
        # Transaction timeout for expensive queries (default 300s for LazyRAG traversals)
        neo4j_transaction_timeout_seconds = float(
            os.getenv("NEO4J_TRANSACTION_TIMEOUT_SECONDS", "300")
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
    if neo4j_transaction_timeout_seconds <= 0:
        raise ValueError("NEO4J_TRANSACTION_TIMEOUT_SECONDS must be > 0.")
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
    strict_default = "true" if is_production_env(app_env) else "false"
    crawler_strict_validation = get_bool_env(
        "CRAWLER_STRICT_VALIDATION",
        strict_default,
    )
    crawl4ai_profile = os.getenv("CRAWL4AI_PROFILE", "fast").strip().lower()
    valid_profiles = {"fast", "thorough", "stealth"}
    if crawl4ai_profile not in valid_profiles:
        if crawler_strict_validation:
            raise ValueError(
                f"Invalid crawl profile: {crawl4ai_profile}. "
                f"Valid profiles: {sorted(valid_profiles)}"
            )
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

    # Epic 14 - Enhanced A2A Protocol settings
    a2a_enabled = get_bool_env("A2A_ENABLED", "true")
    a2a_agent_id = os.getenv("A2A_AGENT_ID", "agentic-rag-001")
    a2a_endpoint_url = os.getenv("A2A_ENDPOINT_URL", "http://localhost:8000")
    a2a_heartbeat_interval_seconds = get_int_env("A2A_HEARTBEAT_INTERVAL_SECONDS", 30, min_val=5)
    a2a_heartbeat_timeout_seconds = get_int_env("A2A_HEARTBEAT_TIMEOUT_SECONDS", 60, min_val=10)
    a2a_task_default_timeout_seconds = get_int_env("A2A_TASK_DEFAULT_TIMEOUT_SECONDS", 300, min_val=1)
    a2a_task_max_retries = get_int_env("A2A_TASK_MAX_RETRIES", 3, min_val=0)

    # Story 22-A2 - A2A Resource Limits settings
    a2a_limits_backend = os.getenv("A2A_LIMITS_BACKEND", "memory").strip().lower()
    if a2a_limits_backend not in {"memory", "redis", "postgres"}:
        logger.warning(
            "invalid_a2a_limits_backend",
            backend=a2a_limits_backend,
            valid_backends=["memory", "redis", "postgres"],
            fallback="memory",
        )
        a2a_limits_backend = "memory"
    a2a_session_limit_per_tenant = get_int_env("A2A_SESSION_LIMIT_PER_TENANT", 100, min_val=1)
    a2a_message_limit_per_session = get_int_env("A2A_MESSAGE_LIMIT_PER_SESSION", 1000, min_val=1)
    a2a_session_ttl_hours = get_int_env("A2A_SESSION_TTL_HOURS", 24, min_val=1)
    a2a_message_rate_limit = get_int_env("A2A_MESSAGE_RATE_LIMIT", 60, min_val=1)
    a2a_limits_cleanup_interval_minutes = get_int_env("A2A_LIMITS_CLEANUP_INTERVAL_MINUTES", 15, min_val=1)

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

    llm_api_key: Optional[str]
    llm_base_url: Optional[str]
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
    embedding_api_key: Optional[str]
    embedding_base_url: Optional[str]
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

    embedding_dimension = get_int_env("EMBEDDING_DIMENSION", 1536, min_val=1)

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
    elif is_development_env(app_env):
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
    # Default grader model: "heuristic" for lightweight scoring, or cross-encoder model name
    # Supported cross-encoder models:
    # - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good accuracy)
    # - cross-encoder/ms-marco-MiniLM-L-12-v2 (higher accuracy)
    # - BAAI/bge-reranker-base (BGE reranker)
    # - BAAI/bge-reranker-large (BGE large, best accuracy)
    grader_model = os.getenv("GRADER_MODEL", "heuristic").strip()
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

    # Story 19-F4 - Heuristic grader length weight configuration
    # Length weight determines how much content length influences the heuristic score
    # when no retrieval scores are available. See docs/guides/advanced-retrieval-configuration.md
    grader_heuristic_length_weight = get_float_env(
        "GRADER_HEURISTIC_LENGTH_WEIGHT", 0.5, min_val=0.0
    )
    # Clamp to 0.0-1.0 range
    grader_heuristic_length_weight = max(0.0, min(1.0, grader_heuristic_length_weight))
    grader_heuristic_min_length = get_int_env(
        "GRADER_HEURISTIC_MIN_LENGTH", 50, min_val=1
    )
    grader_heuristic_max_length = get_int_env(
        "GRADER_HEURISTIC_MAX_LENGTH", 2000, min_val=grader_heuristic_min_length
    )

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
    if crawl_fallback_enabled:
        missing_providers = []
        if "apify" in crawl_fallback_providers and not apify_api_token:
            missing_providers.append("apify")
        if "brightdata" in crawl_fallback_providers and (
            not brightdata_username or not brightdata_password
        ):
            missing_providers.append("brightdata")
        if missing_providers:
            message = (
                "Missing credentials for crawl fallback providers: "
                f"{', '.join(missing_providers)}"
            )
            if crawler_strict_validation:
                raise ValueError(message)
            logger.warning(
                "crawl_fallback_credentials_missing",
                providers=missing_providers,
            )

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

    # Epic 15 - Codebase Intelligence settings
    raw_hallucination_threshold = os.getenv("CODEBASE_HALLUCINATION_THRESHOLD")
    if raw_hallucination_threshold is None:
        codebase_hallucination_threshold = 0.3
    else:
        try:
            codebase_hallucination_threshold = float(raw_hallucination_threshold)
        except ValueError as exc:
            raise ValueError(
                "CODEBASE_HALLUCINATION_THRESHOLD must be a float between 0.0 and 1.0."
            ) from exc
        if not 0.0 <= codebase_hallucination_threshold <= 1.0:
            raise ValueError(
                "CODEBASE_HALLUCINATION_THRESHOLD must be between 0.0 and 1.0."
            )
    codebase_detector_mode = os.getenv("CODEBASE_DETECTOR_MODE", "warn").lower()
    if codebase_detector_mode not in {"warn", "block"}:
        raise ValueError("CODEBASE_DETECTOR_MODE must be 'warn' or 'block'.")
    codebase_cache_ttl_seconds = get_int_env(
        "CODEBASE_CACHE_TTL_SECONDS", 3600, min_val=60
    )
    codebase_symbol_table_max_symbols = get_int_env(
        "CODEBASE_SYMBOL_TABLE_MAX_SYMBOLS", 200000, min_val=0
    )
    codebase_index_rate_limit_max = get_int_env(
        "CODEBASE_INDEX_RATE_LIMIT_MAX", 10, min_val=1
    )
    codebase_index_rate_limit_window_seconds = get_int_env(
        "CODEBASE_INDEX_RATE_LIMIT_WINDOW_SECONDS", 3600, min_val=60
    )
    codebase_detection_slow_ms = get_int_env(
        "CODEBASE_DETECTION_SLOW_MS", 2000, min_val=0
    )
    codebase_allowed_base_path = os.getenv("CODEBASE_ALLOWED_BASE_PATH")
    if codebase_allowed_base_path:
        base_path = Path(codebase_allowed_base_path)
        if not base_path.is_absolute():
            raise ValueError("CODEBASE_ALLOWED_BASE_PATH must be an absolute path.")
        if not base_path.exists():
            raise ValueError(
                f"CODEBASE_ALLOWED_BASE_PATH does not exist: {codebase_allowed_base_path}"
            )
        codebase_allowed_base_path = str(base_path.resolve())
    codebase_rag_enabled = get_bool_env("CODEBASE_RAG_ENABLED", "false")
    codebase_languages_raw = os.getenv("CODEBASE_LANGUAGES", "python,typescript,javascript")
    codebase_languages = [
        lang.strip().lower()
        for lang in codebase_languages_raw.split(",")
        if lang.strip()
    ]
    codebase_exclude_default = [
        "**/node_modules/**",
        "**/__pycache__/**",
        "**/venv/**",
        "**/.venv/**",
        "**/dist/**",
        "**/build/**",
        "**/.git/**",
    ]
    codebase_exclude_raw = os.getenv("CODEBASE_EXCLUDE_PATTERNS")
    if codebase_exclude_raw:
        try:
            parsed = json.loads(codebase_exclude_raw)
            if isinstance(parsed, list):
                codebase_exclude_patterns = [str(p) for p in parsed]
            else:
                codebase_exclude_patterns = codebase_exclude_default
        except (json.JSONDecodeError, ValueError):
            codebase_exclude_patterns = codebase_exclude_default
    else:
        codebase_exclude_patterns = codebase_exclude_default
    codebase_max_chunk_size = get_int_env("CODEBASE_MAX_CHUNK_SIZE", 1000, min_val=200)
    codebase_include_class_context = get_bool_env("CODEBASE_INCLUDE_CLASS_CONTEXT", "true")
    codebase_incremental_indexing = get_bool_env("CODEBASE_INCREMENTAL_INDEXING", "true")
    codebase_index_cache_ttl_seconds = get_int_env(
        "CODEBASE_INDEX_CACHE_TTL_SECONDS", 86400, min_val=60
    )

    # Story 19-G1 - Reranking Cache settings
    reranker_cache_enabled = get_bool_env("RERANKER_CACHE_ENABLED", "false")
    reranker_cache_ttl_seconds = get_int_env("RERANKER_CACHE_TTL_SECONDS", 300, min_val=1)
    reranker_cache_max_size = get_int_env("RERANKER_CACHE_MAX_SIZE", 1000, min_val=1)

    # Story 19-G2 - Contextual Retrieval Prompt Path
    contextual_retrieval_prompt_path = os.getenv("CONTEXTUAL_RETRIEVAL_PROMPT_PATH") or None

    # Story 19-G3 - Model Preloading settings
    grader_preload_model = get_bool_env("GRADER_PRELOAD_MODEL", "false")
    reranker_preload_model = get_bool_env("RERANKER_PRELOAD_MODEL", "false")

    # Story 19-G4 - Score Normalization settings
    grader_normalization_strategy = os.getenv("GRADER_NORMALIZATION_STRATEGY", "min_max").strip().lower()
    valid_normalization_strategies = {"min_max", "z_score", "softmax", "percentile"}
    if grader_normalization_strategy not in valid_normalization_strategies:
        logger.warning(
            "invalid_normalization_strategy",
            strategy=grader_normalization_strategy,
            valid_strategies=list(valid_normalization_strategies),
            fallback="min_max",
        )
        grader_normalization_strategy = "min_max"

    # Epic 20 - Memory Platform settings (Story 20-A1)
    memory_scopes_enabled = get_bool_env("MEMORY_SCOPES_ENABLED", "false")
    memory_default_scope = os.getenv("MEMORY_DEFAULT_SCOPE", "session").strip().lower()
    valid_memory_scopes = {"user", "session", "agent", "global"}
    if memory_default_scope not in valid_memory_scopes:
        logger.warning(
            "invalid_memory_scope",
            scope=memory_default_scope,
            valid_scopes=list(valid_memory_scopes),
            fallback="session",
        )
        memory_default_scope = "session"
    memory_include_parent_scopes = get_bool_env("MEMORY_INCLUDE_PARENT_SCOPES", "true")
    memory_cache_ttl_seconds = get_int_env("MEMORY_CACHE_TTL_SECONDS", 3600, min_val=60)
    memory_max_per_scope = get_int_env("MEMORY_MAX_PER_SCOPE", 10000, min_val=100)

    # Story 20-A2 - Memory Consolidation settings
    memory_consolidation_enabled = get_bool_env("MEMORY_CONSOLIDATION_ENABLED", "false")
    memory_consolidation_schedule = os.getenv("MEMORY_CONSOLIDATION_SCHEDULE", "0 2 * * *")
    memory_similarity_threshold = get_float_env("MEMORY_SIMILARITY_THRESHOLD", 0.9, min_val=0.0)
    # Clamp similarity threshold to valid range
    memory_similarity_threshold = max(0.0, min(1.0, memory_similarity_threshold))
    memory_decay_half_life_days = get_int_env("MEMORY_DECAY_HALF_LIFE_DAYS", 30, min_val=1)
    memory_min_importance = get_float_env("MEMORY_MIN_IMPORTANCE", 0.1, min_val=0.0)
    # Clamp min importance to valid range
    memory_min_importance = max(0.0, min(1.0, memory_min_importance))
    memory_consolidation_batch_size = get_int_env("MEMORY_CONSOLIDATION_BATCH_SIZE", 100, min_val=10)

    # Story 20-B1 - Community Detection settings
    community_detection_enabled = get_bool_env("COMMUNITY_DETECTION_ENABLED", "false")
    community_algorithm = os.getenv("COMMUNITY_ALGORITHM", "louvain").strip().lower()
    valid_community_algorithms = {"louvain", "leiden"}
    if community_algorithm not in valid_community_algorithms:
        logger.warning(
            "invalid_community_algorithm",
            algorithm=community_algorithm,
            valid_algorithms=list(valid_community_algorithms),
            fallback="louvain",
        )
        community_algorithm = "louvain"
    community_min_size = get_int_env("COMMUNITY_MIN_SIZE", 3, min_val=2)
    community_max_levels = get_int_env("COMMUNITY_MAX_LEVELS", 3, min_val=1)
    community_summary_model = os.getenv("COMMUNITY_SUMMARY_MODEL", "gpt-4o-mini")
    community_refresh_schedule = os.getenv("COMMUNITY_REFRESH_SCHEDULE", "0 3 * * 0")

    # Story 20-B2 - LazyRAG Pattern (Query-Time Summarization)
    lazy_rag_enabled = get_bool_env("LAZY_RAG_ENABLED", "false")
    lazy_rag_max_entities = get_int_env("LAZY_RAG_MAX_ENTITIES", 50, min_val=1)
    lazy_rag_max_hops = get_int_env("LAZY_RAG_MAX_HOPS", 2, min_val=1)
    lazy_rag_summary_model = os.getenv("LAZY_RAG_SUMMARY_MODEL", "gpt-4o-mini")
    lazy_rag_use_communities = get_bool_env("LAZY_RAG_USE_COMMUNITIES", "true")

    # Story 20-B3 - Query Routing (Global/Local)
    query_routing_enabled = get_bool_env("QUERY_ROUTING_ENABLED", "false")
    query_routing_use_llm = get_bool_env("QUERY_ROUTING_USE_LLM", "false")
    query_routing_llm_model = os.getenv("QUERY_ROUTING_LLM_MODEL", "gpt-4o-mini")
    query_routing_confidence_threshold = get_float_env(
        "QUERY_ROUTING_CONFIDENCE_THRESHOLD", 0.7, min_val=0.0
    )
    # Clamp threshold to valid range
    query_routing_confidence_threshold = max(0.0, min(1.0, query_routing_confidence_threshold))

    # Story 20-C1 - Graph-Based Rerankers
    graph_reranker_enabled = get_bool_env("GRAPH_RERANKER_ENABLED", "false")
    graph_reranker_type = os.getenv("GRAPH_RERANKER_TYPE", "hybrid").strip().lower()
    valid_graph_reranker_types = {"episode", "distance", "hybrid"}
    if graph_reranker_type not in valid_graph_reranker_types:
        logger.warning(
            "invalid_graph_reranker_type",
            type=graph_reranker_type,
            valid_types=list(valid_graph_reranker_types),
            fallback="hybrid",
        )
        graph_reranker_type = "hybrid"

    # Weight configuration with validation
    graph_reranker_episode_weight = get_float_env(
        "GRAPH_RERANKER_EPISODE_WEIGHT", 0.3, min_val=0.0
    )
    graph_reranker_distance_weight = get_float_env(
        "GRAPH_RERANKER_DISTANCE_WEIGHT", 0.3, min_val=0.0
    )
    graph_reranker_original_weight = get_float_env(
        "GRAPH_RERANKER_ORIGINAL_WEIGHT", 0.4, min_val=0.0
    )

    # Clamp weights to 0-1 range
    graph_reranker_episode_weight = max(0.0, min(1.0, graph_reranker_episode_weight))
    graph_reranker_distance_weight = max(0.0, min(1.0, graph_reranker_distance_weight))
    graph_reranker_original_weight = max(0.0, min(1.0, graph_reranker_original_weight))

    # Validate weights sum to 1.0 (with tolerance)
    weight_sum = (
        graph_reranker_episode_weight
        + graph_reranker_distance_weight
        + graph_reranker_original_weight
    )
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(
            "graph_reranker_weights_not_normalized",
            sum=weight_sum,
            episode=graph_reranker_episode_weight,
            distance=graph_reranker_distance_weight,
            original=graph_reranker_original_weight,
            hint="Weights will be normalized to sum to 1.0",
        )
        # Normalize weights or fall back to safe defaults when sum is non-positive
        if weight_sum > 0:
            graph_reranker_episode_weight /= weight_sum
            graph_reranker_distance_weight /= weight_sum
            graph_reranker_original_weight /= weight_sum
        else:
            graph_reranker_episode_weight = 0.3
            graph_reranker_distance_weight = 0.3
            graph_reranker_original_weight = 0.4

    graph_reranker_episode_window_days = get_int_env(
        "GRAPH_RERANKER_EPISODE_WINDOW_DAYS", 30, min_val=1
    )
    graph_reranker_max_distance = get_int_env(
        "GRAPH_RERANKER_MAX_DISTANCE", 3, min_val=1
    )

    # Story 20-C2 - Dual-Level Retrieval settings
    dual_level_retrieval_enabled = get_bool_env("DUAL_LEVEL_RETRIEVAL_ENABLED", "false")
    dual_level_low_weight = get_float_env("DUAL_LEVEL_LOW_WEIGHT", 0.6, min_val=0.0)
    dual_level_high_weight = get_float_env("DUAL_LEVEL_HIGH_WEIGHT", 0.4, min_val=0.0)
    # Clamp weights to 0-1 range
    dual_level_low_weight = max(0.0, min(1.0, dual_level_low_weight))
    dual_level_high_weight = max(0.0, min(1.0, dual_level_high_weight))
    # Validate weights sum to 1.0 (with tolerance)
    dual_weight_sum = dual_level_low_weight + dual_level_high_weight
    if abs(dual_weight_sum - 1.0) > 0.01:
        logger.warning(
            "dual_level_weights_not_normalized",
            sum=dual_weight_sum,
            low_weight=dual_level_low_weight,
            high_weight=dual_level_high_weight,
            hint="Weights will be normalized to sum to 1.0",
        )
        # Normalize weights or fall back to safe defaults when sum is non-positive
        if dual_weight_sum > 0:
            dual_level_low_weight /= dual_weight_sum
            dual_level_high_weight /= dual_weight_sum
        else:
            dual_level_low_weight = 0.6
            dual_level_high_weight = 0.4
    dual_level_low_limit = get_int_env("DUAL_LEVEL_LOW_LIMIT", 10, min_val=1)
    dual_level_high_limit = get_int_env("DUAL_LEVEL_HIGH_LIMIT", 5, min_val=1)
    dual_level_synthesis_model = os.getenv("DUAL_LEVEL_SYNTHESIS_MODEL", "gpt-4o-mini")
    dual_level_synthesis_temperature = get_float_env(
        "DUAL_LEVEL_SYNTHESIS_TEMPERATURE",
        0.3,
        min_val=0.0,
    )
    dual_level_synthesis_temperature = max(0.0, min(2.0, dual_level_synthesis_temperature))

    # Story 20-C3 - Parent-Child Chunk Hierarchy settings
    hierarchical_chunks_enabled = get_bool_env("HIERARCHICAL_CHUNKS_ENABLED", "false")

    # Parse chunk levels from comma-separated string
    hierarchical_chunk_levels_raw = os.getenv("HIERARCHICAL_CHUNK_LEVELS", "256,512,1024,2048")
    try:
        hierarchical_chunk_levels = [
            int(level.strip())
            for level in hierarchical_chunk_levels_raw.split(",")
            if level.strip()
        ]
        # Validate minimum of 2 levels required for hierarchical chunking
        if not hierarchical_chunk_levels or len(hierarchical_chunk_levels) < 2:
            logger.warning(
                "invalid_hierarchical_chunk_levels",
                levels=hierarchical_chunk_levels,
                hint="At least 2 chunk levels required, using defaults",
            )
            hierarchical_chunk_levels = [256, 512, 1024, 2048]
        else:
            # Validate levels are strictly increasing
            for i in range(1, len(hierarchical_chunk_levels)):
                if hierarchical_chunk_levels[i] <= hierarchical_chunk_levels[i - 1]:
                    logger.warning(
                        "invalid_hierarchical_chunk_levels",
                        levels=hierarchical_chunk_levels,
                        hint="Levels must be strictly increasing, using defaults",
                    )
                    hierarchical_chunk_levels = [256, 512, 1024, 2048]
                    break
    except (ValueError, AttributeError):
        hierarchical_chunk_levels = [256, 512, 1024, 2048]

    hierarchical_overlap_ratio = get_float_env("HIERARCHICAL_OVERLAP_RATIO", 0.1, min_val=0.0)
    # Clamp overlap ratio to valid range (0.0-0.5)
    hierarchical_overlap_ratio = max(0.0, min(0.5, hierarchical_overlap_ratio))

    small_to_big_return_level = get_int_env("SMALL_TO_BIG_RETURN_LEVEL", 2, min_val=0)
    # Validate return level is within chunk levels
    if small_to_big_return_level >= len(hierarchical_chunk_levels):
        logger.warning(
            "invalid_small_to_big_return_level",
            return_level=small_to_big_return_level,
            max_level=len(hierarchical_chunk_levels) - 1,
            fallback=len(hierarchical_chunk_levels) - 1,
        )
        small_to_big_return_level = len(hierarchical_chunk_levels) - 1

    hierarchical_embedding_level = get_int_env("HIERARCHICAL_EMBEDDING_LEVEL", 0, min_val=0)
    # Validate embedding level is within chunk levels
    if hierarchical_embedding_level >= len(hierarchical_chunk_levels):
        logger.warning(
            "invalid_hierarchical_embedding_level",
            embedding_level=hierarchical_embedding_level,
            max_level=len(hierarchical_chunk_levels) - 1,
            fallback=0,
        )
        hierarchical_embedding_level = 0

    # Story 20-D1 - Enhanced Table/Layout Extraction settings
    enhanced_docling_enabled = get_bool_env("ENHANCED_DOCLING_ENABLED", "true")
    docling_table_extraction = get_bool_env("DOCLING_TABLE_EXTRACTION", "true")
    docling_preserve_layout = get_bool_env("DOCLING_PRESERVE_LAYOUT", "true")
    docling_table_as_markdown = get_bool_env("DOCLING_TABLE_AS_MARKDOWN", "true")

    # Story 20-D2 - Multimodal Ingestion settings
    multimodal_ingestion_enabled = get_bool_env("MULTIMODAL_INGESTION_ENABLED", "false")
    office_docs_enabled = get_bool_env("OFFICE_DOCS_ENABLED", "true")

    # Story 20-E1 - Ontology Support settings
    ontology_support_enabled = get_bool_env("ONTOLOGY_SUPPORT_ENABLED", "false")
    ontology_path = os.getenv("ONTOLOGY_PATH") or None
    ontology_auto_type = get_bool_env("ONTOLOGY_AUTO_TYPE", "false")

    # Story 20-E2 - Self-Improving Feedback Loop settings
    feedback_loop_enabled = get_bool_env("FEEDBACK_LOOP_ENABLED", "false")
    feedback_min_samples = get_int_env("FEEDBACK_MIN_SAMPLES", 10, min_val=1)
    feedback_decay_days = get_int_env("FEEDBACK_DECAY_DAYS", 90, min_val=1)
    feedback_boost_max = get_float_env("FEEDBACK_BOOST_MAX", 1.5, min_val=1.0)
    feedback_boost_min = get_float_env("FEEDBACK_BOOST_MIN", 0.5, min_val=0.0)
    # Validate boost range
    if feedback_boost_min >= feedback_boost_max:
        logger.warning(
            "invalid_feedback_boost_range",
            min=feedback_boost_min,
            max=feedback_boost_max,
            hint="FEEDBACK_BOOST_MIN must be less than FEEDBACK_BOOST_MAX, using defaults",
        )
        feedback_boost_min = 0.5
        feedback_boost_max = 1.5
    # Clamp boost values to model's valid range (0.5 to 1.5)
    # This ensures config values match QueryBoost model constraints
    feedback_boost_max = max(1.0, min(1.5, feedback_boost_max))
    feedback_boost_min = max(0.5, min(1.0, feedback_boost_min))

    # Story 20-H1 - Sparse Vector Search (BM42) settings
    sparse_vectors_enabled = get_bool_env("SPARSE_VECTORS_ENABLED", "false")
    sparse_model = os.getenv(
        "SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions"
    )
    hybrid_dense_weight = get_float_env("HYBRID_DENSE_WEIGHT", 0.7, min_val=0.0)
    hybrid_sparse_weight = get_float_env("HYBRID_SPARSE_WEIGHT", 0.3, min_val=0.0)
    # Validate that weights sum to 1.0 (with tolerance for floating point)
    weight_sum = hybrid_dense_weight + hybrid_sparse_weight
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(
            "hybrid_weights_not_normalized",
            dense_weight=hybrid_dense_weight,
            sparse_weight=hybrid_sparse_weight,
            sum=weight_sum,
            hint="Weights will be normalized to sum to 1.0 for balanced hybrid search",
        )
        # Normalize weights to keep hybrid search well behaved,
        # or fall back to defaults when sum is non-positive
        if weight_sum > 0:
            hybrid_dense_weight /= weight_sum
            hybrid_sparse_weight /= weight_sum
        else:
            hybrid_dense_weight = 0.7
            hybrid_sparse_weight = 0.3

    # Story 20-H2 - Cross-Language Query settings
    cross_language_enabled = get_bool_env("CROSS_LANGUAGE_ENABLED", "false")
    cross_language_embedding = os.getenv(
        "CROSS_LANGUAGE_EMBEDDING", "intfloat/multilingual-e5-base"
    )
    cross_language_translation = get_bool_env("CROSS_LANGUAGE_TRANSLATION", "false")

    # Story 20-H3 - External Data Source Sync settings
    external_sync_enabled = get_bool_env("EXTERNAL_SYNC_ENABLED", "false")
    sync_sources = os.getenv("SYNC_SOURCES", "")
    s3_sync_bucket = os.getenv("S3_SYNC_BUCKET", "")
    s3_sync_prefix = os.getenv("S3_SYNC_PREFIX", "")
    confluence_url = os.getenv("CONFLUENCE_URL", "")
    confluence_api_token = os.getenv("CONFLUENCE_API_TOKEN", "")
    confluence_spaces = os.getenv("CONFLUENCE_SPACES", "")
    notion_api_key = os.getenv("NOTION_API_KEY", "")
    notion_database_ids = os.getenv("NOTION_DATABASE_IDS", "")

    # Story 20-H4 - Voice I/O settings
    voice_io_enabled = get_bool_env("VOICE_IO_ENABLED", "false")
    whisper_model = os.getenv("WHISPER_MODEL", "base")
    tts_provider = os.getenv("TTS_PROVIDER", "openai")
    tts_voice = os.getenv("TTS_VOICE", "alloy")
    tts_speed = get_float_env("TTS_SPEED", 1.0, min_val=0.25, max_val=4.0)

    # Story 20-H5 - ColBERT Reranking settings
    colbert_enabled = get_bool_env("COLBERT_ENABLED", "false")
    colbert_model = os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0")
    colbert_max_length = get_int_env("COLBERT_MAX_LENGTH", 512, min_val=64)

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
        embedding_dimension=embedding_dimension,
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
        neo4j_transaction_timeout_seconds=neo4j_transaction_timeout_seconds,
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
        crawler_strict_validation=crawler_strict_validation,
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
        # Epic 14 - Enhanced A2A Protocol settings
        a2a_enabled=a2a_enabled,
        a2a_agent_id=a2a_agent_id,
        a2a_endpoint_url=a2a_endpoint_url,
        a2a_heartbeat_interval_seconds=a2a_heartbeat_interval_seconds,
        a2a_heartbeat_timeout_seconds=a2a_heartbeat_timeout_seconds,
        a2a_task_default_timeout_seconds=a2a_task_default_timeout_seconds,
        a2a_task_max_retries=a2a_task_max_retries,
        # Story 22-A2 - A2A Resource Limits settings
        a2a_limits_backend=a2a_limits_backend,
        a2a_session_limit_per_tenant=a2a_session_limit_per_tenant,
        a2a_message_limit_per_session=a2a_message_limit_per_session,
        a2a_session_ttl_hours=a2a_session_ttl_hours,
        a2a_message_rate_limit=a2a_message_rate_limit,
        a2a_limits_cleanup_interval_minutes=a2a_limits_cleanup_interval_minutes,
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
        grader_model=grader_model,
        grader_threshold=grader_threshold,
        grader_fallback_enabled=grader_fallback_enabled,
        grader_fallback_strategy=grader_fallback_strategy,
        tavily_api_key=tavily_api_key,
        # Story 19-F4 - Heuristic grader length weight settings
        grader_heuristic_length_weight=grader_heuristic_length_weight,
        grader_heuristic_min_length=grader_heuristic_min_length,
        grader_heuristic_max_length=grader_heuristic_max_length,
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
        # Epic 15 - Codebase Intelligence settings
        codebase_hallucination_threshold=codebase_hallucination_threshold,
        codebase_detector_mode=codebase_detector_mode,
        codebase_cache_ttl_seconds=codebase_cache_ttl_seconds,
        codebase_symbol_table_max_symbols=codebase_symbol_table_max_symbols,
        codebase_index_rate_limit_max=codebase_index_rate_limit_max,
        codebase_index_rate_limit_window_seconds=codebase_index_rate_limit_window_seconds,
        codebase_detection_slow_ms=codebase_detection_slow_ms,
        codebase_allowed_base_path=codebase_allowed_base_path,
        codebase_rag_enabled=codebase_rag_enabled,
        codebase_languages=codebase_languages,
        codebase_exclude_patterns=codebase_exclude_patterns,
        codebase_max_chunk_size=codebase_max_chunk_size,
        codebase_include_class_context=codebase_include_class_context,
        codebase_incremental_indexing=codebase_incremental_indexing,
        codebase_index_cache_ttl_seconds=codebase_index_cache_ttl_seconds,
        # Story 19-C5 - Prometheus Observability settings
        prometheus_enabled=get_bool_env("PROMETHEUS_ENABLED", "false"),
        prometheus_path=os.getenv("PROMETHEUS_PATH", "/metrics"),
        # Story 19-G1 - Reranking Cache settings
        reranker_cache_enabled=reranker_cache_enabled,
        reranker_cache_ttl_seconds=reranker_cache_ttl_seconds,
        reranker_cache_max_size=reranker_cache_max_size,
        # Story 19-G2 - Contextual Retrieval Prompt Path
        contextual_retrieval_prompt_path=contextual_retrieval_prompt_path,
        # Story 19-G3 - Model Preloading settings
        grader_preload_model=grader_preload_model,
        reranker_preload_model=reranker_preload_model,
        # Story 19-G4 - Score Normalization settings
        grader_normalization_strategy=grader_normalization_strategy,
        # Epic 20 - Memory Platform settings (Story 20-A1)
        memory_scopes_enabled=memory_scopes_enabled,
        memory_default_scope=memory_default_scope,
        memory_include_parent_scopes=memory_include_parent_scopes,
        memory_cache_ttl_seconds=memory_cache_ttl_seconds,
        memory_max_per_scope=memory_max_per_scope,
        # Story 20-A2 - Memory Consolidation
        memory_consolidation_enabled=memory_consolidation_enabled,
        memory_consolidation_schedule=memory_consolidation_schedule,
        memory_similarity_threshold=memory_similarity_threshold,
        memory_decay_half_life_days=memory_decay_half_life_days,
        memory_min_importance=memory_min_importance,
        memory_consolidation_batch_size=memory_consolidation_batch_size,
        # Story 20-B1 - Community Detection
        community_detection_enabled=community_detection_enabled,
        community_algorithm=community_algorithm,
        community_min_size=community_min_size,
        community_max_levels=community_max_levels,
        community_summary_model=community_summary_model,
        community_refresh_schedule=community_refresh_schedule,
        # Story 20-B2 - LazyRAG Pattern
        lazy_rag_enabled=lazy_rag_enabled,
        lazy_rag_max_entities=lazy_rag_max_entities,
        lazy_rag_max_hops=lazy_rag_max_hops,
        lazy_rag_summary_model=lazy_rag_summary_model,
        lazy_rag_use_communities=lazy_rag_use_communities,
        # Story 20-B3 - Query Routing
        query_routing_enabled=query_routing_enabled,
        query_routing_use_llm=query_routing_use_llm,
        query_routing_llm_model=query_routing_llm_model,
        query_routing_confidence_threshold=query_routing_confidence_threshold,
        # Story 20-C1 - Graph-Based Rerankers
        graph_reranker_enabled=graph_reranker_enabled,
        graph_reranker_type=graph_reranker_type,
        graph_reranker_episode_weight=graph_reranker_episode_weight,
        graph_reranker_distance_weight=graph_reranker_distance_weight,
        graph_reranker_original_weight=graph_reranker_original_weight,
        graph_reranker_episode_window_days=graph_reranker_episode_window_days,
        graph_reranker_max_distance=graph_reranker_max_distance,
        # Story 20-C2 - Dual-Level Retrieval
        dual_level_retrieval_enabled=dual_level_retrieval_enabled,
        dual_level_low_weight=dual_level_low_weight,
        dual_level_high_weight=dual_level_high_weight,
        dual_level_low_limit=dual_level_low_limit,
        dual_level_high_limit=dual_level_high_limit,
        dual_level_synthesis_model=dual_level_synthesis_model,
        dual_level_synthesis_temperature=dual_level_synthesis_temperature,
        # Story 20-C3 - Parent-Child Chunk Hierarchy
        hierarchical_chunks_enabled=hierarchical_chunks_enabled,
        hierarchical_chunk_levels=hierarchical_chunk_levels,
        hierarchical_overlap_ratio=hierarchical_overlap_ratio,
        small_to_big_return_level=small_to_big_return_level,
        hierarchical_embedding_level=hierarchical_embedding_level,
        # Story 20-D1 - Enhanced Table/Layout Extraction
        enhanced_docling_enabled=enhanced_docling_enabled,
        docling_table_extraction=docling_table_extraction,
        docling_preserve_layout=docling_preserve_layout,
        docling_table_as_markdown=docling_table_as_markdown,
        # Story 20-D2 - Multimodal Ingestion
        multimodal_ingestion_enabled=multimodal_ingestion_enabled,
        office_docs_enabled=office_docs_enabled,
        # Story 20-E1 - Ontology Support
        ontology_support_enabled=ontology_support_enabled,
        ontology_path=ontology_path,
        ontology_auto_type=ontology_auto_type,
        # Story 20-E2 - Self-Improving Feedback Loop
        feedback_loop_enabled=feedback_loop_enabled,
        feedback_min_samples=feedback_min_samples,
        feedback_decay_days=feedback_decay_days,
        feedback_boost_max=feedback_boost_max,
        feedback_boost_min=feedback_boost_min,
        # Story 20-H1 - Sparse Vector Search (BM42)
        sparse_vectors_enabled=sparse_vectors_enabled,
        sparse_model=sparse_model,
        hybrid_dense_weight=hybrid_dense_weight,
        hybrid_sparse_weight=hybrid_sparse_weight,
        # Story 20-H2 - Cross-Language Query
        cross_language_enabled=cross_language_enabled,
        cross_language_embedding=cross_language_embedding,
        cross_language_translation=cross_language_translation,
        # Story 20-H3 - External Data Source Sync
        external_sync_enabled=external_sync_enabled,
        sync_sources=sync_sources,
        s3_sync_bucket=s3_sync_bucket,
        s3_sync_prefix=s3_sync_prefix,
        confluence_url=confluence_url,
        confluence_api_token=confluence_api_token,
        confluence_spaces=confluence_spaces,
        notion_api_key=notion_api_key,
        notion_database_ids=notion_database_ids,
        # Story 20-H4 - Voice I/O
        voice_io_enabled=voice_io_enabled,
        whisper_model=whisper_model,
        tts_provider=tts_provider,
        tts_voice=tts_voice,
        tts_speed=tts_speed,
        # Story 20-H5 - ColBERT Reranking
        colbert_enabled=colbert_enabled,
        colbert_model=colbert_model,
        colbert_max_length=colbert_max_length,
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
