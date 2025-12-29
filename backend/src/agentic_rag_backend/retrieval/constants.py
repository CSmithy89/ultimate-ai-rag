"""Shared configuration constants for retrieval services."""

DEFAULT_VECTOR_LIMIT = 8  # Size of pgvector result set before prompt trimming.
MAX_VECTOR_HITS = 6  # Keep evidence concise for prompt length.
MAX_GRAPH_PATHS = 5  # Limit explainability to top paths.
MAX_VECTOR_CONTENT_CHARS = 500  # Truncate long chunks in prompts.
DEFAULT_SIMILARITY_THRESHOLD = 0.7  # Balanced precision/recall for MVP.
DEFAULT_MAX_HOPS = 2  # Balanced for speed and explainability in MVP.
DEFAULT_PATH_LIMIT = 10  # Bounded result size for API responses.
DEFAULT_ENTITY_LIMIT = 12  # Seed entity cap for traversal.
DEFAULT_RETRIEVAL_TIMEOUT_SECONDS = 15.0  # Avoid hanging retrieval calls.
DEFAULT_RETRIEVAL_CACHE_TTL_SECONDS = 30.0  # Short-lived cache for hot queries.
DEFAULT_RETRIEVAL_CACHE_SIZE = 128
