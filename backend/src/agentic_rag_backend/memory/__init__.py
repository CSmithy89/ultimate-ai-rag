"""Memory Platform module for hierarchical scoped memories.

This module implements memory scopes similar to Mem0's approach:
- USER scope: Persists across all sessions for a user
- SESSION scope: Persists within a single conversation session
- AGENT scope: Persists across agent invocations (operational memory)
- GLOBAL scope: Tenant-wide shared memory

The scope hierarchy is:
- SESSION includes USER and GLOBAL memories when searching
- USER includes GLOBAL memories when searching
- AGENT includes GLOBAL memories when searching

Configuration (Story 20-A1):
- MEMORY_SCOPES_ENABLED: Enable/disable the feature (default: false)
- MEMORY_DEFAULT_SCOPE: Default scope for new memories (default: session)
- MEMORY_INCLUDE_PARENT_SCOPES: Include parent scopes in search (default: true)
- MEMORY_CACHE_TTL_SECONDS: Redis cache TTL (default: 3600)
- MEMORY_MAX_PER_SCOPE: Maximum memories per scope (default: 10000)

Memory Consolidation (Story 20-A2):
- MEMORY_CONSOLIDATION_ENABLED: Enable automatic consolidation (default: false)
- MEMORY_CONSOLIDATION_SCHEDULE: Cron schedule (default: "0 2 * * *" = 2 AM daily)
- MEMORY_SIMILARITY_THRESHOLD: Threshold for duplicate detection (default: 0.9)
- MEMORY_DECAY_HALF_LIFE_DAYS: Days for importance to halve (default: 30)
- MEMORY_MIN_IMPORTANCE: Minimum importance before removal (default: 0.1)
"""

from .errors import (
    MemoryLimitExceededError,
    MemoryNotFoundError,
    MemoryScopeError,
)
from .models import (
    ConsolidationRequest,
    ConsolidationResult,
    ConsolidationStatusResponse,
    MemoryScope,
    MemorySearchRequest,
    MemorySearchResponse,
    ScopedMemory,
    ScopedMemoryCreate,
    ScopedMemoryUpdate,
)
from .scopes import get_parent_scopes, validate_scope_context
from .store import ScopedMemoryStore
from .consolidation import (
    MemoryConsolidator,
    calculate_importance,
    cosine_similarity,
)
from .scheduler import (
    MemoryConsolidationScheduler,
    create_consolidation_scheduler,
    parse_cron_schedule,
    APSCHEDULER_AVAILABLE,
)

__all__ = [
    # Errors
    "MemoryLimitExceededError",
    "MemoryNotFoundError",
    "MemoryScopeError",
    # Models (Story 20-A1)
    "MemoryScope",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "ScopedMemory",
    "ScopedMemoryCreate",
    "ScopedMemoryUpdate",
    # Store
    "ScopedMemoryStore",
    # Scopes
    "get_parent_scopes",
    "validate_scope_context",
    # Consolidation Models (Story 20-A2)
    "ConsolidationRequest",
    "ConsolidationResult",
    "ConsolidationStatusResponse",
    # Consolidation Classes (Story 20-A2)
    "MemoryConsolidator",
    "calculate_importance",
    "cosine_similarity",
    # Scheduler (Story 20-A2)
    "MemoryConsolidationScheduler",
    "create_consolidation_scheduler",
    "parse_cron_schedule",
    "APSCHEDULER_AVAILABLE",
]
