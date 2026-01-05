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

Configuration:
- MEMORY_SCOPES_ENABLED: Enable/disable the feature (default: false)
- MEMORY_DEFAULT_SCOPE: Default scope for new memories (default: session)
- MEMORY_INCLUDE_PARENT_SCOPES: Include parent scopes in search (default: true)
- MEMORY_CACHE_TTL_SECONDS: Redis cache TTL (default: 3600)
- MEMORY_MAX_PER_SCOPE: Maximum memories per scope (default: 10000)
"""

from .errors import (
    MemoryLimitExceededError,
    MemoryNotFoundError,
    MemoryScopeError,
)
from .models import (
    MemoryScope,
    MemorySearchRequest,
    MemorySearchResponse,
    ScopedMemory,
    ScopedMemoryCreate,
    ScopedMemoryUpdate,
)
from .scopes import get_parent_scopes, validate_scope_context
from .store import ScopedMemoryStore

__all__ = [
    "MemoryLimitExceededError",
    "MemoryNotFoundError",
    "MemoryScopeError",
    "MemoryScope",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "ScopedMemory",
    "ScopedMemoryCreate",
    "ScopedMemoryUpdate",
    "ScopedMemoryStore",
    "get_parent_scopes",
    "validate_scope_context",
]
