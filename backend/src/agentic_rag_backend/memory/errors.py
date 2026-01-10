"""Memory-specific exceptions for Epic 20 Memory Platform."""


from agentic_rag_backend.core.errors import AppError, ErrorCode


class MemoryNotFoundError(AppError):
    """Error when a memory is not found."""

    def __init__(self, memory_id: str) -> None:
        super().__init__(
            code=ErrorCode.MEMORY_NOT_FOUND,
            message="Requested memory not found",
            status=404,
            details={"memory_id": memory_id},
        )


class MemoryScopeError(AppError):
    """Error for invalid memory scope context."""

    def __init__(self, scope: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.MEMORY_SCOPE_INVALID,
            message=f"Invalid scope context for '{scope}': {reason}",
            status=400,
            details={"scope": scope, "reason": reason},
        )


class MemoryLimitExceededError(AppError):
    """Error when memory limit per scope is exceeded."""

    def __init__(self, scope: str, limit: int, current: int) -> None:
        super().__init__(
            code=ErrorCode.MEMORY_LIMIT_EXCEEDED,
            message=f"Memory limit exceeded for scope '{scope}': {current}/{limit}",
            status=429,
            details={"scope": scope, "limit": limit, "current_count": current},
        )
