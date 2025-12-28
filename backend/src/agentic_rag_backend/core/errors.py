"""Error handling with RFC 7807 Problem Details support."""

from enum import Enum
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse


class ErrorCode(str, Enum):
    """Standardized error codes for the application."""

    VALIDATION_ERROR = "validation_error"
    INVALID_URL = "invalid_url"
    JOB_NOT_FOUND = "job_not_found"
    TENANT_REQUIRED = "tenant_required"
    CRAWL_FAILED = "crawl_failed"
    DATABASE_ERROR = "database_error"
    REDIS_ERROR = "redis_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INTERNAL_ERROR = "internal_error"


class AppError(Exception):
    """
    Structured application error following RFC 7807 Problem Details.

    Attributes:
        code: Error code from ErrorCode enum
        message: Human-readable error message
        status: HTTP status code
        details: Additional error context
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status: int = 500,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}
        super().__init__(message)

    def to_problem_detail(self, instance: str) -> dict[str, Any]:
        """
        Convert error to RFC 7807 Problem Details format.

        Args:
            instance: The request path where the error occurred

        Returns:
            Dictionary in RFC 7807 format
        """
        problem = {
            "type": f"https://api.example.com/errors/{self.code.value.replace('_', '-')}",
            "title": self.code.value.replace("_", " ").title(),
            "status": self.status,
            "detail": self.message,
            "instance": instance,
        }
        if self.details:
            problem["errors"] = self.details
        return problem


class ValidationError(AppError):
    """Validation error for request data."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            status=400,
            details=details,
        )


class InvalidUrlError(AppError):
    """Error for invalid or inaccessible URLs."""

    def __init__(self, url: str, reason: str = "URL is not valid or accessible") -> None:
        super().__init__(
            code=ErrorCode.INVALID_URL,
            message=f"Invalid URL: {reason}",
            status=400,
            details={"url": url},
        )


class JobNotFoundError(AppError):
    """Error when a job is not found."""

    def __init__(self, job_id: str) -> None:
        super().__init__(
            code=ErrorCode.JOB_NOT_FOUND,
            message=f"Job with ID '{job_id}' not found",
            status=404,
            details={"job_id": job_id},
        )


class TenantRequiredError(AppError):
    """Error when tenant_id is missing."""

    def __init__(self) -> None:
        super().__init__(
            code=ErrorCode.TENANT_REQUIRED,
            message="tenant_id is required for this operation",
            status=400,
        )


class CrawlError(AppError):
    """Error during crawling operation."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.CRAWL_FAILED,
            message=f"Crawl failed: {reason}",
            status=500,
            details={"url": url},
        )


class DatabaseError(AppError):
    """Database operation error."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.DATABASE_ERROR,
            message=f"Database error during {operation}: {reason}",
            status=500,
        )


class RedisError(AppError):
    """Redis operation error."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.REDIS_ERROR,
            message=f"Redis error during {operation}: {reason}",
            status=500,
        )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """
    FastAPI exception handler for AppError.

    Converts AppError to RFC 7807 Problem Details JSON response.

    Args:
        request: The FastAPI request object
        exc: The AppError exception

    Returns:
        JSONResponse with Problem Details format
    """
    return JSONResponse(
        status_code=exc.status,
        content=exc.to_problem_detail(str(request.url.path)),
    )
