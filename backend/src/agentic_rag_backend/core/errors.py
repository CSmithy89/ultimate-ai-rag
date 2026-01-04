"""Error handling with RFC 7807 Problem Details support."""

from enum import Enum
from http import HTTPStatus
import json
from typing import Any, Optional

from fastapi import HTTPException, Request
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
    # Story 4.2 - PDF Document Parsing error codes
    INVALID_PDF = "invalid_pdf"
    FILE_TOO_LARGE = "file_too_large"
    PASSWORD_PROTECTED = "password_protected"
    PARSE_FAILED = "parse_failed"
    STORAGE_ERROR = "storage_error"
    # Story 4.3 - Agentic Entity Extraction error codes
    EXTRACTION_FAILED = "extraction_failed"
    EMBEDDING_FAILED = "embedding_failed"
    GRAPH_BUILD_FAILED = "graph_build_failed"
    NEO4J_ERROR = "neo4j_error"
    DEDUPLICATION_FAILED = "deduplication_failed"
    CHUNKING_FAILED = "chunking_failed"
    # Epic 5 - Graphiti Ingestion error codes
    INGESTION_FAILED = "ingestion_failed"
    # Epic 15 - Codebase Intelligence error codes
    CODEBASE_VALIDATION_FAILED = "codebase_validation_failed"
    CODEBASE_INDEX_FAILED = "codebase_index_failed"
    HALLUCINATION_DETECTED = "hallucination_detected"
    # Epic 14 - A2A Protocol error codes
    A2A_AGENT_NOT_FOUND = "a2a_agent_not_found"
    A2A_AGENT_UNHEALTHY = "a2a_agent_unhealthy"
    A2A_CAPABILITY_NOT_FOUND = "a2a_capability_not_found"
    A2A_TASK_NOT_FOUND = "a2a_task_not_found"
    A2A_TASK_TIMEOUT = "a2a_task_timeout"
    A2A_DELEGATION_FAILED = "a2a_delegation_failed"
    A2A_REGISTRATION_FAILED = "a2a_registration_failed"


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


# Story 4.2 - PDF Document Parsing Errors


class InvalidPdfError(AppError):
    """Error for invalid PDF files."""

    def __init__(self, filename: str, reason: str = "File is not a valid PDF document") -> None:
        super().__init__(
            code=ErrorCode.INVALID_PDF,
            message=f"Invalid PDF: {reason}",
            status=400,
            details={"filename": filename},
        )


class FileTooLargeError(AppError):
    """Error when uploaded file exceeds size limit."""

    def __init__(self, filename: str, max_size_mb: int) -> None:
        super().__init__(
            code=ErrorCode.FILE_TOO_LARGE,
            message=f"File size exceeds the maximum allowed size of {max_size_mb}MB",
            status=413,
            details={"filename": filename, "max_size_mb": max_size_mb},
        )


class PasswordProtectedError(AppError):
    """Error for password-protected PDFs (not supported in MVP)."""

    def __init__(self, filename: str) -> None:
        super().__init__(
            code=ErrorCode.PASSWORD_PROTECTED,
            message="Password-protected PDFs are not supported",
            status=400,
            details={"filename": filename},
        )


class ParseError(AppError):
    """Error during document parsing."""

    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.PARSE_FAILED,
            message=f"Failed to parse document: {reason}",
            status=500,
            details={"filename": filename},
        )


class StorageError(AppError):
    """Error during file storage operations."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.STORAGE_ERROR,
            message=f"Storage error during {operation}: {reason}",
            status=500,
        )


# Story 4.3 - Agentic Entity Extraction Errors


class ExtractionError(AppError):
    """Error during entity extraction."""

    def __init__(self, chunk_id: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.EXTRACTION_FAILED,
            message=f"Entity extraction failed: {reason}",
            status=500,
            details={"chunk_id": chunk_id},
        )


class EmbeddingError(AppError):
    """Error during embedding generation."""

    def __init__(self, reason: str, batch_size: Optional[int] = None) -> None:
        details = {}
        if batch_size is not None:
            details["batch_size"] = batch_size
        super().__init__(
            code=ErrorCode.EMBEDDING_FAILED,
            message=f"Embedding generation failed: {reason}",
            status=500,
            details=details,
        )


class GraphBuildError(AppError):
    """Error during knowledge graph construction."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.GRAPH_BUILD_FAILED,
            message=f"Graph build failed during {operation}: {reason}",
            status=500,
            details={"operation": operation},
        )


class Neo4jError(AppError):
    """Neo4j database operation error."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.NEO4J_ERROR,
            message=f"Neo4j error during {operation}: {reason}",
            status=500,
            details={"operation": operation},
        )


class DeduplicationError(AppError):
    """Error during entity deduplication."""

    def __init__(self, entity_name: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.DEDUPLICATION_FAILED,
            message=f"Entity deduplication failed for '{entity_name}': {reason}",
            status=500,
            details={"entity_name": entity_name},
        )


class ChunkingError(AppError):
    """Error during document chunking."""

    def __init__(self, document_id: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.CHUNKING_FAILED,
            message=f"Document chunking failed: {reason}",
            status=500,
            details={"document_id": document_id},
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


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Convert HTTPException errors to RFC 7807 Problem Details responses."""
    status_code = exc.status_code
    title = HTTPStatus(status_code).phrase if status_code in HTTPStatus else "HTTP Error"
    detail = exc.detail
    if not isinstance(detail, str):
        detail = json.dumps(detail)
    problem = {
        "type": f"https://api.example.com/errors/http-{status_code}",
        "title": title,
        "status": status_code,
        "detail": detail,
        "instance": str(request.url.path),
    }
    return JSONResponse(status_code=status_code, content=problem, headers=exc.headers)


# Epic 5 - Graphiti Ingestion Errors


class IngestionError(AppError):
    """Error during Graphiti episode ingestion."""

    def __init__(self, document_id: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.INGESTION_FAILED,
            message=f"Episode ingestion failed: {reason}",
            status=500,
            details={"document_id": document_id},
        )


# Epic 15 - Codebase Intelligence Errors


class CodebaseValidationError(AppError):
    """Error during codebase response validation."""

    def __init__(self, reason: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            code=ErrorCode.CODEBASE_VALIDATION_FAILED,
            message=f"Codebase validation failed: {reason}",
            status=400,
            details=details,
        )


class CodebaseIndexError(AppError):
    """Error during codebase repository indexing."""

    def __init__(self, repo_path: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.CODEBASE_INDEX_FAILED,
            message=f"Repository indexing failed: {reason}",
            status=500,
            details={"repo_path": repo_path},
        )


class HallucinationError(AppError):
    """Error when hallucination threshold exceeded in block mode."""

    def __init__(
        self,
        invalid_count: int,
        total_count: int,
        threshold: float,
        invalid_symbols: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        ratio = invalid_count / total_count if total_count > 0 else 0
        details: dict[str, Any] = {
            "invalid_count": invalid_count,
            "total_count": total_count,
            "ratio": ratio,
            "threshold": threshold,
        }
        if invalid_symbols:
            details["invalid_symbols"] = invalid_symbols
        super().__init__(
            code=ErrorCode.HALLUCINATION_DETECTED,
            message=f"Response blocked: {invalid_count}/{total_count} hallucinated references "
                    f"({ratio:.1%} exceeds {threshold:.1%} threshold)",
            status=422,
            details=details,
        )


# Epic 14 - A2A Protocol Errors


class A2AAgentNotFoundError(AppError):
    """Error when an A2A agent is not found."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(
            code=ErrorCode.A2A_AGENT_NOT_FOUND,
            message=f"Agent '{agent_id}' not found in registry",
            status=404,
            details={"agent_id": agent_id},
        )


class A2AAgentUnhealthyError(AppError):
    """Error when an A2A agent is unhealthy or unresponsive."""

    def __init__(self, agent_id: str, reason: str = "Agent is not responding to heartbeats") -> None:
        super().__init__(
            code=ErrorCode.A2A_AGENT_UNHEALTHY,
            message=f"Agent '{agent_id}' is unhealthy: {reason}",
            status=503,
            details={"agent_id": agent_id, "reason": reason},
        )


class A2ACapabilityNotFoundError(AppError):
    """Error when no agent with required capability is found."""

    def __init__(self, capability_name: str) -> None:
        super().__init__(
            code=ErrorCode.A2A_CAPABILITY_NOT_FOUND,
            message=f"No healthy agent found with capability '{capability_name}'",
            status=404,
            details={"capability_name": capability_name},
        )


class A2ATaskNotFoundError(AppError):
    """Error when an A2A task is not found."""

    def __init__(self, task_id: str) -> None:
        super().__init__(
            code=ErrorCode.A2A_TASK_NOT_FOUND,
            message=f"Task '{task_id}' not found",
            status=404,
            details={"task_id": task_id},
        )


class A2ATaskTimeoutError(AppError):
    """Error when an A2A task times out."""

    def __init__(self, task_id: str, timeout_seconds: int) -> None:
        super().__init__(
            code=ErrorCode.A2A_TASK_TIMEOUT,
            message=f"Task '{task_id}' timed out after {timeout_seconds} seconds",
            status=504,
            details={"task_id": task_id, "timeout_seconds": timeout_seconds},
        )


class A2ADelegationError(AppError):
    """Error when task delegation fails."""

    def __init__(self, reason: str, task_id: Optional[str] = None) -> None:
        details: dict[str, Any] = {}
        if task_id:
            details["task_id"] = task_id
        super().__init__(
            code=ErrorCode.A2A_DELEGATION_FAILED,
            message=f"Task delegation failed: {reason}",
            status=500,
            details=details,
        )


class A2ARegistrationError(AppError):
    """Error when agent registration fails."""

    def __init__(self, agent_id: str, reason: str) -> None:
        super().__init__(
            code=ErrorCode.A2A_REGISTRATION_FAILED,
            message=f"Agent registration failed: {reason}",
            status=400,
            details={"agent_id": agent_id},
        )
