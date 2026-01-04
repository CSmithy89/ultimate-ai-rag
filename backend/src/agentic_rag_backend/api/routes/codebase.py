"""API endpoints for codebase hallucination detection.

Provides endpoints for validating LLM responses against a codebase
to detect hallucinated code references.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from agentic_rag_backend.config import get_settings
from agentic_rag_backend.codebase import (
    DetectorMode,
    HallucinationDetector,
    SymbolTable,
)
from agentic_rag_backend.codebase.indexing import CodebaseIndexer
from agentic_rag_backend.codebase.retrieval import CodeSearchService
from agentic_rag_backend.codebase.detector import index_repository, load_repo_dependencies
from agentic_rag_backend.codebase.symbol_table import (
    cache_symbol_table,
    get_cached_symbol_table,
)
from agentic_rag_backend.core.errors import (
    AppError,
    CodebaseIndexError,
    CodebaseValidationError,
    ErrorCode,
    HallucinationError,
)
from agentic_rag_backend.embeddings import EmbeddingGenerator
from agentic_rag_backend.llm.providers import get_embedding_adapter

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/codebase", tags=["codebase"])


# Pydantic models for request/response
class ValidateResponseRequest(BaseModel):
    """Request model for response validation."""

    response_text: str = Field(
        ...,
        description="The LLM response text to validate",
        min_length=1,
        max_length=100000,
    )
    tenant_id: UUID = Field(
        ...,
        description="Tenant identifier for multi-tenancy",
    )
    repo_path: Optional[str] = Field(
        None,
        description="Path to the repository (uses cached if available)",
    )
    mode: Optional[str] = Field(
        None,
        description="Detector mode: 'warn' or 'block'",
    )
    threshold: Optional[float] = Field(
        None,
        description="Hallucination threshold (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


class ValidationResultResponse(BaseModel):
    """Response model for a single validation result."""

    symbol_name: str
    is_valid: bool
    confidence: float
    reason: str
    suggestions: list[str]
    symbol_type: Optional[str] = None


class HallucinationReportResponse(BaseModel):
    """Response model for hallucination detection report."""

    total_symbols_checked: int
    valid_symbols: int
    invalid_symbols: int
    uncertain_symbols: int
    validation_results: list[ValidationResultResponse]
    files_checked: list[str]
    processing_time_ms: int
    confidence_score: float
    should_block: bool


class IndexRepositoryRequest(BaseModel):
    """Request model for repository indexing."""

    repo_path: str = Field(
        ...,
        description="Absolute path to the repository root",
    )
    tenant_id: UUID = Field(
        ...,
        description="Tenant identifier for multi-tenancy",
    )
    ignore_patterns: Optional[list[str]] = Field(
        None,
        description="Additional glob patterns to ignore",
    )
    cache_ttl_seconds: Optional[int] = Field(
        3600,
        description="TTL for caching the symbol table",
        ge=60,
        le=86400,
    )


class IndexRepositoryResponse(BaseModel):
    """Response model for repository indexing."""

    symbol_count: int
    file_count: int
    cached: bool
    cache_key: Optional[str] = None


class IndexCodebaseRequest(BaseModel):
    """Request model for codebase RAG indexing."""

    repo_path: str = Field(
        ...,
        description="Path to repository to index",
    )
    tenant_id: UUID = Field(
        ...,
        description="Tenant identifier for multi-tenancy",
    )
    languages: Optional[list[str]] = Field(
        None,
        description="Languages to index (defaults to CODEBASE_LANGUAGES)",
    )
    incremental: bool = Field(
        False,
        description="Only index changed files",
    )


class IndexCodebaseResponse(BaseModel):
    """Response model for codebase indexing."""

    files_indexed: int
    symbols_extracted: int
    chunks_created: int
    relationships_created: int
    processing_time_ms: int
    errors: list[str]


class CodeSearchRequest(BaseModel):
    """Request model for codebase search."""

    query: str = Field(..., description="Natural language query about the codebase", min_length=1)
    tenant_id: UUID = Field(
        ...,
        description="Tenant identifier for multi-tenancy",
    )
    limit: int = Field(10, ge=1, le=50)
    include_relationships: bool = Field(True)


class CodeSearchResult(BaseModel):
    """Response model for codebase search result."""

    symbol_name: str
    symbol_type: str
    file_path: str
    line_start: int
    line_end: int
    content: str
    score: float
    relationships: list[dict]


class SymbolTableStatsResponse(BaseModel):
    """Response model for symbol table statistics."""

    tenant_id: str
    repo_path: str
    symbol_count: int
    file_count: int
    symbols_by_type: dict[str, int]


class Meta(BaseModel):
    """Response metadata."""

    requestId: str
    timestamp: str


class SuccessResponse(BaseModel):
    """Standard success response wrapper."""

    data: Any
    meta: Meta


def success_response(data: Any) -> dict[str, Any]:
    """Wrap data in standard success response format."""
    return {
        "data": data,
        "meta": {
            "requestId": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }


# In-memory symbol table cache (per-tenant)
# In production, use Redis caching
_symbol_tables: dict[str, SymbolTable] = {}


async def get_redis_client(request: Request):
    """Get Redis client from app.state."""
    return getattr(request.app.state, "redis_client", None)


@router.post(
    "/validate-response",
    response_model=SuccessResponse,
    summary="Validate LLM response for hallucinations",
    description="Check if an LLM response contains hallucinated code references.",
)
async def validate_response(
    request: Request,
    body: ValidateResponseRequest,
) -> dict[str, Any]:
    """Validate an LLM response for hallucinated code references.

    Extracts code symbols, file paths, API endpoints, and imports from
    the response text and validates them against the indexed codebase.

    Args:
        request: FastAPI request object
        body: Validation request with response text and options

    Returns:
        Success response with HallucinationReport data

    Raises:
        HTTPException: If validation fails or symbol table not found
    """
    settings = get_settings()
    tenant_id = str(body.tenant_id)

    logger.info(
        "validate_response_request",
        tenant_id=tenant_id,
        response_length=len(body.response_text),
        mode=body.mode,
    )

    # Get or load symbol table
    symbol_table = _symbol_tables.get(tenant_id)

    if symbol_table is None:
        # Try to load from Redis cache
        redis_client = await get_redis_client(request)
        if redis_client and body.repo_path:
            symbol_table = await get_cached_symbol_table(
                redis_client,
                tenant_id,
                body.repo_path,
            )
            if symbol_table:
                _symbol_tables[tenant_id] = symbol_table

    if symbol_table is None:
        # If repo_path provided, index it
        if body.repo_path:
            try:
                symbol_table = await index_repository(
                    body.repo_path,
                    tenant_id,
                )
                _symbol_tables[tenant_id] = symbol_table

                # Cache in Redis if available
                redis_client = await get_redis_client(request)
                if redis_client:
                    await cache_symbol_table(
                        redis_client,
                        symbol_table,
                        ttl_seconds=settings.codebase_cache_ttl_seconds,
                    )
            except Exception as e:
                logger.error(
                    "repository_indexing_failed",
                    repo_path=body.repo_path,
                    error=str(e),
                )
                raise CodebaseIndexError(
                    repo_path=body.repo_path,
                    reason=str(e),
                ) from e
        else:
            raise CodebaseValidationError(
                reason="No symbol table found. Provide repo_path to index, or call /index-repository first.",
            )

    # Create detector with specified options
    mode_value = (body.mode or settings.codebase_detector_mode).lower()
    mode = DetectorMode.BLOCK if mode_value == "block" else DetectorMode.WARN
    threshold = body.threshold if body.threshold is not None else settings.codebase_hallucination_threshold

    detector = HallucinationDetector(
        symbol_table=symbol_table,
        mode=mode,
        threshold=threshold,
        openapi_spec=request.app.openapi(),
    )
    declared_packages = load_repo_dependencies(symbol_table.repo_path)
    if declared_packages:
        detector.add_installed_packages(sorted(declared_packages))

    # Validate the response
    report = detector.validate_response(body.response_text, tenant_id)
    should_block = detector.should_block(report)

    if should_block:
        invalid_symbols = [
            {
                "symbol_name": r.symbol_name,
                "reason": r.reason,
                "confidence": r.confidence,
                "suggestions": r.suggestions,
                "symbol_type": r.symbol_type.value if r.symbol_type else None,
            }
            for r in report.validation_results
            if not r.is_valid
        ]
        raise HallucinationError(
            invalid_count=report.invalid_symbols,
            total_count=report.total_symbols_checked,
            threshold=threshold,
            invalid_symbols=invalid_symbols,
        )

    # Convert to response model
    validation_results = [
        ValidationResultResponse(
            symbol_name=r.symbol_name,
            is_valid=r.is_valid,
            confidence=r.confidence,
            reason=r.reason,
            suggestions=r.suggestions,
            symbol_type=r.symbol_type.value if r.symbol_type else None,
        )
        for r in report.validation_results
    ]

    response_data = HallucinationReportResponse(
        total_symbols_checked=report.total_symbols_checked,
        valid_symbols=report.valid_symbols,
        invalid_symbols=report.invalid_symbols,
        uncertain_symbols=report.uncertain_symbols,
        validation_results=validation_results,
        files_checked=report.files_checked,
        processing_time_ms=report.processing_time_ms,
        confidence_score=report.confidence_score,
        should_block=should_block,
    )

    logger.info(
        "validate_response_complete",
        tenant_id=tenant_id,
        total_checked=report.total_symbols_checked,
        invalid=report.invalid_symbols,
        should_block=response_data.should_block,
        processing_time_ms=report.processing_time_ms,
    )

    return success_response(response_data.model_dump())


@router.post(
    "/index-repository",
    response_model=SuccessResponse,
    summary="Index a repository for validation",
    description="Scan a repository and build a symbol table for hallucination detection.",
)
async def index_repository_endpoint(
    request: Request,
    body: IndexRepositoryRequest,
) -> dict[str, Any]:
    """Index a repository and create a symbol table.

    Scans the repository for source files, extracts symbols,
    and caches the result for future validation requests.

    Args:
        request: FastAPI request object
        body: Index request with repository path and options

    Returns:
        Success response with indexing statistics

    Raises:
        HTTPException: If indexing fails
    """
    settings = get_settings()
    tenant_id = str(body.tenant_id)

    logger.info(
        "index_repository_request",
        tenant_id=tenant_id,
        repo_path=body.repo_path,
    )

    try:
        symbol_table = await index_repository(
            body.repo_path,
            tenant_id,
            body.ignore_patterns,
        )
    except Exception as e:
        logger.error(
            "index_repository_failed",
            repo_path=body.repo_path,
            error=str(e),
        )
        raise CodebaseIndexError(
            repo_path=body.repo_path,
            reason=str(e),
        ) from e

    # Store in memory cache
    _symbol_tables[tenant_id] = symbol_table

    # Cache in Redis if available
    cached = False
    cache_key = None
    redis_client = await get_redis_client(request)
    if redis_client:
        try:
            ttl = body.cache_ttl_seconds or settings.codebase_cache_ttl_seconds
            await cache_symbol_table(redis_client, symbol_table, ttl_seconds=ttl)
            cached = True
            cache_key = symbol_table.get_cache_key()
        except Exception as e:
            logger.warning("cache_symbol_table_failed", error=str(e))

    response_data = IndexRepositoryResponse(
        symbol_count=symbol_table.symbol_count(),
        file_count=symbol_table.file_count(),
        cached=cached,
        cache_key=cache_key,
    )

    logger.info(
        "index_repository_complete",
        tenant_id=tenant_id,
        symbol_count=response_data.symbol_count,
        file_count=response_data.file_count,
        cached=cached,
    )

    return success_response(response_data.model_dump())


@router.post(
    "/index",
    response_model=SuccessResponse,
    summary="Index a codebase for RAG queries",
    description="Scan a repository and build codebase RAG indexes.",
)
async def index_codebase(
    request: Request,
    body: IndexCodebaseRequest,
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.codebase_rag_enabled:
        raise CodebaseValidationError(reason="Codebase RAG is disabled.")

    repo_path_obj = Path(body.repo_path)
    if not repo_path_obj.is_absolute() or not repo_path_obj.exists() or not repo_path_obj.is_dir():
        raise CodebaseIndexError(repo_path=body.repo_path, reason="repo_path must be an existing absolute directory")

    postgres = getattr(request.app.state, "postgres", None)
    if postgres is None:
        raise CodebaseIndexError(repo_path=body.repo_path, reason="Postgres client not configured")

    embedding_adapter = get_embedding_adapter(settings)
    embedding_generator = EmbeddingGenerator.from_adapter(
        embedding_adapter,
        cost_tracker=getattr(request.app.state, "cost_tracker", None),
    )
    neo4j = getattr(request.app.state, "neo4j", None)
    redis_client = await get_redis_client(request)

    indexer = CodebaseIndexer(
        tenant_id=str(body.tenant_id),
        repo_path=body.repo_path,
        postgres=postgres,
        embedding_generator=embedding_generator,
        neo4j=neo4j,
        redis_client=redis_client,
        languages=body.languages or settings.codebase_languages,
        exclude_patterns=settings.codebase_exclude_patterns,
        max_chunk_size=settings.codebase_max_chunk_size,
        include_class_context=settings.codebase_include_class_context,
        cache_ttl_seconds=settings.codebase_index_cache_ttl_seconds,
    )

    try:
        if body.incremental and settings.codebase_incremental_indexing:
            result = await indexer.index_incremental()
        else:
            result = await indexer.index_full()
    except Exception as exc:
        logger.error("codebase_index_failed", repo_path=body.repo_path, error=str(exc))
        raise CodebaseIndexError(repo_path=body.repo_path, reason=str(exc)) from exc

    response_data = IndexCodebaseResponse(
        files_indexed=result.files_indexed,
        symbols_extracted=result.symbols_extracted,
        chunks_created=result.chunks_created,
        relationships_created=result.relationships_created,
        processing_time_ms=result.processing_time_ms,
        errors=result.errors,
    )

    return success_response(response_data.model_dump())


@router.post(
    "/search",
    response_model=SuccessResponse,
    summary="Search indexed codebase",
    description="Query the indexed codebase using natural language.",
)
async def search_codebase(
    request: Request,
    body: CodeSearchRequest,
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.codebase_rag_enabled:
        raise CodebaseValidationError(reason="Codebase RAG is disabled.")

    postgres = getattr(request.app.state, "postgres", None)
    if postgres is None:
        raise CodebaseValidationError(reason="Postgres client not configured.")

    embedding_adapter = get_embedding_adapter(settings)
    embedding_generator = EmbeddingGenerator.from_adapter(
        embedding_adapter,
        cost_tracker=getattr(request.app.state, "cost_tracker", None),
    )
    neo4j = getattr(request.app.state, "neo4j", None)

    search_service = CodeSearchService(
        postgres=postgres,
        embedding_generator=embedding_generator,
        neo4j=neo4j,
    )
    try:
        results = await search_service.search(
            tenant_id=str(body.tenant_id),
            query=body.query,
            limit=body.limit,
            include_relationships=body.include_relationships,
        )
    except Exception as exc:
        logger.error("codebase_search_failed", error=str(exc))
        raise CodebaseValidationError(reason=f"Codebase search failed: {exc}") from exc

    return success_response(results)


@router.get(
    "/symbol-table/stats",
    response_model=SuccessResponse,
    summary="Get symbol table statistics",
    description="Retrieve statistics about the indexed symbol table for a tenant.",
)
async def get_symbol_table_stats(
    tenant_id: UUID = Query(..., description="Tenant identifier"),
) -> dict[str, Any]:
    """Get statistics about the symbol table for a tenant.

    Args:
        tenant_id: Tenant identifier

    Returns:
        Success response with symbol table statistics

    Raises:
        HTTPException: If no symbol table found for tenant
    """
    tenant_id_str = str(tenant_id)
    symbol_table = _symbol_tables.get(tenant_id_str)

    if symbol_table is None:
        # Use AppError base class for RFC 7807 compliance
        raise AppError(
            code=ErrorCode.CODEBASE_VALIDATION_FAILED,
            message=f"No symbol table found for tenant {tenant_id}. Call /index-repository first.",
            status=404,
            details={"tenant_id": str(tenant_id)},
        )

    # Calculate symbols by type
    symbols_by_type: dict[str, int] = {}
    for symbol in symbol_table.get_all_symbols():
        type_name = symbol.type.value
        symbols_by_type[type_name] = symbols_by_type.get(type_name, 0) + 1

    response_data = SymbolTableStatsResponse(
        tenant_id=tenant_id_str,
        repo_path=symbol_table.repo_path,
        symbol_count=symbol_table.symbol_count(),
        file_count=symbol_table.file_count(),
        symbols_by_type=symbols_by_type,
    )

    return success_response(response_data.model_dump())


@router.delete(
    "/symbol-table",
    response_model=SuccessResponse,
    summary="Clear symbol table cache",
    description="Clear the cached symbol table for a tenant.",
)
async def clear_symbol_table(
    request: Request,
    tenant_id: UUID = Query(..., description="Tenant identifier"),
) -> dict[str, Any]:
    """Clear the cached symbol table for a tenant.

    Args:
        request: FastAPI request object
        tenant_id: Tenant identifier

    Returns:
        Success response confirming deletion
    """
    tenant_id_str = str(tenant_id)
    memory_cleared = False
    redis_cleared = False

    # Clear from memory
    if tenant_id_str in _symbol_tables:
        del _symbol_tables[tenant_id_str]
        memory_cleared = True

    # Also clear from Redis if available
    redis_client = await get_redis_client(request)
    if redis_client:
        try:
            # Delete all keys matching the tenant's codebase pattern
            pattern = f"codebase:{tenant_id_str}:*"
            async for key in redis_client.scan_iter(match=pattern):
                await redis_client.delete(key)
                redis_cleared = True
        except Exception as e:
            logger.warning(
                "redis_clear_failed",
                tenant_id=tenant_id_str,
                error=str(e),
            )

    logger.info(
        "symbol_table_cleared",
        tenant_id=tenant_id_str,
        memory_cleared=memory_cleared,
        redis_cleared=redis_cleared,
    )

    return success_response({
        "cleared": True,
        "tenant_id": tenant_id_str,
        "memory_cleared": memory_cleared,
        "redis_cleared": redis_cleared,
    })
