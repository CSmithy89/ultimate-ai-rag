"""RAG extension tools for MCP.

Provides MCP tools for RAG operations including vector search,
hybrid retrieval, ingestion, and query processing.

Story 14-1: Expose RAG Engine via MCP Server
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import structlog

from ..types import MCPToolSpec, MCPError, MCPErrorCode, create_tool_input_schema
from ..registry import MCPServerRegistry
from ...validation import is_valid_tenant_id
from ...db.graphiti import GraphitiClient
from ...retrieval.vector_search import VectorSearchService
from ...retrieval.graphiti_retrieval import graphiti_search
from ...retrieval.reranking import (
    RerankerClient,
    RerankedHit,
)
from ...retrieval.types import VectorHit
from ...indexing.youtube_ingestion import ingest_youtube_video, YouTubeIngestionError
from ...indexing.crawler import crawl_url, is_valid_url
from ...indexing.parser import parse_pdf
from ...models.documents import UnifiedDocument, DocumentMetadata, SourceType
from ...config import get_settings

logger = structlog.get_logger(__name__)

# Default timeout values (can be overridden via MCP_TOOL_TIMEOUT_OVERRIDES)
DEFAULT_INGEST_URL_TIMEOUT = 120.0  # seconds
DEFAULT_INGEST_YOUTUBE_TIMEOUT = 60.0  # seconds
DEFAULT_INGEST_PDF_TIMEOUT = 120.0  # seconds

# Maximum content size to prevent DoS attacks (1MB)
MAX_CONTENT_SIZE = 1_000_000


def _build_pdf_page_chunks(parsed_doc) -> dict[int, list[str]]:
    pages: dict[int, list[str]] = {}

    def add(page_number: int, text: str) -> None:
        if text and text.strip():
            pages.setdefault(page_number, []).append(text.strip())

    for section in parsed_doc.sections:
        if section.heading:
            add(
                section.page_number,
                f"{'#' * section.level} {section.heading}",
            )
        add(section.page_number, section.content)

    for table in parsed_doc.tables:
        if table.caption:
            add(table.page_number, f"**{table.caption}**")
        add(table.page_number, table.markdown)

    for footnote in parsed_doc.footnotes:
        add(
            footnote.page_number,
            f"*Footnote [{footnote.reference}]* {footnote.content}",
        )

    return pages


def _get_tool_timeout(tool_name: str, default_timeout: float) -> float:
    """Get configurable timeout for a tool.

    Checks MCP_TOOL_TIMEOUT_OVERRIDES for tool-specific timeout,
    falls back to provided default.

    Args:
        tool_name: The tool name (e.g., "rag.ingest_url")
        default_timeout: Default timeout in seconds

    Returns:
        Timeout value in seconds
    """
    try:
        settings = get_settings()
        overrides = settings.mcp_tool_timeout_overrides
        if tool_name in overrides:
            return overrides[tool_name]
        return default_timeout
    except Exception:
        # If settings fail to load, use default
        return default_timeout


def _validate_content_size(content: str, max_size: int = MAX_CONTENT_SIZE) -> None:
    """Validate that content does not exceed maximum size.

    Args:
        content: Content string to validate
        max_size: Maximum allowed size in bytes

    Raises:
        MCPError: If content exceeds maximum size
    """
    if len(content.encode("utf-8")) > max_size:
        raise MCPError(
            code=MCPErrorCode.INVALID_PARAMS,
            message=f"Content exceeds maximum size of {max_size} bytes",
            data={"max_size": max_size},
        )


def _validate_tenant_id(tenant_id: str) -> None:
    """Validate tenant_id format."""
    if not tenant_id or not is_valid_tenant_id(tenant_id):
        raise MCPError(
            code=MCPErrorCode.INVALID_PARAMS,
            message="Invalid tenant_id format. Must be a valid UUID.",
            data={"tenant_id": tenant_id},
        )


def _vector_hit_to_dict(hit: VectorHit) -> dict[str, Any]:
    """Convert VectorHit to serializable dict."""
    return {
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "content": hit.content,
        "similarity": hit.similarity,
        "metadata": hit.metadata,
    }


def _reranked_hit_to_dict(hit: RerankedHit) -> dict[str, Any]:
    """Convert RerankedHit to serializable dict."""
    return {
        "chunk_id": hit.hit.chunk_id,
        "document_id": hit.hit.document_id,
        "content": hit.hit.content,
        "similarity": hit.hit.similarity,
        "rerank_score": hit.rerank_score,
        "original_rank": hit.original_rank,
        "metadata": hit.hit.metadata,
    }


def create_vector_search_tool(
    vector_service: VectorSearchService,
) -> MCPToolSpec:
    """Create the rag.vector_search tool."""

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        query = arguments.get("query", "")
        if not query or not query.strip():
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Query is required",
            )

        hits = await vector_service.search(query=query, tenant_id=tenant_id)

        return {
            "query": query,
            "tenant_id": tenant_id,
            "hits": [_vector_hit_to_dict(hit) for hit in hits],
            "count": len(hits),
        }

    return MCPToolSpec(
        name="rag.vector_search",
        description=(
            "Search for semantically similar document chunks using vector embeddings. "
            "Returns chunks ranked by cosine similarity."
        ),
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            required=["tenant_id", "query"],
        ),
        handler=handler,
        category="rag",
    )


def create_hybrid_retrieve_tool(
    vector_service: VectorSearchService,
    graphiti_client: GraphitiClient,
    reranker: Optional[RerankerClient] = None,
) -> MCPToolSpec:
    """Create the rag.hybrid_retrieve tool.

    Combines vector search with graph search and optionally reranks results.
    """

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        query = arguments.get("query", "")
        if not query or not query.strip():
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Query is required",
            )

        num_results = arguments.get("num_results", 10)
        use_reranking = arguments.get("use_reranking", True) and reranker is not None

        # Execute vector and graph search in parallel
        vector_task = vector_service.search(query=query, tenant_id=tenant_id)
        graph_task = graphiti_search(
            graphiti_client=graphiti_client,
            query=query,
            tenant_id=tenant_id,
            num_results=num_results,
        )

        vector_hits, graph_result = await asyncio.gather(vector_task, graph_task)

        # Apply reranking if enabled
        reranked_hits: list[RerankedHit] = []
        if use_reranking and reranker and vector_hits:
            reranked_hits = await reranker.rerank(
                query=query,
                hits=vector_hits,
                top_k=num_results,
            )

        # Build response
        result: dict[str, Any] = {
            "query": query,
            "tenant_id": tenant_id,
            "retrieval_mode": "hybrid",
        }

        # Add vector results
        if use_reranking and reranked_hits:
            result["vector_hits"] = [_reranked_hit_to_dict(h) for h in reranked_hits]
            result["reranking_applied"] = True
            result["reranker_model"] = reranker.get_model() if reranker else None
        else:
            result["vector_hits"] = [_vector_hit_to_dict(h) for h in vector_hits[:num_results]]
            result["reranking_applied"] = False

        # Add graph results
        result["graph_nodes"] = [
            {
                "uuid": node.uuid,
                "name": node.name,
                "summary": node.summary,
                "labels": node.labels,
            }
            for node in graph_result.nodes
        ]
        result["graph_edges"] = [
            {
                "uuid": edge.uuid,
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "name": edge.name,
                "fact": edge.fact,
            }
            for edge in graph_result.edges
        ]

        result["processing_time_ms"] = graph_result.processing_time_ms

        return result

    return MCPToolSpec(
        name="rag.hybrid_retrieve",
        description=(
            "Perform hybrid retrieval combining vector semantic search with "
            "knowledge graph traversal. Optionally applies cross-encoder reranking "
            "for improved precision."
        ),
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10,
                },
                "use_reranking": {
                    "type": "boolean",
                    "description": "Whether to apply cross-encoder reranking (default: true)",
                    "default": True,
                },
            },
            required=["tenant_id", "query"],
        ),
        handler=handler,
        category="rag",
    )


def create_ingest_url_tool(
    graphiti_client: GraphitiClient,
) -> MCPToolSpec:
    """Create the rag.ingest_url tool."""

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        url = arguments.get("url", "")
        if not url or not is_valid_url(url):
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Valid URL is required",
                data={"url": url},
            )

        max_depth = arguments.get("max_depth", 1)
        max_pages = arguments.get("max_pages", 10)

        # Crawl URL
        pages_ingested = 0
        errors = []

        try:
            async for page in crawl_url(
                url=url,
                max_depth=max_depth,
                max_pages=max_pages,
                tenant_id=tenant_id,
            ):
                # Ingest each page as an episode
                import hashlib
                from ...indexing.graphiti_ingestion import ingest_document_as_episode

                content_hash = hashlib.sha256(page.content.encode()).hexdigest()
                doc = UnifiedDocument(
                    id=uuid4(),
                    tenant_id=tenant_id,
                    content=page.content,
                    content_hash=content_hash,
                    source_type=SourceType.URL,
                    source_url=page.url,
                    metadata=DocumentMetadata(title=page.title),
                )

                try:
                    await ingest_document_as_episode(
                        graphiti_client=graphiti_client,
                        document=doc,
                    )
                    pages_ingested += 1
                except Exception as e:
                    errors.append({"url": page.url, "error": str(e)})

        except Exception as e:
            raise MCPError(
                code=MCPErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Crawl failed: {e}",
            )

        return {
            "url": url,
            "tenant_id": tenant_id,
            "pages_ingested": pages_ingested,
            "max_depth": max_depth,
            "max_pages": max_pages,
            "errors": errors if errors else None,
        }

    # Get configurable timeout (can be set via MCP_TOOL_TIMEOUT_OVERRIDES)
    timeout = _get_tool_timeout("rag.ingest_url", DEFAULT_INGEST_URL_TIMEOUT)

    return MCPToolSpec(
        name="rag.ingest_url",
        description=(
            "Crawl and ingest content from a URL into the knowledge graph. "
            "Supports depth-limited crawling with automatic entity extraction."
        ),
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "url": {
                    "type": "string",
                    "description": "URL to crawl and ingest",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum crawl depth (default: 1, start page only)",
                    "default": 1,
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to crawl (default: 10)",
                    "default": 10,
                },
            },
            required=["tenant_id", "url"],
        ),
        handler=handler,
        category="rag",
        timeout_seconds=timeout,  # Configurable via MCP_TOOL_TIMEOUT_OVERRIDES
    )


def create_ingest_youtube_tool(
    graphiti_client: GraphitiClient,
) -> MCPToolSpec:
    """Create the rag.ingest_youtube tool."""

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        url = arguments.get("url", "")
        if not url:
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="YouTube URL is required",
            )

        try:
            # Fetch and process transcript
            result = await ingest_youtube_video(url)

            # Ingest chunks as episodes
            from ...indexing.graphiti_ingestion import ingest_document_as_episode

            import hashlib

            chunks_ingested = 0
            for chunk in result.chunks:
                content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
                doc = UnifiedDocument(
                    id=uuid4(),
                    tenant_id=tenant_id,
                    content=chunk.content,
                    content_hash=content_hash,
                    source_type=SourceType.URL,  # YouTube is a URL source
                    source_url=result.source_url,
                    metadata=DocumentMetadata(
                        title=f"YouTube: {result.video_id} - Chunk {chunk.chunk_index}",
                        extra={
                            "video_id": result.video_id,
                            "start_time": chunk.start_time,
                            "end_time": chunk.end_time,
                            "language": result.language,
                        },
                    ),
                )

                await ingest_document_as_episode(
                    graphiti_client=graphiti_client,
                    document=doc,
                )
                chunks_ingested += 1

            return {
                "video_id": result.video_id,
                "tenant_id": tenant_id,
                "source_url": result.source_url,
                "language": result.language,
                "is_generated": result.is_generated,
                "duration_seconds": result.duration_seconds,
                "chunks_ingested": chunks_ingested,
            }

        except YouTubeIngestionError as e:
            raise MCPError(
                code=MCPErrorCode.TOOL_EXECUTION_ERROR,
                message=f"YouTube ingestion failed: {e.reason}",
                data={"video_id": e.video_id},
            )

    # Get configurable timeout (can be set via MCP_TOOL_TIMEOUT_OVERRIDES)
    timeout = _get_tool_timeout("rag.ingest_youtube", DEFAULT_INGEST_YOUTUBE_TIMEOUT)

    return MCPToolSpec(
        name="rag.ingest_youtube",
        description=(
            "Ingest YouTube video transcript into the knowledge graph. "
            "Fetches the transcript, chunks it, and extracts entities."
        ),
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "url": {
                    "type": "string",
                    "description": "YouTube video URL",
                },
            },
            required=["tenant_id", "url"],
        ),
        handler=handler,
        category="rag",
        timeout_seconds=timeout,  # Configurable via MCP_TOOL_TIMEOUT_OVERRIDES
    )


def create_ingest_text_tool(
    graphiti_client: GraphitiClient,
) -> MCPToolSpec:
    """Create the rag.ingest_text tool."""

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        content = arguments.get("content", "")
        if not content or not content.strip():
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Content is required",
            )

        # Validate content size to prevent DoS attacks
        _validate_content_size(content)

        title = arguments.get("title")
        source_url = arguments.get("source_url")

        import hashlib
        from ...indexing.graphiti_ingestion import ingest_document_as_episode

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        doc = UnifiedDocument(
            id=uuid4(),
            tenant_id=tenant_id,
            content=content,
            content_hash=content_hash,
            source_type=SourceType.TEXT,
            source_url=source_url,
            metadata=DocumentMetadata(title=title) if title else None,
        )

        result = await ingest_document_as_episode(
            graphiti_client=graphiti_client,
            document=doc,
        )

        return {
            "document_id": result.document_id,
            "tenant_id": result.tenant_id,
            "episode_uuid": result.episode_uuid,
            "entities_extracted": result.entities_extracted,
            "edges_created": result.edges_created,
            "processing_time_ms": result.processing_time_ms,
        }

    return MCPToolSpec(
        name="rag.ingest_text",
        description=(
            "Ingest text content directly into the knowledge graph. "
            f"Extracts entities and relationships from the provided text. "
            f"Maximum content size: {MAX_CONTENT_SIZE:,} bytes."
        ),
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "content": {
                    "type": "string",
                    "description": f"Text content to ingest (max {MAX_CONTENT_SIZE:,} bytes)",
                },
                "title": {
                    "type": "string",
                    "description": "Optional title for the content",
                },
                "source_url": {
                    "type": "string",
                    "description": "Optional source URL for attribution",
                },
            },
            required=["tenant_id", "content"],
        ),
        handler=handler,
        category="rag",
    )


def create_ingest_pdf_tool(
    graphiti_client: GraphitiClient,
) -> MCPToolSpec:
    """Create the rag.ingest_pdf tool."""

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        from uuid import UUID
        import hashlib
        from ...indexing.graphiti_ingestion import ingest_document_as_episode

        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        file_path = arguments.get("file_path", "")
        if not file_path:
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="file_path is required",
            )

        table_mode = arguments.get("table_mode", "accurate")
        if table_mode not in {"accurate", "fast"}:
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="table_mode must be 'accurate' or 'fast'",
            )

        chunk_by_page = bool(arguments.get("chunk_by_page", False))

        settings = get_settings()
        base_dir = Path(settings.temp_upload_dir).expanduser().resolve()
        path = Path(file_path)
        if not path.is_absolute():
            path = base_dir / path
        try:
            resolved_path = path.resolve()
        except Exception as exc:
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="file_path is invalid",
                data={"file_path": file_path},
            ) from exc
        if not resolved_path.is_relative_to(base_dir):
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="file_path must be under the configured upload directory",
                data={"file_path": file_path, "base_dir": str(base_dir)},
            )
        if not resolved_path.exists():
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="file_path does not exist",
                data={"file_path": file_path},
            )
        if not resolved_path.is_file():
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="file_path must be a file",
                data={"file_path": file_path},
            )

        tenant_uuid = UUID(tenant_id)
        document_id = uuid4()

        try:
            parsed_doc = await asyncio.to_thread(
                parse_pdf,
                resolved_path,
                document_id,
                tenant_uuid,
                table_mode,
            )
        except Exception as exc:
            raise MCPError(
                code=MCPErrorCode.TOOL_EXECUTION_ERROR,
                message=f"PDF parsing failed: {exc}",
            ) from exc

        chunks_ingested = 0
        errors: list[dict[str, Any]] = []

        if chunk_by_page:
            page_chunks = _build_pdf_page_chunks(parsed_doc)
            for page_number, parts in sorted(page_chunks.items()):
                content = "\n\n".join(parts).strip()
                if not content:
                    continue
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                metadata = DocumentMetadata(
                    title=parsed_doc.metadata.title,
                    page_count=parsed_doc.page_count,
                    extra={
                        "page_number": page_number,
                        "filename": parsed_doc.filename,
                    },
                )
                doc = UnifiedDocument(
                    id=uuid4(),
                    tenant_id=tenant_uuid,
                    source_type=SourceType.PDF,
                    filename=parsed_doc.filename,
                    content=content,
                    content_hash=content_hash,
                    metadata=metadata,
                )
                try:
                    await ingest_document_as_episode(
                        graphiti_client=graphiti_client,
                        document=doc,
                    )
                    chunks_ingested += 1
                except Exception as exc:
                    errors.append(
                        {"page_number": page_number, "error": str(exc)}
                    )
        else:
            unified_doc = parsed_doc.to_unified_document()
            try:
                await ingest_document_as_episode(
                    graphiti_client=graphiti_client,
                    document=unified_doc,
                )
                chunks_ingested = 1
            except Exception as exc:
                errors.append({"error": str(exc)})

        return {
            "tenant_id": tenant_id,
            "file_path": file_path,
            "page_count": parsed_doc.page_count,
            "chunk_by_page": chunk_by_page,
            "chunks_ingested": chunks_ingested,
            "errors": errors if errors else None,
        }

    timeout = _get_tool_timeout("rag.ingest_pdf", DEFAULT_INGEST_PDF_TIMEOUT)

    return MCPToolSpec(
        name="rag.ingest_pdf",
        description=(
            "Parse a PDF with Docling and ingest its content into the knowledge graph."
        ),
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file on the server",
                },
                "table_mode": {
                    "type": "string",
                    "description": "Docling table extraction mode (accurate|fast)",
                    "default": "accurate",
                },
                "chunk_by_page": {
                    "type": "boolean",
                    "description": "If true, ingest one chunk per page",
                    "default": False,
                },
            },
            required=["tenant_id", "file_path"],
        ),
        handler=handler,
        category="rag",
        timeout_seconds=timeout,
    )


def create_query_with_reranking_tool(
    vector_service: VectorSearchService,
    reranker: RerankerClient,
) -> MCPToolSpec:
    """Create the rag.query_with_reranking tool."""

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        query = arguments.get("query", "")
        if not query or not query.strip():
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Query is required",
            )

        top_k = arguments.get("top_k", 10)

        # Get vector search results
        hits = await vector_service.search(query=query, tenant_id=tenant_id)

        if not hits:
            return {
                "query": query,
                "tenant_id": tenant_id,
                "hits": [],
                "reranker_model": reranker.get_model(),
                "count": 0,
            }

        # Apply reranking
        reranked = await reranker.rerank(
            query=query,
            hits=hits,
            top_k=top_k,
        )

        return {
            "query": query,
            "tenant_id": tenant_id,
            "hits": [_reranked_hit_to_dict(h) for h in reranked],
            "reranker_model": reranker.get_model(),
            "count": len(reranked),
        }

    return MCPToolSpec(
        name="rag.query_with_reranking",
        description=(
            "Search with cross-encoder reranking for improved precision. "
            "First retrieves candidates via vector search, then reranks using a cross-encoder model."
        ),
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return after reranking (default: 10)",
                    "default": 10,
                },
            },
            required=["tenant_id", "query"],
        ),
        handler=handler,
        category="rag",
    )


def create_explain_answer_tool(
    vector_service: VectorSearchService,
    graphiti_client: GraphitiClient,
) -> MCPToolSpec:
    """Create the rag.explain_answer tool.

    Provides explainability for RAG answers by showing sources and reasoning.
    """

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        tenant_id = arguments.get("tenant_id", "")
        _validate_tenant_id(tenant_id)

        query = arguments.get("query", "")
        if not query or not query.strip():
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Query is required",
            )

        answer = arguments.get("answer", "")
        if not answer:
            raise MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Answer is required for explanation",
            )

        # Validate answer size to prevent DoS attacks
        _validate_content_size(answer)

        # Get sources used for the answer
        vector_task = vector_service.search(query=query, tenant_id=tenant_id)
        graph_task = graphiti_search(
            graphiti_client=graphiti_client,
            query=query,
            tenant_id=tenant_id,
            num_results=5,
        )

        vector_hits, graph_result = await asyncio.gather(vector_task, graph_task)

        # Build explanation
        vector_sources = []
        for i, hit in enumerate(vector_hits[:5]):
            vector_sources.append({
                "rank": i + 1,
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "similarity": hit.similarity,
                "excerpt": hit.content[:200] + "..." if len(hit.content) > 200 else hit.content,
            })

        graph_sources = []
        for node in graph_result.nodes[:5]:
            graph_sources.append({
                "uuid": node.uuid,
                "name": node.name,
                "summary": node.summary,
                "labels": node.labels,
            })

        graph_facts = []
        for edge in graph_result.edges[:5]:
            graph_facts.append({
                "fact": edge.fact,
                "relationship": edge.name,
            })

        return {
            "query": query,
            "tenant_id": tenant_id,
            "answer_provided": answer[:500] + "..." if len(answer) > 500 else answer,
            "explanation": {
                "retrieval_method": "hybrid (vector + graph)",
                "vector_sources": vector_sources,
                "graph_entities": graph_sources,
                "graph_facts": graph_facts,
                "source_count": {
                    "vector_chunks": len(vector_hits),
                    "graph_nodes": len(graph_result.nodes),
                    "graph_edges": len(graph_result.edges),
                },
            },
        }

    return MCPToolSpec(
        name="rag.explain_answer",
        description=(
            "Explain the sources and reasoning behind a RAG answer. "
            "Shows which documents and knowledge graph entities contributed to the answer."
        ),
        input_schema=create_tool_input_schema(
            properties={
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID (UUID format)",
                },
                "query": {
                    "type": "string",
                    "description": "The original query",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer to explain",
                },
            },
            required=["tenant_id", "query", "answer"],
        ),
        handler=handler,
        category="rag",
    )


def register_rag_tools(
    registry: MCPServerRegistry,
    graphiti_client: GraphitiClient,
    vector_service: Optional[VectorSearchService] = None,
    reranker: Optional[RerankerClient] = None,
) -> list[str]:
    """Register all RAG tools with the registry.

    Args:
        registry: MCP server registry
        graphiti_client: Connected Graphiti client
        vector_service: Optional vector search service
        reranker: Optional reranker client

    Returns:
        List of registered tool names
    """
    registered = []

    # Always register ingestion tools
    ingestion_tools = [
        create_ingest_url_tool(graphiti_client),
        create_ingest_pdf_tool(graphiti_client),
        create_ingest_youtube_tool(graphiti_client),
        create_ingest_text_tool(graphiti_client),
    ]

    for tool in ingestion_tools:
        registry.register(tool)
        registered.append(tool.name)

    # Register vector search tools if vector service available
    if vector_service:
        vector_tool = create_vector_search_tool(vector_service)
        registry.register(vector_tool)
        registered.append(vector_tool.name)

        # Hybrid retrieve needs both vector and graphiti
        hybrid_tool = create_hybrid_retrieve_tool(
            vector_service=vector_service,
            graphiti_client=graphiti_client,
            reranker=reranker,
        )
        registry.register(hybrid_tool)
        registered.append(hybrid_tool.name)

        # Explain answer tool
        explain_tool = create_explain_answer_tool(
            vector_service=vector_service,
            graphiti_client=graphiti_client,
        )
        registry.register(explain_tool)
        registered.append(explain_tool.name)

        # Query with reranking if reranker available
        if reranker:
            rerank_tool = create_query_with_reranking_tool(
                vector_service=vector_service,
                reranker=reranker,
            )
            registry.register(rerank_tool)
            registered.append(rerank_tool.name)

    logger.info(
        "rag_tools_registered",
        tools=registered,
        count=len(registered),
        has_vector_service=vector_service is not None,
        has_reranker=reranker is not None,
    )

    return registered
