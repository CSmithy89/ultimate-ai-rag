"""Async parse worker for document processing.

This worker consumes jobs from the Redis Streams 'parse.jobs' queue,
processes documents using Docling (PDFs) or direct parsing (text, markdown,
Office documents), and queues results for indexing.

Supported formats:
- PDF (.pdf) - via Docling
- Text (.txt) - direct reading
- Markdown (.md, .markdown) - direct reading
- Word (.docx) - via python-docx
- Excel (.xlsx) - via openpyxl
- PowerPoint (.pptx) - via python-pptx
- Images (.png, .jpg, etc.) - via Docling OCR
"""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import structlog

from agentic_rag_backend.config import get_settings
from agentic_rag_backend.core.errors import AppError, ParseError
from agentic_rag_backend.db.postgres import PostgresClient, get_postgres_client
from agentic_rag_backend.db.redis import (
    INDEX_JOBS_STREAM,
    PARSE_CONSUMER_GROUP,
    PARSE_JOBS_STREAM,
    RedisClient,
    get_redis_client,
)
from agentic_rag_backend.indexing.parser import parse_pdf
from agentic_rag_backend.indexing.multimodal import (
    DocumentType,
    EXTENSION_TYPE_MAP,
    OfficeParser,
)
from agentic_rag_backend.models.ingest import JobStatusEnum

logger = structlog.get_logger(__name__)


def detect_document_type(file_path: Path) -> DocumentType:
    """Detect document type from file extension.

    Args:
        file_path: Path to the file

    Returns:
        DocumentType enum value
    """
    ext = file_path.suffix.lower()
    return EXTENSION_TYPE_MAP.get(ext, DocumentType.UNKNOWN)


def parse_text_file(file_path: Path) -> tuple[str, dict[str, Any]]:
    """Parse a plain text file.

    Args:
        file_path: Path to the text file

    Returns:
        Tuple of (content, metadata)
    """
    content = file_path.read_text(encoding="utf-8")
    metadata = {
        "source_type": "text",
        "file_size": file_path.stat().st_size,
        "line_count": content.count("\n") + 1,
    }
    return content, metadata


def parse_markdown_file(file_path: Path) -> tuple[str, dict[str, Any]]:
    """Parse a markdown file.

    Args:
        file_path: Path to the markdown file

    Returns:
        Tuple of (content, metadata)
    """
    content = file_path.read_text(encoding="utf-8")

    # Extract headers for section info
    headers = []
    for line in content.split("\n"):
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            headers.append({"level": level, "title": title})

    metadata = {
        "source_type": "markdown",
        "file_size": file_path.stat().st_size,
        "line_count": content.count("\n") + 1,
        "sections": headers,
    }
    return content, metadata


async def process_parse_job(
    job_data: dict,
    postgres: PostgresClient,
    redis: RedisClient,
    table_mode: str = "accurate",
) -> None:
    """
    Process a single parse job.

    This function:
    1. Updates job status to running
    2. Parses the PDF using Docling
    3. Stores parsed content
    4. Queues for indexing
    5. Updates job status and processing time
    6. Cleans up temp file on success

    Args:
        job_data: Job data from Redis Stream containing:
            - job_id: UUID of the job
            - tenant_id: Tenant identifier
            - document_id: Document UUID
            - file_path: Path to temp PDF file
            - filename: Original filename
        postgres: PostgreSQL client for status updates
        redis: Redis client for queueing next stage
        table_mode: Docling table extraction mode

    Raises:
        ParseError: If parsing fails
        AppError: For other application errors
    """
    job_id = UUID(job_data["job_id"])
    tenant_id = UUID(job_data["tenant_id"])
    document_id = UUID(job_data["document_id"])
    file_path = Path(job_data["file_path"])
    filename = job_data.get("filename", file_path.name)

    logger.info(
        "parse_job_started",
        job_id=str(job_id),
        tenant_id=str(tenant_id),
        file_path=str(file_path),
    )

    start_time = time.perf_counter()

    try:
        # Update job status to running
        await postgres.update_job_status(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatusEnum.RUNNING,
            progress={"pages_parsed": 0, "total_pages": 0},
        )

        # Validate file exists
        if not file_path.exists():
            raise ParseError(filename, f"File not found: {file_path}")

        # Detect document type from file extension
        doc_type = detect_document_type(file_path)
        logger.info(
            "document_type_detected",
            job_id=str(job_id),
            doc_type=doc_type.value,
            file_extension=file_path.suffix,
        )

        # Parse based on document type
        loop = asyncio.get_running_loop()
        content = ""
        metadata: dict[str, Any] = {}
        page_count = 1
        source_type = doc_type.value

        if doc_type == DocumentType.PDF:
            # Parse PDF using Docling
            parsed_doc = await loop.run_in_executor(
                None,
                parse_pdf,
                file_path,
                document_id,
                tenant_id,
                table_mode,
            )
            unified_doc = parsed_doc.to_unified_document()
            content = unified_doc.content
            metadata = unified_doc.metadata.model_dump(mode="json")
            page_count = parsed_doc.page_count

        elif doc_type == DocumentType.TEXT:
            # Parse plain text file
            content, metadata = await loop.run_in_executor(
                None, parse_text_file, file_path
            )

        elif doc_type == DocumentType.MARKDOWN:
            # Parse markdown file
            content, metadata = await loop.run_in_executor(
                None, parse_markdown_file, file_path
            )

        elif doc_type in (DocumentType.WORD, DocumentType.EXCEL, DocumentType.POWERPOINT):
            # Parse Office documents
            office_parser = OfficeParser()
            if doc_type == DocumentType.WORD:
                word_content = await loop.run_in_executor(
                    None, office_parser.parse_word, file_path
                )
                content = word_content.full_text
                metadata = {
                    "source_type": "word",
                    "paragraph_count": len(word_content.paragraphs),
                    "table_count": len(word_content.tables),
                    "doc_metadata": word_content.metadata,
                }
            elif doc_type == DocumentType.EXCEL:
                excel_content = await loop.run_in_executor(
                    None, office_parser.parse_excel, file_path
                )
                # Convert sheets to markdown format
                content = "\n\n".join(sheet.to_markdown() for sheet in excel_content.sheets)
                metadata = {
                    "source_type": "excel",
                    "sheet_count": excel_content.sheet_count,
                    "sheets": [s.name for s in excel_content.sheets],
                }
            elif doc_type == DocumentType.POWERPOINT:
                pptx_content = await loop.run_in_executor(
                    None, office_parser.parse_powerpoint, file_path
                )
                content = pptx_content.full_text
                page_count = pptx_content.slide_count
                metadata = {
                    "source_type": "powerpoint",
                    "slide_count": pptx_content.slide_count,
                }

        elif doc_type == DocumentType.IMAGE:
            # For images, use Docling's OCR capabilities (falls back to PDF parser)
            parsed_doc = await loop.run_in_executor(
                None,
                parse_pdf,
                file_path,
                document_id,
                tenant_id,
                table_mode,
            )
            unified_doc = parsed_doc.to_unified_document()
            content = unified_doc.content
            metadata = unified_doc.metadata.model_dump(mode="json")
            source_type = "image"

        else:
            raise ParseError(filename, f"Unsupported document type: {doc_type.value}")

        # Calculate processing time
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Update job processing time
        await postgres.update_job_processing_time(
            job_id=job_id,
            tenant_id=tenant_id,
            processing_time_ms=processing_time_ms,
        )

        # Update document page count
        await postgres.update_document_page_count(
            document_id=document_id,
            tenant_id=tenant_id,
            page_count=page_count,
        )

        # Compute content hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Queue for indexing
        await redis.publish_job(
            stream=INDEX_JOBS_STREAM,
            job_data={
                "job_id": str(job_id),
                "tenant_id": str(tenant_id),
                "document_id": str(document_id),
                "content": content,
                "content_hash": content_hash,
                "metadata": metadata,
                "source_type": source_type,
                "filename": filename,
            },
        )

        # Update job status to completed
        await postgres.update_job_status(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatusEnum.COMPLETED,
            progress={
                "pages_parsed": page_count,
                "total_pages": page_count,
                "doc_type": doc_type.value,
                "processing_time_ms": processing_time_ms,
            },
        )

        # Cleanup temp file on success
        try:
            file_path.unlink(missing_ok=True)
            logger.debug("temp_file_deleted", file_path=str(file_path))
        except Exception as e:
            logger.warning("temp_file_cleanup_failed", file_path=str(file_path), error=str(e))

        logger.info(
            "parse_job_completed",
            job_id=str(job_id),
            doc_type=doc_type.value,
            page_count=page_count,
            content_length=len(content),
            processing_time_ms=processing_time_ms,
        )

    except AppError as e:
        # Update job status to failed
        await postgres.update_job_status(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatusEnum.FAILED,
            error_message=str(e),
        )
        logger.error(
            "parse_job_failed",
            job_id=str(job_id),
            error=str(e),
            error_code=e.code.value if hasattr(e, "code") else None,
        )
        raise

    except Exception as e:
        # Update job status to failed for unexpected errors
        await postgres.update_job_status(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatusEnum.FAILED,
            error_message=f"Unexpected error: {str(e)}",
        )
        logger.error(
            "parse_job_failed_unexpected",
            job_id=str(job_id),
            error=str(e),
        )
        raise ParseError(filename, str(e)) from e


async def run_parse_worker(
    consumer_name: str = "parse-worker-1",
    batch_size: int = 1,
) -> None:
    """
    Run the parse worker as a long-running async task.

    This function:
    1. Connects to Redis and PostgreSQL
    2. Ensures consumer group exists
    3. Continuously consumes jobs from parse.jobs stream
    4. Processes each job through Docling
    5. Acknowledges jobs after processing

    Args:
        consumer_name: Unique identifier for this worker instance
        batch_size: Number of jobs to fetch at once
    """
    settings = get_settings()

    logger.info(
        "parse_worker_starting",
        consumer_name=consumer_name,
        stream=PARSE_JOBS_STREAM,
        group=PARSE_CONSUMER_GROUP,
    )

    # Get database clients
    redis = await get_redis_client(settings.redis_url)
    postgres = await get_postgres_client(settings.database_url)

    # Consume jobs from stream
    async for job_data in redis.consume_jobs(
        stream=PARSE_JOBS_STREAM,
        group=PARSE_CONSUMER_GROUP,
        consumer=consumer_name,
        count=batch_size,
        block_ms=5000,  # 5 second timeout
    ):
        try:
            await process_parse_job(
                job_data=job_data,
                postgres=postgres,
                redis=redis,
                table_mode=settings.docling_table_mode,
            )
        except Exception as e:
            # Log but don't crash the worker
            logger.error(
                "parse_worker_job_error",
                job_id=job_data.get("job_id"),
                error=str(e),
            )
            # Continue processing other jobs


async def main() -> None:
    """Entry point for running the parse worker as a standalone process."""
    import signal
    import sys

    # Handle graceful shutdown
    def signal_handler(sig: int, frame: Optional[object]) -> None:
        logger.info("parse_worker_shutdown_requested")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await run_parse_worker()
    except Exception as e:
        logger.error("parse_worker_crashed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
