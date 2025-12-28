"""Docling parser wrapper for PDF document parsing.

This module provides PDF parsing capabilities using Docling with:
- Structure preservation (headers, sections, footnotes)
- Table extraction with TableFormerMode.ACCURATE
- Content normalization to markdown format
"""

import hashlib
import time
from pathlib import Path
from typing import Generator, Optional
from uuid import UUID

import structlog

from agentic_rag_backend.core.errors import InvalidPdfError, ParseError, PasswordProtectedError
from agentic_rag_backend.models.documents import (
    DocumentMetadata,
    DocumentSection,
    FootnoteContent,
    ParsedDocument,
    TableContent,
)

logger = structlog.get_logger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA-256 hash of file content for deduplication.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal hash string (64 characters)
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_pdf(file_path: Path) -> bool:
    """
    Validate that a file is a valid PDF document.

    Checks:
    - File exists
    - File starts with %PDF magic bytes
    - File is not password protected (basic check)

    Args:
        file_path: Path to the file to validate

    Returns:
        True if valid, False otherwise
    """
    if not file_path.exists():
        return False

    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
            if not header.startswith(b"%PDF"):
                return False
        return True
    except Exception:
        return False


def check_password_protected(file_path: Path) -> bool:
    """
    Check if a PDF is password protected.

    This is a basic check that looks for encryption markers.
    A full check would require attempting to open the document.

    Args:
        file_path: Path to the PDF file

    Returns:
        True if password protected, False otherwise
    """
    try:
        with open(file_path, "rb") as f:
            # Read first 4KB to check for encryption marker
            content = f.read(4096)
            if b"/Encrypt" in content:
                return True
        return False
    except Exception:
        return False


def parse_pdf_elements(
    file_path: Path,
    table_mode: str = "accurate",
) -> Generator[dict, None, dict]:
    """
    Parse PDF with Docling, preserving structure and extracting tables.

    This function uses Docling's DocumentConverter to extract structured
    content from PDF documents, yielding elements as they are processed.

    Args:
        file_path: Path to PDF file
        table_mode: Table extraction mode ('accurate' or 'fast')

    Yields:
        Dictionaries with extracted content:
        - type: 'section' | 'table' | 'text' | 'footnote'
        - content: Extracted text/markdown
        - metadata: Element-specific metadata

    Returns:
        Final metadata dict with page_count and title

    Raises:
        ParseError: If parsing fails
    """
    try:
        # Import Docling here to allow graceful failure if not installed
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
        )
        from docling.datamodel.pipeline_options import TableFormerMode

        # Configure table extraction mode
        if table_mode == "accurate":
            table_former_mode = TableFormerMode.ACCURATE
        else:
            table_former_mode = TableFormerMode.FAST

        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            table_structure_options=TableStructureOptions(mode=table_former_mode),
        )

        logger.info(
            "docling_parse_started",
            file_path=str(file_path),
            table_mode=table_mode,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pipeline_options,
            }
        )

        result = converter.convert(str(file_path))
        doc = result.document

        # Track metadata
        page_count = 0
        title = None

        # Iterate over document items
        for item, level in doc.iterate_items():
            item_type = type(item).__name__

            # Get page number from item's prov if available
            page_num = 1
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page_no"):
                        page_num = prov.page_no
                        page_count = max(page_count, page_num)
                        break

            if item_type == "TableItem":
                # Export table to markdown
                try:
                    markdown = item.export_to_markdown()
                    num_rows = len(item.data.table_cells) if hasattr(item.data, "table_cells") else 0
                    num_cols = item.data.num_cols if hasattr(item.data, "num_cols") else 0
                    caption = None
                    if hasattr(item, "caption") and item.caption:
                        caption = str(item.caption)

                    yield {
                        "type": "table",
                        "content": markdown,
                        "metadata": {
                            "row_count": num_rows,
                            "column_count": num_cols,
                            "page_number": page_num,
                            "caption": caption,
                        },
                    }
                except Exception as e:
                    logger.warning("table_export_failed", error=str(e))

            elif item_type == "SectionHeaderItem":
                # Extract heading
                text = item.text if hasattr(item, "text") else str(item)
                heading_level = level + 1  # Docling levels are 0-indexed
                if heading_level < 1:
                    heading_level = 1
                if heading_level > 6:
                    heading_level = 6

                # First heading is likely the title
                if title is None:
                    title = text

                yield {
                    "type": "section",
                    "content": text,
                    "metadata": {
                        "level": heading_level,
                        "page_number": page_num,
                        "heading": text,
                    },
                }

            elif item_type == "TextItem":
                # Regular text content
                text = item.text if hasattr(item, "text") else str(item)
                yield {
                    "type": "text",
                    "content": text,
                    "metadata": {
                        "page_number": page_num,
                    },
                }

            elif item_type == "FootnoteItem" or "footnote" in item_type.lower():
                # Footnote content
                text = item.text if hasattr(item, "text") else str(item)
                ref = item.reference if hasattr(item, "reference") else "?"
                yield {
                    "type": "footnote",
                    "content": text,
                    "metadata": {
                        "reference": str(ref),
                        "page_number": page_num,
                    },
                }

            elif item_type == "PictureItem" or item_type == "FigureItem":
                # Skip images/figures for now, but log them
                logger.debug("skipping_image_element", item_type=item_type)

            else:
                # Handle other text-like elements
                if hasattr(item, "text") and item.text:
                    yield {
                        "type": "text",
                        "content": item.text,
                        "metadata": {
                            "page_number": page_num,
                        },
                    }

        logger.info(
            "docling_parse_completed",
            file_path=str(file_path),
            page_count=page_count,
        )

        # Return final metadata
        return {"page_count": page_count, "title": title}

    except ImportError as e:
        raise ParseError(str(file_path), f"Docling not installed: {e}")
    except Exception as e:
        logger.error("docling_parse_failed", file_path=str(file_path), error=str(e))
        raise ParseError(str(file_path), str(e))


def parse_pdf(
    file_path: Path,
    document_id: UUID,
    tenant_id: UUID,
    table_mode: str = "accurate",
) -> ParsedDocument:
    """
    Parse a PDF document and return a ParsedDocument model.

    This is the main entry point for PDF parsing. It validates the PDF,
    parses it using Docling, and returns a structured ParsedDocument.

    Args:
        file_path: Path to PDF file
        document_id: UUID for the document
        tenant_id: Tenant identifier
        table_mode: Table extraction mode ('accurate' or 'fast')

    Returns:
        ParsedDocument with extracted content

    Raises:
        InvalidPdfError: If file is not a valid PDF
        PasswordProtectedError: If PDF is password protected
        ParseError: If parsing fails
    """
    # Validate PDF
    if not validate_pdf(file_path):
        raise InvalidPdfError(file_path.name, "File is not a valid PDF document")

    # Check for password protection
    if check_password_protected(file_path):
        raise PasswordProtectedError(file_path.name)

    # Get file info
    file_size = file_path.stat().st_size
    content_hash = compute_file_hash(file_path)

    # Track processing time (NFR2)
    start_time = time.perf_counter()

    # Parse PDF
    sections: list[DocumentSection] = []
    tables: list[TableContent] = []
    footnotes: list[FootnoteContent] = []
    current_section_content: list[str] = []
    current_section_heading: Optional[str] = None
    current_section_level: int = 1
    current_page: int = 1

    # Helper to flush current section
    def flush_section() -> None:
        nonlocal current_section_content, current_section_heading
        if current_section_content:
            sections.append(DocumentSection(
                heading=current_section_heading,
                level=current_section_level,
                content="\n".join(current_section_content),
                page_number=current_page,
            ))
            current_section_content = []
            current_section_heading = None

    # Parse elements
    parser_gen = parse_pdf_elements(file_path, table_mode)
    final_metadata = {"page_count": 0, "title": None}

    try:
        while True:
            try:
                element = next(parser_gen)
            except StopIteration as e:
                final_metadata = e.value or final_metadata
                break

            elem_type = element["type"]
            content = element["content"]
            metadata = element["metadata"]

            if elem_type == "section":
                # New section header - flush previous section
                flush_section()
                current_section_heading = metadata.get("heading")
                current_section_level = metadata.get("level", 1)
                current_page = metadata.get("page_number", 1)

            elif elem_type == "text":
                current_section_content.append(content)
                current_page = metadata.get("page_number", current_page)

            elif elem_type == "table":
                tables.append(TableContent(
                    caption=metadata.get("caption"),
                    markdown=content,
                    row_count=metadata.get("row_count", 0),
                    column_count=metadata.get("column_count", 0),
                    page_number=metadata.get("page_number", 1),
                ))

            elif elem_type == "footnote":
                footnotes.append(FootnoteContent(
                    reference=metadata.get("reference", "?"),
                    content=content,
                    page_number=metadata.get("page_number", 1),
                ))
    except Exception as e:
        raise ParseError(file_path.name, str(e))

    # Flush any remaining section
    flush_section()

    # Calculate processing time
    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    # Build metadata
    doc_metadata = DocumentMetadata(
        title=final_metadata.get("title"),
        page_count=final_metadata.get("page_count", len(set(s.page_number for s in sections)) or 1),
    )

    page_count = final_metadata.get("page_count", 1)
    if page_count < 1:
        # Estimate from sections if Docling didn't provide page count
        page_count = max((s.page_number for s in sections), default=1)

    logger.info(
        "pdf_parsed",
        document_id=str(document_id),
        file_size=file_size,
        page_count=page_count,
        sections_count=len(sections),
        tables_count=len(tables),
        footnotes_count=len(footnotes),
        processing_time_ms=processing_time_ms,
    )

    return ParsedDocument(
        id=document_id,
        tenant_id=tenant_id,
        filename=file_path.name,
        content_hash=content_hash,
        file_size=file_size,
        page_count=page_count,
        sections=sections,
        tables=tables,
        footnotes=footnotes,
        metadata=doc_metadata,
        processing_time_ms=processing_time_ms,
    )
