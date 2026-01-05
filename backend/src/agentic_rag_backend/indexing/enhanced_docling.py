"""Enhanced Docling Parser (Story 20-D1).

This module provides enhanced document parsing with:
- Rich table extraction with headers, rows, and structured data
- Layout preservation with sections, figures, and footnotes
- Markdown and structured data representations
- Table-to-chunk conversion for searchable content

Key Features:
- ExtractedTable: Rich table representation with structured data access
- DocumentLayout: Full document structure with hierarchy
- EnhancedDoclingParser: Enhanced parsing with layout awareness
- Feature flag: ENHANCED_DOCLING_ENABLED

Configuration:
- ENHANCED_DOCLING_ENABLED: Enable/disable enhanced extraction (default: true)
- DOCLING_TABLE_EXTRACTION: Enable table extraction (default: true)
- DOCLING_PRESERVE_LAYOUT: Enable layout preservation (default: true)
- DOCLING_TABLE_AS_MARKDOWN: Store tables as markdown chunks (default: true)

Performance target: <200ms additional latency over standard Docling parsing
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)

# Default configuration
DEFAULT_TABLE_EXTRACTION = True
DEFAULT_PRESERVE_LAYOUT = True
DEFAULT_TABLE_AS_MARKDOWN = True


@dataclass
class Position:
    """Position information for document elements.

    Attributes:
        x: X coordinate (left edge)
        y: Y coordinate (top edge)
        width: Element width
        height: Element height
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize position to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        """Deserialize position from dictionary."""
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            width=float(data.get("width", 0.0)),
            height=float(data.get("height", 0.0)),
        )


@dataclass
class ExtractedTable:
    """Rich table representation extracted from documents.

    This dataclass provides multiple ways to access table data:
    - markdown: Formatted markdown string for display/embedding
    - structured_data: List of dicts for programmatic access
    - headers/rows: Raw table structure

    Attributes:
        id: Unique table identifier
        page_number: Page where table appears
        position: Table position on page
        headers: Column header strings
        rows: List of row data (each row is list of cell values)
        caption: Table caption if available
        markdown: Markdown representation
        structured_data: List of dicts (header->value mapping per row)
        metadata: Additional extraction metadata
    """

    id: str
    page_number: int
    position: Position
    headers: list[str]
    rows: list[list[str]]
    caption: Optional[str] = None
    markdown: str = ""
    structured_data: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate markdown and structured_data if not provided."""
        if not self.markdown and (self.headers or self.rows):
            self.markdown = self._generate_markdown()
        if not self.structured_data and self.headers and self.rows:
            self.structured_data = self._generate_structured_data()

    def _generate_markdown(self) -> str:
        """Generate markdown representation from headers and rows."""
        if not self.headers and not self.rows:
            return ""

        lines = []

        # Add caption if present
        if self.caption:
            lines.append(f"**{self.caption}**\n")

        # Headers
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("|" + "|".join(["---"] * len(self.headers)) + "|")
        elif self.rows:
            # No headers, use first row length for separator
            col_count = len(self.rows[0]) if self.rows else 0
            lines.append("|" + "|".join(["---"] * col_count) + "|")

        # Rows
        for row in self.rows:
            # Escape pipe characters in cell values
            escaped_cells = [cell.replace("|", "\\|") for cell in row]
            lines.append("| " + " | ".join(escaped_cells) + " |")

        return "\n".join(lines)

    def _generate_structured_data(self) -> list[dict[str, str]]:
        """Generate structured data (list of dicts) from headers and rows."""
        if not self.headers:
            return []

        result = []
        for row in self.rows:
            row_dict = {}
            for i, header in enumerate(self.headers):
                value = row[i] if i < len(row) else ""
                row_dict[header] = value
            result.append(row_dict)

        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize table to dictionary."""
        return {
            "id": self.id,
            "page_number": self.page_number,
            "position": self.position.to_dict(),
            "headers": self.headers,
            "rows": self.rows,
            "caption": self.caption,
            "markdown": self.markdown,
            "structured_data": self.structured_data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedTable":
        """Deserialize table from dictionary."""
        position_data = data.get("position", {})
        position = Position.from_dict(position_data) if position_data else Position()

        return cls(
            id=data.get("id", ""),
            page_number=data.get("page_number", 1),
            position=position,
            headers=data.get("headers", []),
            rows=data.get("rows", []),
            caption=data.get("caption"),
            markdown=data.get("markdown", ""),
            structured_data=data.get("structured_data", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DocumentSection:
    """Document section with hierarchy information.

    Attributes:
        id: Section identifier
        heading: Section heading text
        level: Heading level (1-6)
        content: Section text content
        page_number: Page where section starts
        parent_id: Parent section ID for hierarchy
        child_ids: Child section IDs
    """

    id: str
    heading: Optional[str]
    level: int
    content: str
    page_number: int = 1
    parent_id: Optional[str] = None
    child_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize section to dictionary."""
        return {
            "id": self.id,
            "heading": self.heading,
            "level": self.level,
            "content": self.content,
            "page_number": self.page_number,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentSection":
        """Deserialize section from dictionary."""
        return cls(
            id=data.get("id", ""),
            heading=data.get("heading"),
            level=data.get("level", 1),
            content=data.get("content", ""),
            page_number=data.get("page_number", 1),
            parent_id=data.get("parent_id"),
            child_ids=data.get("child_ids", []),
        )


@dataclass
class Figure:
    """Document figure/image reference.

    Attributes:
        id: Figure identifier
        page_number: Page where figure appears
        position: Figure position on page
        caption: Figure caption
        alt_text: Alternative text description
    """

    id: str
    page_number: int
    position: Position
    caption: Optional[str] = None
    alt_text: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize figure to dictionary."""
        return {
            "id": self.id,
            "page_number": self.page_number,
            "position": self.position.to_dict(),
            "caption": self.caption,
            "alt_text": self.alt_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Figure":
        """Deserialize figure from dictionary."""
        position_data = data.get("position", {})
        position = Position.from_dict(position_data) if position_data else Position()

        return cls(
            id=data.get("id", ""),
            page_number=data.get("page_number", 1),
            position=position,
            caption=data.get("caption"),
            alt_text=data.get("alt_text"),
        )


@dataclass
class Footnote:
    """Document footnote.

    Attributes:
        id: Footnote identifier
        reference: Reference marker (e.g., "1", "*")
        content: Footnote text
        page_number: Page where footnote appears
    """

    id: str
    reference: str
    content: str
    page_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize footnote to dictionary."""
        return {
            "id": self.id,
            "reference": self.reference,
            "content": self.content,
            "page_number": self.page_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Footnote":
        """Deserialize footnote from dictionary."""
        return cls(
            id=data.get("id", ""),
            reference=data.get("reference", ""),
            content=data.get("content", ""),
            page_number=data.get("page_number", 1),
        )


@dataclass
class DocumentLayout:
    """Full document layout structure.

    This dataclass represents the complete extracted layout of a document,
    including hierarchical sections, tables, figures, and footnotes.

    Attributes:
        document_id: Document identifier
        tenant_id: Tenant identifier
        page_count: Total pages in document
        sections: Document sections with hierarchy
        tables: Extracted tables
        figures: Detected figures/images
        footnotes: Document footnotes
        headers: Document-level headers (title, subtitle)
        metadata: Additional layout metadata
    """

    document_id: str
    tenant_id: str
    page_count: int
    sections: list[DocumentSection] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    footnotes: list[Footnote] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize layout to dictionary."""
        return {
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "page_count": self.page_count,
            "sections": [s.to_dict() for s in self.sections],
            "tables": [t.to_dict() for t in self.tables],
            "figures": [f.to_dict() for f in self.figures],
            "footnotes": [fn.to_dict() for fn in self.footnotes],
            "headers": self.headers,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentLayout":
        """Deserialize layout from dictionary."""
        return cls(
            document_id=data.get("document_id", ""),
            tenant_id=data.get("tenant_id", ""),
            page_count=data.get("page_count", 0),
            sections=[DocumentSection.from_dict(s) for s in data.get("sections", [])],
            tables=[ExtractedTable.from_dict(t) for t in data.get("tables", [])],
            figures=[Figure.from_dict(f) for f in data.get("figures", [])],
            footnotes=[Footnote.from_dict(fn) for fn in data.get("footnotes", [])],
            headers=data.get("headers", {}),
            metadata=data.get("metadata", {}),
        )

    def get_table_by_id(self, table_id: str) -> Optional[ExtractedTable]:
        """Get table by ID."""
        for table in self.tables:
            if table.id == table_id:
                return table
        return None

    def get_tables_on_page(self, page_number: int) -> list[ExtractedTable]:
        """Get all tables on a specific page."""
        return [t for t in self.tables if t.page_number == page_number]


@dataclass
class TableChunk:
    """Searchable chunk created from a table.

    Attributes:
        id: Chunk identifier
        table_id: Source table identifier
        content: Chunk content (markdown or text)
        document_id: Parent document ID
        tenant_id: Tenant identifier
        page_number: Source page number
        metadata: Chunk metadata
    """

    id: str
    table_id: str
    content: str
    document_id: str
    tenant_id: str
    page_number: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to dictionary."""
        return {
            "id": self.id,
            "table_id": self.table_id,
            "content": self.content,
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }


class EnhancedDoclingParser:
    """Enhanced document parser with rich extraction capabilities.

    This class wraps Docling to provide:
    - Rich table extraction with structure preservation
    - Layout analysis with section hierarchy
    - Figure and footnote detection
    - Table-to-chunk conversion for embedding

    Attributes:
        table_extraction: Enable table extraction
        preserve_layout: Enable layout preservation
        table_as_markdown: Store tables as markdown
        table_mode: Docling table extraction mode
    """

    def __init__(
        self,
        table_extraction: bool = DEFAULT_TABLE_EXTRACTION,
        preserve_layout: bool = DEFAULT_PRESERVE_LAYOUT,
        table_as_markdown: bool = DEFAULT_TABLE_AS_MARKDOWN,
        table_mode: str = "accurate",
    ) -> None:
        """Initialize EnhancedDoclingParser.

        Args:
            table_extraction: Enable table extraction
            preserve_layout: Enable layout preservation
            table_as_markdown: Store tables as markdown chunks
            table_mode: Docling table mode ('accurate' or 'fast')
        """
        self.table_extraction = table_extraction
        self.preserve_layout = preserve_layout
        self.table_as_markdown = table_as_markdown
        self.table_mode = table_mode

        logger.debug(
            "enhanced_docling_parser_initialized",
            table_extraction=table_extraction,
            preserve_layout=preserve_layout,
            table_as_markdown=table_as_markdown,
            table_mode=table_mode,
        )

    def parse_document(
        self,
        file_path: Path,
        document_id: UUID,
        tenant_id: UUID,
    ) -> DocumentLayout:
        """Parse document and extract full layout.

        This is the main entry point for enhanced document parsing.
        It extracts all structural elements including tables, sections,
        figures, and footnotes.

        Args:
            file_path: Path to document file
            document_id: Document identifier
            tenant_id: Tenant identifier

        Returns:
            DocumentLayout with all extracted content

        Raises:
            ImportError: If Docling is not installed
            Exception: If parsing fails
        """
        start_time = time.perf_counter()

        logger.info(
            "enhanced_parse_started",
            file_path=str(file_path),
            document_id=str(document_id),
            tenant_id=str(tenant_id),
        )

        try:
            # Import Docling
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableStructureOptions,
                TableFormerMode,
            )

            # Configure Docling
            table_former_mode = (
                TableFormerMode.ACCURATE
                if self.table_mode == "accurate"
                else TableFormerMode.FAST
            )

            pipeline_options = PdfPipelineOptions(
                do_table_structure=self.table_extraction,
                table_structure_options=TableStructureOptions(mode=table_former_mode),
            )

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pipeline_options,  # type: ignore[dict-item]
                }
            )

            result = converter.convert(str(file_path))
            doc = result.document

            # Extract layout elements
            sections: list[DocumentSection] = []
            tables: list[ExtractedTable] = []
            figures: list[Figure] = []
            footnotes: list[Footnote] = []
            headers: dict[str, str] = {}
            page_count = 0
            table_index = 0
            section_index = 0
            figure_index = 0
            footnote_index = 0

            # Track section hierarchy
            section_stack: list[tuple[int, str]] = []  # (level, id)

            for item, level in doc.iterate_items():
                item_type = type(item).__name__

                # Get page number
                page_num = 1
                if hasattr(item, "prov") and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, "page_no"):
                            page_num = prov.page_no
                            page_count = max(page_count, page_num)
                            break

                # Get position if available
                position = Position()
                if hasattr(item, "prov") and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, "bbox") and prov.bbox:
                            bbox = prov.bbox
                            position = Position(
                                x=getattr(bbox, "l", 0.0),
                                y=getattr(bbox, "t", 0.0),
                                width=getattr(bbox, "r", 0.0) - getattr(bbox, "l", 0.0),
                                height=getattr(bbox, "b", 0.0) - getattr(bbox, "t", 0.0),
                            )
                            break

                if item_type == "TableItem" and self.table_extraction:
                    table = self._extract_table(
                        item=item,
                        table_index=table_index,
                        page_number=page_num,
                        position=position,
                        document_id=str(document_id),
                        tenant_id=str(tenant_id),
                    )
                    if table:
                        tables.append(table)
                        table_index += 1

                elif item_type == "SectionHeaderItem" and self.preserve_layout:
                    text = item.text if hasattr(item, "text") else str(item)
                    heading_level = max(1, min(6, level + 1))

                    # Generate section ID
                    section_id = self._generate_id(
                        f"{tenant_id}:{document_id}:section:{section_index}"
                    )

                    # Update section hierarchy
                    while section_stack and section_stack[-1][0] >= heading_level:
                        section_stack.pop()

                    parent_id = section_stack[-1][1] if section_stack else None

                    section = DocumentSection(
                        id=section_id,
                        heading=text,
                        level=heading_level,
                        content="",
                        page_number=page_num,
                        parent_id=parent_id,
                    )
                    sections.append(section)
                    section_stack.append((heading_level, section_id))
                    section_index += 1

                    # First heading is title
                    if "title" not in headers:
                        headers["title"] = text

                elif item_type == "TextItem":
                    text = item.text if hasattr(item, "text") else str(item)
                    # Append to current section if exists
                    if sections:
                        sections[-1].content += text + "\n"

                elif item_type in ("PictureItem", "FigureItem"):
                    figure_id = self._generate_id(
                        f"{tenant_id}:{document_id}:figure:{figure_index}"
                    )
                    caption = None
                    if hasattr(item, "caption") and item.caption:
                        caption = str(item.caption)

                    figure = Figure(
                        id=figure_id,
                        page_number=page_num,
                        position=position,
                        caption=caption,
                    )
                    figures.append(figure)
                    figure_index += 1

                elif item_type == "FootnoteItem" or "footnote" in item_type.lower():
                    text = item.text if hasattr(item, "text") else str(item)
                    ref = str(item.reference) if hasattr(item, "reference") else str(footnote_index + 1)

                    footnote_id = self._generate_id(
                        f"{tenant_id}:{document_id}:footnote:{footnote_index}"
                    )

                    footnote = Footnote(
                        id=footnote_id,
                        reference=ref,
                        content=text,
                        page_number=page_num,
                    )
                    footnotes.append(footnote)
                    footnote_index += 1

            # Build section hierarchy (link children to parents)
            section_map = {s.id: s for s in sections}
            for section in sections:
                if section.parent_id and section.parent_id in section_map:
                    section_map[section.parent_id].child_ids.append(section.id)

            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            layout = DocumentLayout(
                document_id=str(document_id),
                tenant_id=str(tenant_id),
                page_count=page_count,
                sections=sections,
                tables=tables,
                figures=figures,
                footnotes=footnotes,
                headers=headers,
                metadata={
                    "processing_time_ms": processing_time_ms,
                    "table_count": len(tables),
                    "section_count": len(sections),
                    "figure_count": len(figures),
                    "footnote_count": len(footnotes),
                },
            )

            logger.info(
                "enhanced_parse_completed",
                document_id=str(document_id),
                page_count=page_count,
                tables=len(tables),
                sections=len(sections),
                figures=len(figures),
                footnotes=len(footnotes),
                processing_time_ms=processing_time_ms,
            )

            return layout

        except ImportError as e:
            logger.error(
                "docling_not_installed",
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "enhanced_parse_failed",
                document_id=str(document_id),
                error=str(e),
            )
            raise

    def _extract_table(
        self,
        item: Any,
        table_index: int,
        page_number: int,
        position: Position,
        document_id: str,
        tenant_id: str,
    ) -> Optional[ExtractedTable]:
        """Extract rich table structure from Docling table item.

        Args:
            item: Docling TableItem
            table_index: Table index for ID generation
            page_number: Page number
            position: Table position
            document_id: Document ID
            tenant_id: Tenant ID

        Returns:
            ExtractedTable or None if extraction fails
        """
        try:
            table_id = self._generate_id(
                f"{tenant_id}:{document_id}:table:{table_index}"
            )

            # Get caption
            caption = None
            if hasattr(item, "caption") and item.caption:
                caption = str(item.caption)

            # Extract table structure
            headers: list[str] = []
            rows: list[list[str]] = []

            data = getattr(item, "data", None)
            if data and hasattr(data, "table_cells"):
                # Build grid from cells
                cells = data.table_cells
                if cells:
                    # Find grid dimensions
                    max_row = 0
                    max_col = 0
                    for cell in cells:
                        row_idx = getattr(cell, "row_index", 0)
                        col_idx = getattr(cell, "col_index", 0)
                        max_row = max(max_row, row_idx)
                        max_col = max(max_col, col_idx)

                    # Initialize grid
                    grid: list[list[str]] = [
                        ["" for _ in range(max_col + 1)]
                        for _ in range(max_row + 1)
                    ]

                    # Fill grid
                    for cell in cells:
                        row_idx = getattr(cell, "row_index", 0)
                        col_idx = getattr(cell, "col_index", 0)
                        text = getattr(cell, "text", "")
                        if row_idx < len(grid) and col_idx < len(grid[row_idx]):
                            grid[row_idx][col_idx] = str(text)

                    # First row as headers, rest as data
                    if grid:
                        headers = grid[0]
                        rows = grid[1:] if len(grid) > 1 else []

            # Get markdown representation
            markdown = ""
            export_to_markdown = getattr(item, "export_to_markdown", None)
            if callable(export_to_markdown):
                try:
                    markdown = export_to_markdown()
                except Exception:
                    pass

            table = ExtractedTable(
                id=table_id,
                page_number=page_number,
                position=position,
                headers=headers,
                rows=rows,
                caption=caption,
                markdown=markdown,
                metadata={
                    "row_count": len(rows),
                    "column_count": len(headers),
                    "table_index": table_index,
                },
            )

            return table

        except Exception as e:
            logger.warning(
                "table_extraction_failed",
                table_index=table_index,
                error=str(e),
            )
            return None

    def _table_to_markdown(self, table: ExtractedTable) -> str:
        """Convert table to markdown representation.

        Args:
            table: ExtractedTable to convert

        Returns:
            Markdown string
        """
        if table.markdown:
            return table.markdown
        return table._generate_markdown()

    def table_to_chunks(
        self,
        table: ExtractedTable,
        document_id: str,
        tenant_id: str,
        chunk_rows: bool = True,
        max_rows_per_chunk: int = 10,
    ) -> list[TableChunk]:
        """Convert table to searchable chunks.

        Creates chunks from table content for embedding and search.
        Can create a single chunk for the whole table or split by rows.

        Args:
            table: Table to convert
            document_id: Document ID
            tenant_id: Tenant ID
            chunk_rows: If True, split table into row-based chunks
            max_rows_per_chunk: Max rows per chunk when splitting

        Returns:
            List of TableChunk objects
        """
        chunks: list[TableChunk] = []

        if not chunk_rows or len(table.rows) <= max_rows_per_chunk:
            # Single chunk for whole table
            content = self._table_to_markdown(table)
            if table.caption:
                content = f"**{table.caption}**\n\n{content}"

            chunk_id = self._generate_id(
                f"{tenant_id}:{document_id}:{table.id}:chunk:0"
            )

            chunks.append(TableChunk(
                id=chunk_id,
                table_id=table.id,
                content=content,
                document_id=document_id,
                tenant_id=tenant_id,
                page_number=table.page_number,
                metadata={
                    "chunk_type": "table",
                    "total_rows": len(table.rows),
                    "headers": table.headers,
                },
            ))
        else:
            # Split into multiple chunks
            chunk_index = 0
            for i in range(0, len(table.rows), max_rows_per_chunk):
                row_slice = table.rows[i:i + max_rows_per_chunk]

                # Create sub-table markdown
                sub_table = ExtractedTable(
                    id=f"{table.id}_chunk_{chunk_index}",
                    page_number=table.page_number,
                    position=table.position,
                    headers=table.headers,
                    rows=row_slice,
                    caption=f"{table.caption} (rows {i+1}-{i+len(row_slice)})" if table.caption else None,
                )

                content = self._table_to_markdown(sub_table)
                chunk_id = self._generate_id(
                    f"{tenant_id}:{document_id}:{table.id}:chunk:{chunk_index}"
                )

                chunks.append(TableChunk(
                    id=chunk_id,
                    table_id=table.id,
                    content=content,
                    document_id=document_id,
                    tenant_id=tenant_id,
                    page_number=table.page_number,
                    metadata={
                        "chunk_type": "table",
                        "chunk_index": chunk_index,
                        "row_start": i,
                        "row_end": i + len(row_slice),
                        "total_rows": len(table.rows),
                        "headers": table.headers,
                    },
                ))
                chunk_index += 1

        return chunks

    def _generate_id(self, id_string: str) -> str:
        """Generate deterministic ID from string.

        Args:
            id_string: String to hash

        Returns:
            Short hash-based ID
        """
        hash_digest = hashlib.sha256(id_string.encode()).hexdigest()[:16]
        return hash_digest


class EnhancedDoclingAdapter:
    """Adapter for EnhancedDoclingParser with feature flag support.

    This adapter provides:
    - Feature flag checking (ENHANCED_DOCLING_ENABLED)
    - Configuration from Settings
    - Graceful fallback when disabled
    """

    def __init__(
        self,
        parser: Optional[EnhancedDoclingParser],
        enabled: bool = True,
    ) -> None:
        """Initialize EnhancedDoclingAdapter.

        Args:
            parser: EnhancedDoclingParser instance (or None if disabled)
            enabled: Whether enhanced parsing is enabled
        """
        self._parser = parser
        self.enabled = enabled

    def parse_document(
        self,
        file_path: Path,
        document_id: UUID,
        tenant_id: UUID,
    ) -> Optional[DocumentLayout]:
        """Parse document if enhanced parsing is enabled.

        Args:
            file_path: Path to document file
            document_id: Document identifier
            tenant_id: Tenant identifier

        Returns:
            DocumentLayout if enabled and successful, None otherwise
        """
        if not self.enabled or not self._parser:
            logger.debug(
                "enhanced_docling_disabled",
                enabled=self.enabled,
                has_parser=self._parser is not None,
            )
            return None

        return self._parser.parse_document(
            file_path=file_path,
            document_id=document_id,
            tenant_id=tenant_id,
        )

    def table_to_chunks(
        self,
        table: ExtractedTable,
        document_id: str,
        tenant_id: str,
        **kwargs: Any,
    ) -> list[TableChunk]:
        """Convert table to chunks if enabled.

        Args:
            table: Table to convert
            document_id: Document ID
            tenant_id: Tenant ID
            **kwargs: Additional arguments for table_to_chunks

        Returns:
            List of TableChunk objects (empty if disabled)
        """
        if not self.enabled or not self._parser:
            return []

        return self._parser.table_to_chunks(
            table=table,
            document_id=document_id,
            tenant_id=tenant_id,
            **kwargs,
        )


def get_enhanced_docling_adapter(settings: Any) -> EnhancedDoclingAdapter:
    """Factory function to create EnhancedDoclingAdapter from settings.

    Args:
        settings: Application settings

    Returns:
        Configured EnhancedDoclingAdapter instance
    """
    enabled = getattr(settings, "enhanced_docling_enabled", True)
    table_extraction = getattr(settings, "docling_table_extraction", DEFAULT_TABLE_EXTRACTION)
    preserve_layout = getattr(settings, "docling_preserve_layout", DEFAULT_PRESERVE_LAYOUT)
    table_as_markdown = getattr(settings, "docling_table_as_markdown", DEFAULT_TABLE_AS_MARKDOWN)

    parser = None
    if enabled:
        parser = EnhancedDoclingParser(
            table_extraction=table_extraction,
            preserve_layout=preserve_layout,
            table_as_markdown=table_as_markdown,
        )

    return EnhancedDoclingAdapter(
        parser=parser,
        enabled=enabled,
    )
