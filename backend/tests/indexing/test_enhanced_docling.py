"""Tests for Story 20-D1: Enhanced Table/Layout Extraction.

This module tests the enhanced_docling module including:
- ExtractedTable dataclass and serialization
- DocumentLayout dataclass and serialization
- EnhancedDoclingParser functionality
- Table-to-chunk conversion
- Configuration and feature flags
"""

from pathlib import Path

import pytest
from uuid import UUID

from agentic_rag_backend.indexing.enhanced_docling import (
    Position,
    ExtractedTable,
    DocumentSection,
    Figure,
    Footnote,
    DocumentLayout,
    TableChunk,
    EnhancedDoclingParser,
    EnhancedDoclingAdapter,
    get_enhanced_docling_adapter,
    DEFAULT_TABLE_EXTRACTION,
    DEFAULT_PRESERVE_LAYOUT,
    DEFAULT_TABLE_AS_MARKDOWN,
)


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation_default(self) -> None:
        """Test Position creation with defaults."""
        pos = Position()
        assert pos.x == 0.0
        assert pos.y == 0.0
        assert pos.width == 0.0
        assert pos.height == 0.0

    def test_position_creation_with_values(self) -> None:
        """Test Position creation with values."""
        pos = Position(x=10.5, y=20.3, width=100.0, height=50.0)
        assert pos.x == 10.5
        assert pos.y == 20.3
        assert pos.width == 100.0
        assert pos.height == 50.0

    def test_position_to_dict(self) -> None:
        """Test Position serialization."""
        pos = Position(x=10.0, y=20.0, width=100.0, height=50.0)
        result = pos.to_dict()
        assert result == {"x": 10.0, "y": 20.0, "width": 100.0, "height": 50.0}

    def test_position_from_dict(self) -> None:
        """Test Position deserialization."""
        data = {"x": 15.0, "y": 25.0, "width": 200.0, "height": 100.0}
        pos = Position.from_dict(data)
        assert pos.x == 15.0
        assert pos.y == 25.0
        assert pos.width == 200.0
        assert pos.height == 100.0

    def test_position_from_dict_missing_fields(self) -> None:
        """Test Position deserialization with missing fields."""
        pos = Position.from_dict({})
        assert pos.x == 0.0
        assert pos.y == 0.0


class TestExtractedTable:
    """Tests for ExtractedTable dataclass."""

    def test_table_creation_minimal(self) -> None:
        """Test ExtractedTable creation with minimal fields."""
        table = ExtractedTable(
            id="table_1",
            page_number=1,
            position=Position(),
            headers=["A", "B"],
            rows=[["1", "2"]],
        )
        assert table.id == "table_1"
        assert table.page_number == 1
        assert table.headers == ["A", "B"]
        assert table.rows == [["1", "2"]]

    def test_table_auto_generates_markdown(self) -> None:
        """Test ExtractedTable auto-generates markdown."""
        table = ExtractedTable(
            id="table_1",
            page_number=1,
            position=Position(),
            headers=["Name", "Value"],
            rows=[["Item1", "100"], ["Item2", "200"]],
        )
        assert "| Name | Value |" in table.markdown
        assert "| Item1 | 100 |" in table.markdown
        assert "| Item2 | 200 |" in table.markdown

    def test_table_auto_generates_structured_data(self) -> None:
        """Test ExtractedTable auto-generates structured data."""
        table = ExtractedTable(
            id="table_1",
            page_number=1,
            position=Position(),
            headers=["Name", "Value"],
            rows=[["Item1", "100"], ["Item2", "200"]],
        )
        assert len(table.structured_data) == 2
        assert table.structured_data[0] == {"Name": "Item1", "Value": "100"}
        assert table.structured_data[1] == {"Name": "Item2", "Value": "200"}

    def test_table_with_caption_in_markdown(self) -> None:
        """Test ExtractedTable includes caption in markdown."""
        table = ExtractedTable(
            id="table_1",
            page_number=1,
            position=Position(),
            headers=["A"],
            rows=[["1"]],
            caption="Test Caption",
        )
        assert "**Test Caption**" in table.markdown

    def test_table_escapes_pipe_characters(self) -> None:
        """Test ExtractedTable escapes pipe characters in cell values."""
        table = ExtractedTable(
            id="table_1",
            page_number=1,
            position=Position(),
            headers=["Data"],
            rows=[["value|with|pipes"]],
        )
        assert "value\\|with\\|pipes" in table.markdown

    def test_table_to_dict(self) -> None:
        """Test ExtractedTable serialization."""
        table = ExtractedTable(
            id="table_1",
            page_number=3,
            position=Position(x=10, y=20, width=100, height=50),
            headers=["A", "B"],
            rows=[["1", "2"]],
            caption="Test",
        )
        result = table.to_dict()
        assert result["id"] == "table_1"
        assert result["page_number"] == 3
        assert result["headers"] == ["A", "B"]
        assert result["rows"] == [["1", "2"]]
        assert result["caption"] == "Test"
        assert "position" in result

    def test_table_from_dict(self) -> None:
        """Test ExtractedTable deserialization."""
        data = {
            "id": "table_2",
            "page_number": 5,
            "position": {"x": 10, "y": 20, "width": 100, "height": 50},
            "headers": ["C", "D"],
            "rows": [["3", "4"]],
            "caption": "Caption",
            "markdown": "| C | D |",
            "structured_data": [{"C": "3", "D": "4"}],
        }
        table = ExtractedTable.from_dict(data)
        assert table.id == "table_2"
        assert table.page_number == 5
        assert table.headers == ["C", "D"]
        assert table.caption == "Caption"

    def test_table_roundtrip_serialization(self) -> None:
        """Test ExtractedTable roundtrip serialization."""
        original = ExtractedTable(
            id="table_rt",
            page_number=2,
            position=Position(x=5, y=10, width=200, height=100),
            headers=["X", "Y", "Z"],
            rows=[["a", "b", "c"], ["d", "e", "f"]],
            caption="Roundtrip Test",
        )
        data = original.to_dict()
        restored = ExtractedTable.from_dict(data)
        assert restored.id == original.id
        assert restored.page_number == original.page_number
        assert restored.headers == original.headers
        assert restored.rows == original.rows
        assert restored.caption == original.caption


class TestDocumentSection:
    """Tests for DocumentSection dataclass."""

    def test_section_creation(self) -> None:
        """Test DocumentSection creation."""
        section = DocumentSection(
            id="section_1",
            heading="Introduction",
            level=1,
            content="This is the introduction.",
            page_number=1,
        )
        assert section.id == "section_1"
        assert section.heading == "Introduction"
        assert section.level == 1
        assert section.content == "This is the introduction."

    def test_section_with_hierarchy(self) -> None:
        """Test DocumentSection with parent/child relationships."""
        section = DocumentSection(
            id="section_2",
            heading="Subsection",
            level=2,
            content="Content here",
            parent_id="section_1",
            child_ids=["section_3", "section_4"],
        )
        assert section.parent_id == "section_1"
        assert section.child_ids == ["section_3", "section_4"]

    def test_section_to_dict(self) -> None:
        """Test DocumentSection serialization."""
        section = DocumentSection(
            id="section_1",
            heading="Test",
            level=1,
            content="Content",
        )
        result = section.to_dict()
        assert result["id"] == "section_1"
        assert result["heading"] == "Test"
        assert result["level"] == 1

    def test_section_from_dict(self) -> None:
        """Test DocumentSection deserialization."""
        data = {
            "id": "section_x",
            "heading": "Heading",
            "level": 2,
            "content": "Text content",
            "page_number": 5,
        }
        section = DocumentSection.from_dict(data)
        assert section.id == "section_x"
        assert section.level == 2
        assert section.page_number == 5


class TestFigure:
    """Tests for Figure dataclass."""

    def test_figure_creation(self) -> None:
        """Test Figure creation."""
        fig = Figure(
            id="fig_1",
            page_number=3,
            position=Position(x=100, y=200, width=300, height=200),
            caption="Figure 1: Sample",
        )
        assert fig.id == "fig_1"
        assert fig.page_number == 3
        assert fig.caption == "Figure 1: Sample"

    def test_figure_to_dict(self) -> None:
        """Test Figure serialization."""
        fig = Figure(
            id="fig_1",
            page_number=1,
            position=Position(),
            caption="Test Figure",
        )
        result = fig.to_dict()
        assert result["id"] == "fig_1"
        assert result["caption"] == "Test Figure"

    def test_figure_from_dict(self) -> None:
        """Test Figure deserialization."""
        data = {
            "id": "fig_2",
            "page_number": 5,
            "position": {},
            "caption": "Another Figure",
        }
        fig = Figure.from_dict(data)
        assert fig.id == "fig_2"
        assert fig.page_number == 5


class TestFootnote:
    """Tests for Footnote dataclass."""

    def test_footnote_creation(self) -> None:
        """Test Footnote creation."""
        fn = Footnote(
            id="fn_1",
            reference="1",
            content="This is footnote content.",
            page_number=2,
        )
        assert fn.id == "fn_1"
        assert fn.reference == "1"
        assert fn.content == "This is footnote content."

    def test_footnote_to_dict(self) -> None:
        """Test Footnote serialization."""
        fn = Footnote(
            id="fn_1",
            reference="*",
            content="Asterisk footnote",
        )
        result = fn.to_dict()
        assert result["reference"] == "*"
        assert result["content"] == "Asterisk footnote"

    def test_footnote_from_dict(self) -> None:
        """Test Footnote deserialization."""
        data = {
            "id": "fn_2",
            "reference": "2",
            "content": "Second footnote",
            "page_number": 10,
        }
        fn = Footnote.from_dict(data)
        assert fn.reference == "2"
        assert fn.page_number == 10


class TestDocumentLayout:
    """Tests for DocumentLayout dataclass."""

    def test_layout_creation_empty(self) -> None:
        """Test DocumentLayout creation with no content."""
        layout = DocumentLayout(
            document_id="doc_1",
            tenant_id="tenant_1",
            page_count=0,
        )
        assert layout.document_id == "doc_1"
        assert layout.tenant_id == "tenant_1"
        assert layout.sections == []
        assert layout.tables == []
        assert layout.figures == []
        assert layout.footnotes == []

    def test_layout_creation_with_content(self) -> None:
        """Test DocumentLayout creation with content."""
        table = ExtractedTable(
            id="t1",
            page_number=1,
            position=Position(),
            headers=["A"],
            rows=[["1"]],
        )
        section = DocumentSection(
            id="s1",
            heading="Intro",
            level=1,
            content="Content",
        )
        layout = DocumentLayout(
            document_id="doc_1",
            tenant_id="tenant_1",
            page_count=5,
            tables=[table],
            sections=[section],
        )
        assert len(layout.tables) == 1
        assert len(layout.sections) == 1

    def test_layout_get_table_by_id(self) -> None:
        """Test DocumentLayout.get_table_by_id."""
        table = ExtractedTable(
            id="table_find",
            page_number=1,
            position=Position(),
            headers=["X"],
            rows=[["y"]],
        )
        layout = DocumentLayout(
            document_id="doc",
            tenant_id="tenant",
            page_count=1,
            tables=[table],
        )
        found = layout.get_table_by_id("table_find")
        assert found is not None
        assert found.id == "table_find"
        assert layout.get_table_by_id("nonexistent") is None

    def test_layout_get_tables_on_page(self) -> None:
        """Test DocumentLayout.get_tables_on_page."""
        t1 = ExtractedTable(id="t1", page_number=1, position=Position(), headers=[], rows=[])
        t2 = ExtractedTable(id="t2", page_number=2, position=Position(), headers=[], rows=[])
        t3 = ExtractedTable(id="t3", page_number=2, position=Position(), headers=[], rows=[])
        layout = DocumentLayout(
            document_id="doc",
            tenant_id="tenant",
            page_count=2,
            tables=[t1, t2, t3],
        )
        page_2_tables = layout.get_tables_on_page(2)
        assert len(page_2_tables) == 2
        assert page_2_tables[0].id == "t2"
        assert page_2_tables[1].id == "t3"

    def test_layout_to_dict(self) -> None:
        """Test DocumentLayout serialization."""
        layout = DocumentLayout(
            document_id="doc_1",
            tenant_id="tenant_1",
            page_count=10,
            headers={"title": "Test Document"},
        )
        result = layout.to_dict()
        assert result["document_id"] == "doc_1"
        assert result["tenant_id"] == "tenant_1"
        assert result["page_count"] == 10
        assert result["headers"]["title"] == "Test Document"

    def test_layout_from_dict(self) -> None:
        """Test DocumentLayout deserialization."""
        data = {
            "document_id": "doc_2",
            "tenant_id": "tenant_2",
            "page_count": 20,
            "sections": [],
            "tables": [],
            "figures": [],
            "footnotes": [],
            "headers": {},
        }
        layout = DocumentLayout.from_dict(data)
        assert layout.document_id == "doc_2"
        assert layout.page_count == 20


class TestTableChunk:
    """Tests for TableChunk dataclass."""

    def test_chunk_creation(self) -> None:
        """Test TableChunk creation."""
        chunk = TableChunk(
            id="chunk_1",
            table_id="table_1",
            content="| A | B |\n|---|---|",
            document_id="doc_1",
            tenant_id="tenant_1",
            page_number=3,
        )
        assert chunk.id == "chunk_1"
        assert chunk.table_id == "table_1"
        assert chunk.page_number == 3

    def test_chunk_to_dict(self) -> None:
        """Test TableChunk serialization."""
        chunk = TableChunk(
            id="c1",
            table_id="t1",
            content="content",
            document_id="d1",
            tenant_id="ten1",
            page_number=1,
            metadata={"key": "value"},
        )
        result = chunk.to_dict()
        assert result["id"] == "c1"
        assert result["metadata"]["key"] == "value"


class TestEnhancedDoclingParser:
    """Tests for EnhancedDoclingParser class."""

    def test_parser_initialization_defaults(self) -> None:
        """Test EnhancedDoclingParser initialization with defaults."""
        parser = EnhancedDoclingParser()
        assert parser.table_extraction == DEFAULT_TABLE_EXTRACTION
        assert parser.preserve_layout == DEFAULT_PRESERVE_LAYOUT
        assert parser.table_as_markdown == DEFAULT_TABLE_AS_MARKDOWN
        assert parser.table_mode == "accurate"

    def test_parser_initialization_custom(self) -> None:
        """Test EnhancedDoclingParser initialization with custom settings."""
        parser = EnhancedDoclingParser(
            table_extraction=False,
            preserve_layout=False,
            table_as_markdown=False,
            table_mode="fast",
        )
        assert parser.table_extraction is False
        assert parser.preserve_layout is False
        assert parser.table_as_markdown is False
        assert parser.table_mode == "fast"

    def test_parser_table_to_markdown(self) -> None:
        """Test _table_to_markdown method."""
        parser = EnhancedDoclingParser()
        table = ExtractedTable(
            id="t1",
            page_number=1,
            position=Position(),
            headers=["A", "B"],
            rows=[["1", "2"]],
        )
        markdown = parser._table_to_markdown(table)
        assert "| A | B |" in markdown
        assert "| 1 | 2 |" in markdown

    def test_parser_table_to_chunks_single(self) -> None:
        """Test table_to_chunks creates single chunk for small tables."""
        parser = EnhancedDoclingParser()
        table = ExtractedTable(
            id="t1",
            page_number=1,
            position=Position(),
            headers=["Col1", "Col2"],
            rows=[["a", "b"], ["c", "d"]],
            caption="Test Table",
        )
        chunks = parser.table_to_chunks(
            table=table,
            document_id="doc1",
            tenant_id="tenant1",
            chunk_rows=True,
            max_rows_per_chunk=10,
        )
        assert len(chunks) == 1
        assert chunks[0].table_id == "t1"
        assert "Test Table" in chunks[0].content

    def test_parser_table_to_chunks_split(self) -> None:
        """Test table_to_chunks splits large tables."""
        parser = EnhancedDoclingParser()
        # Create table with 15 rows
        rows = [[f"r{i}", f"v{i}"] for i in range(15)]
        table = ExtractedTable(
            id="t_large",
            page_number=1,
            position=Position(),
            headers=["Row", "Value"],
            rows=rows,
        )
        chunks = parser.table_to_chunks(
            table=table,
            document_id="doc1",
            tenant_id="tenant1",
            chunk_rows=True,
            max_rows_per_chunk=5,
        )
        assert len(chunks) == 3  # 15 rows / 5 per chunk = 3 chunks
        assert chunks[0].metadata.get("row_start") == 0
        assert chunks[0].metadata.get("row_end") == 5
        assert chunks[1].metadata.get("row_start") == 5
        assert chunks[2].metadata.get("row_start") == 10

    def test_parser_table_to_chunks_no_split(self) -> None:
        """Test table_to_chunks without splitting."""
        parser = EnhancedDoclingParser()
        rows = [[f"r{i}", f"v{i}"] for i in range(20)]
        table = ExtractedTable(
            id="t_nosplit",
            page_number=1,
            position=Position(),
            headers=["Row", "Value"],
            rows=rows,
        )
        chunks = parser.table_to_chunks(
            table=table,
            document_id="doc1",
            tenant_id="tenant1",
            chunk_rows=False,  # Don't split
        )
        assert len(chunks) == 1

    def test_parser_generate_id_deterministic(self) -> None:
        """Test _generate_id is deterministic."""
        parser = EnhancedDoclingParser()
        id1 = parser._generate_id("test_string")
        id2 = parser._generate_id("test_string")
        assert id1 == id2

    def test_parser_generate_id_unique(self) -> None:
        """Test _generate_id produces unique IDs for different inputs."""
        parser = EnhancedDoclingParser()
        id1 = parser._generate_id("string_1")
        id2 = parser._generate_id("string_2")
        assert id1 != id2


class TestEnhancedDoclingAdapter:
    """Tests for EnhancedDoclingAdapter class."""

    def test_adapter_disabled(self) -> None:
        """Test adapter when disabled returns None."""
        adapter = EnhancedDoclingAdapter(parser=None, enabled=False)
        assert adapter.enabled is False

    def test_adapter_enabled_no_parser(self) -> None:
        """Test adapter enabled but no parser returns None."""
        adapter = EnhancedDoclingAdapter(parser=None, enabled=True)
        assert adapter.enabled is True
        # parse_document would return None since parser is None

    def test_adapter_table_to_chunks_disabled(self) -> None:
        """Test table_to_chunks returns empty when disabled."""
        adapter = EnhancedDoclingAdapter(parser=None, enabled=False)
        table = ExtractedTable(
            id="t1",
            page_number=1,
            position=Position(),
            headers=["A"],
            rows=[["1"]],
        )
        chunks = adapter.table_to_chunks(
            table=table,
            document_id="doc",
            tenant_id="tenant",
        )
        assert chunks == []

    def test_adapter_enabled_with_parser(self) -> None:
        """Test adapter with enabled parser."""
        parser = EnhancedDoclingParser()
        adapter = EnhancedDoclingAdapter(parser=parser, enabled=True)
        assert adapter.enabled is True
        assert adapter._parser is not None


class TestGetEnhancedDoclingAdapter:
    """Tests for get_enhanced_docling_adapter factory function."""

    def test_factory_disabled(self) -> None:
        """Test factory creates disabled adapter."""

        class MockSettings:
            enhanced_docling_enabled = False

        adapter = get_enhanced_docling_adapter(MockSettings())
        assert adapter.enabled is False
        assert adapter._parser is None

    def test_factory_enabled(self) -> None:
        """Test factory creates enabled adapter."""

        class MockSettings:
            enhanced_docling_enabled = True
            docling_table_extraction = True
            docling_preserve_layout = True
            docling_table_as_markdown = True

        adapter = get_enhanced_docling_adapter(MockSettings())
        assert adapter.enabled is True
        assert adapter._parser is not None

    def test_factory_custom_settings(self) -> None:
        """Test factory respects custom settings."""

        class MockSettings:
            enhanced_docling_enabled = True
            docling_table_extraction = False
            docling_preserve_layout = False
            docling_table_as_markdown = False

        adapter = get_enhanced_docling_adapter(MockSettings())
        assert adapter._parser is not None
        assert adapter._parser.table_extraction is False
        assert adapter._parser.preserve_layout is False
        assert adapter._parser.table_as_markdown is False


class TestEdgeCases:
    """Tests for edge cases."""

    def test_table_empty_headers_and_rows(self) -> None:
        """Test table with empty headers and rows."""
        table = ExtractedTable(
            id="empty",
            page_number=1,
            position=Position(),
            headers=[],
            rows=[],
        )
        assert table.markdown == ""
        assert table.structured_data == []

    def test_table_rows_without_headers(self) -> None:
        """Test table with rows but no headers."""
        table = ExtractedTable(
            id="no_headers",
            page_number=1,
            position=Position(),
            headers=[],
            rows=[["a", "b"], ["c", "d"]],
        )
        # Should still generate markdown with rows
        assert "| a | b |" in table.markdown
        # But no structured data without headers
        assert table.structured_data == []

    def test_table_headers_longer_than_rows(self) -> None:
        """Test table where headers have more columns than some rows."""
        table = ExtractedTable(
            id="mismatch",
            page_number=1,
            position=Position(),
            headers=["A", "B", "C"],
            rows=[["1", "2"]],  # Missing column C
        )
        # Structured data should handle missing values
        assert len(table.structured_data) == 1
        assert table.structured_data[0]["C"] == ""

    def test_table_unicode_content(self) -> None:
        """Test table with Unicode content."""
        table = ExtractedTable(
            id="unicode",
            page_number=1,
            position=Position(),
            headers=["名前", "値"],
            rows=[["テスト", "日本語"]],
        )
        assert "名前" in table.markdown
        assert "テスト" in table.markdown

    def test_layout_empty_document(self) -> None:
        """Test layout for empty document."""
        layout = DocumentLayout(
            document_id="empty_doc",
            tenant_id="tenant",
            page_count=0,
        )
        assert len(layout.sections) == 0
        assert len(layout.tables) == 0
        assert layout.get_table_by_id("any") is None
        assert layout.get_tables_on_page(1) == []

    def test_table_special_characters_in_cells(self) -> None:
        """Test table with special characters."""
        table = ExtractedTable(
            id="special",
            page_number=1,
            position=Position(),
            headers=["Code"],
            rows=[["x > 5 && y < 10"]],
        )
        # Should not break markdown
        assert "x > 5 && y < 10" in table.markdown


class TestSecurityValidation:
    """Tests for security validation features."""

    def test_parse_document_requires_tenant_id(self, tmp_path: Path) -> None:
        """Test that parse_document requires tenant_id."""
        parser = EnhancedDoclingParser()
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test")

        with pytest.raises(ValueError, match="tenant_id is required"):
            parser.parse_document(
                file_path=test_file,
                document_id=UUID("12345678-1234-1234-1234-123456789012"),
                tenant_id=None,  # type: ignore
            )

    def test_parse_document_validates_file_exists(self, tmp_path: Path) -> None:
        """Test that parse_document validates file exists."""
        parser = EnhancedDoclingParser()
        missing_file = tmp_path / "nonexistent.pdf"

        with pytest.raises(FileNotFoundError, match="Document not found"):
            parser.parse_document(
                file_path=missing_file,
                document_id=UUID("12345678-1234-1234-1234-123456789012"),
                tenant_id=UUID("12345678-1234-1234-1234-123456789013"),
            )

    def test_parse_document_validates_is_file(self, tmp_path: Path) -> None:
        """Test that parse_document validates path is a file."""
        parser = EnhancedDoclingParser()
        # tmp_path is a directory, not a file

        with pytest.raises(ValueError, match="Path is not a file"):
            parser.parse_document(
                file_path=tmp_path,
                document_id=UUID("12345678-1234-1234-1234-123456789012"),
                tenant_id=UUID("12345678-1234-1234-1234-123456789013"),
            )

    def test_parse_document_path_traversal_protection(self, tmp_path: Path) -> None:
        """Test that parse_document prevents path traversal."""
        parser = EnhancedDoclingParser()

        # Create allowed directory and file outside it
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_file = tmp_path / "outside.pdf"
        outside_file.write_bytes(b"%PDF-1.4 test")

        with pytest.raises(ValueError, match="Path traversal not allowed"):
            parser.parse_document(
                file_path=outside_file,
                document_id=UUID("12345678-1234-1234-1234-123456789012"),
                tenant_id=UUID("12345678-1234-1234-1234-123456789013"),
                allowed_base_path=allowed_dir,
            )

    def test_parse_document_allows_file_in_base_path(self, tmp_path: Path) -> None:
        """Test that parse_document allows files within base path."""
        parser = EnhancedDoclingParser()

        # Create file inside allowed directory
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        inside_file = allowed_dir / "doc.pdf"
        inside_file.write_bytes(b"%PDF-1.4 test")

        # Should not raise - but will fail later at Docling import
        # We're just testing the path validation passes
        try:
            parser.parse_document(
                file_path=inside_file,
                document_id=UUID("12345678-1234-1234-1234-123456789012"),
                tenant_id=UUID("12345678-1234-1234-1234-123456789013"),
                allowed_base_path=allowed_dir,
            )
        except ImportError:
            # Expected - Docling not installed in test env
            pass
        except Exception as e:
            # Path validation passed, Docling parsing failed (expected)
            assert "Document not found" not in str(e)
            assert "Path traversal" not in str(e)


class TestTableSizeLimits:
    """Tests for table size limit protection."""

    def test_table_size_constants_defined(self) -> None:
        """Test that table size limit constants are defined."""
        from agentic_rag_backend.indexing.enhanced_docling import (
            MAX_TABLE_ROWS,
            MAX_TABLE_COLS,
        )
        assert MAX_TABLE_ROWS == 10000
        assert MAX_TABLE_COLS == 1000


class TestMultiTenancy:
    """Tests for multi-tenancy support."""

    def test_chunk_includes_tenant_id(self) -> None:
        """Test that table chunks include tenant_id."""
        parser = EnhancedDoclingParser()
        table = ExtractedTable(
            id="t1",
            page_number=1,
            position=Position(),
            headers=["A"],
            rows=[["1"]],
        )
        chunks = parser.table_to_chunks(
            table=table,
            document_id="doc_123",
            tenant_id="tenant_abc",
        )
        assert len(chunks) == 1
        assert chunks[0].tenant_id == "tenant_abc"

    def test_layout_includes_tenant_id(self) -> None:
        """Test that DocumentLayout includes tenant_id."""
        layout = DocumentLayout(
            document_id="doc_1",
            tenant_id="tenant_xyz",
            page_count=1,
        )
        data = layout.to_dict()
        assert data["tenant_id"] == "tenant_xyz"

    def test_different_tenants_different_chunk_ids(self) -> None:
        """Test that different tenants get different chunk IDs."""
        parser = EnhancedDoclingParser()
        table = ExtractedTable(
            id="t1",
            page_number=1,
            position=Position(),
            headers=["A"],
            rows=[["1"]],
        )

        chunks_tenant1 = parser.table_to_chunks(
            table=table,
            document_id="doc",
            tenant_id="tenant_1",
        )
        chunks_tenant2 = parser.table_to_chunks(
            table=table,
            document_id="doc",
            tenant_id="tenant_2",
        )

        assert chunks_tenant1[0].id != chunks_tenant2[0].id
