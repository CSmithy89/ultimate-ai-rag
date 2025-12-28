"""Unit tests for the Docling PDF parser wrapper.

Tests cover:
- PDF validation
- Password protection detection
- File hash computation
- PDF parsing with mocked Docling
- ParsedDocument model creation
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from agentic_rag_backend.core.errors import InvalidPdfError, ParseError, PasswordProtectedError
from agentic_rag_backend.indexing.parser import (
    check_password_protected,
    compute_file_hash,
    parse_pdf,
    validate_pdf,
)
from agentic_rag_backend.models.documents import (
    DocumentSection,
    FootnoteContent,
    ParsedDocument,
    TableContent,
)


# Fixtures


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Minimal valid PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture
def sample_pdf(tmp_path: Path, sample_pdf_content: bytes) -> Path:
    """Create a simple test PDF file."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(sample_pdf_content)
    return pdf_path


@pytest.fixture
def encrypted_pdf_content() -> bytes:
    """PDF content with encryption marker."""
    return b"%PDF-1.4\n/Encrypt\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture
def encrypted_pdf(tmp_path: Path, encrypted_pdf_content: bytes) -> Path:
    """Create a PDF file with encryption marker."""
    pdf_path = tmp_path / "encrypted.pdf"
    pdf_path.write_bytes(encrypted_pdf_content)
    return pdf_path


@pytest.fixture
def non_pdf_file(tmp_path: Path) -> Path:
    """Create a non-PDF file."""
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("This is not a PDF file")
    return txt_path


@pytest.fixture
def tenant_id():
    """Generate a random tenant ID."""
    return uuid4()


@pytest.fixture
def document_id():
    """Generate a random document ID."""
    return uuid4()


# Tests for validate_pdf


class TestValidatePdf:
    """Tests for the validate_pdf function."""

    def test_validate_pdf_valid_file(self, sample_pdf: Path):
        """Test PDF validation with valid file."""
        assert validate_pdf(sample_pdf) is True

    def test_validate_pdf_invalid_file(self, non_pdf_file: Path):
        """Test PDF validation with non-PDF file."""
        assert validate_pdf(non_pdf_file) is False

    def test_validate_pdf_nonexistent_file(self):
        """Test PDF validation with missing file."""
        assert validate_pdf(Path("/nonexistent/file.pdf")) is False

    def test_validate_pdf_empty_path(self, tmp_path: Path):
        """Test PDF validation with empty file."""
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")
        assert validate_pdf(empty_file) is False


# Tests for check_password_protected


class TestCheckPasswordProtected:
    """Tests for the check_password_protected function."""

    def test_not_password_protected(self, sample_pdf: Path):
        """Test that normal PDF is not detected as password protected."""
        assert check_password_protected(sample_pdf) is False

    def test_password_protected(self, encrypted_pdf: Path):
        """Test that encrypted PDF is detected as password protected."""
        assert check_password_protected(encrypted_pdf) is True

    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        # Should return False (not detected as encrypted), not raise
        assert check_password_protected(Path("/nonexistent/file.pdf")) is False


# Tests for compute_file_hash


class TestComputeFileHash:
    """Tests for the compute_file_hash function."""

    def test_compute_hash_returns_64_chars(self, sample_pdf: Path):
        """Test that hash is 64 characters (SHA-256 hex)."""
        hash_value = compute_file_hash(sample_pdf)
        assert len(hash_value) == 64

    def test_compute_hash_is_deterministic(self, sample_pdf: Path):
        """Test that same file produces same hash."""
        hash1 = compute_file_hash(sample_pdf)
        hash2 = compute_file_hash(sample_pdf)
        assert hash1 == hash2

    def test_compute_hash_matches_hashlib(self, sample_pdf: Path, sample_pdf_content: bytes):
        """Test that hash matches direct hashlib computation."""
        expected_hash = hashlib.sha256(sample_pdf_content).hexdigest()
        actual_hash = compute_file_hash(sample_pdf)
        assert actual_hash == expected_hash


# Tests for parse_pdf


class TestParsePdf:
    """Tests for the parse_pdf function."""

    def test_parse_pdf_invalid_file(self, non_pdf_file: Path, tenant_id, document_id):
        """Test that invalid PDF raises InvalidPdfError."""
        with pytest.raises(InvalidPdfError):
            parse_pdf(non_pdf_file, document_id, tenant_id)

    def test_parse_pdf_password_protected(self, encrypted_pdf: Path, tenant_id, document_id):
        """Test that password-protected PDF raises PasswordProtectedError."""
        with pytest.raises(PasswordProtectedError):
            parse_pdf(encrypted_pdf, document_id, tenant_id)

    def test_parse_pdf_nonexistent_file(self, tenant_id, document_id):
        """Test that nonexistent file raises InvalidPdfError."""
        with pytest.raises(InvalidPdfError):
            parse_pdf(Path("/nonexistent/file.pdf"), document_id, tenant_id)

    @patch("agentic_rag_backend.indexing.parser.parse_pdf_elements")
    def test_parse_pdf_with_sections(
        self, mock_parse_elements, sample_pdf: Path, tenant_id, document_id
    ):
        """Test that parse_pdf extracts document sections."""

        # Mock the generator to yield section elements
        def mock_generator(*args, **kwargs):
            yield {
                "type": "section",
                "content": "Test Heading",
                "metadata": {"level": 1, "page_number": 1, "heading": "Test Heading"},
            }
            yield {
                "type": "text",
                "content": "This is paragraph text.",
                "metadata": {"page_number": 1},
            }
            # Return final metadata via StopIteration
            return {"page_count": 1, "title": "Test Heading"}

        # Create a generator that properly returns value
        gen = mock_generator()
        mock_parse_elements.return_value = gen

        result = parse_pdf(sample_pdf, document_id, tenant_id)

        assert isinstance(result, ParsedDocument)
        assert result.id == document_id
        assert result.tenant_id == tenant_id
        assert len(result.sections) >= 1
        assert result.sections[0].heading == "Test Heading"
        assert result.sections[0].level == 1

    @patch("agentic_rag_backend.indexing.parser.parse_pdf_elements")
    def test_parse_pdf_with_tables(
        self, mock_parse_elements, sample_pdf: Path, tenant_id, document_id
    ):
        """Test that parse_pdf extracts tables to markdown."""

        def mock_generator(*args, **kwargs):
            yield {
                "type": "table",
                "content": "| A | B |\n|---|---|\n| 1 | 2 |",
                "metadata": {
                    "row_count": 2,
                    "column_count": 2,
                    "page_number": 1,
                    "caption": "Test Table",
                },
            }
            return {"page_count": 1, "title": None}

        gen = mock_generator()
        mock_parse_elements.return_value = gen

        result = parse_pdf(sample_pdf, document_id, tenant_id)

        assert isinstance(result, ParsedDocument)
        assert len(result.tables) == 1
        assert "| A | B |" in result.tables[0].markdown
        assert result.tables[0].row_count == 2
        assert result.tables[0].column_count == 2
        assert result.tables[0].caption == "Test Table"

    @patch("agentic_rag_backend.indexing.parser.parse_pdf_elements")
    def test_parse_pdf_with_footnotes(
        self, mock_parse_elements, sample_pdf: Path, tenant_id, document_id
    ):
        """Test that parse_pdf extracts footnotes."""

        def mock_generator(*args, **kwargs):
            yield {
                "type": "footnote",
                "content": "This is a footnote reference.",
                "metadata": {"reference": "1", "page_number": 2},
            }
            return {"page_count": 2, "title": None}

        gen = mock_generator()
        mock_parse_elements.return_value = gen

        result = parse_pdf(sample_pdf, document_id, tenant_id)

        assert isinstance(result, ParsedDocument)
        assert len(result.footnotes) == 1
        assert result.footnotes[0].reference == "1"
        assert result.footnotes[0].content == "This is a footnote reference."
        assert result.footnotes[0].page_number == 2

    @patch("agentic_rag_backend.indexing.parser.parse_pdf_elements")
    def test_parse_pdf_tracking_processing_time(
        self, mock_parse_elements, sample_pdf: Path, tenant_id, document_id
    ):
        """Test that processing time is tracked (NFR2)."""

        def mock_generator(*args, **kwargs):
            yield {"type": "text", "content": "Test content", "metadata": {"page_number": 1}}
            return {"page_count": 1, "title": None}

        gen = mock_generator()
        mock_parse_elements.return_value = gen

        result = parse_pdf(sample_pdf, document_id, tenant_id)

        # Processing time should be set and >= 0
        assert result.processing_time_ms >= 0

    @patch("agentic_rag_backend.indexing.parser.parse_pdf_elements")
    def test_parse_pdf_returns_content_hash(
        self, mock_parse_elements, sample_pdf: Path, sample_pdf_content: bytes, tenant_id, document_id
    ):
        """Test that content hash is correctly computed."""

        def mock_generator(*args, **kwargs):
            yield {"type": "text", "content": "Test", "metadata": {"page_number": 1}}
            return {"page_count": 1, "title": None}

        gen = mock_generator()
        mock_parse_elements.return_value = gen

        result = parse_pdf(sample_pdf, document_id, tenant_id)

        expected_hash = hashlib.sha256(sample_pdf_content).hexdigest()
        assert result.content_hash == expected_hash


# Tests for ParsedDocument model


class TestParsedDocumentModel:
    """Tests for the ParsedDocument Pydantic model."""

    def test_to_unified_document(self, tenant_id, document_id):
        """Test conversion to UnifiedDocument format."""
        parsed = ParsedDocument(
            id=document_id,
            tenant_id=tenant_id,
            filename="test.pdf",
            content_hash="a" * 64,
            file_size=1024,
            page_count=5,
            sections=[
                DocumentSection(heading="Intro", level=1, content="Introduction text", page_number=1),
                DocumentSection(heading="Details", level=2, content="Detail text", page_number=2),
            ],
            tables=[
                TableContent(
                    caption="Data Table",
                    markdown="| Col1 | Col2 |\n|---|---|\n| a | b |",
                    row_count=2,
                    column_count=2,
                    page_number=3,
                ),
            ],
            footnotes=[
                FootnoteContent(reference="1", content="Footnote text", page_number=2),
            ],
            processing_time_ms=5000,
        )

        unified = parsed.to_unified_document()

        assert unified.id == document_id
        assert unified.tenant_id == tenant_id
        assert unified.source_type.value == "pdf"
        assert unified.filename == "test.pdf"
        assert unified.content_hash == "a" * 64
        # Content should contain sections, tables, and footnotes
        assert "# Intro" in unified.content
        assert "## Details" in unified.content
        assert "| Col1 | Col2 |" in unified.content
        assert "Footnote text" in unified.content

    def test_parsed_document_required_fields(self, tenant_id, document_id):
        """Test that required fields are enforced."""
        with pytest.raises(Exception):  # Pydantic validation error
            ParsedDocument(
                id=document_id,
                tenant_id=tenant_id,
                # Missing filename, content_hash, file_size, page_count
            )

    def test_parsed_document_defaults(self, tenant_id, document_id):
        """Test that optional fields have correct defaults."""
        parsed = ParsedDocument(
            id=document_id,
            tenant_id=tenant_id,
            filename="test.pdf",
            content_hash="a" * 64,
            file_size=1024,
            page_count=1,
        )

        assert parsed.sections == []
        assert parsed.tables == []
        assert parsed.footnotes == []
        assert parsed.processing_time_ms == 0
        assert parsed.source_type == "pdf"
