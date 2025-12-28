"""API endpoint tests for PDF document upload (Story 4.2).

Tests cover:
- Successful PDF upload
- Invalid content type handling
- File size limit enforcement
- Invalid PDF content handling
- Missing tenant_id handling
- Response format validation
"""

from io import BytesIO
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_rag_backend.api.routes.ingest import router, get_redis, get_postgres
from agentic_rag_backend.core.errors import app_error_handler, AppError


# Fixtures


@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL client."""
    mock = MagicMock()
    mock.create_document = AsyncMock(return_value=uuid4())
    mock.create_job = AsyncMock(return_value=uuid4())
    return mock


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = MagicMock()
    mock.publish_job = AsyncMock(return_value="12345-0")
    return mock


@pytest.fixture
def mock_settings():
    """Mock settings."""
    from dataclasses import dataclass

    @dataclass
    class MockSettings:
        redis_url: str = "redis://localhost:6379"
        database_url: str = "postgresql://test:test@localhost/test"
        max_upload_size_mb: int = 100
        temp_upload_dir: str = "/tmp/test_uploads"
        docling_table_mode: str = "accurate"

    return MockSettings()


@pytest.fixture
def app(mock_redis, mock_postgres):
    """Create a FastAPI test app with the ingest router and mocked dependencies."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1")
    test_app.add_exception_handler(AppError, app_error_handler)

    # Override dependencies
    async def override_get_redis():
        return mock_redis

    async def override_get_postgres():
        return mock_postgres

    test_app.dependency_overrides[get_redis] = override_get_redis
    test_app.dependency_overrides[get_postgres] = override_get_postgres

    return test_app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_pdf_content():
    """Valid PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture
def sample_tenant_id():
    """Generate a random tenant ID."""
    return str(uuid4())


# Tests


class TestUploadDocument:
    """Tests for POST /api/v1/ingest/document endpoint."""

    def test_upload_document_success(
        self,
        client: TestClient,
        sample_pdf_content: bytes,
        sample_tenant_id: str,
        mock_redis,
        mock_postgres,
        mock_settings,
        tmp_path,
    ):
        """Test successful PDF upload."""
        mock_settings.temp_upload_dir = str(tmp_path)

        with patch(
            "agentic_rag_backend.api.routes.ingest.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/api/v1/ingest/document",
                files={"file": ("test.pdf", sample_pdf_content, "application/pdf")},
                data={"tenant_id": sample_tenant_id},
            )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "job_id" in data["data"]
        assert data["data"]["status"] == "queued"
        assert data["data"]["filename"] == "test.pdf"
        assert data["data"]["file_size"] == len(sample_pdf_content)
        assert "meta" in data
        assert "requestId" in data["meta"]
        assert "timestamp" in data["meta"]

    def test_upload_document_invalid_content_type(
        self,
        client: TestClient,
        sample_tenant_id: str,
        mock_redis,
        mock_postgres,
        mock_settings,
    ):
        """Test upload with non-PDF content type."""
        with patch(
            "agentic_rag_backend.api.routes.ingest.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/api/v1/ingest/document",
                files={"file": ("test.txt", b"not a pdf", "text/plain")},
                data={"tenant_id": sample_tenant_id},
            )

        assert response.status_code == 400
        data = response.json()
        assert data["title"] == "Invalid Pdf"
        assert "PDF document" in data["detail"]

    def test_upload_document_invalid_pdf_magic_bytes(
        self,
        client: TestClient,
        sample_tenant_id: str,
        mock_redis,
        mock_postgres,
        mock_settings,
    ):
        """Test upload with wrong magic bytes but PDF content type."""
        with patch(
            "agentic_rag_backend.api.routes.ingest.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/api/v1/ingest/document",
                files={"file": ("test.pdf", b"NOT_A_PDF_FILE", "application/pdf")},
                data={"tenant_id": sample_tenant_id},
            )

        assert response.status_code == 400
        data = response.json()
        assert data["title"] == "Invalid Pdf"
        assert "valid PDF" in data["detail"]

    def test_upload_document_file_too_large(
        self,
        client: TestClient,
        sample_tenant_id: str,
        mock_redis,
        mock_postgres,
        mock_settings,
    ):
        """Test upload with file exceeding size limit."""
        # Set a very small limit for testing
        mock_settings.max_upload_size_mb = 1  # 1 MB

        # Create content larger than 1 MB
        large_content = b"%PDF-1.4\n" + b"x" * (2 * 1024 * 1024)  # ~2 MB

        with patch(
            "agentic_rag_backend.api.routes.ingest.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/api/v1/ingest/document",
                files={"file": ("large.pdf", large_content, "application/pdf")},
                data={"tenant_id": sample_tenant_id},
            )

        assert response.status_code == 413
        data = response.json()
        assert data["title"] == "File Too Large"
        assert "1MB" in data["detail"]

    def test_upload_document_missing_tenant_id(
        self,
        client: TestClient,
        sample_pdf_content: bytes,
    ):
        """Test upload without tenant_id."""
        response = client.post(
            "/api/v1/ingest/document",
            files={"file": ("test.pdf", sample_pdf_content, "application/pdf")},
        )

        assert response.status_code == 422  # Validation error

    def test_upload_document_invalid_tenant_id(
        self,
        client: TestClient,
        sample_pdf_content: bytes,
    ):
        """Test upload with invalid tenant_id format."""
        response = client.post(
            "/api/v1/ingest/document",
            files={"file": ("test.pdf", sample_pdf_content, "application/pdf")},
            data={"tenant_id": "not-a-uuid"},
        )

        assert response.status_code == 422  # Validation error

    def test_upload_document_missing_file(
        self,
        client: TestClient,
        sample_tenant_id: str,
    ):
        """Test upload without file."""
        response = client.post(
            "/api/v1/ingest/document",
            data={"tenant_id": sample_tenant_id},
        )

        assert response.status_code == 422  # Validation error

    def test_upload_document_creates_job_record(
        self,
        client: TestClient,
        sample_pdf_content: bytes,
        sample_tenant_id: str,
        mock_redis,
        mock_postgres,
        mock_settings,
        tmp_path,
    ):
        """Test that upload creates a job record in the database."""
        mock_settings.temp_upload_dir = str(tmp_path)

        with patch(
            "agentic_rag_backend.api.routes.ingest.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/api/v1/ingest/document",
                files={"file": ("test.pdf", sample_pdf_content, "application/pdf")},
                data={"tenant_id": sample_tenant_id},
            )

        assert response.status_code == 200

        # Verify create_document was called
        mock_postgres.create_document.assert_called_once()
        call_kwargs = mock_postgres.create_document.call_args.kwargs
        assert call_kwargs["source_type"] == "pdf"
        assert call_kwargs["filename"] == "test.pdf"
        assert call_kwargs["file_size"] == len(sample_pdf_content)

        # Verify create_job was called
        mock_postgres.create_job.assert_called_once()

    def test_upload_document_queues_parse_job(
        self,
        client: TestClient,
        sample_pdf_content: bytes,
        sample_tenant_id: str,
        mock_redis,
        mock_postgres,
        mock_settings,
        tmp_path,
    ):
        """Test that upload queues a job to the parse stream."""
        mock_settings.temp_upload_dir = str(tmp_path)

        with patch(
            "agentic_rag_backend.api.routes.ingest.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/api/v1/ingest/document",
                files={"file": ("test.pdf", sample_pdf_content, "application/pdf")},
                data={"tenant_id": sample_tenant_id},
            )

        assert response.status_code == 200

        # Verify publish_job was called with correct stream
        mock_redis.publish_job.assert_called_once()
        call_args = mock_redis.publish_job.call_args
        assert call_args.kwargs["stream"] == "parse.jobs"
        assert "job_id" in call_args.kwargs["job_data"]
        assert "tenant_id" in call_args.kwargs["job_data"]
        assert "document_id" in call_args.kwargs["job_data"]
        assert "file_path" in call_args.kwargs["job_data"]
        assert "filename" in call_args.kwargs["job_data"]

    def test_upload_document_computes_content_hash(
        self,
        client: TestClient,
        sample_pdf_content: bytes,
        sample_tenant_id: str,
        mock_redis,
        mock_postgres,
        mock_settings,
        tmp_path,
    ):
        """Test that content hash is computed for deduplication."""
        import hashlib

        mock_settings.temp_upload_dir = str(tmp_path)
        expected_hash = hashlib.sha256(sample_pdf_content).hexdigest()

        with patch(
            "agentic_rag_backend.api.routes.ingest.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/api/v1/ingest/document",
                files={"file": ("test.pdf", sample_pdf_content, "application/pdf")},
                data={"tenant_id": sample_tenant_id},
            )

        assert response.status_code == 200

        # Verify create_document was called with content_hash
        call_kwargs = mock_postgres.create_document.call_args.kwargs
        assert call_kwargs["content_hash"] == expected_hash

    def test_upload_document_saves_temp_file(
        self,
        client: TestClient,
        sample_pdf_content: bytes,
        sample_tenant_id: str,
        mock_redis,
        mock_postgres,
        mock_settings,
        tmp_path,
    ):
        """Test that uploaded file is saved to temp storage."""
        mock_settings.temp_upload_dir = str(tmp_path)

        with patch(
            "agentic_rag_backend.api.routes.ingest.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/api/v1/ingest/document",
                files={"file": ("test.pdf", sample_pdf_content, "application/pdf")},
                data={"tenant_id": sample_tenant_id},
            )

        assert response.status_code == 200

        # Verify file was saved to temp directory
        job_data = mock_redis.publish_job.call_args.kwargs["job_data"]
        from pathlib import Path

        temp_file = Path(job_data["file_path"])
        assert temp_file.exists()
        assert temp_file.read_bytes() == sample_pdf_content


class TestResponseFormat:
    """Tests for API response format compliance."""

    def test_success_response_format(
        self,
        client: TestClient,
        sample_pdf_content: bytes,
        sample_tenant_id: str,
        mock_redis,
        mock_postgres,
        mock_settings,
        tmp_path,
    ):
        """Test that success response follows standard format."""
        mock_settings.temp_upload_dir = str(tmp_path)

        with patch(
            "agentic_rag_backend.api.routes.ingest.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/api/v1/ingest/document",
                files={"file": ("test.pdf", sample_pdf_content, "application/pdf")},
                data={"tenant_id": sample_tenant_id},
            )

        data = response.json()

        # Verify standard response structure
        assert "data" in data
        assert "meta" in data
        assert "requestId" in data["meta"]
        assert "timestamp" in data["meta"]

        # Verify data fields
        assert "job_id" in data["data"]
        assert "status" in data["data"]
        assert "filename" in data["data"]
        assert "file_size" in data["data"]

    def test_error_response_rfc7807_format(
        self,
        client: TestClient,
        sample_tenant_id: str,
        mock_redis,
        mock_postgres,
        mock_settings,
    ):
        """Test that error response follows RFC 7807 Problem Details format."""
        with patch(
            "agentic_rag_backend.api.routes.ingest.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/api/v1/ingest/document",
                files={"file": ("test.txt", b"not pdf", "text/plain")},
                data={"tenant_id": sample_tenant_id},
            )

        data = response.json()

        # Verify RFC 7807 fields
        assert "type" in data
        assert "title" in data
        assert "status" in data
        assert "detail" in data
        assert "instance" in data

        # Verify correct values
        assert data["status"] == 400
        assert "/api/v1/ingest/document" in data["instance"]
