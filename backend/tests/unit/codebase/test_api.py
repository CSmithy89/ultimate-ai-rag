"""Tests for codebase API endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4


# Mock the codebase module to avoid tree-sitter dependency in tests
@pytest.fixture(autouse=True)
def mock_codebase_imports():
    """Mock codebase imports to avoid tree-sitter dependency."""
    with patch.dict("sys.modules", {
        "agentic_rag_backend.codebase": MagicMock(),
        "agentic_rag_backend.codebase.detector": MagicMock(),
        "agentic_rag_backend.codebase.symbol_table": MagicMock(),
    }):
        yield


class TestValidateResponseEndpoint:
    """Tests for POST /api/v1/codebase/validate-response endpoint."""

    def test_validate_response_request_model(self):
        """Test the request model validation."""
        from agentic_rag_backend.api.routes.codebase import ValidateResponseRequest

        # Valid request
        request = ValidateResponseRequest(
            response_text="Test response with create_user() function.",
            tenant_id=uuid4(),
        )
        assert request.response_text == "Test response with create_user() function."

    def test_validate_response_request_requires_tenant_id(self):
        """Test that tenant_id is required."""
        from agentic_rag_backend.api.routes.codebase import ValidateResponseRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ValidateResponseRequest(
                response_text="Test",
                # Missing tenant_id
            )

    def test_validate_response_request_text_length_limit(self):
        """Test response_text length validation."""
        from agentic_rag_backend.api.routes.codebase import ValidateResponseRequest
        from pydantic import ValidationError

        # Empty text should fail
        with pytest.raises(ValidationError):
            ValidateResponseRequest(
                response_text="",
                tenant_id=uuid4(),
            )


class TestIndexRepositoryEndpoint:
    """Tests for POST /api/v1/codebase/index-repository endpoint."""

    def test_index_repository_request_model(self):
        """Test the request model validation."""
        from agentic_rag_backend.api.routes.codebase import IndexRepositoryRequest

        request = IndexRepositoryRequest(
            repo_path="/path/to/repo",
            tenant_id=uuid4(),
        )
        assert request.repo_path == "/path/to/repo"
        assert request.cache_ttl_seconds == 3600  # Default

    def test_index_repository_with_ignore_patterns(self):
        """Test request with custom ignore patterns."""
        from agentic_rag_backend.api.routes.codebase import IndexRepositoryRequest

        request = IndexRepositoryRequest(
            repo_path="/path/to/repo",
            tenant_id=uuid4(),
            ignore_patterns=["*.test.py", "fixtures/"],
        )
        assert len(request.ignore_patterns) == 2


class TestResponseModels:
    """Tests for response models."""

    def test_validation_result_response(self):
        """Test ValidationResultResponse model."""
        from agentic_rag_backend.api.routes.codebase import ValidationResultResponse

        result = ValidationResultResponse(
            symbol_name="test_function",
            is_valid=True,
            confidence=1.0,
            reason="Found in codebase",
            suggestions=[],
        )
        assert result.symbol_name == "test_function"
        assert result.is_valid is True

    def test_hallucination_report_response(self):
        """Test HallucinationReportResponse model."""
        from agentic_rag_backend.api.routes.codebase import (
            HallucinationReportResponse,
        )

        report = HallucinationReportResponse(
            total_symbols_checked=10,
            valid_symbols=8,
            invalid_symbols=2,
            uncertain_symbols=0,
            validation_results=[],
            files_checked=["src/main.py"],
            processing_time_ms=100,
            confidence_score=0.8,
            should_block=False,
        )
        assert report.total_symbols_checked == 10
        assert report.confidence_score == 0.8
        assert report.should_block is False

    def test_index_repository_response(self):
        """Test IndexRepositoryResponse model."""
        from agentic_rag_backend.api.routes.codebase import IndexRepositoryResponse

        response = IndexRepositoryResponse(
            symbol_count=150,
            file_count=20,
            cached=True,
            cache_key="codebase:tenant:key",
        )
        assert response.symbol_count == 150
        assert response.cached is True

    def test_symbol_table_stats_response(self):
        """Test SymbolTableStatsResponse model."""
        from agentic_rag_backend.api.routes.codebase import SymbolTableStatsResponse

        stats = SymbolTableStatsResponse(
            tenant_id="test-tenant",
            repo_path="/test/repo",
            symbol_count=100,
            file_count=15,
            symbols_by_type={"function": 50, "class": 20, "method": 30},
        )
        assert stats.symbol_count == 100
        assert stats.symbols_by_type["function"] == 50


class TestSuccessResponseWrapper:
    """Tests for success response wrapper."""

    def test_success_response_format(self):
        """Test that success_response creates correct format."""
        from agentic_rag_backend.api.routes.codebase import success_response

        data = {"key": "value"}
        response = success_response(data)

        assert "data" in response
        assert "meta" in response
        assert response["data"] == data
        assert "requestId" in response["meta"]
        assert "timestamp" in response["meta"]

    def test_success_response_timestamp_format(self):
        """Test that timestamp is in correct format."""
        from agentic_rag_backend.api.routes.codebase import success_response

        response = success_response({})

        # Timestamp should end with Z (UTC)
        assert response["meta"]["timestamp"].endswith("Z")
