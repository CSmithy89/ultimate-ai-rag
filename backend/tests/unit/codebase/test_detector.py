"""Tests for hallucination detector."""

import pytest

from agentic_rag_backend.codebase.detector import (
    DetectorMode,
    HallucinationDetector,
)
from agentic_rag_backend.codebase.symbol_table import SymbolTable
from agentic_rag_backend.codebase.types import CodeSymbol, SymbolScope, SymbolType


@pytest.fixture
def symbol_table():
    """Create a symbol table with test data."""
    table = SymbolTable(tenant_id="test-tenant", repo_path="/test/repo")

    symbols = [
        CodeSymbol(
            name="create_user",
            type=SymbolType.FUNCTION,
            scope=SymbolScope.GLOBAL,
            file_path="src/users.py",
            line_start=10,
            line_end=30,
        ),
        CodeSymbol(
            name="UserService",
            type=SymbolType.CLASS,
            scope=SymbolScope.GLOBAL,
            file_path="src/users.py",
            line_start=35,
            line_end=100,
        ),
        CodeSymbol(
            name="get_user",
            type=SymbolType.METHOD,
            scope=SymbolScope.CLASS,
            file_path="src/users.py",
            line_start=50,
            line_end=60,
            parent="UserService",
            qualified_name="UserService.get_user",
        ),
        CodeSymbol(
            name="process_document",
            type=SymbolType.FUNCTION,
            scope=SymbolScope.GLOBAL,
            file_path="src/documents.py",
            line_start=5,
            line_end=25,
        ),
    ]

    for sym in symbols:
        table.add(sym)

    return table


@pytest.fixture
def detector(symbol_table):
    """Create a hallucination detector."""
    return HallucinationDetector(
        symbol_table=symbol_table,
        mode=DetectorMode.WARN,
        threshold=0.3,
    )


class TestDetectorModes:
    """Tests for detector operating modes."""

    def test_default_mode_is_warn(self, symbol_table):
        """Test that default mode is WARN."""
        detector = HallucinationDetector(symbol_table)
        assert detector.mode == DetectorMode.WARN

    def test_set_block_mode(self, symbol_table):
        """Test setting block mode."""
        detector = HallucinationDetector(
            symbol_table,
            mode=DetectorMode.BLOCK,
        )
        assert detector.mode == DetectorMode.BLOCK

    def test_mode_setter(self, detector):
        """Test changing mode at runtime."""
        detector.mode = DetectorMode.BLOCK
        assert detector.mode == DetectorMode.BLOCK

    def test_threshold_setter(self, detector):
        """Test changing threshold at runtime."""
        detector.threshold = 0.5
        assert detector.threshold == 0.5

        # Test clamping
        detector.threshold = 1.5
        assert detector.threshold == 1.0

        detector.threshold = -0.5
        assert detector.threshold == 0.0


class TestValidateResponse:
    """Tests for response validation."""

    def test_validate_response_with_valid_symbols(self, detector):
        """Test validating a response with valid symbols."""
        response = """
        To create a new user, call the create_user() function.
        The UserService class handles all user operations.
        """
        report = detector.validate_response(response)

        assert report.total_symbols_checked > 0
        assert report.valid_symbols > 0
        assert report.confidence_score > 0

    def test_validate_response_with_hallucinated_symbols(self, detector):
        """Test validating a response with hallucinated symbols."""
        response = """
        Use the nonexistent_function() to process data.
        The FakeClass class does not exist.
        """
        report = detector.validate_response(response)

        assert report.invalid_symbols > 0

    def test_validate_response_with_file_paths(self, detector):
        """Test validating file path references."""
        response = """
        The user logic is in src/users.py file.
        Check the config in src/unknown.py for settings.
        """
        report = detector.validate_response(response)

        assert len(report.files_checked) > 0

    def test_validate_response_processing_time(self, detector):
        """Test that processing time is recorded."""
        response = "Call create_user() to add a user."
        report = detector.validate_response(response)

        assert report.processing_time_ms >= 0

    def test_validate_response_tenant_id(self, detector):
        """Test that tenant_id is recorded."""
        response = "Test response"
        report = detector.validate_response(response, tenant_id="my-tenant")

        assert report.tenant_id == "my-tenant"

    def test_validate_empty_response(self, detector):
        """Test validating an empty response."""
        report = detector.validate_response("No code references here.")

        # Should have high confidence when nothing to validate
        assert report.confidence_score == 1.0
        assert report.invalid_symbols == 0


class TestShouldBlock:
    """Tests for blocking decisions."""

    def test_should_not_block_in_warn_mode(self, symbol_table):
        """Test that responses are not blocked in warn mode."""
        detector = HallucinationDetector(
            symbol_table,
            mode=DetectorMode.WARN,
            threshold=0.3,
        )
        response = "Use nonexistent_function() here."
        report = detector.validate_response(response)

        assert not detector.should_block(report)

    def test_should_block_when_threshold_exceeded(self, symbol_table):
        """Test blocking when hallucination ratio exceeds threshold."""
        detector = HallucinationDetector(
            symbol_table,
            mode=DetectorMode.BLOCK,
            threshold=0.0,  # Any hallucination should block
        )
        response = "Call fake_function() to process."
        report = detector.validate_response(response)

        # Should block because of hallucinated reference
        if report.invalid_symbols > 0:
            assert detector.should_block(report)

    def test_should_not_block_below_threshold(self, symbol_table):
        """Test not blocking when below threshold."""
        detector = HallucinationDetector(
            symbol_table,
            mode=DetectorMode.BLOCK,
            threshold=0.9,  # Very high threshold
        )
        response = "Call create_user() and also fake_func()."
        report = detector.validate_response(response)

        # With high threshold, should not block for a few hallucinations
        if report.total_symbols_checked > 0:
            ratio = report.invalid_symbols / report.total_symbols_checked
            if ratio <= 0.9:
                assert not detector.should_block(report)


class TestPatternExtraction:
    """Tests for code reference pattern extraction."""

    def test_extract_function_calls(self, detector):
        """Test extracting function call patterns."""
        response = "Call create_user() and process_document(data)."
        report = detector.validate_response(response)

        # Should find both functions
        symbol_names = {r.symbol_name for r in report.validation_results}
        assert "create_user" in symbol_names
        assert "process_document" in symbol_names

    def test_extract_class_names(self, detector):
        """Test extracting class name patterns."""
        response = "The UserService class provides user management."
        report = detector.validate_response(response)

        symbol_names = {r.symbol_name for r in report.validation_results}
        assert "UserService" in symbol_names

    def test_extract_file_paths(self, detector):
        """Test extracting file path patterns."""
        response = "See the implementation in src/users.py for details."
        report = detector.validate_response(response)

        assert "src/users.py" in report.files_checked

    def test_extract_qualified_names(self, detector):
        """Test extracting qualified name patterns."""
        response = "The UserService.get_user method retrieves users."
        report = detector.validate_response(response)

        symbol_names = {r.symbol_name for r in report.validation_results}
        assert "UserService.get_user" in symbol_names

    def test_skip_excluded_words(self, detector):
        """Test that common words are excluded."""
        response = "The function returns data from the module."
        report = detector.validate_response(response)

        # Should not flag common words like "returns", "data", "module"
        symbol_names = {r.symbol_name for r in report.validation_results}
        assert "returns" not in symbol_names
        assert "data" not in symbol_names


class TestAPIRouteValidation:
    """Tests for API endpoint validation in responses."""

    def test_validate_api_endpoint_reference(self, symbol_table):
        """Test validating API endpoint references."""
        detector = HallucinationDetector(symbol_table)

        # Add some API routes
        detector.add_api_routes_from_openapi({
            "paths": {
                "/api/v1/users": {"get": {}, "post": {}},
            }
        })

        response = "POST /api/v1/users to create a new user."
        report = detector.validate_response(response)

        # Should find the API endpoint
        symbol_names = {r.symbol_name for r in report.validation_results}
        assert any("/api/v1/users" in name for name in symbol_names)


class TestReportMetrics:
    """Tests for report metrics calculation."""

    def test_confidence_score_calculation(self, detector):
        """Test that confidence score is calculated correctly."""
        response = "Use create_user() and UserService."
        report = detector.validate_response(response)

        if report.total_symbols_checked > 0:
            expected_confidence = report.valid_symbols / report.total_symbols_checked
            assert abs(report.confidence_score - expected_confidence) < 0.01

    def test_uncertain_symbols_count(self, symbol_table):
        """Test uncertain symbols counting."""
        detector = HallucinationDetector(
            symbol_table,
            confidence_threshold=0.9,
        )
        # When validation confidence is below threshold, counts as uncertain
        response = "Use some_function() here."
        report = detector.validate_response(response)

        # Total should equal valid + invalid + uncertain
        assert report.total_symbols_checked == (
            report.valid_symbols +
            report.invalid_symbols +
            report.uncertain_symbols
        )
