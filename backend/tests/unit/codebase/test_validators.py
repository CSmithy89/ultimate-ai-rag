"""Tests for codebase validators."""

import pytest

from agentic_rag_backend.codebase.symbol_table import SymbolTable
from agentic_rag_backend.codebase.types import CodeSymbol, SymbolScope, SymbolType
from agentic_rag_backend.codebase.validators.symbol_validator import SymbolValidator
from agentic_rag_backend.codebase.validators.path_validator import FilePathValidator
from agentic_rag_backend.codebase.validators.api_validator import APIEndpointValidator
from agentic_rag_backend.codebase.validators.import_validator import ImportValidator


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
    ]

    for sym in symbols:
        table.add(sym)

    table.add_known_file("src/config.yaml")
    table.add_known_file("src/__init__.py")

    return table


class TestSymbolValidator:
    """Tests for SymbolValidator."""

    def test_validate_existing_function(self, symbol_table):
        """Test validating an existing function."""
        validator = SymbolValidator(symbol_table)
        result = validator.validate("create_user")

        assert result.is_valid is True
        assert result.confidence == 1.0
        assert "found" in result.reason.lower()

    def test_validate_existing_class(self, symbol_table):
        """Test validating an existing class."""
        validator = SymbolValidator(symbol_table)
        result = validator.validate(
            "UserService",
            context={"expected_type": "class"},
        )

        assert result.is_valid is True
        assert result.confidence == 1.0

    def test_validate_nonexistent_symbol(self, symbol_table):
        """Test validating a nonexistent symbol."""
        validator = SymbolValidator(symbol_table)
        result = validator.validate("nonexistent_function")

        assert result.is_valid is False
        assert result.confidence < 1.0

    def test_validate_with_suggestions(self, symbol_table):
        """Test that suggestions are provided for similar symbols."""
        validator = SymbolValidator(symbol_table)
        result = validator.validate("creat_user")  # Typo

        assert result.is_valid is False
        assert "create_user" in result.suggestions

    def test_validate_builtin_symbol(self, symbol_table):
        """Test that built-in symbols are recognized."""
        validator = SymbolValidator(symbol_table)

        for builtin in ["print", "len", "str", "list"]:
            result = validator.validate(builtin)
            assert result.is_valid is True
            assert "built-in" in result.reason.lower()

    def test_validate_wrong_type(self, symbol_table):
        """Test validating a symbol with wrong expected type."""
        validator = SymbolValidator(symbol_table)
        result = validator.validate(
            "create_user",
            context={"expected_type": "class"},
        )

        # Symbol exists but as function, not class
        assert result.is_valid is False
        assert "exists but as" in result.reason.lower()

    def test_validate_qualified_name(self, symbol_table):
        """Test validating a qualified name."""
        validator = SymbolValidator(symbol_table)
        result = validator.validate_qualified("UserService.get_user")

        assert result.is_valid is True
        assert result.symbol_type == SymbolType.METHOD


class TestFilePathValidator:
    """Tests for FilePathValidator."""

    def test_validate_known_file(self, symbol_table):
        """Test validating a known file path."""
        validator = FilePathValidator(symbol_table)
        result = validator.validate("src/users.py")

        assert result.is_valid is True
        assert result.confidence == 1.0

    def test_validate_nonexistent_file(self, symbol_table):
        """Test validating a nonexistent file."""
        validator = FilePathValidator(symbol_table)
        result = validator.validate("src/unknown_file.py")

        assert result.is_valid is False

    def test_validate_with_path_suggestions(self, symbol_table):
        """Test that suggestions are provided for similar paths."""
        validator = FilePathValidator(symbol_table)
        result = validator.validate("src/user.py")  # Missing 's'

        assert result.is_valid is False
        # Should suggest src/users.py

    def test_validate_normalized_path(self, symbol_table):
        """Test that paths are normalized."""
        validator = FilePathValidator(symbol_table)

        # These should all normalize to src/config.yaml
        result = validator.validate("./src/config.yaml")
        assert result.is_valid is True

    def test_validate_directory(self, symbol_table):
        """Test validating a directory path."""
        validator = FilePathValidator(symbol_table)
        result = validator.validate_directory("src")

        # src directory contains known files
        assert result.is_valid is True


class TestAPIEndpointValidator:
    """Tests for APIEndpointValidator."""

    @pytest.fixture
    def api_validator(self, symbol_table):
        """Create an API validator with sample routes."""
        validator = APIEndpointValidator(symbol_table)
        validator.add_route("/api/v1/users", ["GET", "POST"])
        validator.add_route("/api/v1/users/{id}", ["GET", "PUT", "DELETE"])
        validator.add_route("/api/v1/documents", ["GET", "POST"])
        return validator

    def test_validate_existing_endpoint(self, api_validator):
        """Test validating an existing API endpoint."""
        result = api_validator.validate("/api/v1/users")

        assert result.is_valid is True
        assert result.confidence == 1.0

    def test_validate_endpoint_with_method(self, api_validator):
        """Test validating an endpoint with HTTP method."""
        result = api_validator.validate(
            "/api/v1/users",
            context={"method": "GET"},
        )

        assert result.is_valid is True

    def test_validate_wrong_method(self, api_validator):
        """Test validating an endpoint with wrong method."""
        result = api_validator.validate(
            "/api/v1/users",
            context={"method": "DELETE"},
        )

        assert result.is_valid is False
        assert "not allowed" in result.reason.lower()
        # Should suggest valid methods
        assert "GET" in " ".join(result.suggestions)

    def test_validate_parameterized_endpoint(self, api_validator):
        """Test validating a parameterized endpoint."""
        result = api_validator.validate("/api/v1/users/123")

        assert result.is_valid is True
        assert result.confidence >= 0.9

    def test_validate_nonexistent_endpoint(self, api_validator):
        """Test validating a nonexistent endpoint."""
        result = api_validator.validate("/api/v1/unknown")

        assert result.is_valid is False

    def test_validate_with_endpoint_suggestions(self, api_validator):
        """Test that suggestions are provided for similar endpoints."""
        result = api_validator.validate("/api/v1/user")  # Missing 's'

        assert result.is_valid is False
        assert len(result.suggestions) > 0

    def test_get_all_routes(self, api_validator):
        """Test getting all registered routes."""
        routes = api_validator.get_all_routes()

        assert "/api/v1/users" in routes
        assert "GET" in routes["/api/v1/users"]
        assert "POST" in routes["/api/v1/users"]

    def test_load_from_openapi(self, symbol_table):
        """Test loading routes from OpenAPI spec."""
        openapi_spec = {
            "paths": {
                "/pets": {"get": {}, "post": {}},
                "/pets/{petId}": {"get": {}, "delete": {}},
            }
        }
        validator = APIEndpointValidator(symbol_table, openapi_spec=openapi_spec)

        routes = validator.get_all_routes()
        assert "/pets" in routes
        assert "GET" in routes["/pets"]


class TestImportValidator:
    """Tests for ImportValidator."""

    @pytest.fixture
    def import_validator(self, symbol_table):
        """Create an import validator."""
        validator = ImportValidator(symbol_table)
        validator.add_installed_package("fastapi")
        validator.add_installed_package("pydantic")
        validator.add_installed_package("structlog")
        return validator

    def test_validate_stdlib_import(self, import_validator):
        """Test validating a stdlib import."""
        result = import_validator.validate("os")

        assert result.is_valid is True
        assert "standard library" in result.reason.lower()

    def test_validate_stdlib_submodule(self, import_validator):
        """Test validating a stdlib submodule."""
        result = import_validator.validate("os.path")

        assert result.is_valid is True

    def test_validate_installed_package(self, import_validator):
        """Test validating an installed package."""
        result = import_validator.validate("fastapi")

        assert result.is_valid is True
        assert "installed" in result.reason.lower()

    def test_validate_import_statement(self, import_validator):
        """Test validating a full import statement."""
        result = import_validator.validate("from os import path")

        assert result.is_valid is True

    def test_validate_from_import(self, import_validator):
        """Test validating a from-import statement."""
        result = import_validator.validate("from typing import Optional")

        assert result.is_valid is True

    def test_validate_nonexistent_module(self, import_validator):
        """Test validating a nonexistent module."""
        result = import_validator.validate("nonexistent_module")

        assert result.is_valid is False
        assert "not found" in result.reason.lower()

    def test_load_from_requirements(self, import_validator):
        """Test loading packages from requirements.txt content."""
        requirements = """
fastapi>=0.100.0
pydantic>=2.0
requests>=2.28.0
# comment line
numpy==1.24.0
        """
        count = import_validator.load_from_requirements(requirements)

        assert count >= 4
        result = import_validator.validate("requests")
        assert result.is_valid is True

    def test_validate_local_relative_import(self):
        """Test validating a local relative import path."""
        table = SymbolTable(tenant_id="test-tenant", repo_path="/test/repo")
        table.add_known_file("src/utils/helpers.ts")

        validator = ImportValidator(table)
        result = validator.validate("./src/utils/helpers")

        assert result.is_valid is True
