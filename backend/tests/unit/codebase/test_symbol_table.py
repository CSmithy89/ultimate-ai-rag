"""Tests for symbol_table module."""

import pytest

from agentic_rag_backend.codebase.symbol_table import SymbolTable
from agentic_rag_backend.codebase.types import CodeSymbol, SymbolScope, SymbolType


@pytest.fixture
def empty_table():
    """Create an empty symbol table."""
    return SymbolTable(tenant_id="test-tenant", repo_path="/test/repo")


@pytest.fixture
def sample_symbol():
    """Create a sample code symbol."""
    return CodeSymbol(
        name="sample_function",
        type=SymbolType.FUNCTION,
        scope=SymbolScope.GLOBAL,
        file_path="src/module.py",
        line_start=10,
        line_end=20,
        signature="def sample_function(arg: str) -> bool",
    )


@pytest.fixture
def populated_table(empty_table):
    """Create a symbol table with sample symbols."""
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
            name="User",
            type=SymbolType.CLASS,
            scope=SymbolScope.GLOBAL,
            file_path="src/users.py",
            line_start=35,
            line_end=100,
        ),
        CodeSymbol(
            name="save",
            type=SymbolType.METHOD,
            scope=SymbolScope.CLASS,
            file_path="src/users.py",
            line_start=50,
            line_end=60,
            parent="User",
            qualified_name="User.save",
        ),
        CodeSymbol(
            name="get_document",
            type=SymbolType.FUNCTION,
            scope=SymbolScope.GLOBAL,
            file_path="src/documents.py",
            line_start=5,
            line_end=25,
        ),
    ]
    for sym in symbols:
        empty_table.add(sym)
    return empty_table


class TestSymbolTableBasic:
    """Tests for basic SymbolTable operations."""

    def test_create_empty_table(self, empty_table):
        """Test creating an empty symbol table."""
        assert empty_table.tenant_id == "test-tenant"
        assert empty_table.repo_path == "/test/repo"
        assert empty_table.symbol_count() == 0
        assert empty_table.file_count() == 0

    def test_add_symbol(self, empty_table, sample_symbol):
        """Test adding a symbol to the table."""
        empty_table.add(sample_symbol)
        assert empty_table.symbol_count() == 1
        assert empty_table.contains("sample_function")

    def test_get_symbol(self, empty_table, sample_symbol):
        """Test retrieving a symbol by name."""
        empty_table.add(sample_symbol)
        symbols = empty_table.get("sample_function")
        assert len(symbols) == 1
        assert symbols[0].name == "sample_function"

    def test_get_nonexistent_symbol(self, empty_table):
        """Test getting a symbol that doesn't exist."""
        symbols = empty_table.get("nonexistent")
        assert symbols == []

    def test_contains(self, populated_table):
        """Test checking if a symbol exists."""
        assert populated_table.contains("create_user")
        assert populated_table.contains("User")
        assert not populated_table.contains("nonexistent")

    def test_lookup_alias(self, populated_table):
        """Test that lookup is an alias for get."""
        result = populated_table.lookup("create_user")
        assert len(result) == 1
        assert result[0].name == "create_user"


class TestSymbolTableQualifiedLookup:
    """Tests for qualified name lookup."""

    def test_lookup_qualified(self, populated_table):
        """Test looking up a symbol by qualified name."""
        symbol = populated_table.lookup_qualified("User.save")
        assert symbol is not None
        assert symbol.name == "save"
        assert symbol.parent == "User"

    def test_lookup_qualified_not_found(self, populated_table):
        """Test qualified lookup for nonexistent symbol."""
        symbol = populated_table.lookup_qualified("User.nonexistent")
        assert symbol is None


class TestSymbolTableSimilarSearch:
    """Tests for similar symbol search."""

    def test_find_similar(self, populated_table):
        """Test finding similar symbol names."""
        similar = populated_table.find_similar("creat_user")  # Typo
        assert "create_user" in similar

    def test_find_similar_no_matches(self, populated_table):
        """Test when no similar symbols exist."""
        similar = populated_table.find_similar("xyz123abc")
        assert len(similar) == 0


class TestSymbolTableFileOperations:
    """Tests for file-related operations."""

    def test_get_symbols_in_file(self, populated_table):
        """Test getting all symbols in a file."""
        symbols = populated_table.get_symbols_in_file("src/users.py")
        assert len(symbols) == 3
        names = {s.name for s in symbols}
        assert "create_user" in names
        assert "User" in names
        assert "save" in names

    def test_file_exists(self, populated_table):
        """Test checking if a file is known."""
        assert populated_table.file_exists("src/users.py")
        assert populated_table.file_exists("src/documents.py")
        assert not populated_table.file_exists("src/unknown.py")

    def test_add_known_file(self, empty_table):
        """Test adding a known file path."""
        empty_table.add_known_file("config/settings.yaml")
        assert empty_table.file_exists("config/settings.yaml")

    def test_get_all_known_files(self, populated_table):
        """Test getting all known files."""
        files = populated_table.get_all_known_files()
        assert "src/users.py" in files
        assert "src/documents.py" in files

    def test_remove_file(self, populated_table):
        """Test removing a file and its symbols."""
        populated_table.remove_file("src/users.py")
        assert not populated_table.file_exists("src/users.py")
        assert populated_table.lookup("create_user") == []
        assert populated_table.lookup_qualified("User.save") is None


class TestSymbolTableByType:
    """Tests for getting symbols by type."""

    def test_get_symbols_by_type_function(self, populated_table):
        """Test getting all functions."""
        functions = populated_table.get_symbols_by_type(SymbolType.FUNCTION)
        assert len(functions) == 2
        names = {f.name for f in functions}
        assert "create_user" in names
        assert "get_document" in names

    def test_get_symbols_by_type_class(self, populated_table):
        """Test getting all classes."""
        classes = populated_table.get_symbols_by_type(SymbolType.CLASS)
        assert len(classes) == 1
        assert classes[0].name == "User"

    def test_get_symbols_by_type_method(self, populated_table):
        """Test getting all methods."""
        methods = populated_table.get_symbols_by_type(SymbolType.METHOD)
        assert len(methods) == 1
        assert methods[0].name == "save"


class TestSymbolTableSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict(self, populated_table):
        """Test serializing to dictionary."""
        data = populated_table.to_dict()
        assert data["tenant_id"] == "test-tenant"
        assert data["repo_path"] == "/test/repo"
        assert len(data["symbols"]) == 4
        assert "src/users.py" in data["known_files"]

    def test_from_dict(self, populated_table):
        """Test deserializing from dictionary."""
        data = populated_table.to_dict()
        restored = SymbolTable.from_dict(data)

        assert restored.tenant_id == populated_table.tenant_id
        assert restored.repo_path == populated_table.repo_path
        assert restored.symbol_count() == populated_table.symbol_count()
        assert restored.contains("create_user")
        assert restored.contains("User")

    def test_to_json(self, populated_table):
        """Test JSON serialization."""
        json_str = populated_table.to_json()
        assert isinstance(json_str, str)
        assert "create_user" in json_str
        assert "test-tenant" in json_str

    def test_from_json(self, populated_table):
        """Test JSON deserialization."""
        json_str = populated_table.to_json()
        restored = SymbolTable.from_json(json_str)

        assert restored.symbol_count() == populated_table.symbol_count()
        assert restored.contains("create_user")

    def test_declared_packages_roundtrip(self, populated_table):
        """Test declared package persistence in serialization."""
        populated_table.set_declared_packages({"fastapi", "requests"})
        data = populated_table.to_dict()

        restored = SymbolTable.from_dict(data)
        assert restored.get_declared_packages() == {"fastapi", "requests"}

    def test_get_cache_key(self, populated_table):
        """Test generating cache key."""
        key = populated_table.get_cache_key()
        assert key.startswith("codebase:test-tenant:symbol_table:")
        assert len(key) > len("codebase:test-tenant:symbol_table:")


class TestSymbolTableClear:
    """Tests for clearing the symbol table."""

    def test_clear(self, populated_table):
        """Test clearing all symbols."""
        assert populated_table.symbol_count() > 0
        populated_table.clear()
        assert populated_table.symbol_count() == 0
        assert populated_table.file_count() == 0
        assert not populated_table.contains("create_user")


class TestSymbolTableMultiTenancy:
    """Tests for multi-tenancy isolation."""

    def test_different_tenants(self):
        """Test that different tenants have separate tables."""
        table1 = SymbolTable(tenant_id="tenant-1", repo_path="/repo1")
        table2 = SymbolTable(tenant_id="tenant-2", repo_path="/repo2")

        symbol = CodeSymbol(
            name="shared_function",
            type=SymbolType.FUNCTION,
            scope=SymbolScope.GLOBAL,
            file_path="src/shared.py",
            line_start=1,
            line_end=10,
        )

        table1.add(symbol)

        assert table1.contains("shared_function")
        assert not table2.contains("shared_function")

        assert table1.get_cache_key() != table2.get_cache_key()
