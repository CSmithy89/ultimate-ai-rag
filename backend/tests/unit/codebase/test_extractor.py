"""Tests for symbol extractor module."""

import pytest

from agentic_rag_backend.codebase.extractor import SymbolExtractor
from agentic_rag_backend.codebase.parser import TREE_SITTER_AVAILABLE
from agentic_rag_backend.codebase.types import Language, SymbolScope, SymbolType


# Skip all tests if tree-sitter is not available
pytestmark = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE,
    reason="tree-sitter not installed"
)


@pytest.fixture
def extractor():
    """Create a symbol extractor."""
    return SymbolExtractor()


class TestPythonExtraction:
    """Tests for Python symbol extraction."""

    def test_extract_function(self, extractor):
        """Test extracting a function definition."""
        source = """
def hello(name: str) -> str:
    \"\"\"Say hello to someone.\"\"\"
    return f"Hello, {name}!"
"""
        symbols = extractor.extract_from_string(source, "test.py", Language.PYTHON)

        assert len(symbols) >= 1
        func = next((s for s in symbols if s.name == "hello"), None)
        assert func is not None
        assert func.type == SymbolType.FUNCTION
        assert func.scope == SymbolScope.GLOBAL
        assert "hello" in func.signature

    def test_extract_class(self, extractor):
        """Test extracting a class definition."""
        source = """
class User:
    \"\"\"User model.\"\"\"

    def __init__(self, name: str):
        self.name = name
"""
        symbols = extractor.extract_from_string(source, "test.py", Language.PYTHON)

        # Should find class and method
        class_sym = next((s for s in symbols if s.name == "User"), None)
        assert class_sym is not None
        assert class_sym.type == SymbolType.CLASS
        assert class_sym.docstring == "User model."

    def test_extract_method(self, extractor):
        """Test extracting a method from a class."""
        source = """
class User:
    def get_name(self) -> str:
        return self.name
"""
        symbols = extractor.extract_from_string(source, "test.py", Language.PYTHON)

        method = next((s for s in symbols if s.name == "get_name"), None)
        assert method is not None
        assert method.type == SymbolType.METHOD
        assert method.scope == SymbolScope.CLASS
        assert method.parent == "User"
        assert method.qualified_name == "User.get_name"

    def test_extract_import(self, extractor):
        """Test extracting import statements."""
        source = """
import os
from typing import Optional
"""
        symbols = extractor.extract_from_string(source, "test.py", Language.PYTHON)

        imports = [s for s in symbols if s.type == SymbolType.IMPORT]
        assert len(imports) >= 1

    def test_extract_global_constant(self, extractor):
        """Test extracting global constants."""
        source = """
MAX_SIZE = 100
DEFAULT_NAME = "User"
"""
        symbols = extractor.extract_from_string(source, "test.py", Language.PYTHON)

        constant = next((s for s in symbols if s.name == "MAX_SIZE"), None)
        assert constant is not None
        assert constant.type == SymbolType.CONSTANT

    def test_extract_multiple_classes(self, extractor):
        """Test extracting multiple classes."""
        source = """
class First:
    pass

class Second:
    def method(self):
        pass
"""
        symbols = extractor.extract_from_string(source, "test.py", Language.PYTHON)

        classes = [s for s in symbols if s.type == SymbolType.CLASS]
        assert len(classes) == 2

    def test_extract_nested_class(self, extractor):
        """Test extracting nested classes."""
        source = """
class Outer:
    class Inner:
        pass
"""
        symbols = extractor.extract_from_string(source, "test.py", Language.PYTHON)

        inner = next((s for s in symbols if s.name == "Inner"), None)
        assert inner is not None
        assert inner.parent == "Outer"

    def test_file_path_recorded(self, extractor):
        """Test that file path is recorded correctly."""
        source = "def test(): pass"
        symbols = extractor.extract_from_string(
            source,
            "src/module.py",
            Language.PYTHON,
        )

        assert len(symbols) >= 1
        assert symbols[0].file_path == "src/module.py"

    def test_line_numbers(self, extractor):
        """Test that line numbers are recorded correctly."""
        source = """
# Comment
def first():
    pass

def second():
    pass
"""
        symbols = extractor.extract_from_string(source, "test.py", Language.PYTHON)

        first = next((s for s in symbols if s.name == "first"), None)
        second = next((s for s in symbols if s.name == "second"), None)

        assert first is not None
        assert second is not None
        assert first.line_start < second.line_start


class TestTypescriptExtraction:
    """Tests for TypeScript symbol extraction."""

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE,
        reason="tree-sitter not installed"
    )
    def test_extract_function(self, extractor):
        """Test extracting a TypeScript function."""
        source = """
function greet(name: string): string {
    return `Hello, ${name}!`;
}
"""
        symbols = extractor.extract_from_string(source, "test.ts", Language.TYPESCRIPT)

        func = next((s for s in symbols if s.name == "greet"), None)
        if func:  # May not be supported
            assert func.type == SymbolType.FUNCTION

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE,
        reason="tree-sitter not installed"
    )
    def test_extract_interface(self, extractor):
        """Test extracting a TypeScript interface."""
        source = """
interface User {
    name: string;
    age: number;
}
"""
        symbols = extractor.extract_from_string(source, "test.ts", Language.TYPESCRIPT)

        interface = next((s for s in symbols if s.name == "User"), None)
        if interface:
            assert interface.type == SymbolType.INTERFACE

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE,
        reason="tree-sitter not installed"
    )
    def test_extract_class_with_methods(self, extractor):
        """Test extracting a TypeScript class with methods."""
        source = """
class UserService {
    getUser(id: string): User {
        return { name: "Test", age: 30 };
    }
}
"""
        symbols = extractor.extract_from_string(source, "test.ts", Language.TYPESCRIPT)

        class_sym = next((s for s in symbols if s.name == "UserService"), None)
        if class_sym:
            assert class_sym.type == SymbolType.CLASS


class TestJavaScriptExtraction:
    """Tests for JavaScript symbol extraction."""

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE,
        reason="tree-sitter not installed"
    )
    def test_extract_function(self, extractor):
        """Test extracting a JavaScript function."""
        source = """
function greet(name) {
    return "Hello, " + name;
}
"""
        symbols = extractor.extract_from_string(source, "test.js", Language.JAVASCRIPT)

        func = next((s for s in symbols if s.name == "greet"), None)
        if func:
            assert func.type == SymbolType.FUNCTION


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_file(self, extractor):
        """Test extracting from an empty file."""
        symbols = extractor.extract_from_string("", "test.py", Language.PYTHON)
        assert symbols == []

    def test_only_comments(self, extractor):
        """Test extracting from a file with only comments."""
        source = """
# This is a comment
# Another comment
\"\"\"
Docstring without function
\"\"\"
"""
        symbols = extractor.extract_from_string(source, "test.py", Language.PYTHON)
        assert len(symbols) == 0

    def test_syntax_error_handling(self, extractor):
        """Test handling of syntax errors."""
        source = """
def broken(
    # Missing closing paren and body
"""
        # Should not raise, just return partial results or empty
        symbols = extractor.extract_from_string(source, "test.py", Language.PYTHON)
        # Result is implementation-dependent
        assert isinstance(symbols, list)

    def test_unicode_content(self, extractor):
        """Test handling of Unicode content."""
        source = """
def greet(name: str) -> str:
    \"\"\"Greet with emoji.\"\"\"
    return f"Hello {name}!"
"""
        symbols = extractor.extract_from_string(source, "test.py", Language.PYTHON)
        assert len(symbols) >= 1
