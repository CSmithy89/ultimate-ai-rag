"""Tests for codebase types module."""

import pytest

from agentic_rag_backend.codebase.types import (
    CodeSymbol,
    HallucinationReport,
    Language,
    SymbolScope,
    SymbolType,
    ValidationResult,
    get_language_for_file,
    LANGUAGE_EXTENSIONS,
)


class TestSymbolType:
    """Tests for SymbolType enum."""

    def test_symbol_type_values(self):
        """Test that all expected symbol types exist."""
        assert SymbolType.FUNCTION.value == "function"
        assert SymbolType.CLASS.value == "class"
        assert SymbolType.METHOD.value == "method"
        assert SymbolType.VARIABLE.value == "variable"
        assert SymbolType.IMPORT.value == "import"
        assert SymbolType.CONSTANT.value == "constant"

    def test_symbol_type_is_string_enum(self):
        """Test that SymbolType is a string enum."""
        assert isinstance(SymbolType.FUNCTION, str)
        assert SymbolType.FUNCTION == "function"


class TestSymbolScope:
    """Tests for SymbolScope enum."""

    def test_symbol_scope_values(self):
        """Test that all expected scopes exist."""
        assert SymbolScope.GLOBAL.value == "global"
        assert SymbolScope.CLASS.value == "class"
        assert SymbolScope.FUNCTION.value == "function"
        assert SymbolScope.MODULE.value == "module"


class TestCodeSymbol:
    """Tests for CodeSymbol dataclass."""

    def test_code_symbol_creation(self):
        """Test creating a CodeSymbol."""
        symbol = CodeSymbol(
            name="test_function",
            type=SymbolType.FUNCTION,
            scope=SymbolScope.GLOBAL,
            file_path="src/test.py",
            line_start=10,
            line_end=20,
        )
        assert symbol.name == "test_function"
        assert symbol.type == SymbolType.FUNCTION
        assert symbol.scope == SymbolScope.GLOBAL
        assert symbol.file_path == "src/test.py"
        assert symbol.line_start == 10
        assert symbol.line_end == 20

    def test_code_symbol_with_optional_fields(self):
        """Test CodeSymbol with all optional fields."""
        symbol = CodeSymbol(
            name="my_method",
            type=SymbolType.METHOD,
            scope=SymbolScope.CLASS,
            file_path="src/test.py",
            line_start=5,
            line_end=15,
            signature="def my_method(self, arg: str) -> bool",
            parent="MyClass",
            docstring="A test method.",
            qualified_name="MyClass.my_method",
        )
        assert symbol.signature == "def my_method(self, arg: str) -> bool"
        assert symbol.parent == "MyClass"
        assert symbol.docstring == "A test method."
        assert symbol.qualified_name == "MyClass.my_method"

    def test_code_symbol_is_frozen(self):
        """Test that CodeSymbol is immutable."""
        from dataclasses import FrozenInstanceError

        symbol = CodeSymbol(
            name="test",
            type=SymbolType.FUNCTION,
            scope=SymbolScope.GLOBAL,
            file_path="test.py",
            line_start=1,
            line_end=5,
        )
        with pytest.raises(FrozenInstanceError):
            symbol.name = "changed"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_valid(self):
        """Test creating a valid ValidationResult."""
        result = ValidationResult(
            symbol_name="test_function",
            is_valid=True,
            confidence=1.0,
            reason="Symbol found in codebase",
        )
        assert result.symbol_name == "test_function"
        assert result.is_valid is True
        assert result.confidence == 1.0
        assert result.suggestions == []

    def test_validation_result_invalid_with_suggestions(self):
        """Test ValidationResult with suggestions."""
        result = ValidationResult(
            symbol_name="test_func",
            is_valid=False,
            confidence=0.85,
            reason="Symbol not found",
            suggestions=["test_function", "test_funcs"],
        )
        assert result.is_valid is False
        assert len(result.suggestions) == 2
        assert "test_function" in result.suggestions


class TestHallucinationReport:
    """Tests for HallucinationReport dataclass."""

    def test_hallucination_report_creation(self):
        """Test creating a HallucinationReport."""
        validation = ValidationResult(
            symbol_name="test",
            is_valid=True,
            confidence=1.0,
            reason="Found",
        )
        report = HallucinationReport(
            total_symbols_checked=5,
            valid_symbols=4,
            invalid_symbols=1,
            uncertain_symbols=0,
            validation_results=[validation],
            files_checked=["test.py"],
            processing_time_ms=100,
            confidence_score=0.8,
        )
        assert report.total_symbols_checked == 5
        assert report.valid_symbols == 4
        assert report.invalid_symbols == 1
        assert len(report.validation_results) == 1
        assert report.processing_time_ms == 100


class TestLanguage:
    """Tests for Language enum and utilities."""

    def test_language_values(self):
        """Test that all languages have correct values."""
        assert Language.PYTHON.value == "python"
        assert Language.TYPESCRIPT.value == "typescript"
        assert Language.JAVASCRIPT.value == "javascript"
        assert Language.TSX.value == "tsx"
        assert Language.JSX.value == "jsx"

    def test_get_language_for_file_python(self):
        """Test detecting Python files."""
        assert get_language_for_file("test.py") == Language.PYTHON
        assert get_language_for_file("module.pyi") == Language.PYTHON
        assert get_language_for_file("src/utils/helper.py") == Language.PYTHON

    def test_get_language_for_file_typescript(self):
        """Test detecting TypeScript files."""
        assert get_language_for_file("component.ts") == Language.TYPESCRIPT
        assert get_language_for_file("Component.tsx") == Language.TSX

    def test_get_language_for_file_javascript(self):
        """Test detecting JavaScript files."""
        assert get_language_for_file("script.js") == Language.JAVASCRIPT
        assert get_language_for_file("Component.jsx") == Language.JSX
        assert get_language_for_file("module.mjs") == Language.JAVASCRIPT
        assert get_language_for_file("module.cjs") == Language.JAVASCRIPT

    def test_get_language_for_file_unknown(self):
        """Test that unknown extensions return None."""
        assert get_language_for_file("file.txt") is None
        assert get_language_for_file("image.png") is None
        assert get_language_for_file("doc.md") is None

    def test_language_extensions_mapping(self):
        """Test the language extensions mapping."""
        assert ".py" in LANGUAGE_EXTENSIONS
        assert ".ts" in LANGUAGE_EXTENSIONS
        assert ".js" in LANGUAGE_EXTENSIONS
        assert LANGUAGE_EXTENSIONS[".py"] == Language.PYTHON
