"""Tests for AST parser module."""

import pytest

from agentic_rag_backend.codebase.parser import ASTParser, TREE_SITTER_AVAILABLE
from agentic_rag_backend.codebase.types import Language


# Skip all tests if tree-sitter is not available
pytestmark = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE,
    reason="tree-sitter not installed"
)


@pytest.fixture
def parser():
    """Create an AST parser."""
    return ASTParser()


class TestParserAvailability:
    """Tests for parser availability."""

    def test_is_available(self, parser):
        """Test that parser reports availability correctly."""
        assert parser.is_available() is True

    def test_supports_python(self, parser):
        """Test that Python is supported."""
        assert parser.supports_language(Language.PYTHON)

    def test_languages_property(self, parser):
        """Test that languages property returns supported languages."""
        languages = parser.languages
        assert Language.PYTHON in languages


class TestParseString:
    """Tests for parsing source code strings."""

    def test_parse_python_function(self, parser):
        """Test parsing a Python function."""
        source = """
def hello(name: str) -> str:
    \"\"\"Say hello.\"\"\"
    return f"Hello, {name}!"
"""
        tree = parser.parse_string(source, Language.PYTHON)

        assert tree is not None
        assert tree.root_node is not None
        assert tree.root_node.type == "module"

    def test_parse_python_class(self, parser):
        """Test parsing a Python class."""
        source = """
class User:
    def __init__(self, name: str):
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}"
"""
        tree = parser.parse_string(source, Language.PYTHON)

        assert tree is not None
        assert tree.root_node.type == "module"

    def test_parse_empty_string(self, parser):
        """Test parsing an empty string."""
        tree = parser.parse_string("", Language.PYTHON)

        assert tree is not None
        assert tree.root_node.type == "module"

    def test_parse_unsupported_language(self, parser):
        """Test parsing with unsupported language returns None."""
        tree = parser.parse_string("code", Language.JSX)

        assert tree is None


class TestFindNodesByType:
    """Tests for finding nodes by type."""

    def test_find_function_definitions(self, parser):
        """Test finding all function definitions."""
        source = """
def func1():
    pass

def func2():
    pass

class MyClass:
    def method(self):
        pass
"""
        tree = parser.parse_string(source, Language.PYTHON)
        functions = parser.find_nodes_by_type(tree, "function_definition")

        assert len(functions) == 3

    def test_find_class_definitions(self, parser):
        """Test finding all class definitions."""
        source = """
class First:
    pass

class Second:
    pass
"""
        tree = parser.parse_string(source, Language.PYTHON)
        classes = parser.find_nodes_by_type(tree, "class_definition")

        assert len(classes) == 2

    def test_find_nonexistent_type(self, parser):
        """Test finding a node type that doesn't exist."""
        source = "x = 1"
        tree = parser.parse_string(source, Language.PYTHON)
        nodes = parser.find_nodes_by_type(tree, "nonexistent_type")

        assert nodes == []


class TestWalkTree:
    """Tests for tree walking."""

    def test_walk_tree_visits_all_nodes(self, parser):
        """Test that walk_tree visits all nodes."""
        source = """
def hello():
    return 1
"""
        tree = parser.parse_string(source, Language.PYTHON)
        nodes = list(parser.walk_tree(tree))

        assert len(nodes) > 0
        # Should include module, function_definition, and more
        node_types = {n.type for n in nodes}
        assert "module" in node_types
        assert "function_definition" in node_types

    def test_walk_empty_tree(self, parser):
        """Test walking an empty tree."""
        source = ""
        tree = parser.parse_string(source, Language.PYTHON)
        nodes = list(parser.walk_tree(tree))

        # Should still have the root module node
        assert len(nodes) >= 1


class TestGetNodeText:
    """Tests for extracting node text."""

    def test_get_node_text_function_name(self, parser):
        """Test extracting function name from node."""
        source = "def my_function(): pass"
        tree = parser.parse_string(source, Language.PYTHON)

        functions = parser.find_nodes_by_type(tree, "function_definition")
        assert len(functions) == 1

        func_node = functions[0]
        name_node = func_node.child_by_field_name("name")
        assert name_node is not None

        name_text = parser.get_node_text(name_node, source)
        assert name_text == "my_function"

    def test_get_node_text_full_function(self, parser):
        """Test extracting full function text."""
        source = """def greet():
    return "hello"
"""
        tree = parser.parse_string(source, Language.PYTHON)
        functions = parser.find_nodes_by_type(tree, "function_definition")

        func_text = parser.get_node_text(functions[0], source)
        assert "def greet" in func_text
        assert "return" in func_text


class TestTypescriptParsing:
    """Tests for TypeScript parsing."""

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE,
        reason="tree-sitter not installed"
    )
    def test_parse_typescript_function(self, parser):
        """Test parsing a TypeScript function."""
        if not parser.supports_language(Language.TYPESCRIPT):
            pytest.skip("TypeScript parser not available")

        source = """
function greet(name: string): string {
    return `Hello, ${name}!`;
}
"""
        tree = parser.parse_string(source, Language.TYPESCRIPT)
        assert tree is not None

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE,
        reason="tree-sitter not installed"
    )
    def test_parse_typescript_interface(self, parser):
        """Test parsing a TypeScript interface."""
        if not parser.supports_language(Language.TYPESCRIPT):
            pytest.skip("TypeScript parser not available")

        source = """
interface User {
    name: string;
    age: number;
}
"""
        tree = parser.parse_string(source, Language.TYPESCRIPT)
        assert tree is not None


class TestJavaScriptParsing:
    """Tests for JavaScript parsing."""

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE,
        reason="tree-sitter not installed"
    )
    def test_parse_javascript_function(self, parser):
        """Test parsing a JavaScript function."""
        if not parser.supports_language(Language.JAVASCRIPT):
            pytest.skip("JavaScript parser not available")

        source = """
function greet(name) {
    return "Hello, " + name;
}
"""
        tree = parser.parse_string(source, Language.JAVASCRIPT)
        assert tree is not None

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE,
        reason="tree-sitter not installed"
    )
    def test_parse_javascript_arrow_function(self, parser):
        """Test parsing a JavaScript arrow function."""
        if not parser.supports_language(Language.JAVASCRIPT):
            pytest.skip("JavaScript parser not available")

        source = """
const greet = (name) => {
    return `Hello, ${name}`;
};
"""
        tree = parser.parse_string(source, Language.JAVASCRIPT)
        assert tree is not None
