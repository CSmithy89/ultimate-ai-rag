"""Tree-sitter AST parser wrapper for multi-language support.

Provides a unified interface for parsing Python, TypeScript, and JavaScript
source code into Abstract Syntax Trees (AST) using tree-sitter.
"""

from typing import Optional

import structlog

from .types import Language, get_language_for_file

logger = structlog.get_logger(__name__)

# Tree-sitter availability flag
TREE_SITTER_AVAILABLE = False

try:
    import tree_sitter
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript

    TREE_SITTER_AVAILABLE = True
except ImportError:
    logger.warning(
        "tree_sitter_not_available",
        hint="Install tree-sitter packages for full AST parsing support",
    )
    tree_sitter = None  # type: ignore
    tree_sitter_python = None  # type: ignore
    tree_sitter_javascript = None  # type: ignore
    tree_sitter_typescript = None  # type: ignore


class ASTParser:
    """Tree-sitter AST parser with multi-language support.

    Provides parsing capabilities for Python, TypeScript, and JavaScript
    source code. Falls back gracefully when tree-sitter is not available.

    Attributes:
        languages: Set of supported languages
    """

    def __init__(self, languages: Optional[list[str]] = None) -> None:
        """Initialize the AST parser.

        Args:
            languages: List of languages to support (default: all available)
        """
        self._parsers: dict[Language, "tree_sitter.Parser"] = {}
        self._languages_config = languages or ["python", "typescript", "javascript"]

        if TREE_SITTER_AVAILABLE:
            self._initialize_parsers()
        else:
            logger.warning("ast_parser_limited_mode", reason="tree-sitter not installed")

    def _initialize_parsers(self) -> None:
        """Initialize tree-sitter parsers for configured languages."""
        if not TREE_SITTER_AVAILABLE:
            return

        language_modules = {
            Language.PYTHON: tree_sitter_python,
            Language.JAVASCRIPT: tree_sitter_javascript,
            Language.TYPESCRIPT: tree_sitter_typescript,
            Language.TSX: tree_sitter_typescript,  # TSX uses TypeScript parser
        }

        for lang_str in self._languages_config:
            try:
                lang = Language(lang_str.lower())
            except ValueError:
                logger.warning("unsupported_language", language=lang_str)
                continue

            if lang in language_modules:
                try:
                    parser = tree_sitter.Parser()
                    module = language_modules[lang]

                    # Get language from module
                    if lang == Language.TSX:
                        # TypeScript module has tsx() function
                        ts_lang = tree_sitter.Language(module.language_tsx())
                    elif lang == Language.TYPESCRIPT:
                        ts_lang = tree_sitter.Language(module.language_typescript())
                    elif lang == Language.JAVASCRIPT:
                        ts_lang = tree_sitter.Language(module.language())
                    elif lang == Language.PYTHON:
                        ts_lang = tree_sitter.Language(module.language())
                    else:
                        continue

                    parser.language = ts_lang
                    self._parsers[lang] = parser
                    logger.debug("parser_initialized", language=lang.value)

                except Exception as e:
                    logger.warning(
                        "parser_initialization_failed",
                        language=lang.value,
                        error=str(e),
                    )

    @property
    def languages(self) -> set[Language]:
        """Get set of available languages."""
        return set(self._parsers.keys())

    def is_available(self) -> bool:
        """Check if tree-sitter parsing is available."""
        return TREE_SITTER_AVAILABLE and len(self._parsers) > 0

    def supports_language(self, language: Language) -> bool:
        """Check if a specific language is supported.

        Args:
            language: The language to check

        Returns:
            True if the language is supported, False otherwise
        """
        return language in self._parsers

    def parse_string(
        self,
        source_code: str,
        language: Language,
    ) -> Optional["tree_sitter.Tree"]:
        """Parse source code string into an AST.

        Args:
            source_code: The source code to parse
            language: The programming language of the source

        Returns:
            Parsed tree-sitter Tree, or None if parsing fails
        """
        if not TREE_SITTER_AVAILABLE:
            logger.warning("parse_skipped", reason="tree-sitter not available")
            return None

        parser = self._parsers.get(language)
        if parser is None:
            logger.warning("parser_not_found", language=language.value)
            return None

        try:
            tree = parser.parse(source_code.encode("utf-8"))
            logger.debug(
                "parse_completed",
                language=language.value,
                root_type=tree.root_node.type if tree else None,
            )
            return tree
        except Exception as e:
            logger.warning(
                "parse_failed",
                language=language.value,
                error=str(e),
            )
            return None

    def parse_file(self, file_path: str) -> Optional["tree_sitter.Tree"]:
        """Parse a source file into an AST.

        Automatically detects the language from the file extension.

        Args:
            file_path: Path to the source file

        Returns:
            Parsed tree-sitter Tree, or None if parsing fails
        """
        language = get_language_for_file(file_path)
        if language is None:
            logger.warning("unknown_file_type", file_path=file_path)
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return self.parse_string(source_code, language)
        except (OSError, IOError) as e:
            logger.warning(
                "file_read_failed",
                file_path=file_path,
                error=str(e),
            )
            return None

    async def parse_file_async(self, file_path: str) -> Optional["tree_sitter.Tree"]:
        """Asynchronously parse a source file into an AST.

        Args:
            file_path: Path to the source file

        Returns:
            Parsed tree-sitter Tree, or None if parsing fails
        """
        import aiofiles

        language = get_language_for_file(file_path)
        if language is None:
            logger.warning("unknown_file_type", file_path=file_path)
            return None

        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                source_code = await f.read()
            return self.parse_string(source_code, language)
        except (OSError, IOError) as e:
            logger.warning(
                "file_read_failed",
                file_path=file_path,
                error=str(e),
            )
            return None

    def get_node_text(
        self,
        node: "tree_sitter.Node",
        source_code: str,
    ) -> str:
        """Extract the source text for a tree-sitter node.

        Args:
            node: The tree-sitter node
            source_code: The original source code

        Returns:
            The text content of the node
        """
        return source_code[node.start_byte:node.end_byte]

    def find_nodes_by_type(
        self,
        tree: "tree_sitter.Tree",
        node_type: str,
    ) -> list["tree_sitter.Node"]:
        """Find all nodes of a specific type in the tree.

        Args:
            tree: The tree-sitter Tree to search
            node_type: The type of nodes to find

        Returns:
            List of matching nodes
        """
        matches = []

        def visit(node: "tree_sitter.Node") -> None:
            if node.type == node_type:
                matches.append(node)
            for child in node.children:
                visit(child)

        if tree and tree.root_node:
            visit(tree.root_node)

        return matches

    def walk_tree(
        self,
        tree: "tree_sitter.Tree",
    ):
        """Create a tree cursor for walking the AST.

        Args:
            tree: The tree-sitter Tree to walk

        Yields:
            Each node in the tree (depth-first)
        """
        if not tree or not tree.root_node:
            return

        cursor = tree.walk()

        reached_root = False
        while not reached_root:
            yield cursor.node

            if cursor.goto_first_child():
                continue

            if cursor.goto_next_sibling():
                continue

            retracing = True
            while retracing:
                if not cursor.goto_parent():
                    retracing = False
                    reached_root = True
                elif cursor.goto_next_sibling():
                    retracing = False
