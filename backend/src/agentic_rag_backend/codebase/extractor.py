"""Symbol extraction from AST for codebase intelligence.

Provides extraction of functions, classes, methods, imports, and other
code symbols from tree-sitter AST nodes.
"""

from typing import Optional

import structlog

from .parser import ASTParser, TREE_SITTER_AVAILABLE
from .types import CodeSymbol, Language, SymbolScope, SymbolType, get_language_for_file

logger = structlog.get_logger(__name__)

if TREE_SITTER_AVAILABLE:
    import tree_sitter


class SymbolExtractor:
    """Extract code symbols from AST for multiple languages.

    Supports extraction of:
    - Functions and methods
    - Classes and interfaces
    - Variables and constants
    - Import statements
    - Type aliases and enums

    Uses tree-sitter for AST parsing with language-specific extraction logic.
    """

    def __init__(self, parser: Optional[ASTParser] = None) -> None:
        """Initialize the symbol extractor.

        Args:
            parser: Optional ASTParser instance (creates one if not provided)
        """
        self._parser = parser or ASTParser()

    def extract_from_file(self, file_path: str) -> list[CodeSymbol]:
        """Extract all symbols from a source file.

        Args:
            file_path: Path to the source file

        Returns:
            List of extracted CodeSymbol instances
        """
        language = get_language_for_file(file_path)
        if language is None:
            logger.warning("unsupported_file_type", file_path=file_path)
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return self.extract_from_string(source_code, file_path, language)
        except (OSError, IOError) as e:
            logger.warning(
                "file_read_failed",
                file_path=file_path,
                error=str(e),
            )
            return []

    def extract_from_string(
        self,
        source_code: str,
        file_path: str,
        language: Language,
    ) -> list[CodeSymbol]:
        """Extract all symbols from source code string.

        Args:
            source_code: The source code to extract from
            file_path: Path to the file (for symbol metadata)
            language: The programming language

        Returns:
            List of extracted CodeSymbol instances
        """
        if not TREE_SITTER_AVAILABLE:
            logger.warning("extraction_skipped", reason="tree-sitter not available")
            return []

        tree = self._parser.parse_string(source_code, language)
        if tree is None:
            return []

        if language == Language.PYTHON:
            return self._extract_python(tree, source_code, file_path)
        elif language in (Language.TYPESCRIPT, Language.TSX):
            return self._extract_typescript(tree, source_code, file_path)
        elif language in (Language.JAVASCRIPT, Language.JSX):
            return self._extract_javascript(tree, source_code, file_path)
        else:
            logger.warning("unsupported_language", language=language.value)
            return []

    def _extract_python(
        self,
        tree: "tree_sitter.Tree",
        source_code: str,
        file_path: str,
    ) -> list[CodeSymbol]:
        """Extract symbols from Python source code.

        Extracts:
        - Functions (def statements)
        - Classes and their methods
        - Import statements
        - Global variables/constants

        Args:
            tree: Parsed tree-sitter Tree
            source_code: Original source code
            file_path: Path to the file

        Returns:
            List of extracted CodeSymbol instances
        """
        symbols = []
        source_bytes = source_code.encode("utf-8")

        def get_text(node: "tree_sitter.Node") -> str:
            return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

        def extract_docstring(body_node: "tree_sitter.Node") -> Optional[str]:
            """Extract docstring from function/class body."""
            if not body_node.children:
                return None
            first_stmt = body_node.children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.children[0] if first_stmt.children else None
                if expr and expr.type == "string":
                    docstring = get_text(expr)
                    # Remove quotes
                    if docstring.startswith('"""') or docstring.startswith("'''"):
                        return docstring[3:-3].strip()
                    elif docstring.startswith('"') or docstring.startswith("'"):
                        return docstring[1:-1].strip()
            return None

        def extract_signature(node: "tree_sitter.Node") -> Optional[str]:
            """Extract function signature."""
            name_node = node.child_by_field_name("name")
            params_node = node.child_by_field_name("parameters")
            return_type = node.child_by_field_name("return_type")

            if not name_node or not params_node:
                return None

            sig = f"def {get_text(name_node)}{get_text(params_node)}"
            if return_type:
                sig += f" -> {get_text(return_type)}"
            return sig

        def visit_class(
            node: "tree_sitter.Node",
            parent_class: Optional[str] = None,
        ) -> None:
            """Visit a class definition and extract its symbols."""
            name_node = node.child_by_field_name("name")
            if not name_node:
                return

            class_name = get_text(name_node)

            # Get class signature with bases
            signature = f"class {class_name}"
            superclasses = node.child_by_field_name("superclasses")
            if superclasses:
                signature += get_text(superclasses)

            # Get docstring from body
            body_node = node.child_by_field_name("body")
            docstring = extract_docstring(body_node) if body_node else None

            symbols.append(
                CodeSymbol(
                    name=class_name,
                    type=SymbolType.CLASS,
                    scope=SymbolScope.CLASS if parent_class else SymbolScope.GLOBAL,
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=signature,
                    parent=parent_class,
                    docstring=docstring,
                    qualified_name=f"{parent_class}.{class_name}" if parent_class else class_name,
                )
            )

            # Extract methods from class body
            if body_node:
                for child in body_node.children:
                    if child.type == "function_definition":
                        visit_function(child, parent_class=class_name)
                    elif child.type == "class_definition":
                        visit_class(child, parent_class=class_name)

        def visit_function(
            node: "tree_sitter.Node",
            parent_class: Optional[str] = None,
        ) -> None:
            """Visit a function definition and extract its symbol."""
            name_node = node.child_by_field_name("name")
            if not name_node:
                return

            func_name = get_text(name_node)
            signature = extract_signature(node)

            # Get docstring from body
            body_node = node.child_by_field_name("body")
            docstring = extract_docstring(body_node) if body_node else None

            # Determine symbol type (method vs function)
            symbol_type = SymbolType.METHOD if parent_class else SymbolType.FUNCTION

            symbols.append(
                CodeSymbol(
                    name=func_name,
                    type=symbol_type,
                    scope=SymbolScope.CLASS if parent_class else SymbolScope.GLOBAL,
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=signature,
                    parent=parent_class,
                    docstring=docstring,
                    qualified_name=f"{parent_class}.{func_name}" if parent_class else func_name,
                )
            )

        def visit_import(node: "tree_sitter.Node") -> None:
            """Visit an import statement and extract its symbol."""
            if node.type == "import_statement":
                # import module
                for child in node.children:
                    if child.type == "dotted_name":
                        module_name = get_text(child)
                        symbols.append(
                            CodeSymbol(
                                name=module_name,
                                type=SymbolType.IMPORT,
                                scope=SymbolScope.MODULE,
                                file_path=file_path,
                                line_start=node.start_point[0] + 1,
                                line_end=node.end_point[0] + 1,
                            )
                        )
            elif node.type == "import_from_statement":
                # from module import name
                module_node = node.child_by_field_name("module_name")
                if module_node:
                    module_name = get_text(module_node)
                    symbols.append(
                        CodeSymbol(
                            name=module_name,
                            type=SymbolType.IMPORT,
                            scope=SymbolScope.MODULE,
                            file_path=file_path,
                            line_start=node.start_point[0] + 1,
                            line_end=node.end_point[0] + 1,
                        )
                    )

        def visit_assignment(node: "tree_sitter.Node") -> None:
            """Visit an assignment statement for global variables."""
            # Look for top-level assignments (SCREAMING_SNAKE = value)
            left = node.child_by_field_name("left")
            if left and left.type == "identifier":
                var_name = get_text(left)
                # Check if it's likely a constant (SCREAMING_SNAKE_CASE)
                if var_name.isupper():
                    symbol_type = SymbolType.CONSTANT
                else:
                    symbol_type = SymbolType.VARIABLE

                symbols.append(
                    CodeSymbol(
                        name=var_name,
                        type=symbol_type,
                        scope=SymbolScope.GLOBAL,
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                    )
                )

        # Walk the tree and extract symbols
        for node in self._parser.walk_tree(tree):
            if node.type == "class_definition" and node.parent.type == "module":
                visit_class(node)
            elif node.type == "function_definition" and node.parent.type == "module":
                visit_function(node)
            elif node.type in ("import_statement", "import_from_statement"):
                visit_import(node)
            elif node.type == "assignment" and node.parent.type in ("module", "expression_statement"):
                # Only top-level assignments
                if node.parent.type == "expression_statement" and node.parent.parent.type == "module":
                    visit_assignment(node)

        logger.debug(
            "python_extraction_complete",
            file_path=file_path,
            symbol_count=len(symbols),
        )

        return symbols

    def _extract_typescript(
        self,
        tree: "tree_sitter.Tree",
        source_code: str,
        file_path: str,
    ) -> list[CodeSymbol]:
        """Extract symbols from TypeScript source code.

        Args:
            tree: Parsed tree-sitter Tree
            source_code: Original source code
            file_path: Path to the file

        Returns:
            List of extracted CodeSymbol instances
        """
        symbols = []
        source_bytes = source_code.encode("utf-8")

        def get_text(node: "tree_sitter.Node") -> str:
            return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

        for node in self._parser.walk_tree(tree):
            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append(
                        CodeSymbol(
                            name=get_text(name_node),
                            type=SymbolType.FUNCTION,
                            scope=SymbolScope.GLOBAL,
                            file_path=file_path,
                            line_start=node.start_point[0] + 1,
                            line_end=node.end_point[0] + 1,
                            signature=get_text(node).split("{")[0].strip(),
                        )
                    )

            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    class_name = get_text(name_node)
                    symbols.append(
                        CodeSymbol(
                            name=class_name,
                            type=SymbolType.CLASS,
                            scope=SymbolScope.GLOBAL,
                            file_path=file_path,
                            line_start=node.start_point[0] + 1,
                            line_end=node.end_point[0] + 1,
                        )
                    )

                    # Extract methods from class body
                    body_node = node.child_by_field_name("body")
                    if body_node:
                        for child in body_node.children:
                            if child.type == "method_definition":
                                method_name_node = child.child_by_field_name("name")
                                if method_name_node:
                                    symbols.append(
                                        CodeSymbol(
                                            name=get_text(method_name_node),
                                            type=SymbolType.METHOD,
                                            scope=SymbolScope.CLASS,
                                            file_path=file_path,
                                            line_start=child.start_point[0] + 1,
                                            line_end=child.end_point[0] + 1,
                                            parent=class_name,
                                            qualified_name=f"{class_name}.{get_text(method_name_node)}",
                                        )
                                    )

            elif node.type == "interface_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append(
                        CodeSymbol(
                            name=get_text(name_node),
                            type=SymbolType.INTERFACE,
                            scope=SymbolScope.GLOBAL,
                            file_path=file_path,
                            line_start=node.start_point[0] + 1,
                            line_end=node.end_point[0] + 1,
                        )
                    )

            elif node.type == "type_alias_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append(
                        CodeSymbol(
                            name=get_text(name_node),
                            type=SymbolType.TYPE_ALIAS,
                            scope=SymbolScope.GLOBAL,
                            file_path=file_path,
                            line_start=node.start_point[0] + 1,
                            line_end=node.end_point[0] + 1,
                        )
                    )

            elif node.type == "enum_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append(
                        CodeSymbol(
                            name=get_text(name_node),
                            type=SymbolType.ENUM,
                            scope=SymbolScope.GLOBAL,
                            file_path=file_path,
                            line_start=node.start_point[0] + 1,
                            line_end=node.end_point[0] + 1,
                        )
                    )

            elif node.type == "import_statement":
                symbols.append(
                    CodeSymbol(
                        name=get_text(node),
                        type=SymbolType.IMPORT,
                        scope=SymbolScope.MODULE,
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                    )
                )

        logger.debug(
            "typescript_extraction_complete",
            file_path=file_path,
            symbol_count=len(symbols),
        )

        return symbols

    def _extract_javascript(
        self,
        tree: "tree_sitter.Tree",
        source_code: str,
        file_path: str,
    ) -> list[CodeSymbol]:
        """Extract symbols from JavaScript source code.

        Uses the same logic as TypeScript extraction as they share
        similar AST structures.

        Args:
            tree: Parsed tree-sitter Tree
            source_code: Original source code
            file_path: Path to the file

        Returns:
            List of extracted CodeSymbol instances
        """
        # JavaScript extraction is similar to TypeScript
        return self._extract_typescript(tree, source_code, file_path)

    def extract(
        self,
        tree: "tree_sitter.Tree",
        file_path: str,
        source_code: Optional[str] = None,
    ) -> list[CodeSymbol]:
        """Extract symbols from a pre-parsed AST tree.

        This method requires the source code to be provided separately
        since the tree doesn't contain the original text.

        Args:
            tree: Parsed tree-sitter Tree
            file_path: Path to the source file
            source_code: Original source code (required for text extraction)

        Returns:
            List of extracted CodeSymbol instances
        """
        if source_code is None:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()
            except (OSError, IOError) as e:
                logger.warning(
                    "source_code_read_failed",
                    file_path=file_path,
                    error=str(e),
                )
                return []

        language = get_language_for_file(file_path)
        if language is None:
            return []

        if language == Language.PYTHON:
            return self._extract_python(tree, source_code, file_path)
        elif language in (Language.TYPESCRIPT, Language.TSX):
            return self._extract_typescript(tree, source_code, file_path)
        elif language in (Language.JAVASCRIPT, Language.JSX):
            return self._extract_javascript(tree, source_code, file_path)
        else:
            return []
