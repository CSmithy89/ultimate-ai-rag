"""Import validator for codebase hallucination detection.

Validates that import statements reference existing modules and packages.
"""

from importlib.util import find_spec
from pathlib import Path
from typing import Optional

import structlog

from ..symbol_table import SymbolTable
from ..types import SymbolType, ValidationResult
from .base import BaseValidator

logger = structlog.get_logger(__name__)


class ImportValidator(BaseValidator):
    """Validator for import statement references.

    Checks that import statements reference:
    1. Standard library modules
    2. Installed packages
    3. Declared dependencies
    4. Local modules in the codebase

    Attributes:
        symbol_table: The SymbolTable containing local module info
    """

    def __init__(
        self,
        symbol_table: SymbolTable,
        installed_packages: Optional[set[str]] = None,
        declared_packages: Optional[set[str]] = None,
    ) -> None:
        """Initialize the import validator.

        Args:
            symbol_table: SymbolTable containing indexed codebase symbols
            installed_packages: Optional set of known installed package names
            declared_packages: Optional set of declared dependency names
        """
        self._symbol_table = symbol_table
        self._installed_packages = {
            self._normalize_package_name(pkg)
            for pkg in (installed_packages or set())
            if pkg
        }
        self._declared_packages = {
            self._normalize_package_name(pkg)
            for pkg in (declared_packages or set())
            if pkg
        }
        self._std_lib_modules = self._get_stdlib_modules()

    @property
    def validator_type(self) -> str:
        """Return the validator type identifier."""
        return "import"

    def _get_stdlib_modules(self) -> set[str]:
        """Get a set of Python standard library module names.

        Returns:
            Set of stdlib module names
        """
        import sys

        if hasattr(sys, "stdlib_module_names"):
            return set(sys.stdlib_module_names)

        # Common stdlib modules (not exhaustive but covers most cases)
        return {
            "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
            "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
            "binhex", "bisect", "builtins", "bz2", "calendar", "cgi",
            "cgitb", "chunk", "cmath", "cmd", "code", "codecs", "codeop",
            "collections", "colorsys", "compileall", "concurrent", "configparser",
            "contextlib", "contextvars", "copy", "copyreg", "cProfile", "crypt",
            "csv", "ctypes", "curses", "dataclasses", "datetime", "dbm",
            "decimal", "difflib", "dis", "distutils", "doctest", "email",
            "encodings", "enum", "errno", "faulthandler", "fcntl", "filecmp",
            "fileinput", "fnmatch", "fractions", "ftplib", "functools", "gc",
            "getopt", "getpass", "gettext", "glob", "graphlib", "grp", "gzip",
            "hashlib", "heapq", "hmac", "html", "http", "idlelib", "imaplib",
            "imghdr", "importlib", "inspect", "io", "ipaddress", "itertools",
            "json", "keyword", "lib2to3", "linecache", "locale", "logging",
            "lzma", "mailbox", "mailcap", "marshal", "math", "mimetypes",
            "mmap", "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
            "numbers", "operator", "optparse", "os", "pathlib", "pdb", "pickle",
            "pickletools", "pipes", "pkgutil", "platform", "plistlib", "poplib",
            "posix", "posixpath", "pprint", "profile", "pstats", "pty", "pwd",
            "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random", "re",
            "readline", "reprlib", "resource", "rlcompleter", "runpy", "sched",
            "secrets", "select", "selectors", "shelve", "shlex", "shutil",
            "signal", "site", "smtpd", "smtplib", "sndhdr", "socket",
            "socketserver", "spwd", "sqlite3", "ssl", "stat", "statistics",
            "string", "stringprep", "struct", "subprocess", "sunau", "symtable",
            "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib",
            "tempfile", "termios", "test", "textwrap", "threading", "time",
            "timeit", "tkinter", "token", "tokenize", "tomllib", "trace",
            "traceback", "tracemalloc", "tty", "turtle", "turtledemo", "types",
            "typing", "unicodedata", "unittest", "urllib", "uu", "uuid",
            "venv", "warnings", "wave", "weakref", "webbrowser", "winreg",
            "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp",
            "zipfile", "zipimport", "zlib", "zoneinfo",
        }

    def validate(
        self,
        reference: str,
        location: Optional[tuple[int, int]] = None,
        context: Optional[dict] = None,
    ) -> ValidationResult:
        """Validate an import reference.

        Args:
            reference: The import statement or module name to validate
            location: Optional position in the response text
            context: Optional context (unused)

        Returns:
            ValidationResult indicating if the import is valid
        """
        return self.validate_import(reference, location)

    def validate_import(
        self,
        import_ref: str,
        location: Optional[tuple[int, int]] = None,
    ) -> ValidationResult:
        """Validate an import statement or module reference.

        Checks if the import references a valid module from:
        1. Python standard library
        2. Installed third-party packages
        3. Local modules in the codebase

        Args:
            import_ref: The import reference (e.g., 'os.path' or 'from x import y')
            location: Optional position in the response text

        Returns:
            ValidationResult with validation outcome
        """
        # Parse the import reference
        module_name = self._extract_module_name(import_ref)

        if not module_name:
            return ValidationResult(
                symbol_name=import_ref,
                is_valid=False,
                confidence=0.5,
                reason="Could not parse import reference",
                suggestions=[],
                location_in_response=location,
            )

        if module_name.startswith((".", "/")):
            return self._validate_local_import(module_name, import_ref, location)

        # Check standard library
        top_level = self._get_top_level_module(module_name)
        normalized_top_level = self._normalize_package_name(top_level)
        if normalized_top_level in self._std_lib_modules:
            return ValidationResult(
                symbol_name=import_ref,
                is_valid=True,
                confidence=1.0,
                reason=f"'{module_name}' is a Python standard library module",
                suggestions=[],
                location_in_response=location,
            )

        # Check installed packages
        if self._installed_packages and normalized_top_level in self._installed_packages:
            return ValidationResult(
                symbol_name=import_ref,
                is_valid=True,
                confidence=1.0,
                reason=f"'{module_name}' is from installed package '{top_level}'",
                suggestions=[],
                location_in_response=location,
            )

        # Check declared dependencies (less confident than installed packages)
        if self._declared_packages and normalized_top_level in self._declared_packages:
            return ValidationResult(
                symbol_name=import_ref,
                is_valid=True,
                confidence=0.85,
                reason=f"'{module_name}' matches declared dependency '{top_level}'",
                suggestions=[],
                location_in_response=location,
            )

        # Try to find the module spec
        if not self._installed_packages:
            try:
                spec = find_spec(top_level)
                if spec is not None:
                    return ValidationResult(
                        symbol_name=import_ref,
                        is_valid=True,
                        confidence=1.0,
                        reason=f"Module '{module_name}' is available in the environment",
                        suggestions=[],
                        location_in_response=location,
                    )
            except (ModuleNotFoundError, ImportError, ValueError):
                pass

        # Check local modules in symbol table
        import_symbols = self._symbol_table.get_symbols_by_type(SymbolType.IMPORT)
        local_modules = set()
        for sym in import_symbols:
            local_modules.add(sym.name)

        # Also check file paths for potential modules
        known_files = self._symbol_table.get_all_known_files()
        for file_path in known_files:
            if file_path.endswith((".py", ".ts", ".tsx", ".js", ".jsx")):
                path_no_ext = file_path.rsplit(".", 1)[0]
                module_dot = path_no_ext.replace("/", ".").replace("\\", ".")
                local_modules.add(module_dot)
                if not file_path.endswith(".py"):
                    local_modules.add(path_no_ext.replace("\\", "/"))

                parts = module_dot.split(".")
                for i in range(1, len(parts)):
                    local_modules.add(".".join(parts[:i]))

        if module_name in local_modules or top_level in local_modules:
            return ValidationResult(
                symbol_name=import_ref,
                is_valid=True,
                confidence=0.95,
                reason=f"'{module_name}' appears to be a local module",
                suggestions=[],
                location_in_response=location,
            )

        # Not found - suggest similar
        all_modules = list(
            self._std_lib_modules
            | self._installed_packages
            | self._declared_packages
            | local_modules
        )
        similar = self._find_similar_modules(module_name, all_modules)

        return ValidationResult(
            symbol_name=import_ref,
            is_valid=False,
            confidence=0.7,
            reason=f"Module '{module_name}' not found. May be missing from requirements.",
            suggestions=similar,
            location_in_response=location,
        )

    def _extract_module_name(self, import_ref: str) -> Optional[str]:
        """Extract the module name from an import statement.

        Handles various import formats:
        - 'import os'
        - 'from os import path'
        - 'from os.path import join'
        - 'os.path' (bare module reference)

        Args:
            import_ref: The import statement or module reference

        Returns:
            The module name, or None if parsing fails
        """
        import_ref = import_ref.strip()

        # Handle 'import x' or 'import x.y'
        if import_ref.startswith("import "):
            parts = import_ref[7:].split(",")[0].strip()
            # Handle 'import x as y'
            parts = parts.split(" as ")[0].strip()
            return parts

        # Handle 'from x import y' or 'from x.y import z'
        if import_ref.startswith("from "):
            # Extract module between 'from' and 'import'
            match = import_ref[5:].split(" import ")[0].strip()
            return match

        # Bare module reference (e.g., 'os.path' or '@scope/pkg' or './local')
        if import_ref:
            candidate = import_ref.replace(".", "").replace("_", "").replace("/", "").replace("@", "")
            if candidate.isalnum():
                return import_ref

        return None

    def _find_similar_modules(
        self,
        module_name: str,
        all_modules: list[str],
        limit: int = 5,
    ) -> list[str]:
        """Find similar module names.

        Args:
            module_name: The module to find matches for
            all_modules: List of known module names
            limit: Maximum number of suggestions

        Returns:
            List of similar module names
        """
        from difflib import get_close_matches
        return get_close_matches(module_name, all_modules, n=limit, cutoff=0.6)

    def add_installed_package(self, package_name: str) -> None:
        """Add a package to the known installed packages.

        Args:
            package_name: Name of the package
        """
        if package_name:
            self._installed_packages.add(self._normalize_package_name(package_name))

    def add_declared_package(self, package_name: str) -> None:
        """Add a package to the declared dependency list."""
        if package_name:
            self._declared_packages.add(self._normalize_package_name(package_name))

    def load_from_requirements(self, requirements_content: str) -> int:
        """Load package names from requirements.txt content.

        Args:
            requirements_content: Content of requirements.txt file

        Returns:
            Number of packages added
        """
        count = 0
        for line in requirements_content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                # Extract package name (before version specifier)
                package = line.split("==")[0].split(">=")[0].split("<=")[0]
                package = package.split("[")[0].strip()
                if package:
                    self._installed_packages.add(self._normalize_package_name(package))
                    count += 1
        return count

    def _normalize_package_name(self, name: str) -> str:
        """Normalize package names for comparison."""
        normalized = name.strip().lower()
        if normalized.startswith("@") and "/" in normalized:
            scope, pkg = normalized.split("/", 1)
            return f"{scope}/{pkg.replace('-', '_')}"
        return normalized.replace("-", "_")

    def _get_top_level_module(self, module_name: str) -> str:
        """Get the top-level module or package name."""
        normalized = module_name.replace("\\", "/")
        if normalized.startswith("@") and "/" in normalized:
            parts = normalized.split("/")
            return "/".join(parts[:2])
        if "/" in normalized:
            return normalized.split("/", 1)[0]
        return normalized.split(".", 1)[0]

    def _validate_local_import(
        self,
        module_name: str,
        import_ref: str,
        location: Optional[tuple[int, int]] = None,
    ) -> ValidationResult:
        """Validate local module imports against known files."""
        normalized = module_name.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized.startswith("/"):
            normalized = normalized[1:]

        candidates: list[str] = []
        if Path(normalized).suffix:
            candidates.append(normalized)
        else:
            for ext in (".py", ".ts", ".tsx", ".js", ".jsx", ".json"):
                candidates.append(f"{normalized}{ext}")
                candidates.append(f"{normalized}/index{ext}")

        for candidate in candidates:
            if self._symbol_table.file_exists(candidate):
                return ValidationResult(
                    symbol_name=import_ref,
                    is_valid=True,
                    confidence=0.95,
                    reason=f"Local module '{module_name}' resolved to '{candidate}'",
                    suggestions=[],
                    location_in_response=location,
                )

        return ValidationResult(
            symbol_name=import_ref,
            is_valid=False,
            confidence=0.7,
            reason=f"Local module '{module_name}' not found in repository",
            suggestions=[],
            location_in_response=location,
        )
