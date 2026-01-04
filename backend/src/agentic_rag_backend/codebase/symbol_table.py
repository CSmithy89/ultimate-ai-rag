"""Symbol table implementation for codebase hallucination detection.

Provides an in-memory symbol table with multiple indexes for efficient
lookup by name, qualified name, and file path. Supports Redis caching
for persistence across requests.
"""

import hashlib
import json
from difflib import get_close_matches
from typing import Optional

import structlog

from .types import CodeSymbol, SymbolType

logger = structlog.get_logger(__name__)


class SymbolTable:
    """In-memory symbol table for a codebase with multiple indexes.

    Stores extracted code symbols and provides efficient lookup by:
    - Simple name (may return multiple matches for overloaded symbols)
    - Qualified name (unique, e.g., 'MyClass.my_method')
    - File path (all symbols defined in a specific file)

    Supports tenant isolation for multi-tenancy and Redis caching
    for persistence.

    Attributes:
        tenant_id: Tenant identifier for isolation
        repo_path: Root path of the repository
    """

    def __init__(self, tenant_id: str, repo_path: str) -> None:
        """Initialize an empty symbol table.

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            repo_path: Root path of the repository being indexed
        """
        self.tenant_id = tenant_id
        self.repo_path = repo_path

        # Index by simple name -> list of symbols (handles overloading)
        self._symbols: dict[str, list[CodeSymbol]] = {}

        # Index by file path -> set of symbol names
        self._files: dict[str, set[str]] = {}

        # Index by qualified name -> symbol (unique)
        self._qualified: dict[str, CodeSymbol] = {}

        # Track all file paths for path validation
        self._known_files: set[str] = set()

    def add(self, symbol: CodeSymbol) -> None:
        """Add a symbol to the table.

        Indexes the symbol by name, qualified name, and file path
        for efficient lookup.

        Args:
            symbol: The CodeSymbol to add
        """
        # Index by simple name
        if symbol.name not in self._symbols:
            self._symbols[symbol.name] = []
        self._symbols[symbol.name].append(symbol)

        # Index by file path
        if symbol.file_path not in self._files:
            self._files[symbol.file_path] = set()
        self._files[symbol.file_path].add(symbol.name)

        # Track known files
        self._known_files.add(symbol.file_path)

        # Index by qualified name
        qualified = self._make_qualified_name(symbol)
        self._qualified[qualified] = symbol

        logger.debug(
            "symbol_added",
            name=symbol.name,
            type=symbol.type.value,
            file=symbol.file_path,
            qualified=qualified,
        )

    def get(self, name: str) -> list[CodeSymbol]:
        """Look up symbols by simple name.

        May return multiple symbols for overloaded names.

        Args:
            name: The symbol name to look up

        Returns:
            List of matching symbols, empty if not found
        """
        return self._symbols.get(name, [])

    def lookup(self, name: str) -> list[CodeSymbol]:
        """Alias for get() - look up symbols by simple name.

        Args:
            name: The symbol name to look up

        Returns:
            List of matching symbols, empty if not found
        """
        return self.get(name)

    def contains(self, name: str) -> bool:
        """Check if a symbol name exists in the table.

        Args:
            name: The symbol name to check

        Returns:
            True if the symbol exists, False otherwise
        """
        return name in self._symbols

    def lookup_qualified(self, qualified_name: str) -> Optional[CodeSymbol]:
        """Look up a symbol by its fully qualified name.

        Args:
            qualified_name: Qualified name (e.g., 'MyClass.my_method')

        Returns:
            The matching symbol, or None if not found
        """
        return self._qualified.get(qualified_name)

    def find_similar(self, name: str, limit: int = 5, cutoff: float = 0.6) -> list[str]:
        """Find similar symbol names using edit distance.

        Useful for suggesting corrections when a hallucinated
        symbol name is close to a real one.

        Args:
            name: The name to find similar matches for
            limit: Maximum number of suggestions to return
            cutoff: Minimum similarity ratio (0.0-1.0)

        Returns:
            List of similar symbol names, sorted by similarity
        """
        all_names = list(self._symbols.keys())
        return get_close_matches(name, all_names, n=limit, cutoff=cutoff)

    def get_symbols_in_file(self, file_path: str) -> list[CodeSymbol]:
        """Get all symbols defined in a specific file.

        Args:
            file_path: Relative path to the file

        Returns:
            List of symbols defined in the file
        """
        names = self._files.get(file_path, set())
        symbols: list[CodeSymbol] = []
        for name in names:
            symbols.extend(
                s for s in self._symbols[name] if s.file_path == file_path
            )
        return symbols

    def get_symbols_by_type(self, symbol_type: SymbolType) -> list[CodeSymbol]:
        """Get all symbols of a specific type.

        Args:
            symbol_type: The type of symbols to retrieve

        Returns:
            List of symbols matching the type
        """
        result: list[CodeSymbol] = []
        for symbols in self._symbols.values():
            result.extend(s for s in symbols if s.type == symbol_type)
        return result

    def file_exists(self, file_path: str) -> bool:
        """Check if a file path is known in the symbol table.

        Args:
            file_path: Relative path to check

        Returns:
            True if the file is known, False otherwise
        """
        return file_path in self._known_files

    def add_known_file(self, file_path: str) -> None:
        """Add a file path to the known files set.

        Useful for tracking files that exist but have no symbols
        (e.g., data files, configuration files).

        Args:
            file_path: Relative path to add
        """
        self._known_files.add(file_path)

    def get_all_known_files(self) -> list[str]:
        """Get all known file paths.

        Returns:
            List of all known file paths
        """
        return sorted(self._known_files)

    def get_all_symbols(self) -> list[CodeSymbol]:
        """Get all symbols in the table.

        Returns:
            List of all symbols
        """
        result = []
        for symbols in self._symbols.values():
            result.extend(symbols)
        return result

    def symbol_count(self) -> int:
        """Get the total number of symbols in the table.

        Returns:
            Total symbol count
        """
        return sum(len(symbols) for symbols in self._symbols.values())

    def file_count(self) -> int:
        """Get the number of files with symbols.

        Returns:
            File count
        """
        return len(self._files)

    def clear(self) -> None:
        """Clear all symbols from the table."""
        self._symbols.clear()
        self._files.clear()
        self._qualified.clear()
        self._known_files.clear()

    def remove_file(self, file_path: str) -> None:
        """Remove all symbols and file references for a given file."""
        symbol_names = self._files.pop(file_path, set())
        for name in symbol_names:
            symbols = [s for s in self._symbols.get(name, []) if s.file_path != file_path]
            if symbols:
                self._symbols[name] = symbols
            else:
                self._symbols.pop(name, None)

        for qualified_name, symbol in list(self._qualified.items()):
            if symbol.file_path == file_path:
                self._qualified.pop(qualified_name, None)

        self._known_files.discard(file_path)

    def _make_qualified_name(self, symbol: CodeSymbol) -> str:
        """Create a qualified name for a symbol.

        Format: 'parent.name' if parent exists, otherwise just 'name'.

        Args:
            symbol: The symbol to create a qualified name for

        Returns:
            The qualified name string
        """
        if symbol.qualified_name:
            return symbol.qualified_name
        if symbol.parent:
            return f"{symbol.parent}.{symbol.name}"
        return symbol.name

    def to_dict(self) -> dict:
        """Serialize the symbol table to a dictionary.

        Used for Redis caching and persistence.

        Returns:
            Dictionary representation of the symbol table
        """
        return {
            "tenant_id": self.tenant_id,
            "repo_path": self.repo_path,
            "symbols": [
                {
                    "name": s.name,
                    "type": s.type.value,
                    "scope": s.scope.value,
                    "file_path": s.file_path,
                    "line_start": s.line_start,
                    "line_end": s.line_end,
                    "signature": s.signature,
                    "parent": s.parent,
                    "docstring": s.docstring,
                    "qualified_name": s.qualified_name,
                }
                for s in self.get_all_symbols()
            ],
            "known_files": list(self._known_files),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SymbolTable":
        """Deserialize a symbol table from a dictionary.

        Args:
            data: Dictionary representation from to_dict()

        Returns:
            Reconstructed SymbolTable instance
        """
        from .types import SymbolScope

        table = cls(data["tenant_id"], data["repo_path"])

        for sym_data in data.get("symbols", []):
            symbol = CodeSymbol(
                name=sym_data["name"],
                type=SymbolType(sym_data["type"]),
                scope=SymbolScope(sym_data["scope"]),
                file_path=sym_data["file_path"],
                line_start=sym_data["line_start"],
                line_end=sym_data["line_end"],
                signature=sym_data.get("signature"),
                parent=sym_data.get("parent"),
                docstring=sym_data.get("docstring"),
                qualified_name=sym_data.get("qualified_name"),
            )
            table.add(symbol)

        for file_path in data.get("known_files", []):
            table.add_known_file(file_path)

        return table

    def to_json(self) -> str:
        """Serialize the symbol table to JSON.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "SymbolTable":
        """Deserialize a symbol table from JSON.

        Args:
            json_str: JSON string from to_json()

        Returns:
            Reconstructed SymbolTable instance
        """
        return cls.from_dict(json.loads(json_str))

    def get_cache_key(self) -> str:
        """Generate a cache key for this symbol table.

        The key includes the tenant ID and a hash of the repo path
        for unique identification.

        Returns:
            Cache key string
        """
        repo_hash = hashlib.sha256(self.repo_path.encode()).hexdigest()[:16]
        return f"codebase:{self.tenant_id}:symbol_table:{repo_hash}"


async def cache_symbol_table(
    redis_client: "RedisClient",  # type: ignore[name-defined]  # noqa: F821
    symbol_table: SymbolTable,
    ttl_seconds: int = 3600,
) -> None:
    """Cache a symbol table in Redis.

    Args:
        redis_client: Redis client instance
        symbol_table: SymbolTable to cache
        ttl_seconds: Time-to-live in seconds (default: 1 hour)
    """
    key = symbol_table.get_cache_key()
    data = symbol_table.to_json().encode("utf-8")
    await redis_client.client.setex(key, ttl_seconds, data)
    logger.info(
        "symbol_table_cached",
        key=key,
        symbol_count=symbol_table.symbol_count(),
        file_count=symbol_table.file_count(),
        ttl_seconds=ttl_seconds,
    )


async def get_cached_symbol_table(
    redis_client: "RedisClient",  # type: ignore[name-defined]  # noqa: F821
    tenant_id: str,
    repo_path: str,
) -> Optional[SymbolTable]:
    """Retrieve a cached symbol table from Redis.

    Args:
        redis_client: Redis client instance
        tenant_id: Tenant identifier
        repo_path: Repository path

    Returns:
        Cached SymbolTable if found, None otherwise
    """
    repo_hash = hashlib.sha256(repo_path.encode()).hexdigest()[:16]
    key = f"codebase:{tenant_id}:symbol_table:{repo_hash}"

    data = await redis_client.client.get(key)
    if data is None:
        return None

    try:
        json_str = data.decode("utf-8") if isinstance(data, bytes) else data
        table = SymbolTable.from_json(json_str)
        logger.info(
            "symbol_table_loaded_from_cache",
            key=key,
            symbol_count=table.symbol_count(),
            file_count=table.file_count(),
        )
        return table
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("symbol_table_cache_invalid", key=key, error=str(e))
        return None
