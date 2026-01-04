# Epic 15 Tech Spec: Codebase Intelligence

**Date:** 2025-12-31
**Updated:** 2026-01-04 (Full Technical Specification via epic-tech-context workflow)
**Status:** Contexted
**Epic Owner:** Product and Engineering

---

## Executive Summary

Epic 15 delivers codebase intelligence features that differentiate this platform for developers. The epic focuses on two key capabilities:

1. **Codebase Hallucination Detection (Story 15-1):** AST-based validation of LLM responses to catch references to non-existent code symbols, files, or API endpoints.

2. **Codebase RAG Context (Story 15-2):** Index code repositories as knowledge sources, enabling semantic search over functions, classes, and their relationships.

### Strategic Context

**Key Decision (2026-01-03):** REMOVED multimodal video/image processing. FOCUSED on codebase intelligence.

**Rationale:**
- YouTube transcript API (Epic 13) covers 90%+ of video RAG use cases
- Full video processing (CLIP + Whisper) has high complexity/cost, low ROI
- Codebase hallucination detection is a unique differentiator for developer platform

**Decision Document:** `docs/roadmap-decisions-2026-01-03.md`

---

## Technical Architecture

### High-Level Architecture

```
+-----------------------------------------------------------------------+
|                    CODEBASE INTELLIGENCE SYSTEM                         |
+-----------------------------------------------------------------------+
|                                                                         |
|  +---------------------------+     +-------------------------------+   |
|  |   HALLUCINATION DETECTOR  |     |      CODEBASE RAG INDEXER     |   |
|  +---------------------------+     +-------------------------------+   |
|  |                           |     |                               |   |
|  |  Response Text            |     |  Repository Path              |   |
|  |       |                   |     |       |                       |   |
|  |       v                   |     |       v                       |   |
|  |  +-------------+          |     |  +---------------+            |   |
|  |  | Symbol      |          |     |  | File Scanner  |            |   |
|  |  | Extractor   |          |     |  | (.gitignore)  |            |   |
|  |  +-------------+          |     |  +---------------+            |   |
|  |       |                   |     |       |                       |   |
|  |       v                   |     |       v                       |   |
|  |  +-------------+          |     |  +---------------+            |   |
|  |  | Symbol      |          |     |  | AST Parser    |            |   |
|  |  | Validator   |<---------|-----|>| (Tree-sitter) |            |   |
|  |  +-------------+          |     |  +---------------+            |   |
|  |       |                   |     |       |                       |   |
|  |       v                   |     |       v                       |   |
|  |  +-------------+          |     |  +---------------+            |   |
|  |  | File Path   |          |     |  | Symbol        |            |   |
|  |  | Validator   |          |     |  | Extractor     |            |   |
|  |  +-------------+          |     |  +---------------+            |   |
|  |       |                   |     |       |                       |   |
|  |       v                   |     |       v                       |   |
|  |  +-------------+          |     |  +---------------+            |   |
|  |  | API Matcher |          |     |  | Embedding     |            |   |
|  |  | (OpenAPI)   |          |     |  | Generator     |            |   |
|  |  +-------------+          |     |  +---------------+            |   |
|  |       |                   |     |       |                       |   |
|  |       v                   |     |       +--------+-------+      |   |
|  |  +-----------------+      |     |                |       |      |   |
|  |  | Validation      |      |     |       +--------v----+  |      |   |
|  |  | Report          |      |     |       | Vector Store|  |      |   |
|  |  +-----------------+      |     |       | (pgvector)  |  |      |   |
|  |                           |     |       +-------------+  |      |   |
|  +---------------------------+     |                        |      |   |
|                                    |       +----------------v---+  |   |
|                                    |       | Knowledge Graph    |  |   |
|                                    |       | (Neo4j/Graphiti)   |  |   |
|                                    |       +--------------------+  |   |
|                                    +-------------------------------+   |
|                                                                         |
+-----------------------------------------------------------------------+
```

### Technology Choices

| Component | Technology | Rationale |
|-----------|------------|-----------|
| AST Parsing | tree-sitter (via py-tree-sitter) | Multi-language support, incremental parsing, CST for exact positions |
| Python Fallback | Python ast module | Native, no dependencies for Python-only use cases |
| Symbol Extraction | Tree-sitter queries + custom extractors | Language-specific query patterns for functions, classes, etc. |
| Embeddings | Existing multi-provider (Epic 11-12) | text-embedding-3-small or configured embedding provider |
| Vector Store | pgvector (existing) | Already integrated for document chunks |
| Graph Store | Neo4j/Graphiti (existing) | Symbol relationships map naturally to graphs |
| Caching | Redis (existing) | Cache parsed ASTs for performance |

### New Dependencies

```toml
# backend/pyproject.toml additions for Epic 15
dependencies = [
  # ... existing deps ...
  # Epic 15 - Codebase Intelligence
  "tree-sitter>=0.23.0",           # Core parser
  "tree-sitter-python>=0.23.0",    # Python grammar
  "tree-sitter-javascript>=0.23.0", # JavaScript grammar
  "tree-sitter-typescript>=0.23.0", # TypeScript grammar
  "tree-sitter-language-pack>=0.1.0",  # Additional languages (optional)
  "gitignore-parser>=0.1.11",      # .gitignore parsing
]
```

---

## Story 15-1: Implement Codebase Hallucination Detector

### Objective

Detect LLM responses that reference non-existent code symbols, files, or API endpoints.

### Why This Matters

LLMs frequently hallucinate when generating code:
- Non-existent function names (e.g., `user.getFullName()` when method is `user.get_full_name()`)
- Incorrect file paths (e.g., `src/utils/helpers.py` when file is `src/util/helpers.py`)
- Made-up API endpoints (e.g., `/api/v1/users/search` when endpoint is `/api/v1/users/query`)
- Wrong class/method signatures
- Non-existent imports

For a developer platform, catching these hallucinations is critical for trust.

### Detection Capabilities

| Element | Detection Method | Confidence Level |
|---------|------------------|------------------|
| Functions/Methods | AST parsing + symbol table lookup | High |
| Classes | AST parsing + symbol table lookup | High |
| File paths | Filesystem validation | High |
| API endpoints | OpenAPI spec matching (if available) | High |
| Import statements | Module existence check + pyproject.toml/package.json | Medium |
| Variables | AST scope analysis | Medium |
| String literals (paths) | Heuristic pattern matching | Low |

### Technical Design

#### Module Structure

```
backend/src/agentic_rag_backend/
+-- codebase/                           # NEW: Codebase intelligence module
|   +-- __init__.py
|   +-- parser.py                       # Tree-sitter wrapper, AST parsing
|   +-- symbol_table.py                 # In-memory symbol table management
|   +-- extractors/
|   |   +-- __init__.py
|   |   +-- base.py                     # BaseSymbolExtractor abstract class
|   |   +-- python_extractor.py         # Python-specific extraction
|   |   +-- typescript_extractor.py     # TypeScript/JavaScript extraction
|   +-- validators/
|   |   +-- __init__.py
|   |   +-- symbol_validator.py         # Symbol existence validation
|   |   +-- path_validator.py           # File path validation
|   |   +-- api_validator.py            # OpenAPI endpoint matching
|   +-- detector.py                     # Main HallucinationDetector class
|   +-- types.py                        # Pydantic models for codebase types
```

#### Core Classes

```python
# backend/src/agentic_rag_backend/codebase/types.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class SymbolType(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    VARIABLE = "variable"
    IMPORT = "import"
    MODULE = "module"
    CONSTANT = "constant"
    TYPE_ALIAS = "type_alias"
    INTERFACE = "interface"  # TypeScript

class SymbolScope(str, Enum):
    GLOBAL = "global"
    CLASS = "class"
    FUNCTION = "function"
    MODULE = "module"

@dataclass(frozen=True)
class CodeSymbol:
    """A code symbol extracted from AST."""
    name: str
    type: SymbolType
    scope: SymbolScope
    file_path: str
    line_start: int
    line_end: int
    signature: Optional[str] = None  # For functions/methods
    parent: Optional[str] = None     # Parent class/module
    docstring: Optional[str] = None

@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a symbol reference."""
    symbol_name: str
    is_valid: bool
    confidence: float  # 0.0 - 1.0
    reason: str
    suggestions: list[str]  # Similar symbols if invalid
    location_in_response: Optional[tuple[int, int]] = None  # (start, end) positions

@dataclass
class HallucinationReport:
    """Complete hallucination detection report."""
    total_symbols_checked: int
    valid_symbols: int
    invalid_symbols: int
    uncertain_symbols: int
    validation_results: list[ValidationResult]
    files_checked: list[str]
    processing_time_ms: int
    confidence_score: float  # Overall confidence (0.0 - 1.0)
```

#### Symbol Table Implementation

```python
# backend/src/agentic_rag_backend/codebase/symbol_table.py
from typing import Optional
import structlog
from .types import CodeSymbol, SymbolType

logger = structlog.get_logger(__name__)

class SymbolTable:
    """In-memory symbol table for a codebase."""

    def __init__(self, tenant_id: str, repo_path: str):
        self.tenant_id = tenant_id
        self.repo_path = repo_path
        self._symbols: dict[str, list[CodeSymbol]] = {}  # name -> symbols
        self._files: dict[str, set[str]] = {}  # file_path -> symbol names
        self._qualified: dict[str, CodeSymbol] = {}  # full.qualified.name -> symbol

    def add(self, symbol: CodeSymbol) -> None:
        """Add a symbol to the table."""
        # Index by name
        if symbol.name not in self._symbols:
            self._symbols[symbol.name] = []
        self._symbols[symbol.name].append(symbol)

        # Index by file
        if symbol.file_path not in self._files:
            self._files[symbol.file_path] = set()
        self._files[symbol.file_path].add(symbol.name)

        # Index by qualified name
        qualified = self._make_qualified_name(symbol)
        self._qualified[qualified] = symbol

    def lookup(self, name: str) -> list[CodeSymbol]:
        """Look up symbols by name (may return multiple for overloads)."""
        return self._symbols.get(name, [])

    def lookup_qualified(self, qualified_name: str) -> Optional[CodeSymbol]:
        """Look up by fully qualified name (e.g., 'MyClass.my_method')."""
        return self._qualified.get(qualified_name)

    def find_similar(self, name: str, limit: int = 5) -> list[str]:
        """Find similar symbol names using edit distance."""
        from difflib import get_close_matches
        all_names = list(self._symbols.keys())
        return get_close_matches(name, all_names, n=limit, cutoff=0.6)

    def get_symbols_in_file(self, file_path: str) -> list[CodeSymbol]:
        """Get all symbols defined in a file."""
        names = self._files.get(file_path, set())
        symbols = []
        for name in names:
            symbols.extend(self._symbols[name])
        return symbols

    def _make_qualified_name(self, symbol: CodeSymbol) -> str:
        """Create qualified name like 'module.Class.method'."""
        parts = []
        if symbol.parent:
            parts.append(symbol.parent)
        parts.append(symbol.name)
        return ".".join(parts)
```

#### Hallucination Detector

```python
# backend/src/agentic_rag_backend/codebase/detector.py
import re
import time
from typing import Optional
from enum import Enum
import structlog

from .types import HallucinationReport, ValidationResult, SymbolType
from .symbol_table import SymbolTable
from .validators.symbol_validator import SymbolValidator
from .validators.path_validator import PathValidator
from .validators.api_validator import APIValidator

logger = structlog.get_logger(__name__)

class DetectorMode(str, Enum):
    WARN = "warn"    # Annotate response with warnings
    BLOCK = "block"  # Reject response if hallucinations found

class HallucinationDetector:
    """Detect code hallucinations in LLM responses."""

    def __init__(
        self,
        symbol_table: SymbolTable,
        repo_path: str,
        mode: DetectorMode = DetectorMode.WARN,
        openapi_spec_path: Optional[str] = None,
    ):
        self.symbol_table = symbol_table
        self.repo_path = repo_path
        self.mode = mode

        self.symbol_validator = SymbolValidator(symbol_table)
        self.path_validator = PathValidator(repo_path)
        self.api_validator = APIValidator(openapi_spec_path) if openapi_spec_path else None

        # Regex patterns for extracting code references
        self._patterns = {
            "function_call": re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('),
            "method_call": re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\('),
            "class_name": re.compile(r'\b([A-Z][a-zA-Z0-9_]*)\b'),
            "import": re.compile(r'(?:from\s+|import\s+)([a-zA-Z_][a-zA-Z0-9_.]+)'),
            "file_path": re.compile(r'[`"\']([a-zA-Z0-9_./\\-]+\.[a-zA-Z]+)[`"\']'),
            "api_endpoint": re.compile(r'[`"\']/(api/[a-zA-Z0-9_/{}.-]+)[`"\']'),
        }

    async def validate_response(
        self,
        response_text: str,
        context: Optional[dict] = None,
    ) -> HallucinationReport:
        """Validate an LLM response for code hallucinations.

        Args:
            response_text: The LLM response text to validate
            context: Optional context (e.g., the query, file being discussed)

        Returns:
            HallucinationReport with validation results
        """
        start_time = time.perf_counter()
        results: list[ValidationResult] = []
        files_checked: set[str] = set()

        # Extract and validate symbols
        for pattern_name, pattern in self._patterns.items():
            matches = pattern.finditer(response_text)
            for match in matches:
                reference = match.group(1)
                location = (match.start(1), match.end(1))

                if pattern_name == "file_path":
                    result = self._validate_file_path(reference, location)
                    if result.is_valid:
                        files_checked.add(reference)
                elif pattern_name == "api_endpoint":
                    result = self._validate_api_endpoint(reference, location)
                elif pattern_name in ("function_call", "method_call"):
                    result = self._validate_symbol(reference, SymbolType.FUNCTION, location)
                elif pattern_name == "class_name":
                    result = self._validate_symbol(reference, SymbolType.CLASS, location)
                elif pattern_name == "import":
                    result = self._validate_import(reference, location)
                else:
                    continue

                results.append(result)

        # Calculate metrics
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        valid = sum(1 for r in results if r.is_valid)
        invalid = sum(1 for r in results if not r.is_valid and r.confidence > 0.7)
        uncertain = sum(1 for r in results if not r.is_valid and r.confidence <= 0.7)

        # Overall confidence based on proportion of valid symbols
        total = len(results)
        confidence_score = valid / total if total > 0 else 1.0

        report = HallucinationReport(
            total_symbols_checked=total,
            valid_symbols=valid,
            invalid_symbols=invalid,
            uncertain_symbols=uncertain,
            validation_results=results,
            files_checked=list(files_checked),
            processing_time_ms=processing_time_ms,
            confidence_score=confidence_score,
        )

        logger.info(
            "hallucination_detection_completed",
            total_checked=total,
            valid=valid,
            invalid=invalid,
            uncertain=uncertain,
            confidence=confidence_score,
            processing_time_ms=processing_time_ms,
        )

        return report

    def _validate_symbol(
        self,
        name: str,
        expected_type: SymbolType,
        location: tuple[int, int],
    ) -> ValidationResult:
        """Validate a symbol reference."""
        symbols = self.symbol_table.lookup(name)

        if symbols:
            # Symbol exists
            matching = [s for s in symbols if s.type == expected_type]
            if matching:
                return ValidationResult(
                    symbol_name=name,
                    is_valid=True,
                    confidence=1.0,
                    reason=f"Symbol '{name}' found as {expected_type.value}",
                    suggestions=[],
                    location_in_response=location,
                )
            else:
                # Wrong type
                actual_types = set(s.type.value for s in symbols)
                return ValidationResult(
                    symbol_name=name,
                    is_valid=False,
                    confidence=0.9,
                    reason=f"Symbol '{name}' exists but as {actual_types}, not {expected_type.value}",
                    suggestions=[f"{name} is a {t}" for t in actual_types],
                    location_in_response=location,
                )

        # Symbol not found - check for similar names
        similar = self.symbol_table.find_similar(name)
        return ValidationResult(
            symbol_name=name,
            is_valid=False,
            confidence=0.85,
            reason=f"Symbol '{name}' not found in codebase",
            suggestions=similar,
            location_in_response=location,
        )

    def _validate_file_path(
        self,
        path: str,
        location: tuple[int, int],
    ) -> ValidationResult:
        """Validate a file path reference."""
        return self.path_validator.validate(path, location)

    def _validate_api_endpoint(
        self,
        endpoint: str,
        location: tuple[int, int],
    ) -> ValidationResult:
        """Validate an API endpoint reference."""
        if self.api_validator:
            return self.api_validator.validate(endpoint, location)

        return ValidationResult(
            symbol_name=endpoint,
            is_valid=True,  # Can't validate without OpenAPI spec
            confidence=0.3,
            reason="No OpenAPI spec available for validation",
            suggestions=[],
            location_in_response=location,
        )

    def _validate_import(
        self,
        module_path: str,
        location: tuple[int, int],
    ) -> ValidationResult:
        """Validate an import statement."""
        # Check if it's a local module in the codebase
        file_path = module_path.replace(".", "/") + ".py"
        if self.path_validator.exists(file_path):
            return ValidationResult(
                symbol_name=module_path,
                is_valid=True,
                confidence=1.0,
                reason=f"Local module '{module_path}' found",
                suggestions=[],
                location_in_response=location,
            )

        # Check if it's a standard library or installed package
        # (Lower confidence since we can't always verify)
        return ValidationResult(
            symbol_name=module_path,
            is_valid=True,
            confidence=0.5,
            reason=f"Module '{module_path}' assumed to be external dependency",
            suggestions=[],
            location_in_response=location,
        )
```

### API Endpoints

```python
# backend/src/agentic_rag_backend/api/routes/codebase.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from ...codebase.detector import HallucinationDetector, DetectorMode
from ...codebase.types import HallucinationReport

router = APIRouter(prefix="/api/v1/codebase", tags=["codebase"])

class ValidateResponseRequest(BaseModel):
    response_text: str = Field(..., description="LLM response to validate")
    repo_path: Optional[str] = Field(None, description="Repository path (uses default if not set)")
    mode: str = Field("warn", description="Detection mode: warn or block")

class ValidateResponseResponse(BaseModel):
    valid: bool
    confidence: float
    total_checked: int
    invalid_count: int
    invalid_symbols: list[dict]
    suggestions: list[dict]
    processing_time_ms: int

@router.post("/validate-response", response_model=ValidateResponseResponse)
async def validate_llm_response(
    request: ValidateResponseRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """Validate an LLM response for code hallucinations.

    Returns validation report with invalid symbols and suggestions.
    """
    # Get or build symbol table for tenant's codebase
    symbol_table = await get_symbol_table(tenant_id, request.repo_path)

    detector = HallucinationDetector(
        symbol_table=symbol_table,
        repo_path=request.repo_path or get_default_repo_path(tenant_id),
        mode=DetectorMode(request.mode),
    )

    report = await detector.validate_response(request.response_text)

    if request.mode == "block" and report.invalid_symbols > 0:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Hallucinations detected",
                "invalid_count": report.invalid_symbols,
                "details": [r.__dict__ for r in report.validation_results if not r.is_valid],
            }
        )

    return ValidateResponseResponse(
        valid=report.invalid_symbols == 0,
        confidence=report.confidence_score,
        total_checked=report.total_symbols_checked,
        invalid_count=report.invalid_symbols,
        invalid_symbols=[
            {"name": r.symbol_name, "reason": r.reason}
            for r in report.validation_results if not r.is_valid
        ],
        suggestions=[
            {"name": r.symbol_name, "suggestions": r.suggestions}
            for r in report.validation_results if r.suggestions
        ],
        processing_time_ms=report.processing_time_ms,
    )
```

### Configuration

```bash
# Epic 15 - Codebase Hallucination Detector
HALLUCINATION_DETECTOR_ENABLED=true|false  # Default: false
HALLUCINATION_DETECTOR_MODE=warn|block     # Default: warn
HALLUCINATION_DETECTOR_LANGUAGES=python,typescript,javascript  # Supported languages
HALLUCINATION_DETECTOR_CONFIDENCE_THRESHOLD=0.7  # Min confidence to report (0.0-1.0)
HALLUCINATION_DETECTOR_CACHE_TTL_SECONDS=3600    # Symbol table cache TTL
```

### Acceptance Criteria

- [ ] Given an LLM response that references code elements, when validation runs, then AST and symbol search detect unknown classes, functions, or files
- [ ] The detector reports a warning with a list of missing symbols and similar alternatives
- [ ] Detection can be configured to block or annotate responses
- [ ] Supports Python, TypeScript, and JavaScript initially
- [ ] Detection is opt-in and configurable
- [ ] Symbol table is cached in Redis with configurable TTL
- [ ] Processing time < 100ms for typical responses (< 5000 chars)

---

## Story 15-2: Implement Codebase RAG Context

### Objective

Index a code repository as a knowledge source for RAG queries, enabling semantic search over functions, classes, and their relationships.

### Use Cases

- "How does authentication work in this codebase?"
- "What functions call the UserService?"
- "Explain the data flow from API to database"
- "Find all error handling patterns in the codebase"

### Technical Design

#### Indexing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       CODEBASE INDEXING PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Repository Path                                                          │
│       │                                                                   │
│       v                                                                   │
│  ┌──────────────────┐                                                    │
│  │ File Scanner     │  .gitignore, configurable exclusions               │
│  │ (gitignore-parser)│                                                    │
│  └──────────────────┘                                                    │
│       │                                                                   │
│       v                                                                   │
│  ┌──────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │ Python Files     │   │ TypeScript      │   │ JavaScript      │       │
│  │ (.py)            │   │ (.ts, .tsx)     │   │ (.js, .jsx)     │       │
│  └──────────────────┘   └─────────────────┘   └─────────────────┘       │
│       │                        │                      │                   │
│       v                        v                      v                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Tree-sitter AST Parser                         │    │
│  │    (py-tree-sitter with language grammars)                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                   │
│       v                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Symbol Extractor                               │    │
│  │    - Functions with signatures and docstrings                    │    │
│  │    - Classes with methods and attributes                         │    │
│  │    - Imports and dependencies                                    │    │
│  │    - Call graph analysis                                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                   │
│       ├──────────────────────┬───────────────────────────┐               │
│       v                      v                           v               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ Chunk with   │    │ Embed with   │    │ Build Relationships      │   │
│  │ Context      │    │ Provider     │    │ (calls, imports, extends)│   │
│  │ (class+method)│    │              │    │                          │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│       │                      │                           │               │
│       v                      v                           v               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Multi-Store Indexing                            │   │
│  │    pgvector (embeddings) + Neo4j/Graphiti (relationships)         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Module Structure

```
backend/src/agentic_rag_backend/
+-- codebase/
|   +-- indexing/
|   |   +-- __init__.py
|   |   +-- scanner.py           # File discovery with .gitignore
|   |   +-- ast_parser.py        # Tree-sitter parsing wrapper
|   |   +-- symbol_extractor.py  # Extract symbols from AST
|   |   +-- chunker.py           # Create semantic chunks from code
|   |   +-- embedder.py          # Generate embeddings for code
|   |   +-- graph_builder.py     # Build call graph / dependency graph
|   |   +-- indexer.py           # Main CodebaseIndexer orchestrator
|   +-- retrieval/
|   |   +-- __init__.py
|   |   +-- code_search.py       # Semantic code search
|   |   +-- symbol_search.py     # Symbol-specific search
```

#### Core Classes

```python
# backend/src/agentic_rag_backend/codebase/indexing/indexer.py
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, AsyncGenerator
import structlog

from ..types import CodeSymbol
from ..symbol_table import SymbolTable
from .scanner import FileScanner
from .ast_parser import ASTParser
from .symbol_extractor import SymbolExtractor
from .chunker import CodeChunker
from .embedder import CodeEmbedder
from .graph_builder import CodeGraphBuilder

logger = structlog.get_logger(__name__)

@dataclass
class IndexingResult:
    """Result of codebase indexing."""
    tenant_id: str
    repo_path: str
    files_indexed: int
    symbols_extracted: int
    chunks_created: int
    relationships_created: int
    processing_time_ms: int
    errors: list[str]

class CodebaseIndexer:
    """Index a codebase for RAG queries."""

    def __init__(
        self,
        tenant_id: str,
        repo_path: str,
        embedding_provider: str = "openai",
        languages: list[str] = None,
        exclude_patterns: list[str] = None,
    ):
        self.tenant_id = tenant_id
        self.repo_path = Path(repo_path)
        self.languages = languages or ["python", "typescript", "javascript"]
        self.exclude_patterns = exclude_patterns or [
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.venv/**",
            "**/dist/**",
            "**/build/**",
        ]

        self.scanner = FileScanner(self.repo_path, self.exclude_patterns)
        self.parser = ASTParser(self.languages)
        self.extractor = SymbolExtractor()
        self.chunker = CodeChunker()
        self.embedder = CodeEmbedder(embedding_provider)
        self.graph_builder = CodeGraphBuilder()

    async def index_full(self) -> IndexingResult:
        """Perform full codebase indexing.

        This scans all files and rebuilds the index from scratch.
        """
        import time
        start_time = time.perf_counter()
        errors: list[str] = []

        files_indexed = 0
        symbols_extracted = 0
        chunks_created = 0
        relationships_created = 0

        # Build symbol table for the repo
        symbol_table = SymbolTable(self.tenant_id, str(self.repo_path))

        # Scan files
        files = list(self.scanner.scan())
        logger.info("codebase_scan_complete", file_count=len(files))

        for file_path in files:
            try:
                # Parse AST
                ast_tree = await self.parser.parse_file(file_path)
                if not ast_tree:
                    continue

                # Extract symbols
                symbols = self.extractor.extract(ast_tree, file_path)
                for symbol in symbols:
                    symbol_table.add(symbol)
                    symbols_extracted += 1

                # Create chunks with context
                chunks = self.chunker.chunk_file(file_path, symbols)
                for chunk in chunks:
                    # Generate embedding
                    embedding = await self.embedder.embed(chunk.content)
                    # Store in pgvector
                    await self._store_chunk(chunk, embedding)
                    chunks_created += 1

                # Build relationships
                relationships = self.graph_builder.build_relationships(symbols)
                for rel in relationships:
                    await self._store_relationship(rel)
                    relationships_created += 1

                files_indexed += 1

            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")
                logger.error("codebase_index_file_error", file=file_path, error=str(e))

        # Cache symbol table in Redis
        await self._cache_symbol_table(symbol_table)

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        result = IndexingResult(
            tenant_id=self.tenant_id,
            repo_path=str(self.repo_path),
            files_indexed=files_indexed,
            symbols_extracted=symbols_extracted,
            chunks_created=chunks_created,
            relationships_created=relationships_created,
            processing_time_ms=processing_time_ms,
            errors=errors,
        )

        logger.info(
            "codebase_indexing_complete",
            files=files_indexed,
            symbols=symbols_extracted,
            chunks=chunks_created,
            relationships=relationships_created,
            processing_time_ms=processing_time_ms,
            error_count=len(errors),
        )

        return result

    async def index_incremental(self, changed_files: list[str]) -> IndexingResult:
        """Perform incremental indexing for changed files only.

        This is called when file changes are detected.
        """
        # Only re-index the changed files
        # Invalidate affected symbols and re-extract
        pass

    async def _store_chunk(self, chunk, embedding) -> None:
        """Store a code chunk in pgvector."""
        # Implementation uses existing pgvector patterns from Epic 3
        pass

    async def _store_relationship(self, relationship) -> None:
        """Store a code relationship in Neo4j/Graphiti."""
        # Implementation uses existing Graphiti patterns from Epic 5
        pass

    async def _cache_symbol_table(self, symbol_table: SymbolTable) -> None:
        """Cache symbol table in Redis."""
        pass
```

#### Code Chunker with Context

```python
# backend/src/agentic_rag_backend/codebase/indexing/chunker.py
from dataclasses import dataclass
from typing import Optional
from ..types import CodeSymbol, SymbolType

@dataclass
class CodeChunk:
    """A chunk of code with context for embedding."""
    content: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: Optional[str]
    symbol_type: Optional[SymbolType]
    parent_context: Optional[str]  # Class definition for methods
    imports_context: Optional[str]  # Relevant imports

class CodeChunker:
    """Create semantic chunks from code symbols."""

    def __init__(
        self,
        max_chunk_size: int = 1000,
        include_class_context: bool = True,
        include_imports: bool = True,
    ):
        self.max_chunk_size = max_chunk_size
        self.include_class_context = include_class_context
        self.include_imports = include_imports

    def chunk_file(
        self,
        file_path: str,
        symbols: list[CodeSymbol],
    ) -> list[CodeChunk]:
        """Create chunks from file symbols.

        Strategy (following Qodo's approach):
        - Each function/method becomes a chunk
        - Include class definition header for methods
        - Include relevant imports
        - For large classes, chunk individual methods separately
        """
        chunks: list[CodeChunk] = []

        # Group symbols by parent
        class_symbols: dict[str, list[CodeSymbol]] = {}
        standalone: list[CodeSymbol] = []

        for symbol in symbols:
            if symbol.parent:
                if symbol.parent not in class_symbols:
                    class_symbols[symbol.parent] = []
                class_symbols[symbol.parent].append(symbol)
            else:
                standalone.append(symbol)

        # Chunk standalone functions
        for symbol in standalone:
            if symbol.type in (SymbolType.FUNCTION, SymbolType.CLASS):
                chunks.append(self._create_chunk(symbol, None))

        # Chunk class methods with class context
        for class_name, methods in class_symbols.items():
            class_header = self._get_class_header(class_name, symbols)
            for method in methods:
                chunks.append(self._create_chunk(method, class_header))

        return chunks

    def _create_chunk(
        self,
        symbol: CodeSymbol,
        parent_context: Optional[str],
    ) -> CodeChunk:
        """Create a chunk for a symbol with context."""
        # Read file content
        content = self._read_symbol_content(symbol)

        # Prepend parent context if available
        if parent_context and self.include_class_context:
            full_content = f"{parent_context}\n\n{content}"
        else:
            full_content = content

        # Truncate if too long
        if len(full_content) > self.max_chunk_size:
            full_content = full_content[:self.max_chunk_size] + "\n# ... truncated"

        return CodeChunk(
            content=full_content,
            file_path=symbol.file_path,
            start_line=symbol.line_start,
            end_line=symbol.line_end,
            symbol_name=symbol.name,
            symbol_type=symbol.type,
            parent_context=parent_context,
            imports_context=None,  # TODO: Extract imports
        )

    def _read_symbol_content(self, symbol: CodeSymbol) -> str:
        """Read the content of a symbol from its file."""
        with open(symbol.file_path, "r") as f:
            lines = f.readlines()
        return "".join(lines[symbol.line_start - 1:symbol.line_end])

    def _get_class_header(
        self,
        class_name: str,
        symbols: list[CodeSymbol],
    ) -> Optional[str]:
        """Get the class definition header (first few lines)."""
        for symbol in symbols:
            if symbol.name == class_name and symbol.type == SymbolType.CLASS:
                return symbol.signature or f"class {class_name}:"
        return None
```

### Graph Relationships

The codebase indexer creates the following relationship types in Neo4j/Graphiti:

| Relationship | Description | Example |
|-------------|-------------|---------|
| `CALLS` | Function/method call | `UserService.create_user` CALLS `validate_email` |
| `IMPORTS` | Module import | `auth.py` IMPORTS `jwt` |
| `EXTENDS` | Class inheritance | `AdminUser` EXTENDS `User` |
| `IMPLEMENTS` | Interface implementation | `PostgresRepo` IMPLEMENTS `Repository` |
| `DEFINED_IN` | Symbol defined in file | `create_user` DEFINED_IN `services/user.py` |
| `USES_TYPE` | Type annotation usage | `get_user` USES_TYPE `User` |

### API Endpoints

```python
# backend/src/agentic_rag_backend/api/routes/codebase.py (continued)

class IndexCodebaseRequest(BaseModel):
    repo_path: str = Field(..., description="Path to repository to index")
    languages: list[str] = Field(["python", "typescript", "javascript"])
    incremental: bool = Field(False, description="Only index changed files")

class IndexCodebaseResponse(BaseModel):
    files_indexed: int
    symbols_extracted: int
    chunks_created: int
    relationships_created: int
    processing_time_ms: int
    errors: list[str]

class CodeSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query about the codebase")
    limit: int = Field(10, ge=1, le=50)
    include_relationships: bool = Field(True)

class CodeSearchResult(BaseModel):
    symbol_name: str
    symbol_type: str
    file_path: str
    line_start: int
    line_end: int
    content: str
    score: float
    relationships: list[dict]

@router.post("/index", response_model=IndexCodebaseResponse)
async def index_codebase(
    request: IndexCodebaseRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id),
):
    """Index a codebase for RAG queries.

    For large repositories, this runs in the background.
    """
    indexer = CodebaseIndexer(
        tenant_id=tenant_id,
        repo_path=request.repo_path,
        languages=request.languages,
    )

    if request.incremental:
        # Get changed files from git or filesystem
        changed = await get_changed_files(request.repo_path)
        result = await indexer.index_incremental(changed)
    else:
        result = await indexer.index_full()

    return IndexCodebaseResponse(**result.__dict__)

@router.post("/search", response_model=list[CodeSearchResult])
async def search_codebase(
    request: CodeSearchRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """Search the indexed codebase using natural language.

    Uses hybrid search (vector + graph) for best results.
    """
    # Use existing hybrid retrieval patterns from Epic 3
    results = await hybrid_code_search(
        tenant_id=tenant_id,
        query=request.query,
        limit=request.limit,
        include_relationships=request.include_relationships,
    )

    return results
```

### Configuration

```bash
# Epic 15 - Codebase RAG Context
CODEBASE_RAG_ENABLED=true|false           # Default: false
CODEBASE_LANGUAGES=python,typescript,javascript  # Languages to index
CODEBASE_EXCLUDE_PATTERNS=["**/node_modules/**","**/__pycache__/**"]  # JSON array
CODEBASE_MAX_CHUNK_SIZE=1000              # Max chars per code chunk
CODEBASE_INCLUDE_CLASS_CONTEXT=true       # Include class header with methods
CODEBASE_INCREMENTAL_INDEXING=true        # Enable file change detection
CODEBASE_INDEX_CACHE_TTL_SECONDS=86400    # 24 hours
```

### Acceptance Criteria

- [ ] Given a code repository, when indexing runs, then symbols are extracted and embedded
- [ ] Symbol relationships are captured in the knowledge graph (e.g., "function A calls function B")
- [ ] Queries about the codebase return relevant code context
- [ ] Indexing respects .gitignore and configurable exclusion patterns
- [ ] Incremental indexing is supported for changed files
- [ ] Indexing completes within 5 minutes for a 100K LOC repository
- [ ] Search results include file paths, line numbers, and code snippets

---

## Integration with Existing Systems

### Graphiti Integration

Codebase symbols and relationships are stored in Graphiti alongside document knowledge:

```python
# Custom entity types for code symbols
class CodeFunction(EntityModel):
    """A function or method in the codebase."""
    language: str = Field(description="Programming language")
    signature: str = Field(description="Function signature")
    is_async: bool = Field(False, description="Whether function is async")

class CodeClass(EntityModel):
    """A class definition in the codebase."""
    language: str = Field(description="Programming language")
    base_classes: list[str] = Field(default_factory=list)
    is_abstract: bool = Field(False)

class CodeModule(EntityModel):
    """A module/file in the codebase."""
    language: str = Field(description="Programming language")
    file_path: str = Field(description="Relative file path")
```

### Hybrid Retrieval Integration

Code search uses the existing hybrid retrieval pipeline (Epic 3 + Epic 12):

```python
# Integration with existing retrieval
async def hybrid_code_search(
    tenant_id: str,
    query: str,
    limit: int = 10,
    include_relationships: bool = True,
) -> list[CodeSearchResult]:
    """Hybrid search over code with reranking and grading."""

    # Stage 1: Vector search over code chunks
    vector_results = await vector_search_code(
        tenant_id=tenant_id,
        query=query,
        limit=limit * 2,  # Get more for reranking
    )

    # Stage 2: Rerank with cross-encoder (Epic 12)
    if settings.reranker_enabled:
        reranked = await rerank_results(query, vector_results)
    else:
        reranked = vector_results

    # Stage 3: Grade results (Epic 12)
    if settings.grader_enabled:
        graded = await grade_results(query, reranked[:limit])
        if not graded.passed:
            # Could expand to include more context
            pass

    # Stage 4: Enrich with graph relationships
    if include_relationships:
        for result in reranked[:limit]:
            result.relationships = await get_symbol_relationships(
                tenant_id=tenant_id,
                symbol_name=result.symbol_name,
            )

    return reranked[:limit]
```

---

## Data Models and Schemas

### Database Schema (PostgreSQL)

```sql
-- New table for code chunks
CREATE TABLE code_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    repo_path TEXT NOT NULL,
    file_path TEXT NOT NULL,
    symbol_name TEXT,
    symbol_type TEXT,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),  -- Using pgvector
    parent_context TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT fk_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- Index for efficient tenant-scoped queries
CREATE INDEX idx_code_chunks_tenant ON code_chunks(tenant_id);
CREATE INDEX idx_code_chunks_repo ON code_chunks(tenant_id, repo_path);
CREATE INDEX idx_code_chunks_symbol ON code_chunks(tenant_id, symbol_name);

-- Vector similarity index
CREATE INDEX idx_code_chunks_embedding ON code_chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### Neo4j Schema

```cypher
// Node labels for code entities
(:CodeSymbol {
    name: String,
    type: String,  // function, class, method, etc.
    language: String,
    filePath: String,
    lineStart: Integer,
    lineEnd: Integer,
    signature: String,
    docstring: String,
    tenantId: String
})

// Relationship types
-[:CALLS {lineNumber: Integer}]->
-[:IMPORTS]->
-[:EXTENDS]->
-[:IMPLEMENTS]->
-[:DEFINED_IN]->
-[:USES_TYPE]->

// Indexes
CREATE INDEX code_symbol_name FOR (s:CodeSymbol) ON (s.name);
CREATE INDEX code_symbol_tenant FOR (s:CodeSymbol) ON (s.tenantId);
CREATE INDEX code_symbol_file FOR (s:CodeSymbol) ON (s.filePath);
```

---

## Testing Strategy

### Unit Tests

```python
# backend/tests/codebase/test_parser.py
@pytest.fixture
def sample_python_code():
    return '''
class UserService:
    """Service for user operations."""

    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        validate_email(email)
        return User(name=name, email=email)

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.repo.find(user_id)

def validate_email(email: str) -> bool:
    """Validate email format."""
    return "@" in email
'''

def test_extract_functions(sample_python_code):
    parser = ASTParser(["python"])
    extractor = SymbolExtractor()

    ast_tree = parser.parse_string(sample_python_code, "python")
    symbols = extractor.extract(ast_tree, "test.py")

    # Should find: UserService, create_user, get_user, validate_email
    assert len(symbols) == 4

    class_symbol = next(s for s in symbols if s.name == "UserService")
    assert class_symbol.type == SymbolType.CLASS

    method_symbols = [s for s in symbols if s.parent == "UserService"]
    assert len(method_symbols) == 2

def test_hallucination_detection():
    # Build symbol table
    symbol_table = SymbolTable("test-tenant", "/test/repo")
    symbol_table.add(CodeSymbol(
        name="create_user",
        type=SymbolType.FUNCTION,
        scope=SymbolScope.GLOBAL,
        file_path="services/user.py",
        line_start=10,
        line_end=15,
    ))

    detector = HallucinationDetector(
        symbol_table=symbol_table,
        repo_path="/test/repo",
    )

    # Test with hallucinated function
    response = "You can use the `create_users()` function to add multiple users."
    report = await detector.validate_response(response)

    assert report.invalid_symbols == 1
    assert "create_user" in report.validation_results[0].suggestions
```

### Integration Tests

```python
# backend/tests/integration/test_codebase_intelligence.py
@pytest.mark.integration
async def test_full_indexing_pipeline():
    """Test complete codebase indexing and search."""
    indexer = CodebaseIndexer(
        tenant_id="test-tenant",
        repo_path="./tests/fixtures/sample_repo",
        languages=["python"],
    )

    result = await indexer.index_full()

    assert result.files_indexed > 0
    assert result.symbols_extracted > 0
    assert result.chunks_created > 0
    assert len(result.errors) == 0

@pytest.mark.integration
async def test_code_search_with_reranking():
    """Test code search with hybrid retrieval."""
    results = await hybrid_code_search(
        tenant_id="test-tenant",
        query="How do I create a new user?",
        limit=5,
    )

    assert len(results) > 0
    # Top result should be related to user creation
    assert "user" in results[0].symbol_name.lower()
```

---

## Migration and Deployment

### Phase 1: Feature Flag Introduction

1. Add configuration flags (disabled by default)
2. Add new dependencies to pyproject.toml
3. Create database migrations for new tables

### Phase 2: Core Implementation

1. Implement AST parser with tree-sitter
2. Implement symbol extraction for Python
3. Implement hallucination detector
4. Add API endpoints

### Phase 3: Extended Language Support

1. Add TypeScript/JavaScript support
2. Add incremental indexing
3. Performance optimization

### Phase 4: Production Hardening

1. Add comprehensive test coverage
2. Benchmark and optimize
3. Documentation and examples

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Hallucination detection accuracy | >= 80% | Precision on evaluation set of known hallucinations |
| Detection latency | < 100ms | P95 for typical responses (< 5000 chars) |
| Indexing speed | 100K LOC in 5 min | Benchmark on reference codebase |
| Search precision | >= 70% | Relevant results in top 5 |
| Symbol extraction recall | >= 90% | Symbols found vs. manual count |

---

## Dependencies

- **Epic 3:** Hybrid retrieval infrastructure (pgvector, vector search)
- **Epic 5:** Graphiti for symbol relationships
- **Epic 11-12:** Multi-provider embeddings, reranking, grading
- **Epic 13:** Ingestion patterns (file scanning, async processing)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| AST parsing complexity varies by language | Medium | Start with Python, add languages incrementally |
| Large codebases require significant indexing time | Medium | Implement incremental indexing, background processing |
| Tree-sitter grammar maintenance | Low | Use official tree-sitter grammars with good community support |
| Symbol extraction may miss dynamic code | Medium | Document limitations, focus on static analysis |
| False positives in hallucination detection | Medium | Configurable confidence thresholds, "warn" mode default |

---

## References

- [Tree-sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [py-tree-sitter](https://github.com/tree-sitter/py-tree-sitter)
- [CocoIndex Codebase Indexing](https://cocoindex.io/blogs/index-code-base-for-rag)
- [Qodo RAG for Large Codebases](https://www.qodo.ai/blog/rag-for-large-scale-code-repos/)
- [LanceDB Building RAG on Codebases](https://blog.lancedb.com/building-rag-on-codebases-part-2/)
- [Package Hallucinations Research (USENIX 2025)](https://www.usenix.org/publications/loginonline/we-have-package-you-comprehensive-analysis-package-hallucinations-code)
- `docs/roadmap-decisions-2026-01-03.md` - Decision rationale
- `_bmad-output/prd.md`
- `_bmad-output/architecture.md`
- `_bmad-output/project-planning-artifacts/epics.md`
- `docs/recommendations_2025.md`
