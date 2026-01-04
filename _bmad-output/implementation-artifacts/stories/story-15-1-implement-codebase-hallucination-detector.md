# Story 15-1: Implement Codebase Hallucination Detector

**Status:** done
**Epic:** 15 - Codebase Intelligence
**Priority:** High
**Complexity:** Medium-High (4-5 days estimated)

---

## User Story

As a **developer using the RAG platform**,
I want **LLM responses validated against actual codebase symbols, files, and API endpoints**,
So that **I can trust that code suggestions reference real elements in my codebase rather than hallucinated constructs**.

---

## Background

LLMs frequently hallucinate when generating code-related content:
- Non-existent function names (e.g., `user.getFullName()` when method is `user.get_full_name()`)
- Incorrect file paths (e.g., `src/utils/helpers.py` when file is `src/util/helpers.py`)
- Made-up API endpoints (e.g., `/api/v1/users/search` when endpoint is `/api/v1/users/query`)
- Wrong class/method signatures
- Non-existent imports

For a developer platform, catching these hallucinations is critical for trust and code quality.

---

## Acceptance Criteria

### AC-1: Symbol Validation
- Given an LLM response that references function names
- When the hallucination detector processes the response
- Then it identifies functions that exist in the codebase symbol table
- And flags functions that do not exist with suggestions for similar symbols

### AC-2: Class Validation
- Given an LLM response that references class names
- When the hallucination detector processes the response
- Then it validates classes against the AST-extracted symbol table
- And reports classes with incorrect types (e.g., referenced as function but is a class)

### AC-3: File Path Validation
- Given an LLM response that contains file path references
- When the hallucination detector processes the response
- Then it validates paths against the actual filesystem
- And suggests similar paths for non-existent files

### AC-4: API Endpoint Validation
- Given an LLM response that references API endpoints (e.g., `/api/v1/users`)
- When an OpenAPI spec is available
- Then the detector validates endpoints against the spec
- And reports non-matching endpoints with alternatives

### AC-5: Import Statement Validation
- Given an LLM response that contains import statements
- When the hallucination detector processes the response
- Then it validates local module imports against the codebase
- And validates external dependencies against pyproject.toml/package.json

### AC-6: Detector Modes
- Given the detector is configured with mode `warn`
- When hallucinations are detected
- Then the response is annotated with warnings but not blocked

- Given the detector is configured with mode `block`
- When hallucinations are detected
- Then the response is rejected with HTTP 422 and detailed error information

### AC-7: Detection Performance
- Given a typical LLM response (< 5000 characters)
- When validation runs
- Then processing completes in < 100ms (P95)

### AC-8: Symbol Table Caching
- Given a codebase has been indexed
- When the symbol table is built
- Then it is cached in Redis with configurable TTL
- And subsequent validations use the cached table

### AC-9: Multi-Tenancy
- Given multiple tenants using the platform
- When symbol tables are built and cached
- Then each tenant's symbols are isolated by tenant_id

### AC-10: API Endpoint Availability
- Given the backend is running with hallucination detection enabled
- When a client calls `POST /api/v1/codebase/validate-response`
- Then it receives a structured validation report

---

## Technical Details

### Module Structure

Create new `codebase/` module under backend:

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

### Core Types (types.py)

```python
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
    name: str
    type: SymbolType
    scope: SymbolScope
    file_path: str
    line_start: int
    line_end: int
    signature: Optional[str] = None
    parent: Optional[str] = None
    docstring: Optional[str] = None

@dataclass(frozen=True)
class ValidationResult:
    symbol_name: str
    is_valid: bool
    confidence: float  # 0.0 - 1.0
    reason: str
    suggestions: list[str]
    location_in_response: Optional[tuple[int, int]] = None

@dataclass
class HallucinationReport:
    total_symbols_checked: int
    valid_symbols: int
    invalid_symbols: int
    uncertain_symbols: int
    validation_results: list[ValidationResult]
    files_checked: list[str]
    processing_time_ms: int
    confidence_score: float  # Overall confidence (0.0 - 1.0)
```

### Main Components

1. **SymbolTable** (`symbol_table.py`):
   - In-memory symbol storage with multiple indexes
   - Lookup by name, qualified name, file
   - Similar symbol suggestions using difflib

2. **HallucinationDetector** (`detector.py`):
   - Regex patterns for extracting code references
   - Validates symbols, paths, API endpoints, imports
   - Configurable mode (warn/block)
   - Returns structured HallucinationReport

3. **Extractors** (`extractors/`):
   - `BaseSymbolExtractor`: Abstract base with interface
   - `PythonExtractor`: Uses tree-sitter-python
   - `TypeScriptExtractor`: Uses tree-sitter-typescript

4. **Validators** (`validators/`):
   - `SymbolValidator`: Symbol table lookups
   - `PathValidator`: Filesystem validation
   - `APIValidator`: OpenAPI spec matching

### API Endpoints

```python
# POST /api/v1/codebase/validate-response
class ValidateResponseRequest(BaseModel):
    response_text: str
    repo_path: Optional[str] = None
    mode: str = "warn"  # warn | block

class ValidateResponseResponse(BaseModel):
    valid: bool
    confidence: float
    total_checked: int
    invalid_count: int
    invalid_symbols: list[dict]
    suggestions: list[dict]
    processing_time_ms: int
```

### Configuration

Environment variables:

```bash
HALLUCINATION_DETECTOR_ENABLED=true|false  # Default: false
HALLUCINATION_DETECTOR_MODE=warn|block     # Default: warn
HALLUCINATION_DETECTOR_LANGUAGES=python,typescript,javascript
HALLUCINATION_DETECTOR_CONFIDENCE_THRESHOLD=0.7
HALLUCINATION_DETECTOR_CACHE_TTL_SECONDS=3600
```

### Dependencies

Add to `backend/pyproject.toml`:

```toml
dependencies = [
  # Epic 15 - Codebase Intelligence
  "tree-sitter>=0.23.0",
  "tree-sitter-python>=0.23.0",
  "tree-sitter-javascript>=0.23.0",
  "tree-sitter-typescript>=0.23.0",
  "gitignore-parser>=0.1.11",
]
```

---

## Implementation Tasks

### Phase 1: Core Types and Symbol Table (Day 1)

- [ ] **Task 1.1**: Create `codebase/` module structure
  - Create `__init__.py` files for all submodules
  - Set up module exports

- [ ] **Task 1.2**: Implement `types.py`
  - Define SymbolType, SymbolScope enums
  - Define CodeSymbol, ValidationResult, HallucinationReport dataclasses
  - Add Pydantic models for API request/response

- [ ] **Task 1.3**: Implement `symbol_table.py`
  - Implement SymbolTable class with add, lookup, lookup_qualified methods
  - Implement find_similar using difflib.get_close_matches
  - Add file-based index for get_symbols_in_file

### Phase 2: AST Parser and Extractors (Day 2)

- [ ] **Task 2.1**: Install tree-sitter dependencies
  - Add dependencies to pyproject.toml
  - Verify tree-sitter grammars install correctly

- [ ] **Task 2.2**: Implement `parser.py`
  - Create ASTParser class with tree-sitter initialization
  - Implement parse_file and parse_string methods
  - Handle language detection from file extension

- [ ] **Task 2.3**: Implement `extractors/base.py`
  - Define BaseSymbolExtractor abstract class
  - Define extract() method signature

- [ ] **Task 2.4**: Implement `extractors/python_extractor.py`
  - Extract functions, classes, methods, imports
  - Capture signatures and docstrings
  - Handle nested definitions

- [ ] **Task 2.5**: Implement `extractors/typescript_extractor.py`
  - Extract functions, classes, interfaces, imports
  - Handle TypeScript-specific constructs (types, enums)

### Phase 3: Validators (Day 3)

- [ ] **Task 3.1**: Implement `validators/symbol_validator.py`
  - Symbol existence validation
  - Type mismatch detection
  - Similar symbol suggestions

- [ ] **Task 3.2**: Implement `validators/path_validator.py`
  - Filesystem path validation
  - Relative/absolute path handling
  - Similar path suggestions

- [ ] **Task 3.3**: Implement `validators/api_validator.py`
  - OpenAPI spec loading and parsing
  - Endpoint matching with path parameters
  - Method validation

### Phase 4: Detector and Integration (Day 4)

- [ ] **Task 4.1**: Implement `detector.py`
  - Regex patterns for code reference extraction
  - Integration with validators
  - HallucinationReport generation

- [ ] **Task 4.2**: Add Redis caching for symbol tables
  - Serialize/deserialize symbol tables
  - Cache key with tenant_id prefix
  - TTL configuration

- [ ] **Task 4.3**: Create API routes
  - `POST /api/v1/codebase/validate-response`
  - Request validation with Pydantic
  - Multi-tenancy enforcement

- [ ] **Task 4.4**: Add configuration
  - Add settings to config.py
  - Document environment variables

### Phase 5: Testing and Documentation (Day 5)

- [ ] **Task 5.1**: Unit tests for symbol table
- [ ] **Task 5.2**: Unit tests for extractors (Python, TypeScript)
- [ ] **Task 5.3**: Unit tests for validators
- [ ] **Task 5.4**: Unit tests for detector
- [ ] **Task 5.5**: Integration tests for API endpoint
- [ ] **Task 5.6**: Performance benchmarks

---

## Testing Requirements

### Unit Tests

| Test File | Description |
|-----------|-------------|
| `backend/tests/codebase/test_types.py` | Test type definitions and validation |
| `backend/tests/codebase/test_symbol_table.py` | Test symbol table operations |
| `backend/tests/codebase/test_parser.py` | Test AST parsing for Python/TypeScript |
| `backend/tests/codebase/test_python_extractor.py` | Test Python symbol extraction |
| `backend/tests/codebase/test_typescript_extractor.py` | Test TypeScript symbol extraction |
| `backend/tests/codebase/test_symbol_validator.py` | Test symbol validation logic |
| `backend/tests/codebase/test_path_validator.py` | Test file path validation |
| `backend/tests/codebase/test_api_validator.py` | Test OpenAPI endpoint validation |
| `backend/tests/codebase/test_detector.py` | Test main detector orchestration |
| `backend/tests/api/routes/test_codebase.py` | Test API endpoints |

### Test Scenarios

**Symbol Validation:**
```python
def test_valid_function_reference():
    """Function referenced in response exists in symbol table."""

def test_hallucinated_function():
    """Function name does not exist, similar suggestions provided."""

def test_wrong_symbol_type():
    """Symbol exists but is wrong type (e.g., class used as function)."""
```

**File Path Validation:**
```python
def test_valid_file_path():
    """File path exists in repository."""

def test_hallucinated_path():
    """File path does not exist, similar paths suggested."""

def test_relative_path_resolution():
    """Relative paths resolved correctly."""
```

**API Endpoint Validation:**
```python
def test_valid_endpoint():
    """Endpoint exists in OpenAPI spec."""

def test_hallucinated_endpoint():
    """Endpoint not in spec, alternatives suggested."""

def test_endpoint_with_path_params():
    """Endpoint with {id} style parameters matched."""
```

**Import Validation:**
```python
def test_local_module_import():
    """Local module import validated against codebase."""

def test_external_dependency():
    """External package assumed valid with low confidence."""
```

### Integration Tests

```python
@pytest.mark.integration
async def test_full_validation_pipeline():
    """Test complete detection with real code samples."""

@pytest.mark.integration
async def test_symbol_table_caching():
    """Test Redis caching of symbol tables."""

@pytest.mark.integration
async def test_multi_tenant_isolation():
    """Test symbol table isolation between tenants."""
```

### Performance Benchmarks

```python
@pytest.mark.benchmark
def test_detection_latency():
    """Validation completes in < 100ms for 5000 char response."""

@pytest.mark.benchmark
def test_symbol_extraction_speed():
    """Symbol extraction for 1000 LOC file under 50ms."""
```

---

## Files to Create

| File Path | Purpose |
|-----------|---------|
| `backend/src/agentic_rag_backend/codebase/__init__.py` | Module exports |
| `backend/src/agentic_rag_backend/codebase/types.py` | Type definitions |
| `backend/src/agentic_rag_backend/codebase/symbol_table.py` | Symbol table management |
| `backend/src/agentic_rag_backend/codebase/parser.py` | Tree-sitter AST wrapper |
| `backend/src/agentic_rag_backend/codebase/detector.py` | Main detector class |
| `backend/src/agentic_rag_backend/codebase/extractors/__init__.py` | Extractor exports |
| `backend/src/agentic_rag_backend/codebase/extractors/base.py` | Base extractor class |
| `backend/src/agentic_rag_backend/codebase/extractors/python_extractor.py` | Python extraction |
| `backend/src/agentic_rag_backend/codebase/extractors/typescript_extractor.py` | TypeScript extraction |
| `backend/src/agentic_rag_backend/codebase/validators/__init__.py` | Validator exports |
| `backend/src/agentic_rag_backend/codebase/validators/symbol_validator.py` | Symbol validation |
| `backend/src/agentic_rag_backend/codebase/validators/path_validator.py` | Path validation |
| `backend/src/agentic_rag_backend/codebase/validators/api_validator.py` | API validation |
| `backend/src/agentic_rag_backend/api/routes/codebase.py` | API routes |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `backend/pyproject.toml` | Add tree-sitter dependencies |
| `backend/src/agentic_rag_backend/config.py` | Add hallucination detector settings |
| `backend/src/agentic_rag_backend/api/routes/__init__.py` | Export codebase router |
| `backend/src/agentic_rag_backend/main.py` | Register codebase routes |
| `.env.example` | Document new environment variables |

---

## Definition of Done

- [ ] All core types defined with proper Pydantic validation
- [ ] Symbol table implemented with add, lookup, and similar-search
- [ ] Tree-sitter parser working for Python and TypeScript
- [ ] Python symbol extractor extracts functions, classes, methods, imports
- [ ] TypeScript symbol extractor extracts functions, classes, interfaces, imports
- [ ] Symbol validator validates against symbol table with suggestions
- [ ] Path validator validates file paths with suggestions
- [ ] API validator validates against OpenAPI spec (when available)
- [ ] Detector class orchestrates all validators
- [ ] Redis caching for symbol tables with TTL
- [ ] API endpoint exposed at `/api/v1/codebase/validate-response`
- [ ] Multi-tenancy enforced with tenant_id isolation
- [ ] Configuration via environment variables
- [ ] Unit tests for all components (>80% coverage)
- [ ] Integration tests for full pipeline
- [ ] Performance benchmark passing (<100ms P95)
- [ ] Dependencies added to pyproject.toml
- [ ] Environment variables documented in .env.example

---

## Dependencies

- **Epic 3:** Existing Redis infrastructure for caching
- **Epic 11:** Multi-provider embedding architecture (for future RAG integration)
- **Epic 12:** Retrieval patterns (for potential integration with grader)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tree-sitter grammar complexity | Medium | Start with Python, add TypeScript incrementally |
| False positives in detection | Medium | Configurable confidence threshold, default to "warn" mode |
| Performance for large codebases | Medium | Symbol table caching, incremental parsing |
| Dynamic code patterns missed | Low | Document static analysis limitations |

---

## Implementation Notes (2026-01-04)

### Architecture Decisions

1. **Simplified Module Structure**: Instead of separate `extractors/` subdirectory with per-language files, consolidated into a single `extractor.py` that handles Python, TypeScript, and JavaScript extraction. This reduces complexity while maintaining language support.

2. **Graceful Tree-sitter Degradation**: All tree-sitter dependent code checks `TREE_SITTER_AVAILABLE` flag and degrades gracefully when not installed. Tests skip appropriately.

3. **Unified Validator Base**: Created abstract `BaseValidator` class in `validators/base.py` that all validators inherit from, ensuring consistent interface.

4. **Import Validator Addition**: Added `ImportValidator` class to validate import statements against standard library and installed packages (from requirements.txt).

5. **In-Memory + Redis Caching**: API routes use in-memory `_symbol_table_cache` dict with Redis fallback via `cache_symbol_table()` and `get_cached_symbol_table()` functions.

### Files Created

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/codebase/__init__.py` | Module exports |
| `backend/src/agentic_rag_backend/codebase/types.py` | SymbolType, SymbolScope, CodeSymbol, ValidationResult, HallucinationReport, Language |
| `backend/src/agentic_rag_backend/codebase/symbol_table.py` | SymbolTable class with indexes and Redis caching |
| `backend/src/agentic_rag_backend/codebase/parser.py` | ASTParser with tree-sitter wrapper |
| `backend/src/agentic_rag_backend/codebase/extractor.py` | SymbolExtractor for Python/TS/JS |
| `backend/src/agentic_rag_backend/codebase/detector.py` | HallucinationDetector with DetectorMode enum |
| `backend/src/agentic_rag_backend/codebase/validators/__init__.py` | Validator exports |
| `backend/src/agentic_rag_backend/codebase/validators/base.py` | BaseValidator abstract class |
| `backend/src/agentic_rag_backend/codebase/validators/symbol_validator.py` | Symbol validation with built-in recognition |
| `backend/src/agentic_rag_backend/codebase/validators/path_validator.py` | File path validation with normalization |
| `backend/src/agentic_rag_backend/codebase/validators/api_validator.py` | API endpoint validation with OpenAPI support |
| `backend/src/agentic_rag_backend/codebase/validators/import_validator.py` | Import statement validation |
| `backend/src/agentic_rag_backend/api/routes/codebase.py` | API routes and Pydantic models |

### Files Modified

| File | Change |
|------|--------|
| `backend/pyproject.toml` | Added tree-sitter and gitignore-parser dependencies |
| `backend/src/agentic_rag_backend/config.py` | Added codebase_hallucination_threshold, codebase_detector_mode, codebase_cache_ttl_seconds |
| `backend/src/agentic_rag_backend/core/errors.py` | Added CODEBASE_VALIDATION_FAILED, CODEBASE_INDEX_FAILED, HALLUCINATION_DETECTED error codes |
| `backend/src/agentic_rag_backend/api/routes/__init__.py` | Added codebase_router export |
| `backend/src/agentic_rag_backend/main.py` | Registered codebase_router |
| `.env.example` | Added Epic 15 environment variables |

### Test Files Created

| File | Coverage |
|------|----------|
| `backend/tests/unit/codebase/__init__.py` | Package init |
| `backend/tests/unit/codebase/test_types.py` | Type definitions and Language detection |
| `backend/tests/unit/codebase/test_symbol_table.py` | SymbolTable operations and caching |
| `backend/tests/unit/codebase/test_parser.py` | ASTParser for Python/TS/JS |
| `backend/tests/unit/codebase/test_extractor.py` | SymbolExtractor for all languages |
| `backend/tests/unit/codebase/test_detector.py` | HallucinationDetector modes and validation |
| `backend/tests/unit/codebase/test_validators.py` | All four validators |
| `backend/tests/unit/codebase/test_api.py` | API models and endpoint structure |

### Key Implementation Details

1. **Regex Patterns for Code Extraction**: The detector uses comprehensive regex patterns to extract:
   - Function calls: `(\w+)\s*\(`
   - Class names: `\b([A-Z][a-zA-Z0-9]*)\b`
   - File paths: `[a-zA-Z_][a-zA-Z0-9_/.-]*\.(py|ts|js|tsx|jsx|yaml|yml|json|md)`
   - API endpoints: `/api/[^\s\"'<>]+`
   - Imports: `(?:from\s+(\S+)\s+)?import\s+`

2. **Common Word Exclusion**: Built-in list of common words (returns, data, module, etc.) excluded from validation to reduce false positives.

3. **Built-in Symbol Recognition**: SymbolValidator recognizes Python built-ins (print, len, str, etc.) and marks them valid without symbol table lookup.

4. **Multi-Tenancy**: All symbol tables are scoped by tenant_id, with cache keys prefixed as `codebase:symbols:{tenant_id}:{hash}`.

5. **Environment Variables**:
   - `CODEBASE_HALLUCINATION_THRESHOLD=0.3` - Ratio of invalid symbols to trigger blocking
   - `CODEBASE_DETECTOR_MODE=warn` - Either `warn` or `block`
   - `CODEBASE_CACHE_TTL_SECONDS=3600` - Symbol table cache duration

### Deviations from Original Design

1. **No Separate Extractor Files**: Combined Python, TypeScript, and JavaScript extraction into single `extractor.py` instead of per-language files.

2. **Added ImportValidator**: Not in original spec but essential for validating import statement hallucinations.

3. **index_repository Function**: Added async function in `detector.py` for indexing repositories with gitignore support.

4. **Additional API Endpoints**: Added `GET /symbol-table/stats` and `DELETE /symbol-table` endpoints beyond the original spec.

---

## References

- [Tree-sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [py-tree-sitter GitHub](https://github.com/tree-sitter/py-tree-sitter)
- [Package Hallucinations Research (USENIX 2025)](https://www.usenix.org/publications/loginonline/we-have-package-you-comprehensive-analysis-package-hallucinations-code)
- Epic 15 Tech Spec: `_bmad-output/epics/epic-15-tech-spec.md`

---

## Senior Developer Review (2026-01-04)

### Review Summary

The codebase hallucination detector implementation is well-structured with proper separation of concerns, comprehensive validation logic, and good test coverage. However, several security, compliance, and quality issues were identified that require attention. The path validation lacked path traversal protection, API error handling used generic HTTPException instead of RFC 7807-compliant AppError responses, the repository indexing endpoint lacked input validation for SSRF protection, and the Redis cache clearing was incomplete. These issues have been addressed in this review.

### Issues Found

1. **[HIGH] Path Traversal Vulnerability in FilePathValidator**
   - File: `/backend/src/agentic_rag_backend/codebase/validators/path_validator.py`
   - Lines: 98-109, 228-239
   - Problem: The `validate_path()` and `validate_directory()` methods construct filesystem paths from user input without validating that the resolved path remains within the repository bounds. An attacker could use paths like `../../../etc/passwd` to probe filesystem locations outside the repository.
   - Fix: Added path resolution with `Path.resolve()` and validation that resolved path starts with the repository root path. Added logging for blocked traversal attempts.

2. **[HIGH] Missing RFC 7807 Error Responses in API Routes**
   - File: `/backend/src/agentic_rag_backend/api/routes/codebase.py`
   - Lines: 238-251, 343-352, 411-415
   - Problem: API endpoints used generic `HTTPException` with status codes 400, 404, and 500 instead of RFC 7807-compliant `AppError` responses as mandated by the project's API response conventions.
   - Fix: Replaced `HTTPException` with `CodebaseValidationError`, `CodebaseIndexError`, and `AppError` to ensure proper error format.

3. **[HIGH] Missing Input Validation for Repository Path (SSRF Risk)**
   - File: `/backend/src/agentic_rag_backend/codebase/detector.py`
   - Lines: 421-443
   - Problem: The `index_repository()` function accepted any path without validation. This could allow attackers to index sensitive system directories or trigger scanning of unintended locations.
   - Fix: Added validation requiring absolute paths, rejecting paths containing `..`, resolving paths to canonical form, and verifying the path exists and is a directory.

4. **[MEDIUM] CodebaseValidationError Used Incorrect HTTP Status**
   - File: `/backend/src/agentic_rag_backend/core/errors.py`
   - Lines: 366-375
   - Problem: `CodebaseValidationError` used status code 500 (internal server error) when it should use 400 (bad request) since validation failures are client errors, not server errors.
   - Fix: Changed status from 500 to 400.

5. **[MEDIUM] Incomplete Redis Cache Clearing**
   - File: `/backend/src/agentic_rag_backend/api/routes/codebase.py`
   - Lines: 443-467
   - Problem: The `clear_symbol_table()` endpoint only cleared the in-memory cache but did not clear the corresponding Redis cache entries, leading to stale data being loaded on subsequent requests.
   - Fix: Added Redis cache clearing using `scan_iter()` and `delete()` for all keys matching the tenant's codebase pattern.

6. **[LOW] Imprecise Exception Type in Test**
   - File: `/backend/tests/unit/codebase/test_types.py`
   - Lines: 85-97
   - Problem: The test for `CodeSymbol` immutability caught a generic `Exception` instead of the specific `FrozenInstanceError` that dataclasses raise. This could mask unrelated exceptions.
   - Fix: Changed `pytest.raises(Exception)` to `pytest.raises(FrozenInstanceError)` with proper import.

### Outcome

**APPROVE** - All identified issues have been fixed directly in the codebase. The implementation is now ready to merge with the following security enhancements:

- Path traversal protection added to file path validation
- SSRF protection added to repository indexing
- Proper RFC 7807 error responses
- Complete cache invalidation including Redis

No architectural changes were required. The fixes maintain backward compatibility with existing API contracts while improving security posture.

---

## Senior Developer Re-Review (2026-01-04)

### Review Summary

Performed a full re-review focused on detector enforcement, configuration defaults, and import/API validation completeness. Found several correctness gaps affecting AC-4/AC-5/AC-6 and tightened handling accordingly.

### Issues Found

1. **[HIGH] Block Mode Did Not Enforce HTTP 422**
   - File: `/backend/src/agentic_rag_backend/api/routes/codebase.py`
   - Problem: In block mode, responses were still returned as success with `should_block=true` rather than rejected.
   - Fix: Raised `HallucinationError` (RFC 7807) with invalid symbol details when threshold exceeded.

2. **[HIGH] CODEBASE_DETECTOR_MODE Ignored**
   - File: `/backend/src/agentic_rag_backend/api/routes/codebase.py`
   - Problem: Request model defaulted to `"warn"`, overriding env default and making `CODEBASE_DETECTOR_MODE` ineffective.
   - Fix: Request default now uses settings when mode is omitted.

3. **[MEDIUM] Import Validation Not Tied to Declared Dependencies**
   - Files: `/backend/src/agentic_rag_backend/codebase/detector.py`, `/backend/src/agentic_rag_backend/codebase/validators/import_validator.py`
   - Problem: Validation relied on runtime `find_spec` rather than repo dependencies, allowing undeclared packages to pass.
   - Fix: Load dependencies from `pyproject.toml`, `requirements*.txt`, and `package.json`; prefer declared dependencies over environment; added JS/TS import extraction and local relative import checks.

4. **[MEDIUM] OpenAPI Spec Not Wired for API Validation**
   - File: `/backend/src/agentic_rag_backend/api/routes/codebase.py`
   - Problem: API validation always operated with empty routes because no spec was supplied.
   - Fix: Pass `app.openapi()` into the detector when available.

5. **[LOW] Path Normalization Stripped `../` Prefixes**
   - File: `/backend/src/agentic_rag_backend/codebase/validators/path_validator.py`
   - Problem: Normalization removed `../`, masking traversal indicators and enabling false-positive validations.
   - Fix: Only strip `./` to preserve traversal signals for validation.

### Outcome

**APPROVE** - All issues were fixed directly in the codebase. The detector now enforces block mode correctly, respects configuration defaults, validates imports against declared dependencies, and loads OpenAPI specs for endpoint checks.
