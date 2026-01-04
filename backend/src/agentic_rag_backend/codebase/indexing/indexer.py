"""Codebase indexing orchestration for RAG context."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid5, NAMESPACE_URL

import structlog

from agentic_rag_backend.codebase import SymbolExtractor, SymbolTable
from agentic_rag_backend.codebase.symbol_table import cache_symbol_table, get_cached_symbol_table
from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.embeddings import EmbeddingGenerator
from agentic_rag_backend.indexing.chunker import count_tokens

from ..types import CodeSymbol, SymbolType, get_language_for_file
from .chunker import CodeChunk, CodeChunker
from .graph_builder import CodeGraphBuilder, CodeRelationship
from .scanner import FileScanner

logger = structlog.get_logger(__name__)

_INDEX_STATE_CACHE: dict[str, tuple[float, dict[str, float]]] = {}


@dataclass(frozen=True)
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
        postgres: PostgresClient,
        embedding_generator: EmbeddingGenerator,
        neo4j: Optional[Neo4jClient] = None,
        redis_client: Optional[object] = None,
        languages: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        max_chunk_size: int = 1000,
        include_class_context: bool = True,
        cache_ttl_seconds: int = 86400,
    ) -> None:
        self.tenant_id = tenant_id
        self.repo_path = Path(repo_path).resolve()
        self.postgres = postgres
        self.embedding_generator = embedding_generator
        self.neo4j = neo4j
        self.redis_client = redis_client
        self.languages = languages or ["python", "typescript", "javascript"]
        self.exclude_patterns = exclude_patterns or []
        self.max_chunk_size = max_chunk_size
        self.include_class_context = include_class_context
        self.cache_ttl_seconds = cache_ttl_seconds

        self.scanner = FileScanner(
            repo_path=self.repo_path,
            exclude_patterns=self.exclude_patterns,
            languages=self.languages,
        )
        self.extractor = SymbolExtractor()
        self.chunker = CodeChunker(
            max_chunk_size=self.max_chunk_size,
            include_class_context=self.include_class_context,
        )
        self.graph_builder = CodeGraphBuilder()

    async def index_full(self) -> IndexingResult:
        files = self.scanner.scan()
        result = await self._index_files(files)
        self._update_index_state(files)
        return result

    async def index_incremental(self) -> IndexingResult:
        all_files = self.scanner.scan()
        changed = self._get_changed_files(all_files)
        symbol_table = None
        if self.redis_client:
            try:
                symbol_table = await get_cached_symbol_table(
                    self.redis_client,
                    self.tenant_id,
                    str(self.repo_path),
                )
            except Exception as exc:
                logger.warning("codebase_symbol_cache_load_failed", error=str(exc))

        if symbol_table is None:
            symbol_table = SymbolTable(self.tenant_id, str(self.repo_path))

        for file_path in changed:
            rel_path = Path(file_path).relative_to(self.repo_path).as_posix()
            symbol_table.remove_file(rel_path)

        return await self._index_files(changed, symbol_table=symbol_table)

    async def _index_files(
        self,
        files: list[str],
        symbol_table: Optional[SymbolTable] = None,
    ) -> IndexingResult:
        start_time = time.perf_counter()
        errors: list[str] = []

        files_indexed = 0
        symbols_extracted = 0
        chunks_created = 0
        relationships_created = 0

        tenant_uuid = UUID(self.tenant_id)
        symbol_table = symbol_table or SymbolTable(self.tenant_id, str(self.repo_path))
        symbol_entity_ids: dict[str, str] = {}
        file_entity_ids: dict[str, str] = {}

        file_contents: dict[str, str] = {}

        for file_path in files:
            rel_path = Path(file_path).relative_to(self.repo_path).as_posix()
            try:
                content = Path(file_path).read_text(encoding="utf-8")
                file_contents[rel_path] = content
            except OSError as exc:
                errors.append(f"{rel_path}: {exc}")
                logger.warning("codebase_read_failed", file_path=rel_path, error=str(exc))
                continue

            language = get_language_for_file(rel_path)
            if language is None:
                continue

            symbols = self.extractor.extract_from_string(content, rel_path, language)
            symbol_table.add_known_file(rel_path)
            for symbol in symbols:
                symbol_table.add(symbol)
            symbols_extracted += len(symbols)

            if symbols and self.neo4j:
                try:
                    await self._index_graph_entities(
                        rel_path,
                        symbols,
                        symbol_entity_ids,
                        file_entity_ids,
                    )
                except Exception as exc:
                    errors.append(f"{rel_path}: graph entity indexing failed ({exc})")
                    logger.warning("codebase_graph_entity_failed", file_path=rel_path, error=str(exc))

            chunks = self.chunker.chunk_file(
                file_path=file_path,
                symbols=symbols,
                display_path=rel_path,
            )
            if not chunks and content.strip():
                line_count = len(content.splitlines()) or 1
                chunks = [
                    CodeChunk(
                        content=content.strip(),
                        symbol_name=rel_path,
                        symbol_type="module",
                        file_path=rel_path,
                        line_start=1,
                        line_end=line_count,
                        chunk_index=0,
                    )
                ]
            if chunks:
                try:
                    chunk_embeddings = await self.embedding_generator.generate_embeddings(
                        [chunk.content for chunk in chunks],
                        tenant_id=self.tenant_id,
                    )
                except Exception as exc:
                    errors.append(f"{rel_path}: embedding failed ({exc})")
                    logger.warning("codebase_embedding_failed", file_path=rel_path, error=str(exc))
                    continue

                document_id = await self._ensure_document(tenant_uuid, rel_path)
                await self.postgres.delete_chunks_by_document(document_id, tenant_uuid)

                for idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    symbol_key = f"{chunk.file_path}:{chunk.symbol_name}:{chunk.symbol_type}"
                    metadata = {
                        "source_type": "codebase",
                        "repo_path": str(self.repo_path),
                        "file_path": rel_path,
                        "symbol_name": chunk.symbol_name,
                        "symbol_type": chunk.symbol_type,
                        "line_start": chunk.line_start,
                        "line_end": chunk.line_end,
                        "symbol_entity_id": symbol_entity_ids.get(symbol_key),
                        "language": language.value,
                    }
                    token_count = count_tokens(chunk.content)
                    try:
                        await self.postgres.create_chunk(
                            tenant_id=tenant_uuid,
                            document_id=document_id,
                            content=chunk.content,
                            chunk_index=idx,
                            token_count=token_count,
                            embedding=embedding,
                            metadata=metadata,
                        )
                        chunks_created += 1
                    except Exception as exc:
                        errors.append(f"{rel_path}: chunk {idx} failed ({exc})")
                        logger.warning(
                            "codebase_chunk_create_failed",
                            file_path=rel_path,
                            chunk_index=idx,
                            error=str(exc),
                        )

            files_indexed += 1

        if self.neo4j and file_contents:
            try:
                relationships_created += await self._index_graph_relationships(
                    symbol_table,
                    file_contents,
                    symbol_entity_ids,
                    file_entity_ids,
                )
            except Exception as exc:
                errors.append(f"graph relationship indexing failed ({exc})")
                logger.warning("codebase_graph_relationship_failed", error=str(exc))

        if self.redis_client:
            try:
                await cache_symbol_table(self.redis_client, symbol_table, ttl_seconds=self.cache_ttl_seconds)
            except Exception as exc:
                logger.warning("codebase_symbol_cache_failed", error=str(exc))

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        return IndexingResult(
            tenant_id=self.tenant_id,
            repo_path=str(self.repo_path),
            files_indexed=files_indexed,
            symbols_extracted=symbols_extracted,
            chunks_created=chunks_created,
            relationships_created=relationships_created,
            processing_time_ms=processing_time_ms,
            errors=errors,
        )

    async def _ensure_document(self, tenant_id: UUID, file_path: str) -> UUID:
        content_hash = hashlib.sha256(f"{self.repo_path}:{file_path}".encode("utf-8")).hexdigest()
        return await self.postgres.create_document(
            tenant_id=tenant_id,
            source_type="codebase",
            content_hash=content_hash,
            source_url=str(self.repo_path),
            filename=file_path,
            metadata={
                "repo_path": str(self.repo_path),
                "file_path": file_path,
                "source_type": "codebase",
            },
        )

    async def _index_graph_entities(
        self,
        rel_path: str,
        symbols: list[CodeSymbol],
        symbol_entity_ids: dict[str, str],
        file_entity_ids: dict[str, str],
    ) -> None:
        if not self.neo4j:
            return

        file_entity_id = file_entity_ids.get(rel_path)
        if not file_entity_id:
            file_entity_id = self._make_entity_id("file", rel_path)
            file_entity_ids[rel_path] = file_entity_id
            await self.neo4j.create_entity(
                entity_id=file_entity_id,
                tenant_id=self.tenant_id,
                name=rel_path,
                entity_type="CodeFile",
                description=f"File {rel_path}",
                properties={"file_path": rel_path},
            )

        for symbol in symbols:
            symbol_key = f"{rel_path}:{symbol.name}:{symbol.type.value}"
            symbol_id = self._make_entity_id(symbol.type.value, symbol_key)
            symbol_entity_ids[symbol_key] = symbol_id
            entity_type = self._map_symbol_type(symbol.type)
            await self.neo4j.create_entity(
                entity_id=symbol_id,
                tenant_id=self.tenant_id,
                name=symbol.name,
                entity_type=entity_type,
                description=symbol.signature or symbol.docstring,
                properties={
                    "file_path": rel_path,
                    "line_start": symbol.line_start,
                    "line_end": symbol.line_end,
                },
            )
            await self.neo4j.create_relationship(
                source_id=symbol_id,
                target_id=file_entity_id,
                relationship_type="DEFINED_IN",
                tenant_id=self.tenant_id,
                confidence=1.0,
                description=f"{symbol.name} defined in {rel_path}",
            )

    async def _index_graph_relationships(
        self,
        symbol_table: SymbolTable,
        file_contents: dict[str, str],
        symbol_entity_ids: dict[str, str],
        file_entity_ids: dict[str, str],
    ) -> int:
        if not self.neo4j:
            return 0

        relationship_count = 0
        for rel_path, content in file_contents.items():
            symbols = symbol_table.get_symbols_in_file(rel_path)
            relationships = self.graph_builder.build_relationships(
                symbol_table,
                rel_path,
                content,
                symbols,
            )
            for relationship in relationships:
                created = await self._create_relationship(relationship, symbol_entity_ids, file_entity_ids)
                if created:
                    relationship_count += 1

        return relationship_count

    async def _create_relationship(
        self,
        relationship: CodeRelationship,
        symbol_entity_ids: dict[str, str],
        file_entity_ids: dict[str, str],
    ) -> bool:
        if not self.neo4j:
            return False

        if relationship.source_type == "file":
            source_id = file_entity_ids.get(relationship.source_name)
            if not source_id:
                source_id = self._make_entity_id("file", relationship.source_name)
                file_entity_ids[relationship.source_name] = source_id
                await self.neo4j.create_entity(
                    entity_id=source_id,
                    tenant_id=self.tenant_id,
                    name=relationship.source_name,
                    entity_type="CodeFile",
                    description=f"File {relationship.source_name}",
                    properties={"file_path": relationship.source_name},
                )
        else:
            source_key = f"{relationship.source_file}:{relationship.source_name}:{relationship.source_type}"
            source_id = symbol_entity_ids.get(source_key)
            if not source_id:
                source_id = self._make_entity_id(
                    relationship.source_type,
                    source_key,
                )
                symbol_entity_ids[source_key] = source_id
                symbol_type = self._coerce_symbol_type(relationship.source_type)
                await self.neo4j.create_entity(
                    entity_id=source_id,
                    tenant_id=self.tenant_id,
                    name=relationship.source_name,
                    entity_type=self._map_symbol_type(symbol_type) if symbol_type else "CodeSymbol",
                    description=f"{relationship.source_type} {relationship.source_name}",
                )

        if relationship.target_type == "module":
            target_id = self._make_entity_id("module", relationship.target_name)
            await self.neo4j.create_entity(
                entity_id=target_id,
                tenant_id=self.tenant_id,
                name=relationship.target_name,
                entity_type="CodeModule",
                description=f"Imported module {relationship.target_name}",
            )
        else:
            target_file = relationship.target_file or relationship.source_file
            target_key = f"{target_file}:{relationship.target_name}:{relationship.target_type}"
            target_id = symbol_entity_ids.get(target_key)
            if not target_id:
                target_id = self._make_entity_id(
                    relationship.target_type,
                    target_key,
                )
                symbol_entity_ids[target_key] = target_id
                symbol_type = self._coerce_symbol_type(relationship.target_type)
                await self.neo4j.create_entity(
                    entity_id=target_id,
                    tenant_id=self.tenant_id,
                    name=relationship.target_name,
                    entity_type=self._map_symbol_type(symbol_type) if symbol_type else "CodeSymbol",
                    description=f"{relationship.target_type} {relationship.target_name}",
                )

        return await self.neo4j.create_relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship.relationship_type,
            tenant_id=self.tenant_id,
            confidence=0.8,
            description=f"{relationship.source_name} {relationship.relationship_type} {relationship.target_name}",
        )

    def _get_changed_files(self, files: list[str]) -> list[str]:
        key = self._get_cache_key()
        state = self._load_index_state(key)
        changed: list[str] = []
        updated_state: dict[str, float] = {}

        for file_path in files:
            rel_path = Path(file_path).relative_to(self.repo_path).as_posix()
            try:
                mtime = Path(file_path).stat().st_mtime
            except OSError:
                continue
            updated_state[rel_path] = mtime
            if state.get(rel_path) != mtime:
                changed.append(file_path)

        _INDEX_STATE_CACHE[key] = (time.time(), updated_state)
        return changed

    def _load_index_state(self, key: str) -> dict[str, float]:
        entry = _INDEX_STATE_CACHE.get(key)
        if not entry:
            return {}
        timestamp, state = entry
        if time.time() - timestamp > self.cache_ttl_seconds:
            return {}
        return state

    def _update_index_state(self, files: list[str]) -> None:
        key = self._get_cache_key()
        state: dict[str, float] = {}
        for file_path in files:
            rel_path = Path(file_path).relative_to(self.repo_path).as_posix()
            try:
                state[rel_path] = Path(file_path).stat().st_mtime
            except OSError:
                continue
        _INDEX_STATE_CACHE[key] = (time.time(), state)

    def _get_cache_key(self) -> str:
        return f"{self.tenant_id}:{self.repo_path}"

    def _make_entity_id(self, category: str, name: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"{self.tenant_id}:{category}:{name}"))

    def _map_symbol_type(self, symbol_type: SymbolType) -> str:
        if symbol_type in {SymbolType.FUNCTION, SymbolType.METHOD}:
            return "CodeFunction"
        if symbol_type == SymbolType.CLASS:
            return "CodeClass"
        return "CodeSymbol"

    def _coerce_symbol_type(self, value: str) -> Optional[SymbolType]:
        try:
            return SymbolType(value)
        except ValueError:
            return None
