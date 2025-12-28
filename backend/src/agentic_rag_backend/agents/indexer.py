"""Agentic Indexer agent for autonomous entity extraction and graph building.

This agent uses Agno patterns for trajectory logging (log_thought, log_action,
log_observation) to trace its decision-making process during entity extraction
and knowledge graph construction.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog

from agentic_rag_backend.core.errors import ExtractionError
from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.indexing.chunker import ChunkData, chunk_document
from agentic_rag_backend.indexing.embeddings import EmbeddingGenerator
from agentic_rag_backend.indexing.entity_extractor import EntityExtractor
from agentic_rag_backend.models.graphs import (
    ExtractedEntity,
    ExtractedRelationship,
    IndexingResult,
)

logger = structlog.get_logger(__name__)


@dataclass
class TrajectoryEntry:
    """A single entry in the agent's trajectory log."""

    timestamp: datetime
    entry_type: str  # "thought", "action", "observation"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexerTrajectory:
    """Complete trajectory of an indexing run."""

    run_id: str
    document_id: str
    tenant_id: str
    started_at: datetime
    entries: list[TrajectoryEntry] = field(default_factory=list)
    completed_at: Optional[datetime] = None


class IndexerAgent:
    """
    Agentic Indexer for autonomous document indexing.

    This agent orchestrates the complete indexing pipeline:
    1. Chunk document into semantic units
    2. Generate embeddings for chunks
    3. Extract entities and relationships using LLM
    4. Deduplicate entities
    5. Build knowledge graph in Neo4j
    6. Store chunks with embeddings in pgvector

    All decisions are logged to the trajectory for debugging and auditing.
    """

    def __init__(
        self,
        postgres: PostgresClient,
        neo4j: Neo4jClient,
        embedding_generator: EmbeddingGenerator,
        entity_extractor: EntityExtractor,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        """
        Initialize the IndexerAgent.

        Args:
            postgres: PostgreSQL client for chunks
            neo4j: Neo4j client for knowledge graph
            embedding_generator: OpenAI embedding generator
            entity_extractor: LLM entity extractor
            chunk_size: Target tokens per chunk
            chunk_overlap: Token overlap between chunks
        """
        self.postgres = postgres
        self.neo4j = neo4j
        self.embedding_generator = embedding_generator
        self.entity_extractor = entity_extractor
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Current trajectory for this agent instance
        self._trajectory: Optional[IndexerTrajectory] = None
        self._entity_cache: dict[str, str] = {}  # name -> entity_id mapping

    def _start_trajectory(self, document_id: str, tenant_id: str) -> None:
        """Start a new trajectory for an indexing run."""
        self._trajectory = IndexerTrajectory(
            run_id=str(uuid4()),
            document_id=document_id,
            tenant_id=tenant_id,
            started_at=datetime.now(timezone.utc),
        )
        self._entity_cache = {}  # Reset cache for new document

    def _finish_trajectory(self) -> IndexerTrajectory:
        """Complete the current trajectory."""
        if self._trajectory:
            self._trajectory.completed_at = datetime.now(timezone.utc)
        return self._trajectory

    def log_thought(self, content: str, **metadata: Any) -> None:
        """
        Log an agent thought (reasoning step).

        Args:
            content: Description of the thought/reasoning
            **metadata: Additional context
        """
        if self._trajectory:
            entry = TrajectoryEntry(
                timestamp=datetime.now(timezone.utc),
                entry_type="thought",
                content=content,
                metadata=metadata,
            )
            self._trajectory.entries.append(entry)
            logger.debug("agent_thought", content=content, **metadata)

    def log_action(self, action: str, details: dict[str, Any]) -> None:
        """
        Log an agent action (tool/API call).

        Args:
            action: Name of the action
            details: Action parameters and context
        """
        if self._trajectory:
            entry = TrajectoryEntry(
                timestamp=datetime.now(timezone.utc),
                entry_type="action",
                content=action,
                metadata=details,
            )
            self._trajectory.entries.append(entry)
            logger.debug("agent_action", action=action, **details)

    def log_observation(self, content: str, **metadata: Any) -> None:
        """
        Log an agent observation (result of action).

        Args:
            content: Description of what was observed
            **metadata: Additional context
        """
        if self._trajectory:
            entry = TrajectoryEntry(
                timestamp=datetime.now(timezone.utc),
                entry_type="observation",
                content=content,
                metadata=metadata,
            )
            self._trajectory.entries.append(entry)
            logger.debug("agent_observation", content=content, **metadata)

    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name for deduplication.

        Args:
            name: Raw entity name

        Returns:
            Normalized name (lowercase, trimmed, standardized spacing)
        """
        return " ".join(name.lower().strip().split())

    async def _deduplicate_entity(
        self,
        entity: ExtractedEntity,
        tenant_id: str,
    ) -> tuple[str, bool]:
        """
        Check if entity already exists and return its ID.

        Uses name normalization and type matching for deduplication.

        Args:
            entity: Entity to check
            tenant_id: Tenant identifier

        Returns:
            Tuple of (entity_id, is_new) where is_new is True if entity was created
        """
        normalized_name = self._normalize_entity_name(entity.name)
        cache_key = f"{normalized_name}:{entity.type}"

        # Check local cache first
        if cache_key in self._entity_cache:
            self.log_thought(
                "Entity found in local cache",
                entity_name=entity.name,
                cached_id=self._entity_cache[cache_key],
            )
            return self._entity_cache[cache_key], False

        # Check Neo4j for existing entity
        existing = await self.neo4j.find_similar_entity(
            tenant_id=tenant_id,
            name=entity.name,
            entity_type=entity.type,
        )

        if existing:
            entity_id = existing["id"]
            self._entity_cache[cache_key] = entity_id
            self.log_observation(
                "Found existing entity in graph",
                entity_name=entity.name,
                existing_id=entity_id,
            )
            return entity_id, False

        # Create new entity
        entity_id = str(uuid4())
        self._entity_cache[cache_key] = entity_id
        return entity_id, True

    async def _process_chunk(
        self,
        chunk: ChunkData,
        chunk_id: UUID,
        document_id: UUID,
        tenant_id: UUID,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelationship]]:
        """
        Process a single chunk: extract entities and relationships.

        Args:
            chunk: Chunk data
            chunk_id: Chunk UUID
            document_id: Document UUID
            tenant_id: Tenant UUID

        Returns:
            Tuple of (entities, relationships)
        """
        self.log_action(
            "extract_entities",
            {
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
            },
        )

        result = await self.entity_extractor.extract_from_chunk(
            chunk_content=chunk.content,
            chunk_id=str(chunk_id),
        )

        self.log_observation(
            "Entity extraction completed",
            entities_found=len(result.entities),
            relationships_found=len(result.relationships),
            processing_time_ms=result.processing_time_ms,
        )

        return result.entities, result.relationships

    async def index_document(
        self,
        document_id: UUID,
        tenant_id: UUID,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IndexingResult:
        """
        Index a document: chunk, embed, extract entities, build graph.

        This is the main entry point for document indexing. It orchestrates
        the complete pipeline and logs all decisions to the trajectory.

        Args:
            document_id: Document UUID
            tenant_id: Tenant UUID
            content: Document text content
            metadata: Optional document metadata

        Returns:
            IndexingResult with counts and processing time

        Raises:
            ExtractionError: If indexing fails
        """
        start_time = time.perf_counter()
        self._start_trajectory(str(document_id), str(tenant_id))

        self.log_thought(
            "Starting document indexing",
            document_id=str(document_id),
            content_length=len(content),
        )

        try:
            # Step 1: Chunk the document
            self.log_action("chunk_document", {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap})

            chunks = chunk_document(
                content=content,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            self.log_observation(f"Document chunked into {len(chunks)} chunks")

            if not chunks:
                self.log_thought("No chunks created, document may be empty")
                return IndexingResult(
                    document_id=str(document_id),
                    tenant_id=str(tenant_id),
                    chunks_created=0,
                    entities_extracted=0,
                    relationships_extracted=0,
                    entities_deduplicated=0,
                    processing_time_ms=0,
                )

            # Step 2: Generate embeddings for all chunks
            self.log_action("generate_embeddings", {"chunk_count": len(chunks)})

            chunk_texts = [c.content for c in chunks]
            embeddings = await self.embedding_generator.generate_embeddings(chunk_texts)

            self.log_observation(f"Generated {len(embeddings)} embeddings")

            # Step 3: Store chunks in pgvector
            self.log_action("store_chunks", {"count": len(chunks)})

            chunk_ids = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = await self.postgres.create_chunk(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    embedding=embedding,
                    metadata={"start_char": chunk.start_char, "end_char": chunk.end_char},
                )
                chunk_ids.append(chunk_id)

            self.log_observation(f"Stored {len(chunk_ids)} chunks in pgvector")

            # Step 4: Create document node in Neo4j
            self.log_action("create_document_node", {"document_id": str(document_id)})

            await self.neo4j.create_document_node(
                document_id=str(document_id),
                tenant_id=str(tenant_id),
                title=metadata.get("title") if metadata else None,
                source_url=metadata.get("source_url") if metadata else None,
                source_type=metadata.get("source_type") if metadata else None,
            )

            # Step 5: Create chunk nodes in Neo4j
            for chunk_id, chunk in zip(chunk_ids, chunks):
                await self.neo4j.create_chunk_node(
                    chunk_id=str(chunk_id),
                    tenant_id=str(tenant_id),
                    document_id=str(document_id),
                    chunk_index=chunk.chunk_index,
                    preview=chunk.content[:200],
                )

            # Step 6: Extract entities from each chunk
            self.log_thought("Beginning entity extraction for all chunks")

            all_entities: list[ExtractedEntity] = []
            all_relationships: list[ExtractedRelationship] = []
            entity_to_chunks: dict[str, list[str]] = {}  # entity_name -> [chunk_ids]

            for chunk_id, chunk in zip(chunk_ids, chunks):
                entities, relationships = await self._process_chunk(
                    chunk=chunk,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    tenant_id=tenant_id,
                )

                for entity in entities:
                    all_entities.append(entity)
                    norm_name = self._normalize_entity_name(entity.name)
                    if norm_name not in entity_to_chunks:
                        entity_to_chunks[norm_name] = []
                    entity_to_chunks[norm_name].append(str(chunk_id))

                all_relationships.extend(relationships)

            self.log_observation(
                "Entity extraction complete",
                total_entities=len(all_entities),
                total_relationships=len(all_relationships),
            )

            # Step 7: Deduplicate and create entities in Neo4j
            self.log_action("deduplicate_entities", {"count": len(all_entities)})

            entities_created = 0
            entities_deduplicated = 0
            entity_name_to_id: dict[str, str] = {}

            for entity in all_entities:
                norm_name = self._normalize_entity_name(entity.name)

                # Skip if we've already processed this entity in this document
                if norm_name in entity_name_to_id:
                    continue

                entity_id, is_new = await self._deduplicate_entity(entity, str(tenant_id))

                if is_new:
                    # Create the entity in Neo4j
                    await self.neo4j.create_entity(
                        entity_id=entity_id,
                        tenant_id=str(tenant_id),
                        name=entity.name,
                        entity_type=entity.type,
                        description=entity.description,
                        source_chunk_id=entity_to_chunks.get(norm_name, [None])[0],
                    )
                    entities_created += 1
                else:
                    # Update existing entity with new chunk reference
                    for chunk_id_str in entity_to_chunks.get(norm_name, []):
                        await self.neo4j.create_entity(
                            entity_id=entity_id,
                            tenant_id=str(tenant_id),
                            name=entity.name,
                            entity_type=entity.type,
                            source_chunk_id=chunk_id_str,
                        )
                    entities_deduplicated += 1

                entity_name_to_id[norm_name] = entity_id

                # Link chunks to entities
                for chunk_id_str in entity_to_chunks.get(norm_name, []):
                    await self.neo4j.link_chunk_to_entity(
                        chunk_id=chunk_id_str,
                        entity_id=entity_id,
                        tenant_id=str(tenant_id),
                    )

            self.log_observation(
                "Entity deduplication complete",
                entities_created=entities_created,
                entities_deduplicated=entities_deduplicated,
            )

            # Step 8: Create relationships in Neo4j
            self.log_action("create_relationships", {"count": len(all_relationships)})

            relationships_created = 0
            for rel in all_relationships:
                source_norm = self._normalize_entity_name(rel.source)
                target_norm = self._normalize_entity_name(rel.target)

                source_id = entity_name_to_id.get(source_norm)
                target_id = entity_name_to_id.get(target_norm)

                if source_id and target_id:
                    created = await self.neo4j.create_relationship(
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type=rel.type,
                        tenant_id=str(tenant_id),
                        confidence=rel.confidence,
                    )
                    if created:
                        relationships_created += 1

            self.log_observation(f"Created {relationships_created} relationships")

            # Finalize
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            result = IndexingResult(
                document_id=str(document_id),
                tenant_id=str(tenant_id),
                chunks_created=len(chunks),
                entities_extracted=len(all_entities),
                relationships_extracted=len(all_relationships),
                entities_deduplicated=entities_deduplicated,
                processing_time_ms=processing_time_ms,
            )

            self.log_thought(
                "Indexing completed successfully",
                chunks=result.chunks_created,
                entities=result.entities_extracted,
                relationships=result.relationships_extracted,
                deduplicated=result.entities_deduplicated,
                time_ms=result.processing_time_ms,
            )

            trajectory = self._finish_trajectory()
            # Note: result.model_dump() already includes document_id and tenant_id
            logger.info(
                "document_indexed",
                trajectory_id=trajectory.run_id if trajectory else None,
                **result.model_dump(),
            )

            return result

        except Exception as e:
            self.log_observation(f"Indexing failed with error: {str(e)}")
            self._finish_trajectory()
            raise ExtractionError(str(document_id), str(e)) from e

    def get_trajectory(self) -> Optional[IndexerTrajectory]:
        """Get the current trajectory for debugging."""
        return self._trajectory
