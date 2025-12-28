"""Pydantic models for graph entities and relationships."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities that can be extracted."""

    PERSON = "Person"
    ORGANIZATION = "Organization"
    TECHNOLOGY = "Technology"
    CONCEPT = "Concept"
    LOCATION = "Location"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""

    MENTIONS = "MENTIONS"
    AUTHORED_BY = "AUTHORED_BY"
    PART_OF = "PART_OF"
    USES = "USES"
    RELATED_TO = "RELATED_TO"


class ExtractedEntity(BaseModel):
    """Entity extracted from text by the IndexerAgent."""

    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type (Person, Organization, Technology, Concept, Location)")
    description: Optional[str] = Field(None, description="Brief description of the entity")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "OpenAI",
                    "type": "Organization",
                    "description": "AI research company that developed GPT-4",
                }
            ]
        }
    }


class ExtractedRelationship(BaseModel):
    """Relationship extracted from text by the IndexerAgent."""

    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    type: str = Field(..., description="Relationship type (MENTIONS, AUTHORED_BY, USES, PART_OF, RELATED_TO)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source": "GPT-4",
                    "target": "OpenAI",
                    "type": "AUTHORED_BY",
                    "confidence": 0.95,
                }
            ]
        }
    }


class EntityGraph(BaseModel):
    """Collection of entities and relationships from extraction."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """A chunk of document content with embedding."""

    id: UUID = Field(..., description="Unique chunk identifier")
    tenant_id: UUID = Field(..., description="Tenant identifier")
    document_id: UUID = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="Position in document")
    token_count: int = Field(..., ge=0, description="Number of tokens")
    embedding: Optional[list[float]] = Field(None, description="1536-dim embedding vector")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Additional chunk metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkData(BaseModel):
    """Lightweight chunk data for processing pipeline."""

    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="Position in document")
    token_count: int = Field(..., ge=0, description="Number of tokens")
    start_char: int = Field(..., ge=0, description="Start character position")
    end_char: int = Field(..., ge=0, description="End character position")


class Neo4jEntity(BaseModel):
    """Entity as stored in Neo4j."""

    id: str = Field(..., description="UUID string")
    tenant_id: str = Field(..., description="Tenant UUID string")
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    description: Optional[str] = Field(None, description="Entity description")
    properties: Optional[dict[str, Any]] = Field(default=None, description="Additional properties")
    source_chunks: list[str] = Field(
        default_factory=list, description="Chunk IDs where entity appears"
    )
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Neo4jRelationship(BaseModel):
    """Relationship as stored in Neo4j."""

    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: str = Field(..., description="Relationship type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source_chunk: Optional[str] = Field(None, description="Chunk ID where relationship found")
    created_at: Optional[datetime] = None


class IndexProgress(BaseModel):
    """Progress metrics for an indexing job."""

    chunks_processed: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    entities_extracted: int = Field(default=0, ge=0)
    relationships_extracted: int = Field(default=0, ge=0)
    processing_time_ms: Optional[int] = Field(None, ge=0)


class ExtractionResult(BaseModel):
    """Result of entity extraction from a chunk."""

    chunk_id: str = Field(..., description="Chunk identifier")
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    processing_time_ms: int = Field(default=0, ge=0)


class GraphBuildResult(BaseModel):
    """Result of building graph from extracted entities."""

    entities_created: int = Field(default=0, ge=0)
    entities_deduplicated: int = Field(default=0, ge=0)
    relationships_created: int = Field(default=0, ge=0)
    relationships_skipped: int = Field(default=0, ge=0)


class IndexingResult(BaseModel):
    """Complete result of indexing a document."""

    document_id: str = Field(..., description="Document identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    chunks_created: int = Field(default=0, ge=0)
    entities_extracted: int = Field(default=0, ge=0)
    relationships_extracted: int = Field(default=0, ge=0)
    entities_deduplicated: int = Field(default=0, ge=0)
    processing_time_ms: int = Field(default=0, ge=0)
