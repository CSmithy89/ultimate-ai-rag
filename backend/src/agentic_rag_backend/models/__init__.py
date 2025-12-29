"""Pydantic models for the Agentic RAG Backend."""

from .ingest import (
    CrawlOptions,
    CrawlRequest,
    CrawlResponse,
    JobProgress,
    JobStatus,
)
from .documents import (
    DocumentMetadata,
    DocumentStatus,
    SourceType,
    UnifiedDocument,
)
from .graphs import (
    ChunkData,
    DocumentChunk,
    EntityGraph,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    GraphBuildResult,
    IndexingResult,
    IndexProgress,
    Neo4jEntity,
    Neo4jRelationship,
    RelationshipType,
)
from .copilot import (
    AGUIEvent,
    AGUIEventType,
    ActionRequestEvent,
    CopilotConfig,
    CopilotMessage,
    CopilotRequest,
    MessageRole,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextDeltaEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallEvent,
)

__all__ = [
    # Ingest models
    "CrawlOptions",
    "CrawlRequest",
    "CrawlResponse",
    "JobProgress",
    "JobStatus",
    # Document models
    "DocumentMetadata",
    "DocumentStatus",
    "SourceType",
    "UnifiedDocument",
    # Graph models
    "ChunkData",
    "DocumentChunk",
    "EntityGraph",
    "EntityType",
    "ExtractedEntity",
    "ExtractedRelationship",
    "ExtractionResult",
    "GraphBuildResult",
    "IndexingResult",
    "IndexProgress",
    "Neo4jEntity",
    "Neo4jRelationship",
    "RelationshipType",
    # Copilot models
    "AGUIEvent",
    "AGUIEventType",
    "ActionRequestEvent",
    "CopilotConfig",
    "CopilotMessage",
    "CopilotRequest",
    "MessageRole",
    "RunFinishedEvent",
    "RunStartedEvent",
    "StateSnapshotEvent",
    "TextDeltaEvent",
    "TextMessageEndEvent",
    "TextMessageStartEvent",
    "ToolCallEvent",
]
