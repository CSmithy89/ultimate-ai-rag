"""LLM-based entity and relationship extraction with structured output."""

import json
import time
from typing import Optional

import structlog
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agentic_rag_backend.core.errors import ExtractionError
from agentic_rag_backend.models.graphs import (
    EntityGraph,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)

logger = structlog.get_logger(__name__)

# Default extraction model
DEFAULT_EXTRACTION_MODEL = "gpt-4o"

# Entity extraction prompt template
# Note: Curly braces in JSON example are escaped with {{ and }} for Python formatting
EXTRACTION_PROMPT = """You are an expert at extracting structured information from text.

Given the following text chunk, extract:
1. Named entities (people, organizations, technologies, concepts, locations)
2. Relationships between entities

Entity Types:
- Person: Named individuals (e.g., "Elon Musk", "Ada Lovelace")
- Organization: Companies, institutions, groups (e.g., "OpenAI", "MIT")
- Technology: Software, frameworks, tools, systems (e.g., "Python", "Kubernetes", "GPT-4")
- Concept: Abstract ideas, methodologies, principles (e.g., "Machine Learning", "Agile")
- Location: Places, regions, addresses (e.g., "San Francisco", "Europe")

Relationship Types:
- MENTIONS: Entity A mentions or references entity B
- AUTHORED_BY: Entity A was created/written by entity B
- USES: Entity A uses or depends on entity B
- PART_OF: Entity A is part of or belongs to entity B
- RELATED_TO: Entity A has a general relationship with entity B

Output your response as a valid JSON object with this exact structure:
{{
  "entities": [
    {{"name": "exact entity name", "type": "Person|Organization|Technology|Concept|Location", "description": "brief description"}}
  ],
  "relationships": [
    {{"source": "source entity name", "target": "target entity name", "type": "MENTIONS|AUTHORED_BY|USES|PART_OF|RELATED_TO", "confidence": 0.0-1.0}}
  ]
}}

Guidelines:
- Extract 5-15 entities if present (don't over-extract generic terms)
- Only extract relationships between entities you've identified
- Confidence should reflect how clearly the relationship is stated (0.5-1.0)
- Description should be 10-30 words explaining the entity's role in context
- Normalize entity names (e.g., "GPT-4" not "gpt4" or "GPT 4")
- Do not extract common words or phrases that aren't true named entities

Text to analyze:
{chunk_content}

Respond only with the JSON object, no additional text."""


class EntityExtractor:
    """
    LLM-based entity and relationship extractor.

    Uses GPT-4o with structured JSON output for reliable extraction
    of entities and relationships from document chunks.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EXTRACTION_MODEL,
    ) -> None:
        """
        Initialize entity extractor.

        Args:
            api_key: OpenAI API key
            model: Model ID for extraction (default: gpt-4o)

        Raises:
            ValueError: If api_key is not provided
        """
        if not api_key:
            raise ValueError("OpenAI API key is required for entity extraction")
        self.client = AsyncOpenAI(api_key=api_key, timeout=30.0)
        self.model = model
        logger.info("entity_extractor_initialized", model=model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            "extraction_retry",
            attempt=retry_state.attempt_number,
            error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
        ),
    )
    async def _call_llm(self, prompt: str) -> str:
        """
        Make LLM API call with retry logic.

        Args:
            prompt: Prompt to send to the model

        Returns:
            Raw response content

        Raises:
            ExtractionError: If API call fails after retries
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise information extraction assistant that outputs only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ExtractionError("unknown", f"LLM API call failed: {str(e)}") from e

    def _parse_extraction_response(
        self,
        response: str,
        chunk_id: str,
    ) -> EntityGraph:
        """
        Parse LLM response into EntityGraph.

        Args:
            response: Raw JSON response from LLM
            chunk_id: Chunk identifier for error reporting

        Returns:
            EntityGraph with extracted entities and relationships
        """
        try:
            data = json.loads(response)

            entities = []
            for ent_data in data.get("entities", []):
                # Validate entity type
                ent_type = ent_data.get("type", "Concept")
                valid_types = {"Person", "Organization", "Technology", "Concept", "Location"}
                if ent_type not in valid_types:
                    ent_type = "Concept"  # Default fallback

                entities.append(ExtractedEntity(
                    name=ent_data.get("name", "").strip(),
                    type=ent_type,
                    description=ent_data.get("description"),
                ))

            relationships = []
            # Get set of entity names for validation
            entity_names = {e.name.lower() for e in entities}

            for rel_data in data.get("relationships", []):
                source = rel_data.get("source", "").strip()
                target = rel_data.get("target", "").strip()
                rel_type = rel_data.get("type", "RELATED_TO")

                # Validate relationship type
                valid_rel_types = {"MENTIONS", "AUTHORED_BY", "USES", "PART_OF", "RELATED_TO"}
                if rel_type not in valid_rel_types:
                    rel_type = "RELATED_TO"

                # Only include relationships between extracted entities
                if source.lower() in entity_names and target.lower() in entity_names:
                    confidence = rel_data.get("confidence", 0.7)
                    # Clamp confidence to valid range
                    confidence = max(0.0, min(1.0, confidence))

                    relationships.append(ExtractedRelationship(
                        source=source,
                        target=target,
                        type=rel_type,
                        confidence=confidence,
                    ))

            # Filter out entities with empty names
            entities = [e for e in entities if e.name]

            return EntityGraph(entities=entities, relationships=relationships)

        except json.JSONDecodeError as e:
            logger.warning(
                "extraction_parse_failed",
                chunk_id=chunk_id,
                error=str(e),
                response_preview=response[:200] if response else None,
            )
            return EntityGraph(entities=[], relationships=[])

    async def extract_from_chunk(
        self,
        chunk_content: str,
        chunk_id: str,
    ) -> ExtractionResult:
        """
        Extract entities and relationships from a single chunk.

        Args:
            chunk_content: Text content of the chunk
            chunk_id: Chunk identifier for logging

        Returns:
            ExtractionResult with entities and relationships
        """
        start_time = time.perf_counter()

        # Skip very short chunks
        if len(chunk_content.strip()) < 50:
            logger.debug("chunk_too_short", chunk_id=chunk_id, length=len(chunk_content))
            return ExtractionResult(
                chunk_id=chunk_id,
                entities=[],
                relationships=[],
                processing_time_ms=0,
            )

        prompt = EXTRACTION_PROMPT.format(chunk_content=chunk_content)

        try:
            response = await self._call_llm(prompt)
            graph = self._parse_extraction_response(response, chunk_id)

            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            logger.info(
                "chunk_extracted",
                chunk_id=chunk_id,
                entities=len(graph.entities),
                relationships=len(graph.relationships),
                processing_time_ms=processing_time_ms,
            )

            return ExtractionResult(
                chunk_id=chunk_id,
                entities=graph.entities,
                relationships=graph.relationships,
                processing_time_ms=processing_time_ms,
            )

        except ExtractionError:
            raise
        except Exception as e:
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(
                "extraction_failed",
                chunk_id=chunk_id,
                error=str(e),
            )
            raise ExtractionError(chunk_id, str(e)) from e

    async def extract_from_chunks(
        self,
        chunks: list[tuple[str, str]],  # List of (chunk_id, content) tuples
    ) -> list[ExtractionResult]:
        """
        Extract entities and relationships from multiple chunks.

        Processes chunks sequentially to respect API rate limits.
        Failed chunks are logged but don't stop processing.

        Args:
            chunks: List of (chunk_id, content) tuples

        Returns:
            List of ExtractionResult objects
        """
        results = []

        for chunk_id, content in chunks:
            try:
                result = await self.extract_from_chunk(content, chunk_id)
                results.append(result)
            except ExtractionError as e:
                # Log but continue processing other chunks
                logger.error(
                    "chunk_extraction_failed",
                    chunk_id=chunk_id,
                    error=str(e),
                )
                # Add empty result for failed chunk
                results.append(ExtractionResult(
                    chunk_id=chunk_id,
                    entities=[],
                    relationships=[],
                    processing_time_ms=0,
                ))

        total_entities = sum(len(r.entities) for r in results)
        total_relationships = sum(len(r.relationships) for r in results)

        logger.info(
            "batch_extraction_complete",
            chunks_processed=len(results),
            total_entities=total_entities,
            total_relationships=total_relationships,
        )

        return results


# Global entity extractor instance
_entity_extractor: Optional[EntityExtractor] = None


async def get_entity_extractor(
    api_key: Optional[str] = None,
    model: str = DEFAULT_EXTRACTION_MODEL,
) -> EntityExtractor:
    """
    Get or create the global entity extractor instance.

    Args:
        api_key: OpenAI API key. Required on first call.
        model: Model to use for extraction

    Returns:
        EntityExtractor instance
    """
    global _entity_extractor
    if _entity_extractor is None:
        if api_key is None:
            raise ExtractionError("unknown", "API key required for first initialization")
        _entity_extractor = EntityExtractor(api_key, model)
    return _entity_extractor
