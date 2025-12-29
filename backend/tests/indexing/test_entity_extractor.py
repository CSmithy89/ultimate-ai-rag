"""Tests for the entity extractor module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag_backend.indexing.entity_extractor import (
    EntityExtractor,
    DEFAULT_EXTRACTION_MODEL,
)


class TestEntityExtractor:
    """Tests for EntityExtractor class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        return AsyncMock()

    @pytest.fixture
    def extractor(self, mock_openai_client):
        """Create an EntityExtractor with mocked client."""
        with patch("agentic_rag_backend.indexing.entity_extractor.AsyncOpenAI") as mock_class:
            mock_class.return_value = mock_openai_client
            ext = EntityExtractor(api_key="test-key", model=DEFAULT_EXTRACTION_MODEL)
            ext.client = mock_openai_client
            return ext

    def _create_mock_response(self, content: str):
        """Create a mock API response."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = content
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    @pytest.mark.asyncio
    async def test_extract_entities_empty_response(self, extractor, mock_openai_client):
        """Test extraction with empty entities response."""
        response_json = '{"entities": [], "relationships": []}'
        mock_response = self._create_mock_response(response_json)
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await extractor.extract_from_chunk(
            chunk_content="Some content here that is long enough to process.",
            chunk_id="test-chunk-2",
        )

        assert len(result.entities) == 0
        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_extract_entities_invalid_json(self, extractor, mock_openai_client):
        """Test extraction with invalid JSON response."""
        mock_response = self._create_mock_response("not valid json {")
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await extractor.extract_from_chunk(
            chunk_content="Some content that is long enough to process properly.",
            chunk_id="test-chunk-3",
        )

        # Should return empty result on parse failure
        assert len(result.entities) == 0
        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_extract_entities_short_chunk(self, extractor):
        """Test extraction skips very short chunks."""
        result = await extractor.extract_from_chunk(
            chunk_content="Hi",
            chunk_id="test-chunk-4",
        )

        assert len(result.entities) == 0
        assert len(result.relationships) == 0
        assert result.processing_time_ms == 0

    @pytest.mark.asyncio
    async def test_parse_valid_extraction_response(self, extractor, mock_openai_client):
        """Test parsing a valid extraction response."""
        response_json = '''
        {
            "entities": [
                {"name": "Python", "type": "Technology", "description": "Programming language"}
            ],
            "relationships": []
        }
        '''
        mock_response = self._create_mock_response(response_json)
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await extractor.extract_from_chunk(
            chunk_content="Python is a great programming language for data science and machine learning.",
            chunk_id="test-chunk-5",
        )

        assert len(result.entities) == 1
        assert result.entities[0].name == "Python"
        assert result.entities[0].type == "Technology"
