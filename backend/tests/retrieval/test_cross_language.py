"""Unit tests for cross-language query support (Story 20-H2).

Tests cover:
- LanguageDetector for language detection
- CrossLanguageEmbedding (with mock)
- QueryTranslator (with mock)
- CrossLanguageAdapter feature flag behavior
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag_backend.retrieval.cross_language import (
    LanguageDetector,
    LanguageDetectionResult,
    CrossLanguageEmbedding,
    QueryTranslator,
    CrossLanguageAdapter,
    DEFAULT_CROSS_LANGUAGE_ENABLED,
    DEFAULT_CROSS_LANGUAGE_EMBEDDING,
    DEFAULT_CROSS_LANGUAGE_TRANSLATION,
)


# ============================================================================
# LanguageDetectionResult Tests
# ============================================================================

class TestLanguageDetectionResult:
    """Tests for LanguageDetectionResult dataclass."""

    def test_create_result(self):
        """Test creating a detection result."""
        result = LanguageDetectionResult(
            language="en",
            confidence=0.9,
            script="latin",
        )
        assert result.language == "en"
        assert result.confidence == 0.9
        assert result.script == "latin"

    def test_unknown_result(self):
        """Test creating unknown result."""
        result = LanguageDetectionResult.unknown()
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.script == "unknown"


# ============================================================================
# LanguageDetector Tests
# ============================================================================

class TestLanguageDetector:
    """Tests for LanguageDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a LanguageDetector instance."""
        return LanguageDetector()

    def test_detect_english(self, detector):
        """Test detecting English text."""
        result = detector.detect("Hello, how are you today?")
        assert result.language == "en"
        assert result.confidence > 0.0
        assert result.script == "latin"

    def test_detect_spanish(self, detector):
        """Test detecting Spanish text."""
        # Use sentence with marker words: el, la, está, es
        result = detector.detect("El gato está en la casa y es muy lindo")
        assert result.language == "es"
        assert result.confidence > 0.0
        assert result.script == "latin"

    def test_detect_french(self, detector):
        """Test detecting French text."""
        # Use sentence with marker words: le, la, est, et
        result = detector.detect("Le chat est dans la maison et le jardin")
        assert result.language == "fr"
        assert result.confidence > 0.0
        assert result.script == "latin"

    def test_detect_german(self, detector):
        """Test detecting German text."""
        # Use sentence with marker words: der, die, ist, und
        result = detector.detect("Der Hund ist in der Küche und die Katze")
        assert result.language == "de"
        assert result.confidence > 0.0
        assert result.script == "latin"

    def test_detect_chinese(self, detector):
        """Test detecting Chinese text."""
        result = detector.detect("你好世界")
        assert result.language == "zh"
        assert result.confidence > 0.0
        assert result.script == "cjk"

    def test_detect_japanese(self, detector):
        """Test detecting Japanese text (hiragana)."""
        result = detector.detect("こんにちは")
        assert result.language == "ja"
        assert result.confidence > 0.0
        assert result.script == "cjk"

    def test_detect_korean(self, detector):
        """Test detecting Korean text."""
        result = detector.detect("안녕하세요")
        assert result.language == "ko"
        assert result.confidence > 0.0
        assert result.script == "cjk"

    def test_detect_russian(self, detector):
        """Test detecting Russian (Cyrillic) text."""
        result = detector.detect("Привет мир")
        assert result.language == "ru"
        assert result.confidence > 0.0
        assert result.script == "cyrillic"

    def test_detect_arabic(self, detector):
        """Test detecting Arabic text."""
        result = detector.detect("مرحبا بالعالم")
        assert result.language == "ar"
        assert result.confidence > 0.0
        assert result.script == "arabic"

    def test_detect_empty_text(self, detector):
        """Test detecting empty text returns unknown."""
        result = detector.detect("")
        assert result.language == "unknown"
        assert result.confidence == 0.0

    def test_detect_whitespace_only(self, detector):
        """Test detecting whitespace returns unknown."""
        result = detector.detect("   ")
        assert result.language == "unknown"

    def test_detect_mixed_script(self, detector):
        """Test detecting mixed script text."""
        result = detector.detect("Hello 你好 World")
        # Should detect based on dominant script or Chinese chars
        assert result.language in ("en", "zh")

    def test_detect_numbers_only(self, detector):
        """Test detecting numbers only defaults to English."""
        result = detector.detect("12345")
        # No alphabetic chars, defaults to English for Latin
        assert result.language in ("en", "unknown")


# ============================================================================
# CrossLanguageEmbedding Tests
# ============================================================================

class TestCrossLanguageEmbedding:
    """Tests for CrossLanguageEmbedding class."""

    def test_initialization(self):
        """Test embedding initialization."""
        embedding = CrossLanguageEmbedding()
        assert embedding.model_name == DEFAULT_CROSS_LANGUAGE_EMBEDDING
        assert embedding._model is None  # Lazy initialization

    def test_custom_model_name(self):
        """Test custom model name."""
        embedding = CrossLanguageEmbedding(model_name="custom/model")
        assert embedding.model_name == "custom/model"

    @pytest.mark.asyncio
    async def test_embed_with_mock(self):
        """Test embedding with mocked model."""
        import numpy as np

        embedding = CrossLanguageEmbedding()

        # Mock the model - must return numpy array (has tolist method)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        embedding._model = mock_model

        result = await embedding.embed("test query")

        assert result == [0.1, 0.2, 0.3, 0.4]
        # Check e5 prefix was added
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        assert "query:" in call_args

    @pytest.mark.asyncio
    async def test_embed_batch_with_mock(self):
        """Test batch embedding with mocked model."""
        import numpy as np

        embedding = CrossLanguageEmbedding()

        # Mock the model - must return numpy array (has tolist method)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        embedding._model = mock_model

        result = await embedding.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        """Test batch embedding with empty list."""
        embedding = CrossLanguageEmbedding()
        result = await embedding.embed_batch([])
        assert result == []


# ============================================================================
# QueryTranslator Tests
# ============================================================================

class TestQueryTranslator:
    """Tests for QueryTranslator class."""

    def test_initialization(self):
        """Test translator initialization."""
        translator = QueryTranslator()
        assert translator.llm_provider is None
        assert translator.target_language == "en"

    @pytest.mark.asyncio
    async def test_translate_without_provider(self):
        """Test translation without LLM provider returns original."""
        translator = QueryTranslator()
        result = await translator.translate("Bonjour")
        assert result == "Bonjour"  # Returns original

    @pytest.mark.asyncio
    async def test_translate_with_mock_provider(self):
        """Test translation with mocked LLM provider."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="Hello")

        translator = QueryTranslator(llm_provider=mock_llm)
        result = await translator.translate("Bonjour", "en")

        assert result == "Hello"
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_translate_caching(self):
        """Test translation caching."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="Hello")

        translator = QueryTranslator(llm_provider=mock_llm)

        # First call
        result1 = await translator.translate("Bonjour", "en")
        # Second call (should use cache)
        result2 = await translator.translate("Bonjour", "en")

        assert result1 == result2 == "Hello"
        # LLM should only be called once due to caching
        assert mock_llm.generate.call_count == 1

    def test_clear_cache(self):
        """Test clearing the translation cache."""
        translator = QueryTranslator()
        translator._cache = {"test:en": "translation"}

        count = translator.clear_cache()

        assert count == 1
        assert len(translator._cache) == 0


# ============================================================================
# CrossLanguageAdapter Tests
# ============================================================================

class TestCrossLanguageAdapter:
    """Tests for CrossLanguageAdapter class."""

    def test_adapter_disabled_by_default(self):
        """Test adapter is disabled by default."""
        adapter = CrossLanguageAdapter()
        assert not adapter.enabled

    def test_adapter_enabled(self):
        """Test adapter can be enabled."""
        mock_base = MagicMock()
        adapter = CrossLanguageAdapter(enabled=True, base_embedding=mock_base)
        assert adapter.enabled

    @pytest.mark.asyncio
    async def test_embed_when_disabled(self):
        """Test embed falls back to base when disabled."""
        mock_base = MagicMock()
        mock_base.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        adapter = CrossLanguageAdapter(enabled=False, base_embedding=mock_base)
        result = await adapter.embed("test query")

        assert result == [0.1, 0.2, 0.3]
        mock_base.embed.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_embed_when_enabled(self):
        """Test embed uses cross-language model when enabled."""
        mock_base = MagicMock()
        adapter = CrossLanguageAdapter(enabled=True, base_embedding=mock_base)

        # Mock the cross-language embedding
        adapter._cross_language_embedding = MagicMock()
        adapter._cross_language_embedding.embed = AsyncMock(
            return_value=[0.4, 0.5, 0.6]
        )

        result = await adapter.embed("test query")

        assert result == [0.4, 0.5, 0.6]
        adapter._cross_language_embedding.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_with_translation(self):
        """Test embed with translation enabled."""
        mock_base = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="Hello world")

        adapter = CrossLanguageAdapter(
            enabled=True,
            base_embedding=mock_base,
            translation_enabled=True,
            llm_provider=mock_llm,
        )

        # Mock the cross-language embedding
        adapter._cross_language_embedding = MagicMock()
        adapter._cross_language_embedding.embed = AsyncMock(
            return_value=[0.1, 0.2]
        )

        # Query in French (should be translated)
        result = await adapter.embed("Bonjour le monde")

        # Translation should be called for non-English
        assert result == [0.1, 0.2]

    def test_detect_language(self):
        """Test language detection through adapter."""
        adapter = CrossLanguageAdapter(enabled=False)
        result = adapter.detect_language("Hello world")

        assert result.language == "en"
        assert result.script == "latin"

    @pytest.mark.asyncio
    async def test_embed_batch_when_disabled(self):
        """Test batch embed falls back to base when disabled."""
        mock_base = MagicMock()
        mock_base.embed_batch = AsyncMock(return_value=[[0.1], [0.2]])

        adapter = CrossLanguageAdapter(enabled=False, base_embedding=mock_base)
        result = await adapter.embed_batch(["text1", "text2"])

        assert len(result) == 2
        mock_base.embed_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_raises_without_provider(self):
        """Test embed raises when disabled and no base provider."""
        adapter = CrossLanguageAdapter(enabled=False, base_embedding=None)

        with pytest.raises(RuntimeError, match="No embedding provider"):
            await adapter.embed("test")


# ============================================================================
# Default Constants Tests
# ============================================================================

class TestDefaultConstants:
    """Tests for default configuration constants."""

    def test_default_cross_language_disabled(self):
        """Test cross-language is disabled by default."""
        assert DEFAULT_CROSS_LANGUAGE_ENABLED is False

    def test_default_embedding_model(self):
        """Test default embedding model is multilingual-e5."""
        assert "multilingual" in DEFAULT_CROSS_LANGUAGE_EMBEDDING.lower()
        assert "e5" in DEFAULT_CROSS_LANGUAGE_EMBEDDING.lower()

    def test_default_translation_disabled(self):
        """Test translation is disabled by default."""
        assert DEFAULT_CROSS_LANGUAGE_TRANSLATION is False


# ============================================================================
# Integration-Style Tests
# ============================================================================

class TestCrossLanguageIntegration:
    """Integration-style tests for cross-language support."""

    @pytest.mark.asyncio
    async def test_multilingual_detection_and_embedding(self):
        """Test language detection with embedding flow."""
        adapter = CrossLanguageAdapter(enabled=True)

        # Mock the embedding model
        adapter._cross_language_embedding = MagicMock()
        adapter._cross_language_embedding.embed = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )

        # Test with different languages
        languages = [
            ("Hello world", "en"),
            ("你好世界", "zh"),
            ("Bonjour le monde", "fr"),
        ]

        for text, expected_lang in languages:
            detected = adapter.detect_language(text)
            result = await adapter.embed(text)

            assert detected.language == expected_lang
            assert len(result) == 3

    def test_language_detection_confidence(self):
        """Test that detection has reasonable confidence."""
        detector = LanguageDetector()

        # Sentence with many English marker words: the, is, are, and, for, but
        result = detector.detect("The dog is here and the cat is there for the day but the weather is bad")
        assert result.language == "en"
        assert result.confidence > 0.3

        # CJK characters should have high confidence
        result = detector.detect("这是一个中文句子")
        assert result.language == "zh"
        assert result.confidence > 0.5
