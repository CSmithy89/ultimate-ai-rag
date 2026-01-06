"""Cross-language query support for multilingual search.

Story 20-H2: Implement Cross-Language Query

This module provides cross-language query capabilities using multilingual
embeddings and optional query translation.

Components:
- LanguageDetector: Detect query language from text
- CrossLanguageEmbedding: Multilingual embedding using sentence-transformers
- QueryTranslator: Translate queries using LLM
- CrossLanguageAdapter: Feature flag wrapper for cross-language features
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import structlog

logger = structlog.get_logger(__name__)

# Default configuration values
DEFAULT_CROSS_LANGUAGE_ENABLED = False
DEFAULT_CROSS_LANGUAGE_EMBEDDING = "intfloat/multilingual-e5-base"
DEFAULT_CROSS_LANGUAGE_TRANSLATION = False

# Language detection patterns (Unicode blocks)
LANGUAGE_PATTERNS: dict[str, re.Pattern] = {
    "zh": re.compile(r"[\u4e00-\u9fff]"),  # Chinese
    "ja": re.compile(r"[\u3040-\u309f\u30a0-\u30ff]"),  # Japanese hiragana/katakana
    "ko": re.compile(r"[\uac00-\ud7af]"),  # Korean
    "ar": re.compile(r"[\u0600-\u06ff]"),  # Arabic
    "he": re.compile(r"[\u0590-\u05ff]"),  # Hebrew
    "ru": re.compile(r"[\u0400-\u04ff]"),  # Cyrillic (Russian, etc.)
    "el": re.compile(r"[\u0370-\u03ff]"),  # Greek
    "th": re.compile(r"[\u0e00-\u0e7f]"),  # Thai
    "hi": re.compile(r"[\u0900-\u097f]"),  # Devanagari (Hindi, etc.)
}

# Common words for Latin-script language detection
LATIN_LANGUAGE_MARKERS: dict[str, set[str]] = {
    "en": {"the", "is", "are", "was", "were", "have", "has", "and", "or", "but", "for"},
    "es": {"el", "la", "los", "las", "es", "son", "está", "están", "y", "de", "que"},
    "fr": {"le", "la", "les", "est", "sont", "et", "de", "que", "pour", "avec"},
    "de": {"der", "die", "das", "ist", "sind", "und", "oder", "für", "mit", "von"},
    "pt": {"o", "a", "os", "as", "é", "são", "está", "e", "de", "que", "para"},
    "it": {"il", "la", "lo", "gli", "le", "è", "sono", "e", "di", "che", "per"},
    "nl": {"de", "het", "een", "is", "zijn", "en", "van", "voor", "met", "dat"},
}


@dataclass
class LanguageDetectionResult:
    """Result of language detection.

    Attributes:
        language: ISO 639-1 language code (e.g., 'en', 'zh', 'es')
        confidence: Confidence score (0.0 to 1.0)
        script: Detected script type (e.g., 'latin', 'cjk', 'cyrillic')
    """

    language: str
    confidence: float = 0.0
    script: str = "unknown"

    @classmethod
    def unknown(cls) -> "LanguageDetectionResult":
        """Return unknown language result."""
        return cls(language="unknown", confidence=0.0, script="unknown")


class LanguageDetector:
    """Detect language from text using Unicode patterns and word markers.

    This is a lightweight detector that doesn't require external libraries.
    For production use with higher accuracy, consider langdetect or fasttext.

    Example:
        detector = LanguageDetector()
        result = detector.detect("Hello, how are you?")
        print(f"Language: {result.language}, Confidence: {result.confidence}")
    """

    def detect(self, text: str) -> LanguageDetectionResult:
        """Detect the language of the given text.

        Args:
            text: Text to analyze

        Returns:
            LanguageDetectionResult with language code and confidence
        """
        if not text or not text.strip():
            return LanguageDetectionResult.unknown()

        text = text.strip()

        # First check for non-Latin scripts
        script_result = self._detect_script(text)
        if script_result.language != "unknown":
            return script_result

        # For Latin script, use word markers
        return self._detect_latin_language(text)

    def _detect_script(self, text: str) -> LanguageDetectionResult:
        """Detect language by Unicode script patterns.

        Args:
            text: Text to analyze

        Returns:
            LanguageDetectionResult or unknown if Latin script
        """
        char_counts: dict[str, int] = {}
        total_chars = 0

        for char in text:
            if char.isalpha():
                total_chars += 1
                for lang, pattern in LANGUAGE_PATTERNS.items():
                    if pattern.match(char):
                        char_counts[lang] = char_counts.get(lang, 0) + 1
                        break

        if total_chars == 0:
            return LanguageDetectionResult.unknown()

        # Find dominant script
        if char_counts:
            best_lang = max(char_counts, key=char_counts.get)  # type: ignore
            confidence = char_counts[best_lang] / total_chars

            if confidence > 0.3:  # At least 30% of chars match
                script = self._get_script_name(best_lang)
                return LanguageDetectionResult(
                    language=best_lang,
                    confidence=min(1.0, confidence * 1.5),  # Boost confidence
                    script=script,
                )

        return LanguageDetectionResult.unknown()

    def _detect_latin_language(self, text: str) -> LanguageDetectionResult:
        """Detect language for Latin-script text using word markers.

        Args:
            text: Text to analyze

        Returns:
            LanguageDetectionResult with detected language
        """
        # Normalize and tokenize
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        if not words:
            return LanguageDetectionResult(
                language="en", confidence=0.3, script="latin"
            )

        # Count marker word matches
        scores: dict[str, int] = {}
        for lang, markers in LATIN_LANGUAGE_MARKERS.items():
            matches = len(words & markers)
            if matches > 0:
                scores[lang] = matches

        if not scores:
            # Default to English for Latin script with no markers
            return LanguageDetectionResult(
                language="en", confidence=0.3, script="latin"
            )

        best_lang = max(scores, key=scores.get)  # type: ignore
        confidence = min(1.0, scores[best_lang] / 5)  # Normalize by max expected

        return LanguageDetectionResult(
            language=best_lang,
            confidence=confidence,
            script="latin",
        )

    @staticmethod
    def _get_script_name(lang: str) -> str:
        """Get script name for a language code."""
        script_map = {
            "zh": "cjk",
            "ja": "cjk",
            "ko": "cjk",
            "ar": "arabic",
            "he": "hebrew",
            "ru": "cyrillic",
            "el": "greek",
            "th": "thai",
            "hi": "devanagari",
        }
        return script_map.get(lang, "other")


class EmbeddingProtocol(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...


@dataclass
class CrossLanguageEmbedding:
    """Multilingual embedding using sentence-transformers.

    Uses models like multilingual-e5 that map text from different
    languages to the same vector space.

    Example:
        embedding = CrossLanguageEmbedding()
        vector = await embedding.embed("Hello world")
        vector_zh = await embedding.embed("你好世界")  # Same semantic space
    """

    model_name: str = DEFAULT_CROSS_LANGUAGE_EMBEDDING
    _model: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize (model loaded lazily)."""
        self._model = None

    def _ensure_model(self) -> None:
        """Lazily load the model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info(
                "cross_language_embedding_initialized",
                model_name=self.model_name,
            )
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for cross-language embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed (any language)

        Returns:
            Embedding vector as list of floats
        """
        self._ensure_model()

        try:
            # Add query prefix for e5 models
            if "e5" in self.model_name.lower():
                text = f"query: {text}"

            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(
                "cross_language_embed_failed",
                text=text[:50],
                error=str(e),
            )
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self._ensure_model()

        try:
            # Add query prefix for e5 models
            if "e5" in self.model_name.lower():
                texts = [f"query: {t}" for t in texts]

            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(
                "cross_language_embed_batch_failed",
                num_texts=len(texts),
                error=str(e),
            )
            raise


class LLMTranslatorProtocol(Protocol):
    """Protocol for LLM-based translation."""

    async def translate(self, text: str, target_language: str) -> str:
        """Translate text to target language."""
        ...


@dataclass
class QueryTranslator:
    """Translate queries using LLM for cross-language search.

    Uses the existing LLM provider to translate queries to a target
    language (typically English for English-indexed content).

    Example:
        translator = QueryTranslator(llm_provider=llm)
        english = await translator.translate("Bonjour le monde", "en")
    """

    llm_provider: Optional[Any] = None
    target_language: str = "en"
    _cache: dict[str, str] = field(default_factory=dict)

    async def translate(
        self,
        text: str,
        target_language: Optional[str] = None,
    ) -> str:
        """Translate text to target language.

        Args:
            text: Text to translate
            target_language: Target language code (default: self.target_language)

        Returns:
            Translated text, or original if translation fails
        """
        target = target_language or self.target_language

        # Check cache
        cache_key = f"{text}:{target}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.llm_provider is None:
            logger.warning("query_translator_no_provider")
            return text

        try:
            # Use LLM for translation
            prompt = (
                f"Translate the following text to {target}. "
                f"Return only the translation, nothing else:\n\n{text}"
            )

            # This assumes the LLM provider has a generate/complete method
            if hasattr(self.llm_provider, "generate"):
                result = await self.llm_provider.generate(prompt)
            elif hasattr(self.llm_provider, "complete"):
                result = await self.llm_provider.complete(prompt)
            else:
                logger.warning(
                    "query_translator_unknown_provider",
                    provider_type=type(self.llm_provider).__name__,
                )
                return text

            translated = result.strip()
            self._cache[cache_key] = translated

            logger.debug(
                "query_translated",
                original=text[:50],
                translated=translated[:50],
                target=target,
            )

            return translated
        except Exception as e:
            logger.warning(
                "query_translation_failed",
                text=text[:50],
                target=target,
                error=str(e),
            )
            return text

    def clear_cache(self) -> int:
        """Clear the translation cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        return count


class CrossLanguageAdapter:
    """Feature flag wrapper for cross-language query support.

    When disabled, passes through to the underlying embedding provider.
    When enabled, uses multilingual embeddings and optional translation.

    Example:
        adapter = CrossLanguageAdapter(
            enabled=settings.cross_language_enabled,
            base_embedding=base_embeddings,
            translation_enabled=settings.cross_language_translation,
        )

        # Automatically handles multilingual queries
        vector = await adapter.embed("Comment ça marche?")
    """

    def __init__(
        self,
        enabled: bool = DEFAULT_CROSS_LANGUAGE_ENABLED,
        base_embedding: Optional[EmbeddingProtocol] = None,
        cross_language_model: str = DEFAULT_CROSS_LANGUAGE_EMBEDDING,
        translation_enabled: bool = DEFAULT_CROSS_LANGUAGE_TRANSLATION,
        llm_provider: Optional[Any] = None,
        target_language: str = "en",
    ) -> None:
        """Initialize the adapter.

        Args:
            enabled: Whether cross-language features are enabled
            base_embedding: Base embedding provider (used when disabled)
            cross_language_model: Multilingual model name
            translation_enabled: Whether to translate queries
            llm_provider: LLM provider for translation
            target_language: Target language for translation
        """
        self._enabled = enabled
        self._base_embedding = base_embedding
        self._cross_language_embedding: Optional[CrossLanguageEmbedding] = None
        self._translator: Optional[QueryTranslator] = None
        self._translation_enabled = translation_enabled
        self._language_detector = LanguageDetector()

        if enabled:
            self._cross_language_embedding = CrossLanguageEmbedding(
                model_name=cross_language_model
            )

            if translation_enabled and llm_provider:
                self._translator = QueryTranslator(
                    llm_provider=llm_provider,
                    target_language=target_language,
                )

            logger.info(
                "cross_language_adapter_enabled",
                model=cross_language_model,
                translation=translation_enabled,
            )
        else:
            logger.info("cross_language_adapter_disabled")

    @property
    def enabled(self) -> bool:
        """Check if cross-language features are enabled."""
        return self._enabled

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        When enabled, uses multilingual embedding. When disabled,
        falls back to base embedding provider.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self._enabled:
            if self._base_embedding:
                return await self._base_embedding.embed(text)
            raise RuntimeError("No embedding provider available")

        # Optionally translate first
        if self._translator and self._translation_enabled:
            detected = self._language_detector.detect(text)
            if detected.language != "en" and detected.confidence > 0.5:
                text = await self._translator.translate(text)

        # Use multilingual embedding
        return await self._cross_language_embedding.embed(text)  # type: ignore

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if not self._enabled:
            if self._base_embedding and hasattr(self._base_embedding, "embed_batch"):
                return await self._base_embedding.embed_batch(texts)  # type: ignore
            # Fall back to individual embedding
            return [await self.embed(t) for t in texts]

        # Optionally translate non-English texts
        if self._translator and self._translation_enabled:
            translated_texts = []
            for text in texts:
                detected = self._language_detector.detect(text)
                if detected.language != "en" and detected.confidence > 0.5:
                    text = await self._translator.translate(text)
                translated_texts.append(text)
            texts = translated_texts

        return await self._cross_language_embedding.embed_batch(texts)  # type: ignore

    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect the language of text.

        Args:
            text: Text to analyze

        Returns:
            LanguageDetectionResult with language code and confidence
        """
        return self._language_detector.detect(text)
