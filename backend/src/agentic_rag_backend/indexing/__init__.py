"""Indexing components for the Knowledge Ingestion Pipeline."""

from .crawler import (
    CrawlerService,
    crawl_url,
    compute_content_hash,
    is_valid_url,
    normalize_url,
    is_same_domain,
    extract_links_from_html,
    extract_links_from_markdown,
    extract_title_from_html,
    extract_title_from_markdown,
    CRAWL4AI_AVAILABLE,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_JS_WAIT_SECONDS,
)
from .chunker import chunk_document, chunk_sections, count_tokens, estimate_chunks, ChunkData
from .graphiti_ingestion import (
    ingest_document_as_episode,
    EpisodeIngestionResult,
    EPISODE_ENTITY_TYPES,
)
from .contextual import (
    ContextualChunkEnricher,
    DocumentContext,
    EnrichedChunk,
    create_contextual_enricher,
)
from .fallback_providers import (
    CrawlResult,
    CrawlProvider,
    ApifyProvider,
    BrightDataProvider,
    FallbackCrawler,
    SimpleCrawlProvider,
    create_fallback_crawler,
)
from .youtube_ingestion import (
    YouTubeIngestionError,
    TranscriptSegment,
    YouTubeTranscriptResult,
    TranscriptChunk,
    YouTubeIngestionResult,
    extract_video_id,
    fetch_transcript,
    chunk_transcript,
    ingest_youtube_video,
)

__all__ = [
    # Crawler (Story 13.3: Crawl4AI migration)
    "CrawlerService",
    "crawl_url",
    "compute_content_hash",
    "is_valid_url",
    "normalize_url",
    "is_same_domain",
    "extract_links_from_html",
    "extract_links_from_markdown",
    "extract_title_from_html",
    "extract_title_from_markdown",
    "CRAWL4AI_AVAILABLE",
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_JS_WAIT_SECONDS",
    # Chunker
    "chunk_document",
    "chunk_sections",
    "count_tokens",
    "estimate_chunks",
    "ChunkData",
    # Graphiti ingestion
    "ingest_document_as_episode",
    "EpisodeIngestionResult",
    "EPISODE_ENTITY_TYPES",
    # Epic 12: Contextual Retrieval
    "ContextualChunkEnricher",
    "DocumentContext",
    "EnrichedChunk",
    "create_contextual_enricher",
    # Epic 13: Fallback Providers
    "CrawlResult",
    "CrawlProvider",
    "ApifyProvider",
    "BrightDataProvider",
    "FallbackCrawler",
    "SimpleCrawlProvider",
    "create_fallback_crawler",
    # Epic 13: YouTube Transcript Ingestion
    "YouTubeIngestionError",
    "TranscriptSegment",
    "YouTubeTranscriptResult",
    "TranscriptChunk",
    "YouTubeIngestionResult",
    "extract_video_id",
    "fetch_transcript",
    "chunk_transcript",
    "ingest_youtube_video",
]
