# Ingestion Pipeline Guide

This guide documents the content ingestion pipeline for the Agentic RAG platform, covering all supported data sources, processing stages, and configuration options.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [URL Ingestion (Crawl4AI)](#url-ingestion-crawl4ai)
- [PDF Ingestion (Docling)](#pdf-ingestion-docling)
- [YouTube Transcription](#youtube-transcription)
- [Codebase Ingestion](#codebase-ingestion)
- [External Sync Sources](#external-sync-sources)
- [Chunking Strategies](#chunking-strategies)
- [Graph Ingestion (Graphiti)](#graph-ingestion-graphiti)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The ingestion pipeline transforms raw content into searchable, indexed knowledge using a multi-stage architecture:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Source    │────>│   Parser    │────>│   Chunker   │────>│   Indexer   │
│ (URL/PDF/   │     │ (Crawl4AI/  │     │ (Semantic/  │     │ (pgvector/  │
│  YouTube)   │     │  Docling)   │     │ Hierarchical│     │  Graphiti)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Processing Flow

1. **Source Acquisition**: Content is fetched from URLs, uploaded PDFs, or external APIs
2. **Parsing**: Raw content is converted to structured markdown with metadata extraction
3. **Chunking**: Content is split into semantically coherent chunks with optional contextual enhancement
4. **Indexing**: Chunks are embedded and stored in pgvector and optionally ingested into Graphiti for temporal knowledge graph

### Redis Streams Job Queue

All ingestion jobs are processed asynchronously via Redis Streams:

- `crawl_jobs` - URL crawling tasks
- `parse_jobs` - PDF parsing tasks
- `index_jobs` - Embedding and indexing tasks

---

## URL Ingestion (Crawl4AI)

The platform uses [Crawl4AI](https://github.com/unclecode/crawl4ai) for high-performance web crawling with JavaScript rendering support.

### Features

- **JavaScript Rendering**: Full browser-based rendering for SPAs, React sites, and dynamic content
- **Parallel Crawling**: `MemoryAdaptiveDispatcher` for intelligent parallel crawling that adapts to system resources
- **Intelligent Caching**: Unchanged pages are not re-fetched
- **Proxy Support**: Configurable proxy for accessing blocked sites
- **10x Throughput**: 50 pages in under 30 seconds (vs. traditional httpx approach)
- **Profile-based Configuration**: Fast, thorough, and stealth profiles for different scenarios

### API Endpoint

```http
POST /api/v1/ingest/url
Content-Type: application/json

{
  "url": "https://docs.example.com",
  "tenant_id": "uuid",
  "max_depth": 3,
  "options": {
    "follow_links": true,
    "include_patterns": [".*\\/docs\\/.*"],
    "exclude_patterns": [".*\\/api\\/.*"],
    "rate_limit": 2.0
  }
}
```

### Response

```json
{
  "data": {
    "job_id": "uuid",
    "status": "queued"
  },
  "meta": {
    "requestId": "uuid",
    "timestamp": "2026-01-12T10:00:00Z"
  }
}
```

### Crawl Profiles

Three pre-defined profiles optimize crawling for different scenarios:

#### Fast Profile
For static documentation sites (GitHub Pages, ReadTheDocs, etc.):

```python
CrawlProfile(
    name="fast",
    headless=True,
    stealth=False,
    max_concurrent=10,
    rate_limit=5.0,  # requests/second
    wait_for=None,
    wait_timeout=5.0,
    cache_enabled=True,
)
```

#### Thorough Profile
For SPAs and dynamic content with JavaScript rendering:

```python
CrawlProfile(
    name="thorough",
    headless=True,
    stealth=False,
    max_concurrent=5,
    rate_limit=2.0,
    wait_for="css:body",
    wait_timeout=15.0,
    cache_enabled=True,
)
```

#### Stealth Profile
For bot-protected sites with anti-detection measures:

```python
CrawlProfile(
    name="stealth",
    headless=False,  # Non-headless is less detectable
    stealth=True,
    max_concurrent=3,
    rate_limit=0.5,
    wait_for=None,
    wait_timeout=30.0,
    cache_enabled=False,
)
```

### Auto-Detection

The crawler can automatically select a profile based on URL domain heuristics:

```python
# Priority order: exact domain > suffix > prefix
_EXACT_DOMAIN_PROFILES = {
    "linkedin.com": "stealth",
    "google.com": "stealth",
    # ...
}

_PREFIX_DOMAIN_PROFILES = {
    "docs.": "fast",
    "app.": "thorough",
    # ...
}
```

Enable auto-detection:

```python
from agentic_rag_backend.indexing.crawler import crawl_url

async for page in crawl_url(url, auto_detect_profile=True):
    print(page.title)
```

### User Agent Rotation

For stealth mode, the crawler supports dynamic user agent rotation:

```bash
# Environment variables
CRAWLER_USER_AGENT_STRATEGY=rotate  # rotate, random, static
CRAWLER_USER_AGENT_LIST_PATH=config/user-agents.txt
CRAWLER_USER_AGENT_USE_FAKE=true  # Use fake-useragent library
```

### Fallback Providers

For sites that block standard crawling, fallback to commercial scraping services:

| Provider | Use Case | Configuration |
|----------|----------|---------------|
| Apify | General scraping | `APIFY_API_KEY` |
| Bright Data | Enterprise scraping | `BRIGHTDATA_API_KEY` |

```bash
CRAWL_FALLBACK_ENABLED=true
CRAWL_FALLBACK_PROVIDER=apify  # or brightdata
```

### SSRF Protection

The crawler includes built-in SSRF protection:

- Blocks localhost and 127.0.0.1
- Blocks private IP ranges (10.x, 172.16.x, 192.168.x)
- Blocks link-local and reserved addresses

---

## PDF Ingestion (Docling)

PDF documents are parsed using [Docling](https://github.com/DS4SD/docling) with enhanced table extraction and layout preservation.

### Features

- **Structure Preservation**: Headers, sections, footnotes maintained
- **Table Extraction**: `TableFormerMode.ACCURATE` for precise table structure
- **Content Normalization**: Converts to markdown format
- **Layout Analysis**: Section hierarchy with parent-child relationships
- **Multimodal Support**: Image and figure detection (metadata only)

### API Endpoint

```http
POST /api/v1/ingest/document
Content-Type: multipart/form-data

file: <pdf-file>
tenant_id: "uuid"
metadata: "{\"source\": \"manual\"}"
```

### Response

```json
{
  "data": {
    "job_id": "uuid",
    "status": "queued",
    "filename": "document.pdf",
    "file_size": 1234567
  },
  "meta": {
    "requestId": "uuid",
    "timestamp": "2026-01-12T10:00:00Z"
  }
}
```

### Enhanced Docling Parser

The `EnhancedDoclingParser` provides rich document extraction:

```python
from agentic_rag_backend.indexing.enhanced_docling import EnhancedDoclingParser

parser = EnhancedDoclingParser(
    table_extraction=True,
    preserve_layout=True,
    table_as_markdown=True,
    table_mode="accurate",  # or "fast"
)

layout = parser.parse_document(file_path, document_id, tenant_id)

# Access extracted content
print(f"Pages: {layout.page_count}")
print(f"Tables: {len(layout.tables)}")
print(f"Sections: {len(layout.sections)}")
print(f"Figures: {len(layout.figures)}")
print(f"Footnotes: {len(layout.footnotes)}")
```

### Table-to-Chunk Conversion

Large tables are split into searchable chunks:

```python
chunks = parser.table_to_chunks(
    table=layout.tables[0],
    document_id="doc-123",
    tenant_id="tenant-456",
    chunk_rows=True,
    max_rows_per_chunk=10,
)
```

### Configuration

```bash
# Feature flags
ENHANCED_DOCLING_ENABLED=true
DOCLING_TABLE_EXTRACTION=true
DOCLING_PRESERVE_LAYOUT=true
DOCLING_TABLE_AS_MARKDOWN=true

# Limits
MAX_UPLOAD_SIZE_MB=50
DOCLING_PARSE_TIMEOUT_SECONDS=120
MAX_TABLE_ROWS=10000
MAX_TABLE_COLS=1000
```

### Validation

- PDF magic bytes validation (`%PDF`)
- Password protection detection
- File size limits (100MB max)
- Path traversal protection

---

## YouTube Transcription

YouTube videos are ingested via transcript extraction using the `youtube-transcript-api` library.

### Features

- **Transcript-First**: Fast processing without full video download
- **Multi-Language Support**: Configurable preferred language order
- **Auto-Generated Detection**: Identifies auto-generated vs. manual subtitles
- **Time-Based Chunking**: Preserves timestamp metadata for deep linking

### Supported URL Formats

- `youtube.com/watch?v=VIDEO_ID`
- `youtu.be/VIDEO_ID`
- `youtube.com/embed/VIDEO_ID`
- `youtube.com/shorts/VIDEO_ID`

### Usage

```python
from agentic_rag_backend.indexing.youtube_ingestion import ingest_youtube_video

result = await ingest_youtube_video(
    url="https://youtube.com/watch?v=dQw4w9WgXcQ",
    languages=["en", "en-US"],
    chunk_duration_seconds=60,
)

print(f"Video ID: {result.video_id}")
print(f"Language: {result.language}")
print(f"Auto-generated: {result.is_generated}")
print(f"Chunks: {len(result.chunks)}")
print(f"Duration: {result.duration_seconds}s")
```

### Transcript Chunks

Each chunk includes timestamp metadata:

```python
TranscriptChunk(
    content="This is the transcript text...",
    start_time=60.0,
    end_time=120.0,
    video_id="dQw4w9WgXcQ",
    chunk_index=1,
)
```

### Configuration

```bash
# Preferred languages (comma-separated)
YOUTUBE_PREFERRED_LANGUAGES=en,en-US,en-GB

# Chunk duration in seconds
YOUTUBE_CHUNK_DURATION_SECONDS=60
```

### Error Handling

The module handles common YouTube API errors:

| Error | Cause | Response |
|-------|-------|----------|
| `TranscriptsDisabled` | Video owner disabled subtitles | `YouTubeIngestionError` |
| `NoTranscriptFound` | No transcript in preferred languages | `YouTubeIngestionError` |
| `VideoUnavailable` | Video is private or deleted | `YouTubeIngestionError` |

---

## Codebase Ingestion

**Status**: Implemented (Epic 15)

The codebase intelligence features enable indexing of code repositories for RAG context.

### Features

- AST-based symbol extraction
- File path verification
- API endpoint matching
- Hallucination detection for code generation

### Implementation

See the codebase intelligence module at:
- `/backend/src/agentic_rag_backend/codebase/`

---

## External Sync Sources

The platform supports synchronization with external data sources through pluggable connectors.

### Supported Sources

| Source | Connector | Configuration |
|--------|-----------|---------------|
| Confluence | `ConfluenceConnector` | URL, email, API token |
| AWS S3 | `S3Connector` | Access key, secret key, region |
| Notion | `NotionConnector` | API key, database IDs |
| Google Drive | Planned | - |
| Discord | Planned | - |

### Confluence Connector

Syncs pages from Atlassian Confluence spaces:

```python
from agentic_rag_backend.sync import SyncConfig, SyncSourceType
from agentic_rag_backend.sync.confluence_connector import ConfluenceConnector

config = SyncConfig(
    source_type=SyncSourceType.CONFLUENCE,
    source_id="confluence-main",
    credentials={
        "url": "https://your-domain.atlassian.net/wiki",
        "email": "user@example.com",
        "api_token": "your-api-token",
    },
    settings={
        "spaces": ["SPACE1", "SPACE2"],
    },
)

connector = ConfluenceConnector(config)
result = await connector.sync(incremental=True)

print(f"Items found: {result.items_found}")
print(f"Items synced: {result.items_synced}")
```

### S3 Connector

Syncs documents from AWS S3 buckets:

```python
from agentic_rag_backend.sync.s3_connector import S3Connector

config = SyncConfig(
    source_type=SyncSourceType.S3,
    source_id="s3-docs",
    credentials={
        "aws_access_key_id": "xxx",
        "aws_secret_access_key": "xxx",
        "region_name": "us-east-1",
    },
    settings={
        "bucket": "my-docs",
        "prefix": "documents/",
    },
)

connector = S3Connector(config)
result = await connector.sync()
```

**Supported file types**:
- Text: `.txt`, `.md`, `.json`, `.yaml`, `.yml`, `.csv`
- HTML: `.html`, `.htm`, `.xml`
- Documents: `.pdf`, `.doc`, `.docx`

**Size limit**: 100MB per file

### Notion Connector

Syncs pages from Notion workspaces:

```python
from agentic_rag_backend.sync.notion_connector import NotionConnector

config = SyncConfig(
    source_type=SyncSourceType.NOTION,
    source_id="notion-workspace",
    credentials={
        "api_key": "secret_xxx",
    },
    settings={
        "database_ids": ["db-1", "db-2"],  # Optional: empty = all pages
    },
)

connector = NotionConnector(config)
result = await connector.sync()
```

### Sync Manager

Orchestrate multiple connectors:

```python
from agentic_rag_backend.sync.manager import SyncManager, create_sync_manager

manager = create_sync_manager(
    enabled=True,
    configs=[confluence_config, s3_config, notion_config],
    max_concurrent=3,
)

# Sync all sources
results = await manager.sync_all(incremental=True)

# Sync specific source
result = await manager.sync_source("confluence-main")

# Validate connections
status = await manager.validate_all()
```

### Incremental Sync

Connectors support incremental sync via ETags or last-modified timestamps:

```python
# First sync: full
result = await connector.sync(incremental=False)

# Subsequent syncs: only changed items
result = await connector.sync(incremental=True)
```

### Configuration

```bash
# Enable external sync
EXTERNAL_SYNC_ENABLED=true
SYNC_INTERVAL_MINUTES=60
MAX_ITEMS_PER_SYNC=1000
HTTP_TIMEOUT_SECONDS=30

# Confluence
CONFLUENCE_URL=https://your-domain.atlassian.net/wiki
CONFLUENCE_EMAIL=user@example.com
CONFLUENCE_API_TOKEN=xxx
CONFLUENCE_SPACES=SPACE1,SPACE2

# S3
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION_NAME=us-east-1
S3_BUCKET=my-docs
S3_PREFIX=documents/

# Notion
NOTION_API_KEY=secret_xxx
NOTION_DATABASE_IDS=db-1,db-2
```

---

## Chunking Strategies

### Semantic Chunker

The default chunker uses semantic boundaries for coherent chunks:

```python
from agentic_rag_backend.indexing.chunker import SemanticChunker

chunker = SemanticChunker(
    chunk_size=1000,
    chunk_overlap=200,
    min_chunk_size=100,
)

chunks = chunker.chunk(document)
```

### Hierarchical Chunker

For documents with clear structure (sections, subsections):

```python
from agentic_rag_backend.indexing.hierarchical_chunker import HierarchicalChunker

chunker = HierarchicalChunker(
    max_chunk_size=1000,
    preserve_headers=True,
)

chunks = chunker.chunk(document)
```

### Contextual Retrieval

Adds LLM-generated context to each chunk for improved retrieval:

```python
from agentic_rag_backend.indexing.contextual import ContextualRetrieval

contextual = ContextualRetrieval(
    enabled=True,
    model="claude-3-5-haiku-20241022",  # Cost-effective model
)

enhanced_chunks = await contextual.enhance_chunks(chunks, document)
```

### Configuration

```bash
# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MIN_CHUNK_SIZE=100

# Contextual retrieval
CONTEXTUAL_RETRIEVAL_ENABLED=true
CONTEXTUAL_RETRIEVAL_MODEL=claude-3-5-haiku-20241022
```

---

## Graph Ingestion (Graphiti)

Content is also ingested into the temporal knowledge graph via Graphiti.

```python
from agentic_rag_backend.indexing.graphiti_ingestion import GraphitiIngestor

ingestor = GraphitiIngestor(
    tenant_id=tenant_id,
    source_type="pdf",
)

await ingestor.ingest_document(
    document_id=doc_id,
    chunks=chunks,
    metadata={
        "filename": "report.pdf",
        "source_url": None,
    },
)
```

See the [Memory Platform Guide](memory-platform.md) for details on Graphiti integration.

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Target chunk size in characters |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `MAX_UPLOAD_SIZE_MB` | 50 | Maximum PDF upload size |
| `TEMP_UPLOAD_DIR` | `/tmp/uploads` | Temporary file storage |
| `CRAWL4AI_USER_AGENT` | AgenticRAG-Crawler/1.0 | Default user agent |
| `CRAWL_FALLBACK_ENABLED` | false | Enable fallback scraping services |
| `EXTERNAL_SYNC_ENABLED` | false | Enable external data source sync |
| `CONTEXTUAL_RETRIEVAL_ENABLED` | false | Enable contextual enhancement |
| `ENHANCED_DOCLING_ENABLED` | true | Enable enhanced PDF parsing |

### Rate Limits

| Endpoint | Limit |
|----------|-------|
| `POST /ingest/url` | 10/minute |
| `POST /ingest/document` | 5/minute |

---

## Troubleshooting

### URL Crawling Issues

**Problem**: Pages return empty content

**Solution**:
1. Try the `thorough` profile for JavaScript-heavy sites
2. Increase `js_wait_seconds`
3. Check if site requires authentication

**Problem**: Blocked by bot detection

**Solution**:
1. Use the `stealth` profile
2. Enable user agent rotation
3. Configure proxy URL
4. Try fallback providers (Apify/Bright Data)

### PDF Parsing Issues

**Problem**: Tables not extracted correctly

**Solution**:
1. Ensure `DOCLING_TABLE_EXTRACTION=true`
2. Try `table_mode="accurate"` (slower but more precise)
3. Check if PDF is scanned (OCR may be needed)

**Problem**: Password-protected PDF

**Solution**: Password-protected PDFs are rejected with `PasswordProtectedError`. Remove protection before upload.

### YouTube Issues

**Problem**: No transcript found

**Solution**:
1. Check if video has captions enabled
2. Try different language codes
3. Video may be private or deleted

### External Sync Issues

**Problem**: Confluence connection fails

**Solution**:
1. Verify API token (not password)
2. Check URL format includes `/wiki`
3. Ensure user has space access permissions

**Problem**: S3 access denied

**Solution**:
1. Verify IAM credentials
2. Check bucket policy allows GetObject
3. Verify region configuration

---

## Related Documentation

- [Advanced Retrieval Configuration](advanced-retrieval-configuration.md)
- [Memory Platform Guide](memory-platform.md)
- [Provider Configuration](provider-configuration.md)
- [Protocol Integration Guide](../protocol-integration/README.md)
