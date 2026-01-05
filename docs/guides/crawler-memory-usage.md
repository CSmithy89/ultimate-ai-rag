# Crawler Memory Usage

Large crawls can use significant memory, especially when many URLs are held in
memory for batching and link discovery.

## Large Crawl Considerations

- For crawls over 100 pages, expect higher memory usage due to:
  - in-memory page content
  - link discovery queues
  - crawl result buffering

## Recommendations

- Use streaming (`stream=true`) for large crawls to reduce memory pressure.
- Keep `max_pages` bounded for predictable memory use.
- Consider enabling the bloom filter for very large crawls:
  - `CRAWLER_BLOOM_FILTER_THRESHOLD`
  - `CRAWLER_BLOOM_FILTER_ERROR_RATE`

## Related Settings

- `max_pages`
- `max_concurrent`
- `CRAWL4AI_CACHE_ENABLED`
