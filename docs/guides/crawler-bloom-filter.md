# Crawler Bloom Filter

For very large crawls, the crawler can switch the visited set to a Bloom filter
for lower memory usage at the cost of a small false-positive rate.

## Configuration

- `CRAWLER_BLOOM_FILTER_THRESHOLD` (default: 10000)
- `CRAWLER_BLOOM_FILTER_ERROR_RATE` (default: 0.001)

When `max_pages` is greater than or equal to the threshold, the crawler uses a
Bloom filter for the `visited` set. Smaller crawls continue to use a normal set.

## Tradeoffs

- Bloom filters can produce false positives (skipping a small number of URLs).
- Memory usage is significantly lower for very large crawls.
