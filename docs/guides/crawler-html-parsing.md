# Crawler HTML Parsing

Large HTML documents are parsed off the event loop to avoid blocking async crawls.

## Threshold

- Async parsing is used when HTML size is >= 1,000,000 bytes (1 MB).
- Threshold constant: `ASYNC_HTML_PARSE_THRESHOLD_BYTES`.

## Behavior

- Small documents use in-process parsing for simplicity.
- Large documents use `asyncio.to_thread()` to offload parsing.
- Logs include `async_html_parse_enabled` with size and threshold.
