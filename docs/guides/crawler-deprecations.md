# Crawler Deprecations

The following crawler helper aliases are deprecated and will be removed in v2.0.

## Deprecated Functions

- `extract_links(html, base_url)` → use `get_links(html, base_url)`
- `extract_title(html)` → use `get_title(html)`

## Timeline

- Deprecation warnings are emitted starting now.
- Removal target: v2.0.

## Migration

Update any imports or calls to use the new canonical names.
