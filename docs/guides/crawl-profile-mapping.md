# Crawl Profile Domain Mapping

This project uses a YAML config file to map domains to crawl profiles.

## Default Config Location

- `config/crawl-profiles.yaml`
- Override with `CRAWL_PROFILE_CONFIG_PATH`

Changes require a service restart (rules load at startup).

## Rule Format

```yaml
exact:
  linkedin.com: stealth
suffix:
  .github.io: fast
prefix:
  docs.: fast
```

### Matching Order

1. `exact` domain match (highest priority)
2. `suffix` match
3. `prefix` match

## Adding Custom Mappings

- Add the domain to the appropriate section.
- Use one of: `fast`, `thorough`, `stealth`.
- Restart the backend to apply updates.
