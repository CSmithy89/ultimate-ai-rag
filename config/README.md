# Configuration Profiles

Profiles provide opinionated defaults for the platform. Environment variables always override profile values.

## Profiles

- `minimal.yaml`: low-resource defaults for local testing.
- `standard.yaml`: balanced defaults for typical deployments.
- `enterprise.yaml`: full-featured configuration.
- `custom.yaml.template`: starting point for custom overrides (copy to `custom.yaml`).

## Usage

Set `CONFIG_PROFILE` in `.env` or the environment:

```bash
export CONFIG_PROFILE=standard
```

## Overrides

Any environment variable overrides profile defaults. For example:

```bash
export LLM_PROVIDER=anthropic
```

## Schema

See `config/schema.json` for validation guidance.
