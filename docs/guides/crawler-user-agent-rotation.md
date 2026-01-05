# Crawler User-Agent Rotation

The crawler supports configurable user-agent strategies to reduce bot detection.

## Environment Variables

- `CRAWLER_USER_AGENT_STRATEGY`: `rotate` (default), `random`, or `static`
- `CRAWLER_USER_AGENT_LIST_PATH`: path to a newline-delimited list
- `CRAWL4AI_USER_AGENT`: static user-agent override (used when strategy=static)
- `CRAWLER_USER_AGENT_USE_FAKE`: use `fake-useragent` when strategy=random (optional)

## Default List

The default list is stored at `config/user-agents.txt` and contains 10+ realistic user agents.

## Strategy Behavior

- `rotate`: cycles through the list per crawl batch when `stealth` is enabled
- `random`: selects a random entry per crawl batch when `stealth` is enabled
- `static`: uses `CRAWL4AI_USER_AGENT` if set, otherwise the first entry (session-wide)

Notes:
- Rotation only applies when `stealth=True` because user-agent is set on the browser config.
- Restart the backend after updating the list.
