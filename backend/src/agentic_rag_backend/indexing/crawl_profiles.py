"""Crawl configuration profiles for different crawling scenarios.

Story 13-4: Implement crawl configuration profiles.

Provides pre-defined profiles for common crawling scenarios:
- FAST: High-speed crawling for static documentation sites
- THOROUGH: For SPAs and dynamic content with wait-for conditions
- STEALTH: For bot-protected sites with anti-detection measures
"""

from dataclasses import dataclass, replace
import os
from pathlib import Path
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

import structlog
import yaml


logger = structlog.get_logger(__name__)


class CrawlProfileName(str, Enum):
    """Available crawl profile names."""

    FAST = "fast"
    THOROUGH = "thorough"
    STEALTH = "stealth"


@dataclass(frozen=True)
class CrawlProfile:
    """
    Configuration profile for web crawling.

    Frozen dataclass ensures profiles are immutable after creation.

    Rate limit values:
    - fast: 5.0 req/s - Below typical rate limit thresholds for static sites
    - thorough: 2.0 req/s - Balanced for SPAs with JS rendering
    - stealth: 0.5 req/s - Slow to avoid triggering bot detection
    """

    name: str
    description: str
    headless: bool
    stealth: bool
    max_concurrent: int
    rate_limit: float  # requests per second (must be > 0)
    wait_for: Optional[str]  # CSS selector or JavaScript expression to wait for
    wait_timeout: float  # seconds to wait for wait_for condition
    proxy_config: Optional[str]  # proxy URL or None
    cache_enabled: bool

    def __post_init__(self) -> None:
        """Validate profile configuration."""
        if self.rate_limit <= 0:
            raise ValueError(
                f"rate_limit must be positive, got {self.rate_limit}. "
                "Use a small value like 0.1 for very slow crawling."
            )
        if self.max_concurrent < 1:
            raise ValueError(
                f"max_concurrent must be at least 1, got {self.max_concurrent}"
            )


# Pre-defined crawl profiles
CRAWL_PROFILES: dict[str, CrawlProfile] = {
    "fast": CrawlProfile(
        name="fast",
        description="High-speed crawling for static documentation sites",
        headless=True,
        stealth=False,
        max_concurrent=10,  # Balanced default: parallelism without overloading typical docs sites.
        rate_limit=5.0,  # Keeps per-host request rate below common 10 rps limits.
        wait_for=None,
        wait_timeout=5.0,  # Short wait for static content; avoids unnecessary delay.
        proxy_config=None,
        cache_enabled=True,
    ),
    "thorough": CrawlProfile(
        name="thorough",
        description="For SPAs with dynamic content, waits for JavaScript rendering",
        headless=True,
        stealth=False,
        max_concurrent=5,  # Lower concurrency to reduce load on JS-heavy pages.
        rate_limit=2.0,  # Slower rate for dynamic content and stability.
        wait_for="css:body",
        wait_timeout=15.0,  # Longer wait to allow SPA hydration.
        proxy_config=None,
        cache_enabled=True,
    ),
    "stealth": CrawlProfile(
        name="stealth",
        description="For bot-protected sites with anti-detection measures",
        headless=False,  # Non-headless mode is less detectable
        stealth=True,
        max_concurrent=3,  # Conservative concurrency for stealth targets.
        rate_limit=0.5,  # Slow rate to reduce detection risk.
        wait_for=None,
        wait_timeout=30.0,  # Extra buffer for sites with heavy defenses.
        proxy_config=None,  # Can be overridden via settings
        cache_enabled=False,  # Disable cache to ensure fresh content
    ),
}


def get_crawl_profile(name: str) -> CrawlProfile:
    """
    Get a crawl profile by name.

    Args:
        name: Profile name (fast, thorough, or stealth)

    Returns:
        CrawlProfile instance

    Raises:
        ValueError: If profile name is not recognized
    """
    name_lower = name.lower().strip()

    if name_lower not in CRAWL_PROFILES:
        valid_names = ", ".join(sorted(CRAWL_PROFILES.keys()))
        raise ValueError(
            f"Unknown crawl profile: {name!r}. Valid profiles: {valid_names}"
        )

    profile = CRAWL_PROFILES[name_lower]
    logger.debug(
        "crawl_profile_selected",
        profile_name=profile.name,
        description=profile.description,
        headless=profile.headless,
        stealth=profile.stealth,
        max_concurrent=profile.max_concurrent,
        rate_limit=profile.rate_limit,
    )

    return profile


_DEFAULT_DOMAIN_PROFILE_RULES: dict[str, dict[str, str]] = {
    "exact": {
        "linkedin.com": "stealth",
        "facebook.com": "stealth",
        "twitter.com": "stealth",
        "x.com": "stealth",
        "instagram.com": "stealth",
        "cloudflare.com": "stealth",
        "indeed.com": "stealth",
        "glassdoor.com": "stealth",
        "amazon.com": "stealth",
        "google.com": "stealth",
    },
    "suffix": {
        ".github.io": "fast",
    },
    "prefix": {
        "docs.": "fast",
        "documentation.": "fast",
        "readthedocs.": "fast",
        "gitbook.": "fast",
        "docusaurus.": "fast",
        "wiki.": "fast",
        "app.": "thorough",
        "dashboard.": "thorough",
        "console.": "thorough",
        "portal.": "thorough",
    },
}
_DOMAIN_RULES_PATH = Path(os.getenv("CRAWL_PROFILE_CONFIG_PATH", "config/crawl-profiles.yaml"))


def _load_domain_profile_rules() -> dict[str, dict[str, str]]:
    config_path = os.getenv("CRAWL_PROFILE_DOMAIN_CONFIG")
    path = Path(config_path) if config_path else _DOMAIN_RULES_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("domain rules must be a JSON object")

        valid_profiles = {profile.value for profile in CrawlProfileName}

        def _clean_rules(value: object, rule_type: str) -> dict[str, str]:
            if not isinstance(value, dict):
                raise ValueError(f"{rule_type} rules must be a JSON object")
            cleaned: dict[str, str] = {}
            for pattern, profile in value.items():
                if not isinstance(pattern, str) or not isinstance(profile, str):
                    logger.warning(
                        "crawl_profile_rules_invalid_entry",
                        rule_type=rule_type,
                        pattern=pattern,
                        profile=profile,
                    )
                    continue
                profile_name = profile.strip().lower()
                if profile_name not in valid_profiles:
                    logger.warning(
                        "crawl_profile_rules_invalid_profile",
                        rule_type=rule_type,
                        pattern=pattern,
                        profile=profile,
                    )
                    continue
                cleaned[pattern.strip().lower()] = profile_name
            return cleaned

        exact_rules = _clean_rules(raw.get("exact", {}), "exact")
        suffix_rules = _clean_rules(raw.get("suffix", {}), "suffix")
        prefix_rules = _clean_rules(raw.get("prefix", {}), "prefix")
        return {"exact": exact_rules, "suffix": suffix_rules, "prefix": prefix_rules}
    except Exception as exc:
        logger.warning(
            "crawl_profile_rules_load_failed",
            path=str(path),
            error=str(exc),
        )
        return _DEFAULT_DOMAIN_PROFILE_RULES


# Domain heuristics for auto-detection
# Organized by match priority: exact domains > suffixes > prefixes
_DOMAIN_RULES = _load_domain_profile_rules()
_EXACT_DOMAIN_PROFILES = _DOMAIN_RULES.get("exact", {})
_SUFFIX_DOMAIN_PROFILES = _DOMAIN_RULES.get("suffix", {})
_PREFIX_DOMAIN_PROFILES = _DOMAIN_RULES.get("prefix", {})


def get_profile_for_url(url: str) -> str:
    """
    Auto-detect the appropriate crawl profile based on URL domain heuristics.

    This function analyzes the URL's domain to suggest the most appropriate
    crawl profile. Uses prioritized matching:
    1. Exact domain matches (e.g., google.com) - highest priority
    2. Domain suffix matches (e.g., .github.io)
    3. Subdomain prefix matches (e.g., docs.) - lowest priority

    This ensures docs.google.com matches google.com (stealth) not docs. (fast).

    Args:
        url: URL to analyze

    Returns:
        Profile name string (fast, thorough, or stealth)

    Examples:
        >>> get_profile_for_url("https://docs.python.org")
        'fast'
        >>> get_profile_for_url("https://app.example.com/dashboard")
        'thorough'
        >>> get_profile_for_url("https://linkedin.com/jobs")
        'stealth'
        >>> get_profile_for_url("https://docs.google.com")  # google.com takes priority
        'stealth'
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Priority 1: Check exact domain matches (most specific)
        # e.g., "google.com" matches "google.com" or "www.google.com" or "docs.google.com"
        for exact_domain, profile in _EXACT_DOMAIN_PROFILES.items():
            if domain == exact_domain or domain.endswith("." + exact_domain):
                logger.debug(
                    "crawl_profile_auto_detected",
                    url=url,
                    match_type="exact_domain",
                    pattern=exact_domain,
                    suggested_profile=profile,
                )
                return profile

        # Priority 2: Check domain suffix matches
        # e.g., ".github.io" matches "example.github.io"
        for suffix, profile in _SUFFIX_DOMAIN_PROFILES.items():
            if domain.endswith(suffix):
                logger.debug(
                    "crawl_profile_auto_detected",
                    url=url,
                    match_type="suffix",
                    pattern=suffix,
                    suggested_profile=profile,
                )
                return profile

        # Priority 3: Check subdomain prefix matches (lowest priority)
        # e.g., "docs." matches "docs.python.org"
        for prefix, profile in _PREFIX_DOMAIN_PROFILES.items():
            if domain.startswith(prefix):
                logger.debug(
                    "crawl_profile_auto_detected",
                    url=url,
                    match_type="prefix",
                    pattern=prefix,
                    suggested_profile=profile,
                )
                return profile

        # Default to thorough for unknown sites (safe middle ground)
        logger.debug(
            "crawl_profile_auto_detected",
            url=url,
            match_type="default",
            pattern=None,
            suggested_profile="thorough",
        )
        return "thorough"

    except (ValueError, AttributeError) as e:
        # ValueError: malformed URL in urlparse
        # AttributeError: None or unexpected type in URL
        logger.warning(
            "crawl_profile_detection_failed",
            url=url,
            error=str(e),
            error_type=type(e).__name__,
            fallback_profile="thorough",
        )
        return "thorough"


def apply_proxy_override(profile: CrawlProfile, proxy_url: Optional[str]) -> CrawlProfile:
    """
    Create a new profile with an overridden proxy configuration.

    Since CrawlProfile is frozen, this creates a new instance with the
    updated proxy_config field using dataclasses.replace().

    Args:
        profile: Original profile
        proxy_url: Proxy URL to apply, or None to keep original

    Returns:
        New CrawlProfile with updated proxy_config
    """
    if proxy_url is None:
        return profile

    return replace(profile, proxy_config=proxy_url)
