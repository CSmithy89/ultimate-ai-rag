"""Crawl configuration profiles for different crawling scenarios.

Story 13-4: Implement crawl configuration profiles.

Provides pre-defined profiles for common crawling scenarios:
- FAST: High-speed crawling for static documentation sites
- THOROUGH: For SPAs and dynamic content with wait-for conditions
- STEALTH: For bot-protected sites with anti-detection measures
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

import structlog


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
    """

    name: str
    description: str
    headless: bool
    stealth: bool
    max_concurrent: int
    rate_limit: float  # requests per second
    wait_for: Optional[str]  # CSS selector or JavaScript expression to wait for
    wait_timeout: float  # seconds to wait for wait_for condition
    proxy_config: Optional[str]  # proxy URL or None
    cache_enabled: bool


# Pre-defined crawl profiles
CRAWL_PROFILES: dict[str, CrawlProfile] = {
    "fast": CrawlProfile(
        name="fast",
        description="High-speed crawling for static documentation sites",
        headless=True,
        stealth=False,
        max_concurrent=10,
        rate_limit=5.0,
        wait_for=None,
        wait_timeout=5.0,
        proxy_config=None,
        cache_enabled=True,
    ),
    "thorough": CrawlProfile(
        name="thorough",
        description="For SPAs with dynamic content, waits for JavaScript rendering",
        headless=True,
        stealth=False,
        max_concurrent=5,
        rate_limit=2.0,
        wait_for="css:body",
        wait_timeout=15.0,
        proxy_config=None,
        cache_enabled=True,
    ),
    "stealth": CrawlProfile(
        name="stealth",
        description="For bot-protected sites with anti-detection measures",
        headless=False,  # Non-headless mode is less detectable
        stealth=True,
        max_concurrent=3,
        rate_limit=0.5,  # Slower to avoid triggering rate limits
        wait_for=None,
        wait_timeout=30.0,
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


# Domain heuristics for auto-detection
# Organized by match priority: exact domains > suffixes > prefixes

# Exact domain matches (highest priority) - checked first
_EXACT_DOMAIN_PROFILES: dict[str, str] = {
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
}

# Domain suffix matches (e.g., ".github.io")
_SUFFIX_DOMAIN_PROFILES: dict[str, str] = {
    ".github.io": "fast",
}

# Subdomain prefix matches (lowest priority, e.g., "docs.")
_PREFIX_DOMAIN_PROFILES: dict[str, str] = {
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
}


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
