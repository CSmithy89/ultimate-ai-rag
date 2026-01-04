"""Tests for crawl configuration profiles.

Story 13-4: Implement crawl configuration profiles.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from agentic_rag_backend.indexing.crawl_profiles import (
    CrawlProfile,
    CrawlProfileName,
    CRAWL_PROFILES,
    get_crawl_profile,
    get_profile_for_url,
    apply_proxy_override,
)


class TestCrawlProfileName:
    """Tests for CrawlProfileName enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert CrawlProfileName.FAST.value == "fast"
        assert CrawlProfileName.THOROUGH.value == "thorough"
        assert CrawlProfileName.STEALTH.value == "stealth"

    def test_enum_is_string(self):
        """Test that enum values are strings."""
        assert isinstance(CrawlProfileName.FAST.value, str)
        assert CrawlProfileName.FAST == "fast"


class TestCrawlProfile:
    """Tests for CrawlProfile dataclass."""

    def test_create_profile(self):
        """Test creating a CrawlProfile."""
        profile = CrawlProfile(
            name="test",
            description="Test profile",
            headless=True,
            stealth=False,
            max_concurrent=5,
            rate_limit=2.0,
            wait_for=None,
            wait_timeout=10.0,
            proxy_config=None,
            cache_enabled=True,
        )
        assert profile.name == "test"
        assert profile.description == "Test profile"
        assert profile.headless is True
        assert profile.stealth is False
        assert profile.max_concurrent == 5
        assert profile.rate_limit == 2.0
        assert profile.wait_for is None
        assert profile.wait_timeout == 10.0
        assert profile.proxy_config is None
        assert profile.cache_enabled is True

    def test_profile_is_frozen(self):
        """Test that CrawlProfile is immutable."""
        profile = CrawlProfile(
            name="test",
            description="Test",
            headless=True,
            stealth=False,
            max_concurrent=5,
            rate_limit=2.0,
            wait_for=None,
            wait_timeout=10.0,
            proxy_config=None,
            cache_enabled=True,
        )
        with pytest.raises(AttributeError):
            profile.name = "modified"


class TestCrawlProfiles:
    """Tests for pre-defined CRAWL_PROFILES."""

    def test_fast_profile_exists(self):
        """Test that fast profile exists with correct settings."""
        profile = CRAWL_PROFILES["fast"]
        assert profile.name == "fast"
        assert profile.headless is True
        assert profile.stealth is False
        assert profile.max_concurrent == 10
        assert profile.rate_limit == 5.0
        assert profile.cache_enabled is True

    def test_thorough_profile_exists(self):
        """Test that thorough profile exists with correct settings."""
        profile = CRAWL_PROFILES["thorough"]
        assert profile.name == "thorough"
        assert profile.headless is True
        assert profile.stealth is False
        assert profile.max_concurrent == 5
        assert profile.rate_limit == 2.0
        assert profile.wait_for == "css:body"
        assert profile.wait_timeout == 15.0
        assert profile.cache_enabled is True

    def test_stealth_profile_exists(self):
        """Test that stealth profile exists with correct settings."""
        profile = CRAWL_PROFILES["stealth"]
        assert profile.name == "stealth"
        assert profile.headless is False  # Non-headless for stealth
        assert profile.stealth is True
        assert profile.max_concurrent == 3
        assert profile.rate_limit == 0.5
        assert profile.wait_timeout == 30.0
        assert profile.cache_enabled is False  # Disabled for fresh content

    def test_all_profiles_have_required_fields(self):
        """Test that all profiles have all required fields."""
        required_fields = [
            "name",
            "description",
            "headless",
            "stealth",
            "max_concurrent",
            "rate_limit",
            "wait_for",
            "wait_timeout",
            "proxy_config",
            "cache_enabled",
        ]
        for profile_name, profile in CRAWL_PROFILES.items():
            for field in required_fields:
                assert hasattr(profile, field), f"Profile {profile_name} missing {field}"


class TestGetCrawlProfile:
    """Tests for get_crawl_profile function."""

    def test_get_fast_profile(self):
        """Test getting fast profile by name."""
        profile = get_crawl_profile("fast")
        assert profile.name == "fast"
        assert profile == CRAWL_PROFILES["fast"]

    def test_get_thorough_profile(self):
        """Test getting thorough profile by name."""
        profile = get_crawl_profile("thorough")
        assert profile.name == "thorough"
        assert profile == CRAWL_PROFILES["thorough"]

    def test_get_stealth_profile(self):
        """Test getting stealth profile by name."""
        profile = get_crawl_profile("stealth")
        assert profile.name == "stealth"
        assert profile == CRAWL_PROFILES["stealth"]

    def test_case_insensitive(self):
        """Test that profile names are case-insensitive."""
        assert get_crawl_profile("FAST") == CRAWL_PROFILES["fast"]
        assert get_crawl_profile("Fast") == CRAWL_PROFILES["fast"]
        assert get_crawl_profile("THOROUGH") == CRAWL_PROFILES["thorough"]
        assert get_crawl_profile("STEALTH") == CRAWL_PROFILES["stealth"]

    def test_strips_whitespace(self):
        """Test that whitespace is stripped from profile names."""
        assert get_crawl_profile("  fast  ") == CRAWL_PROFILES["fast"]
        assert get_crawl_profile("\nthorough\t") == CRAWL_PROFILES["thorough"]

    def test_invalid_profile_raises_error(self):
        """Test that invalid profile name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_crawl_profile("invalid")
        assert "Unknown crawl profile" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_invalid_profile_shows_valid_options(self):
        """Test that error message shows valid profile options."""
        with pytest.raises(ValueError) as exc_info:
            get_crawl_profile("unknown")
        error_msg = str(exc_info.value)
        assert "fast" in error_msg
        assert "thorough" in error_msg
        assert "stealth" in error_msg


class TestGetProfileForUrl:
    """Tests for get_profile_for_url auto-detection function."""

    # Documentation sites should use fast profile
    @pytest.mark.parametrize(
        "url",
        [
            "https://docs.python.org/3/",
            "https://documentation.example.com/api",
            "https://readthedocs.io/projects/example",
            "https://gitbook.example.com/docs",
            "https://example.github.io/docs",
            "https://wiki.example.org/Main_Page",
        ],
    )
    def test_docs_sites_suggest_fast(self, url):
        """Test that documentation sites suggest fast profile."""
        assert get_profile_for_url(url) == "fast"

    # App/dashboard sites should use thorough profile
    @pytest.mark.parametrize(
        "url",
        [
            "https://app.example.com/dashboard",
            "https://dashboard.example.com/",
            "https://console.example.com/settings",
            "https://portal.example.com/login",
        ],
    )
    def test_app_sites_suggest_thorough(self, url):
        """Test that app/dashboard sites suggest thorough profile."""
        assert get_profile_for_url(url) == "thorough"

    # Bot-protected sites should use stealth profile
    @pytest.mark.parametrize(
        "url",
        [
            "https://linkedin.com/in/example",
            "https://www.linkedin.com/jobs",
            "https://facebook.com/page",
            "https://twitter.com/user",
            "https://x.com/user",
            "https://indeed.com/jobs",
            "https://glassdoor.com/company",
            "https://amazon.com/product",
        ],
    )
    def test_protected_sites_suggest_stealth(self, url):
        """Test that bot-protected sites suggest stealth profile."""
        assert get_profile_for_url(url) == "stealth"

    def test_unknown_domain_defaults_to_thorough(self):
        """Test that unknown domains default to thorough profile."""
        assert get_profile_for_url("https://example.com") == "thorough"
        assert get_profile_for_url("https://random-site.org/page") == "thorough"

    def test_handles_invalid_url_gracefully(self):
        """Test that invalid URLs don't crash and return default."""
        assert get_profile_for_url("not-a-valid-url") == "thorough"
        assert get_profile_for_url("") == "thorough"

    def test_handles_http_urls(self):
        """Test that HTTP URLs are detected correctly."""
        assert get_profile_for_url("http://docs.example.com") == "fast"
        assert get_profile_for_url("http://linkedin.com") == "stealth"


class TestApplyProxyOverride:
    """Tests for apply_proxy_override function."""

    def test_applies_proxy_to_profile(self):
        """Test that proxy is applied to a new profile."""
        original = CRAWL_PROFILES["fast"]
        proxy_url = "http://user:pass@proxy.example.com:8080"

        modified = apply_proxy_override(original, proxy_url)

        assert modified.proxy_config == proxy_url
        assert modified.name == original.name
        assert modified.headless == original.headless
        assert modified.max_concurrent == original.max_concurrent

    def test_none_proxy_returns_original(self):
        """Test that None proxy returns the original profile."""
        original = CRAWL_PROFILES["fast"]
        result = apply_proxy_override(original, None)
        assert result is original

    def test_original_profile_unchanged(self):
        """Test that original profile is not modified."""
        original = CRAWL_PROFILES["stealth"]
        original_proxy = original.proxy_config

        apply_proxy_override(original, "http://new-proxy.com:8080")

        assert original.proxy_config == original_proxy


class TestCrawlProfileIntegration:
    """Integration tests for crawl profile usage with CrawlerService."""

    @pytest.fixture
    def mock_crawl4ai(self):
        """Mock Crawl4AI imports."""
        with patch(
            "agentic_rag_backend.indexing.crawler.CRAWL4AI_AVAILABLE", True
        ), patch(
            "agentic_rag_backend.indexing.crawler.AsyncWebCrawler"
        ) as mock_crawler, patch(
            "agentic_rag_backend.indexing.crawler.BrowserConfig"
        ) as mock_browser_config, patch(
            "agentic_rag_backend.indexing.crawler.CrawlerRunConfig"
        ) as mock_run_config, patch(
            "agentic_rag_backend.indexing.crawler.CacheMode"
        ) as mock_cache_mode:
            mock_cache_mode.ENABLED = "enabled"
            mock_cache_mode.BYPASS = "bypass"
            yield {
                "crawler": mock_crawler,
                "browser_config": mock_browser_config,
                "run_config": mock_run_config,
                "cache_mode": mock_cache_mode,
            }

    def test_crawler_service_accepts_profile(self, mock_crawl4ai):
        """Test that CrawlerService accepts profile parameter."""
        from agentic_rag_backend.indexing.crawler import CrawlerService

        profile = get_crawl_profile("thorough")
        service = CrawlerService(profile=profile)

        assert service.headless == profile.headless
        assert service.max_concurrent == profile.max_concurrent
        assert service.cache_enabled == profile.cache_enabled
        assert service.stealth == profile.stealth
        assert service.wait_for == profile.wait_for
        assert service.profile_name == "thorough"

    def test_crawler_service_stealth_profile(self, mock_crawl4ai):
        """Test that stealth profile is applied correctly."""
        from agentic_rag_backend.indexing.crawler import CrawlerService

        profile = get_crawl_profile("stealth")
        service = CrawlerService(profile=profile)

        assert service.headless is False
        assert service.stealth is True
        assert service.cache_enabled is False
        assert service.max_concurrent == 3
        assert service.profile_name == "stealth"

    def test_crawler_service_fast_profile(self, mock_crawl4ai):
        """Test that fast profile is applied correctly."""
        from agentic_rag_backend.indexing.crawler import CrawlerService

        profile = get_crawl_profile("fast")
        service = CrawlerService(profile=profile)

        assert service.headless is True
        assert service.stealth is False
        assert service.cache_enabled is True
        assert service.max_concurrent == 10
        assert service.profile_name == "fast"


class TestConfigIntegration:
    """Tests for config.py integration with crawl profiles."""

    def test_config_has_profile_setting(self):
        """Test that Settings class has crawl4ai_profile field."""
        from agentic_rag_backend.config import Settings
        import inspect

        # Check that the field exists in Settings
        sig = inspect.signature(Settings)
        param_names = list(sig.parameters.keys())
        assert "crawl4ai_profile" in param_names
        assert "crawl4ai_stealth_proxy" in param_names

    def test_load_settings_parses_profile(self):
        """Test that load_settings correctly parses CRAWL4AI_PROFILE."""
        import os
        from agentic_rag_backend.config import load_settings

        # Set required env vars for load_settings
        env_vars = {
            "DATABASE_URL": "postgresql://test:test@localhost/test",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "REDIS_URL": "redis://localhost:6379",
            "OPENAI_API_KEY": "test-key",
            "CRAWL4AI_PROFILE": "stealth",
            "CRAWL4AI_STEALTH_PROXY": "http://proxy.example.com:8080",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = load_settings()
            assert settings.crawl4ai_profile == "stealth"
            assert settings.crawl4ai_stealth_proxy == "http://proxy.example.com:8080"

    def test_load_settings_defaults_to_fast(self):
        """Test that CRAWL4AI_PROFILE defaults to 'fast'."""
        import os
        from agentic_rag_backend.config import load_settings

        # Set required env vars without CRAWL4AI_PROFILE
        env_vars = {
            "DATABASE_URL": "postgresql://test:test@localhost/test",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "REDIS_URL": "redis://localhost:6379",
            "OPENAI_API_KEY": "test-key",
        }

        # Remove CRAWL4AI_PROFILE if it exists
        env_to_remove = ["CRAWL4AI_PROFILE", "CRAWL4AI_STEALTH_PROXY"]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_to_remove}
        clean_env.update(env_vars)

        with patch.dict(os.environ, clean_env, clear=True):
            settings = load_settings()
            assert settings.crawl4ai_profile == "fast"
            assert settings.crawl4ai_stealth_proxy is None
