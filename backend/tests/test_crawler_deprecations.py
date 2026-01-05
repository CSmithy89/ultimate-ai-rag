"""Tests for crawler legacy alias deprecations."""

import warnings

from agentic_rag_backend.indexing import crawler


def test_extract_links_warns_and_returns_links():
    html = '<a href="https://example.com">Example</a>'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        links = crawler.extract_links(html, "https://base.example")

    assert "https://example.com" in links
    assert any(w.category is DeprecationWarning for w in caught)


def test_extract_title_warns_and_returns_title():
    html = "<html><head><title>Example Title</title></head><body></body></html>"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        title = crawler.extract_title(html)

    assert title == "Example Title"
    assert any(w.category is DeprecationWarning for w in caught)
