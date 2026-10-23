
"""Tests for bounded documentation discovery."""

import pytest

from harness.tools.discovery import (
    discover_documentation,
    DiscoveryError,
)


def test_discovers_documentation_urls():
    def fake_search(query, max_results):
        assert "GitHub" in query
        assert max_results > 0

        return [
            {
                "title": "GitHub API documentation",
                "href": (
                    "https://docs.github.com/en/rest/orgs/"
                    "members"
                ),
            },
            {
                "title": "Other documentation",
                "href": "https://example.com/api",
            },
        ]

    urls = discover_documentation(
        "Build a GitHub organization members API client",
        domains=["docs.github.com"],
        searcher=fake_search,
    )

    assert urls == [
        "https://docs.github.com/en/rest/orgs/members"
    ]


def test_rejects_unsafe_and_duplicate_urls():
    def fake_search(query, max_results):
        return [
            {"href": "http://example.com/docs"},
            {"href": "https://example.com/docs"},
            {"href": "https://example.com/docs#section"},
            {"href": "https://example.com:8443/docs"},
            {"href": "https://example.com.evil.test/docs"},
        ]

    urls = discover_documentation(
        "Find API documentation",
        domains=["example.com"],
        searcher=fake_search,
    )

    assert urls == ["https://example.com/docs"]


def test_discovery_result_limit():
    def fake_search(query, max_results):
        return [
            {"href": f"https://example.com/docs/{i}"}
            for i in range(8)
        ]

    urls = discover_documentation(
        "Find API documentation",
        searcher=fake_search,
        max_urls=2,
    )

    assert len(urls) == 2


def test_search_failure_is_wrapped():
    def failing_search(query, max_results):
        raise RuntimeError("Search unavailable")

    with pytest.raises(
        DiscoveryError,
        match="Documentation search failed",
    ):
        discover_documentation(
            "Find API documentation",
            searcher=failing_search,
        )


def test_invalid_documentation_domain_rejected():
    with pytest.raises(
        DiscoveryError,
        match="Invalid documentation domain",
    ):
        discover_documentation(
            "Find API documentation",
            domains=["https://example.com/path"],
            searcher=lambda query, limit: [],
        )
