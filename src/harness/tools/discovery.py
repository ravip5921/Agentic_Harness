
"""Bounded documentation URL discovery for Research.

Search results are untrusted URL candidates, not verified evidence.
Actual retrieval and evidence hashing belong to tools/evidence.py.
"""

from collections.abc import Callable
from urllib.parse import urlparse

from ddgs import DDGS

MAX_SEARCH_RESULTS = 8
MAX_DISCOVERED_URLS = 3


class DiscoveryError(Exception):
    """Documentation discovery could not complete."""


def default_search(query: str, max_results: int) -> list[dict]:
    """Search for documentation using DDGS."""
    with DDGS(timeout=10) as client:
        return list(
            client.text(
                query,
                max_results=max_results,
            )
        )


def domain_matches(host: str, domain: str) -> bool:
    """Match a domain or its subdomains, not lookalike domains."""
    return host == domain or host.endswith("." + domain)


def discover_documentation(
    task_description: str,
    *,
    domains: list[str] | None = None,
    searcher: Callable = default_search,
    max_urls: int = MAX_DISCOVERED_URLS,
) -> list[str]:
    """Discover candidate HTTPS documentation URLs.

    Args:
        task_description: Original natural-language task.
        domains: Optional allowed documentation domains.
        searcher: Injectable search provider for testing.
        max_urls: Maximum number of URLs returned.

    Returns:
        Candidate documentation URLs, not verified evidence.
    """

    if not task_description.strip():
        raise DiscoveryError("Task description is empty.")

    if not 1 <= max_urls <= MAX_DISCOVERED_URLS:
        raise DiscoveryError("Invalid discovery result limit.")

    allowed_domains = []

    for domain in domains or []:
        if not isinstance(domain, str):
            raise DiscoveryError("Domain must be a string.")

        normalized = domain.lower().strip().rstrip(".")

        if (
            not normalized
            or "://" in normalized
            or "/" in normalized
            or ":" in normalized
            or any(ch.isspace() for ch in normalized)
        ):
            raise DiscoveryError(
                f"Invalid documentation domain: {domain}"
            )

        allowed_domains.append(normalized)

    query = (
        f"{task_description[:500]} "
        "official API documentation reference"
    )

    try:
        results = searcher(
            query,
            MAX_SEARCH_RESULTS,
        )
    except Exception as exc:
        raise DiscoveryError(
            "Documentation search failed."
        ) from exc

    urls = []
    seen = set()

    for result in results:
        if not isinstance(result, dict):
            continue

        url = result.get("href")

        if not isinstance(url, str):
            continue

        try:
            parsed = urlparse(url)

            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port not in (None, 443)
            ):
                continue

            host = parsed.hostname.lower().rstrip(".")

            if allowed_domains and not any(
                domain_matches(host, domain)
                for domain in allowed_domains
            ):
                continue

            # Discard URL fragments and redundant duplicates.
            normalized_url = parsed._replace(
                fragment=""
            ).geturl()

        except ValueError:
            continue

        if normalized_url in seen:
            continue

        seen.add(normalized_url)
        urls.append(normalized_url)

        if len(urls) >= max_urls:
            break

    return urls
