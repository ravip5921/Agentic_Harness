
"""Bounded retrieval of explicitly approved documentation sources."""

import hashlib
import ipaddress
import socket

from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

from harness.models import EvidenceSource


MAX_RESPONSE_BYTES = 100_000


class EvidenceError(Exception):
    pass


def validate_source_url(
    url: str,
) -> None:
    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise EvidenceError("Only HTTPS documentation URLs are allowed.")

    if not parsed.hostname or parsed.username or parsed.password:
        raise EvidenceError("Invalid documentation URL.")

    if parsed.port not in (None, 443):
        raise EvidenceError("Nonstandard HTTPS ports are not permitted.")

    host = parsed.hostname.lower().rstrip(".")

    # Basic protection against private-network targets.
    # Production deployment also requires network-level egress controls.
    try:
        addresses = socket.getaddrinfo(
            host, 443, type=socket.SOCK_STREAM
        )
    except OSError as exc:
        raise EvidenceError("Could not resolve documentation host.") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])

        if not ip.is_global:
            raise EvidenceError(
                "Documentation host resolves to a non-public address."
            )


def retrieve_evidence(
    urls: list[str],
) -> list[EvidenceSource]:
    """Retrieve supplied documentation; do not discover or invent URLs."""

    sources = []

    for index, url in enumerate(urls, start=1):
        validate_source_url(url)

        try:
            with requests.get(
                url,
                timeout=(5, 15),
                allow_redirects=False,
                stream=True,
                headers={"Accept": "text/html, text/plain"},
            ) as response:
                if response.is_redirect:
                    raise EvidenceError("Documentation redirect rejected.")

                response.raise_for_status()

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if not (
                    "text/html" in content_type
                    or "text/plain" in content_type
                ):
                    raise EvidenceError(
                        f"Unsupported content type: {content_type}"
                    )

                chunks = []
                total = 0

                for chunk in response.iter_content(chunk_size=8192):
                    total += len(chunk)

                    if total > MAX_RESPONSE_BYTES:
                        raise EvidenceError(
                            "Documentation exceeds retrieval limit."
                        )

                    chunks.append(chunk)

                content = b"".join(chunks)

        except requests.RequestException as exc:
            raise EvidenceError(
                f"Could not retrieve approved source: {url}"
            ) from exc

        digest = hashlib.sha256(content).hexdigest()

        # Bounded raw excerpt for the initial prototype.
        excerpt = content.decode(
            "utf-8", errors="replace"
        )[:6000]

        sources.append(
            EvidenceSource(
                source_id=f"source-{index}",
                url=url,
                retrieved_at=datetime.now(
                    timezone.utc
                ).isoformat(),
                content_sha256=digest,
                excerpt=excerpt,
            )
        )

    return sources
