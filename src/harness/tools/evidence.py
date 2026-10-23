
"""Bounded retrieval of documentation sources.

Supports HTML and plain-text documentation.
HTML is converted into readable text before being supplied to Research.

Retrieved source content is untrusted.
"""

import hashlib
import ipaddress
import socket

from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from harness.models import EvidenceSource


MAX_RESPONSE_BYTES = 1_000_000
MAX_EXCERPT_CHARS = 12_000


class EvidenceError(Exception):
    """Raised when documentation cannot be safely retrieved."""


def validate_source_url(url: str) -> None:
    """Validate a public HTTPS documentation URL."""

    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise EvidenceError(
            "Only HTTPS documentation URLs are allowed."
        )

    if not parsed.hostname or parsed.username or parsed.password:
        raise EvidenceError("Invalid documentation URL.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise EvidenceError("Invalid documentation port.") from exc

    if port not in (None, 443):
        raise EvidenceError(
            "Nonstandard HTTPS ports are not permitted."
        )

    host = parsed.hostname.lower().rstrip(".")

    try:
        addresses = socket.getaddrinfo(
            host,
            443,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise EvidenceError(
            "Could not resolve documentation host."
        ) from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])

        if not ip.is_global:
            raise EvidenceError(
                "Documentation host resolves to a non-public address."
            )


def extract_documentation_text(
    content: bytes,
    content_type: str,
) -> str:
    """Extract bounded readable text from HTML or plain text."""

    text = content.decode("utf-8", errors="replace")

    if "text/html" not in content_type.lower():
        return text[:MAX_EXCERPT_CHARS]

    soup = BeautifulSoup(text, "html.parser")

    # Remove common non-documentation content.
    for element in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
        "svg",
    ]):
        element.decompose()

    # Prefer the main documentation content.
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    readable_text = main.get_text(
        separator="\n",
        strip=True,
    )

    # Remove empty lines and normalize whitespace.
    lines = [
        line.strip()
        for line in readable_text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)[:MAX_EXCERPT_CHARS]


def retrieve_evidence(
    urls: list[str],
) -> list[EvidenceSource]:
    """Retrieve and extract supplied documentation URLs.

    Each response is bounded and independently validated.

    Source IDs are local to this call. The Research stage must
    renumber them after combining individual retrieval results.
    """

    sources = []

    for index, url in enumerate(urls, start=1):
        validate_source_url(url)

        try:
            with requests.get(
                url,
                timeout=(5, 15),
                allow_redirects=False,
                stream=True,
                headers={
                    "Accept": "text/html, text/plain",
                    "Accept-Encoding": "identity",
                },
            ) as response:
                if response.is_redirect:
                    raise EvidenceError(
                        "Documentation redirect rejected."
                    )

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

                for chunk in response.iter_content(
                    chunk_size=8192
                ):
                    if not chunk:
                        continue

                    total += len(chunk)

                    if total > MAX_RESPONSE_BYTES:
                        raise EvidenceError(
                            "Documentation exceeds retrieval limit."
                        )

                    chunks.append(chunk)

                content = b"".join(chunks)

        except requests.RequestException as exc:
            raise EvidenceError(
                f"Could not retrieve documentation: {url}"
            ) from exc

        digest = hashlib.sha256(content).hexdigest()

        excerpt = extract_documentation_text(
            content,
            content_type,
        )

        if not excerpt.strip():
            raise EvidenceError(
                f"Documentation contains no usable text: {url}"
            )

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
