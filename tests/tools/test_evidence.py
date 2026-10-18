"""Tests for documentation URL validation."""

import pytest

from harness.tools.evidence import (
    EvidenceError,
    validate_source_url,
)

def test_rejects_non_https_url():
    with pytest.raises(EvidenceError):
        validate_source_url(
            "http://docs.example.com/api",
        )


def test_rejects_credentials_in_url():
    with pytest.raises(EvidenceError):
        validate_source_url(
            "https://user:secret@docs.example.com/api",
        )