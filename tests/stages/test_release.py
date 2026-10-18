
"""Tests for the Release stage."""

from harness.models import ReleaseResult, ReleaseStatus
from harness.stages.release import release


def test_release_returns_valid_result():
    result = release({})

    assert "release_result" in result
    assert isinstance(result["release_result"], ReleaseResult)


def test_release_is_blocked_until_implemented():
    result = release({})["release_result"]

    assert result.status == ReleaseStatus.BLOCKED
    assert result.change_ref is None
    assert result.errors
