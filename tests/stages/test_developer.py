
"""Tests for the Developer stage."""

from harness.models import (
    DevelopmentResult,
    StageOutcome,
)
from harness.stages.developer import developer


def test_developer_returns_valid_result():
    result = developer({})

    assert "development_result" in result
    assert isinstance(result["development_result"], DevelopmentResult)


def test_developer_is_blocked_until_implemented():
    result = developer({})["development_result"]

    assert result.outcome == StageOutcome.BLOCKED
    assert result.artifacts == []
    assert result.errors
