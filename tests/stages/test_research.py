
"""Tests for the Research stage."""

from harness.models import (
    ResearchResult,
    StageOutcome,
    TaskRequest,
)
from harness.stages.research import research


def test_research_returns_valid_result():
    """Research must return a validated ResearchResult."""

    state = {
        "task": TaskRequest(task="Sample development task"),
        "revision_count": 0,
    }

    result = research(state)

    assert "research_result" in result
    assert isinstance(result["research_result"], ResearchResult)


def test_research_is_blocked_until_implemented():
    """Unimplemented research must not report success."""

    state = {
        "task": TaskRequest(task="Sample development task"),
    }

    result = research(state)["research_result"]

    assert result.outcome == StageOutcome.BLOCKED
    assert result.artifacts == []
    assert result.errors


def test_research_preserves_task_description():
    """Research should acknowledge the supplied task."""

    task = TaskRequest(task="Build a sample integration")

    result = research({"task": task})["research_result"]

    assert task.task in result.summary
