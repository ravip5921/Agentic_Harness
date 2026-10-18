"""Tests for the complete harness workflow using simulated stage results."""

from harness.graph import build_graph
from harness.models import (
    DevelopmentResult,
    QAResult,
    QAVerdict,
    ReleaseResult,
    ReleaseStatus,
    ResearchResult,
    RunStatus,
    StageOutcome,
    TaskRequest,
)


def test_complete_workflow():
    """Verify Research -> Developer -> QA -> Release."""

    executed = []

    def mock_research(state):
        executed.append("research")
        return {
            "research_result": ResearchResult(
                outcome=StageOutcome.COMPLETED,
                summary="Simulated research completed.",
                artifacts=["mock/engineering_spec.md"],
            )
        }

    def mock_developer(state):
        executed.append("developer")

        assert state["research_result"].outcome == StageOutcome.COMPLETED

        return {
            "development_result": DevelopmentResult(
                outcome=StageOutcome.COMPLETED,
                summary="Simulated implementation completed.",
                artifacts=["mock/implementation.py"],
            )
        }

    def mock_qa(state):
        executed.append("qa")

        assert state["development_result"].outcome == StageOutcome.COMPLETED

        return {
            "qa_result": QAResult(
                verdict=QAVerdict.PASS,
                summary="Simulated QA passed.",
            )
        }

    def mock_release(state):
        executed.append("release")

        assert state["qa_result"].verdict == QAVerdict.PASS

        return {
            "release_result": ReleaseResult(
                status=ReleaseStatus.STAGED,
                summary="Simulated release staged.",
                change_ref="mock/changes.patch",
            )
        }

    graph = build_graph(
        research=mock_research,
        developer=mock_developer,
        qa=mock_qa,
        release=mock_release,
    )

    result = graph.invoke({
        "task": TaskRequest(task="Example task"),
        "revision_count": 0,
        "status": RunStatus.RUNNING,
    })

    assert executed == [
        "research",
        "developer",
        "qa",
        "release",
    ]

    assert result["status"] == RunStatus.STAGED
    assert result["revision_count"] == 0

def test_qa_failure_triggers_development_retry():
    """Verify a failed QA attempt returns to Developer."""

    executed = []

    def mock_research(state):
        executed.append("research")
        return {
            "research_result": ResearchResult(
                outcome=StageOutcome.COMPLETED,
            )
        }

    def mock_developer(state):
        executed.append("developer")
        return {
            "development_result": DevelopmentResult(
                outcome=StageOutcome.COMPLETED,
            )
        }

    def mock_qa(state):
        executed.append("qa")

        if state.get("revision_count", 0) == 0:
            return {
                "qa_result": QAResult(
                    verdict=QAVerdict.FAIL,
                    feedback=["Simulated test failure."],
                )
            }

        return {
            "qa_result": QAResult(
                verdict=QAVerdict.PASS,
            )
        }

    def mock_release(state):
        executed.append("release")
        return {
            "release_result": ReleaseResult(
                status=ReleaseStatus.STAGED,
            )
        }

    graph = build_graph(
        research=mock_research,
        developer=mock_developer,
        qa=mock_qa,
        release=mock_release,
    )

    result = graph.invoke({
        "task": TaskRequest(
            task="Example task",
            max_dev_revisions=2,
        ),
        "revision_count": 0,
        "status": RunStatus.RUNNING,
    })

    assert executed == [
        "research",
        "developer",
        "qa",
        "developer",
        "qa",
        "release",
    ]

    assert result["revision_count"] == 1
    assert result["status"] == RunStatus.STAGED