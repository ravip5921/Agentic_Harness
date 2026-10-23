"""Integration test: real Research orchestration with placeholder Developer."""

from harness.graph import build_graph

from harness.models import (
    DevelopmentResult,
    RunStatus,
    StageOutcome,
    TaskRequest,
)

from harness.stages.research import make_research_stage

from test_research import (
    successful_generator,
    fake_repository_reader,
    fake_evidence_reader,
)


def test_research_advances_to_developer(tmp_path):
    executed = []

    def blocked_developer(state):
        executed.append("developer")
        assert state["research_result"].outcome == StageOutcome.COMPLETED
        assert len(state["research_result"].artifacts) == 3
        return {
            "development_result": DevelopmentResult(
                outcome=StageOutcome.BLOCKED,
                summary="Placeholder Developer for Research routing verification.",
            )
        }

    def unexpected_downstream(state):
        raise AssertionError("QA and Release must not execute after blocked Development.")

    research = make_research_stage(
        generator=successful_generator,
        repository_reader=fake_repository_reader,
        evidence_reader=fake_evidence_reader,
        runs_dir=tmp_path,
    )

    graph = build_graph(
        research=research,
        developer=blocked_developer,
        qa=unexpected_downstream,
        release=unexpected_downstream,
    )

    result = graph.invoke({
        "task": TaskRequest(
            task="Implement a sample feature"
        ),
        "revision_count": 0,
        "status": RunStatus.RUNNING,
    })

    assert (
        result["research_result"].outcome
        == StageOutcome.COMPLETED
    )

    assert (
        result["development_result"].outcome
        == StageOutcome.BLOCKED
    )

    assert result["status"] == RunStatus.BLOCKED
    assert executed == ["developer"]

    # QA and Release must not execute after blocked Development.
    assert "qa_result" not in result
    assert "release_result" not in result
