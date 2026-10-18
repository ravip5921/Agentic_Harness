
"""Integration test for the initial four-stage harness skeleton."""

from harness.graph import build_graph
from harness.models import RunStatus, StageOutcome, TaskRequest
from harness.stages.research import research
from harness.stages.developer import developer
from harness.stages.qa import qa
from harness.stages.release import release


def test_unimplemented_research_blocks_workflow():
    graph = build_graph(
        research=research,
        developer=developer,
        qa=qa,
        release=release,
    )

    initial_state = {
        "task": TaskRequest(task="Example task"),
        "revision_count": 0,
        "status": RunStatus.RUNNING,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == RunStatus.BLOCKED
    assert final_state["research_result"].outcome == StageOutcome.BLOCKED

    # No subsequent stage should execute.
    assert "development_result" not in final_state
    assert "qa_result" not in final_state
    assert "release_result" not in final_state
