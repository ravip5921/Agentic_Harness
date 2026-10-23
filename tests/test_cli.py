import pytest
from threading import Event

from harness.cli import harness_cli, stream_workflow
from harness.events import emit_event
from harness.graph import build_graph
from harness.models import (
    DevelopmentResult, QAResult, QAVerdict, ReleaseResult, ReleaseStatus,
    ResearchResult, RunStatus, StageOutcome, TaskRequest,
)


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        harness_cli(["--help"])

    assert exc.value.code == 0


def workflow(research=None, verdict=QAVerdict.PASS, release_status=ReleaseStatus.STAGED):
    return build_graph(
        research=research or (lambda _: {"research_result": ResearchResult(
            outcome=StageOutcome.COMPLETED, artifacts=["runs/spec.json"])}),
        developer=lambda _: {"development_result": DevelopmentResult(
            outcome=StageOutcome.COMPLETED,
            artifacts=["runs/manifest.json", "runs/staging"])},
        qa=lambda _: {"qa_result": QAResult(verdict=verdict, report_path="runs/test_report.json")},
        release=lambda _: {"release_result": ReleaseResult(
            status=release_status, change_ref="runs/candidate")},
    )


def test_custom_event_is_consumed_before_node_finishes():
    acknowledged = Event()
    completed = Event()

    def research(_):
        emit_event("research", "specification_started", "Generating engineering specification", level="start")
        assert acknowledged.wait(timeout=5), "CLI did not receive live progress"
        completed.set()
        return {"research_result": ResearchResult(outcome=StageOutcome.COMPLETED)}

    class Progress:
        def event(self, event):
            assert event["event"] == "specification_started"
            assert not completed.is_set()
            acknowledged.set()

        def update(self, update, state):
            pass

    initial = {"task": TaskRequest(task="Demo"), "revision_count": 0, "status": RunStatus.RUNNING}
    result = stream_workflow(workflow(research), initial, Progress())
    assert completed.is_set()
    assert result["task"] == initial["task"]
    assert result["status"] == RunStatus.STAGED
    assert result["development_result"].artifacts == ["runs/manifest.json", "runs/staging"]
    assert result["qa_result"].report_path == "runs/test_report.json"
    assert result["release_result"].change_ref == "runs/candidate"
    assert set(result) == set(initial) | {"research_result", "development_result", "qa_result", "release_result"}


@pytest.mark.parametrize("verdict,release_status,expected_status,exit_code", [
    (QAVerdict.PASS, ReleaseStatus.STAGED, "staged", 0),
    (QAVerdict.PASS, ReleaseStatus.FAILED, "failed", 1),
    (QAVerdict.BLOCKED, ReleaseStatus.STAGED, "blocked", 2),
    (QAVerdict.FAIL, ReleaseStatus.STAGED, "failed", 1),
])
def test_cli_preserves_final_summary_and_exit_codes(
    monkeypatch, tmp_path, capsys, verdict, release_status, expected_status, exit_code,
):
    def research(_):
        emit_event("research", "specification_started", "Generating engineering specification", level="start")
        return {"research_result": ResearchResult(
            outcome=StageOutcome.COMPLETED, artifacts=["runs/spec.json"])}

    monkeypatch.setattr("harness.cli.create_workflow", lambda _: workflow(research, verdict, release_status))
    assert harness_cli(["Demo", "--runs-dir", str(tmp_path), "--max-dev-revisions", "0"]) == exit_code
    output = capsys.readouterr().out
    assert output.index("Generating engineering specification") < output.index("=== Agentic Harness Result ===")
    for expected in ("Run ID:", f"Status: {expected_status}", "Development revisions: 0",
                     "Research:", "Developer:", "QA:", "Release:",
                     "runs/spec.json", "runs/manifest.json", "runs/staging", "runs/test_report.json",
                     f"Workflow completed: {expected_status.upper()}"):
        assert expected in output
    if verdict == QAVerdict.PASS:
        assert "runs/candidate" in output
    else:
        assert "Release: Not executed" in output
    assert "\x1b" not in output
