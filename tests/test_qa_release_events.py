"""Progress reports validated QA and Release results without changing artifacts."""

import importlib
import shutil
from types import SimpleNamespace

import pytest

from harness.graph import build_graph
from harness.models import (
    DevelopmentResult,
    EngineeringSpec,
    ImplementationManifest,
    QACheckResult,
    QACheckStatus,
    QACheckType,
    QAReport,
    QARequirementAssessment,
    QAResult,
    QAVerdict,
    ReleaseAssessment,
    ReleaseStatus,
    RequirementFinding,
    RequirementStatus,
    ResearchResult,
    StageOutcome,
    TaskRequest,
)
from harness.tools.release_artifacts import ReleaseArtifactError
from harness.tools.qa_workspace import QAWorkspaceError
from harness.tools.release_workspace import ReleaseWorkspaceError


qa_module = importlib.import_module("harness.stages.qa")
release_module = importlib.import_module("harness.stages.release")


@pytest.fixture
def capture_events(monkeypatch):
    events = []

    def capture(stage, event, message, **fields):
        events.append({"stage": stage, "event": event, "message": message, **fields})

    monkeypatch.setattr(qa_module, "emit_event", capture)
    monkeypatch.setattr(release_module, "emit_event", capture)
    return events


@pytest.fixture
def manifest():
    return ImplementationManifest(
        summary="Addition utility",
        files=[{"path": "add.py", "operation": "create", "sha256": "a" * 64}],
    )


def make_qa(tmp_path, manifest, check_status, finding_status, invalid_coverage=False):
    task = TaskRequest(task="Implement addition.")
    specification = EngineeringSpec(
        title="Addition", objective="Add integers.", requirements=["Implement add()."],
        acceptance_criteria=["add(2, 3) returns 5."], technical_approach="Use Python.",
        implementation_steps=["Implement addition."],
    )
    assessment = QARequirementAssessment(
        findings=[
            RequirementFinding(
                requirement_id=requirement_id, requirement=requirement,
                status=finding_status, explanation="Reviewed implementation.",
                implementation_files=["add.py"],
            )
            for requirement_id, requirement in (
                ("REQ-1", specification.requirements[0]),
                ("AC-1", specification.acceptance_criteria[0]),
            )
        ],
        overall_summary="Assessment completed.",
    )
    if invalid_coverage:
        assessment.findings.pop()

    calls = []

    def executor(check_type, workspace):
        calls.append(check_type)
        status = QACheckStatus.PASSED if check_type == QACheckType.SYNTAX else check_status
        return QACheckResult(
            name=check_type.value, check_type=check_type, status=status,
            exit_code=0 if status == QACheckStatus.PASSED else 1,
            stdout="Private test output is not a progress message.",
        )

    stage = qa_module.make_qa_stage(
        artifact_loader=lambda *args, **kwargs: (manifest, tmp_path / "staging"),
        workspace_preparer=lambda **kwargs: kwargs["workspace_dir"],
        test_executor=executor,
        specification_loader=lambda *args, **kwargs: specification,
        context_loader=lambda *args, **kwargs: object(),
        requirements_assessor=lambda *args: assessment,
        runs_dir=tmp_path,
    )
    state = {
        "task": task, "revision_count": 0,
        "development_result": DevelopmentResult(outcome=StageOutcome.COMPLETED),
    }
    return stage, state, calls


@pytest.mark.parametrize(
    "check_status,finding_status,verdict,level",
    [
        (QACheckStatus.PASSED, RequirementStatus.SATISFIED, QAVerdict.PASS, "success"),
        (QACheckStatus.FAILED, RequirementStatus.NOT_SATISFIED, QAVerdict.FAIL, "failure"),
        (QACheckStatus.PASSED, RequirementStatus.UNCERTAIN, QAVerdict.BLOCKED, "warning"),
        (QACheckStatus.BLOCKED, RequirementStatus.SATISFIED, QAVerdict.BLOCKED, "warning"),
        (QACheckStatus.SKIPPED, RequirementStatus.SATISFIED, QAVerdict.BLOCKED, "warning"),
    ],
)
def test_qa_events_report_actual_outcomes(
    tmp_path, manifest, capture_events, check_status, finding_status, verdict, level,
):
    stage, state, calls = make_qa(tmp_path, manifest, check_status, finding_status)
    result = stage(state)["qa_result"]

    assert result.verdict == verdict
    assert calls == [QACheckType.SYNTAX, QACheckType.UNIT_TEST]
    checks = [event for event in capture_events if event["event"] == "check_completed"]
    assert checks[1]["data"]["status"] == check_status.value
    assert checks[1]["level"] == (
        "success" if check_status == QACheckStatus.PASSED
        else "failure" if check_status == QACheckStatus.FAILED else "warning"
    )
    assessments = [
        event for event in capture_events
        if event["event"] == "requirements_assessment_completed"
    ]
    if check_status in (QACheckStatus.BLOCKED, QACheckStatus.SKIPPED):
        assert assessments == []
    else:
        assert assessments[0]["data"] == {
            "satisfied": 2 if finding_status == RequirementStatus.SATISFIED else 0,
            "not_satisfied": 2 if finding_status == RequirementStatus.NOT_SATISFIED else 0,
            "uncertain": 2 if finding_status == RequirementStatus.UNCERTAIN else 0,
            "total": 2,
        }
        assert assessments[0]["level"] == level

    feedback_events = [event for event in capture_events if "feedback" in event["event"]]
    assert bool(feedback_events) == (verdict == QAVerdict.FAIL)
    assert "Private test output" not in str(capture_events)
    expected_report = tmp_path / state["task"].run_id / "qa/attempt-0/test_report.json"
    assert result.report_path == str(expected_report)
    assert QAReport.model_validate_json(expected_report.read_text()).verdict == verdict


def test_invalid_requirement_coverage_is_never_reported_as_complete(
    tmp_path, manifest, capture_events,
):
    stage, state, _ = make_qa(
        tmp_path, manifest, QACheckStatus.PASSED, RequirementStatus.SATISFIED,
        invalid_coverage=True,
    )
    result = stage(state)["qa_result"]

    assert result.verdict == QAVerdict.BLOCKED
    assert not any(
        event["event"] == "requirements_assessment_completed" for event in capture_events
    )
    failed = next(
        event for event in capture_events if event["event"] == "requirements_assessment_failed"
    )
    assert failed["data"]["reason"] == "RequirementCoverageError"
    assert "Missing requirement assessments" not in failed["message"]


def test_qa_progress_accepts_opaque_injected_manifest(tmp_path, capture_events):
    manifest = object()
    calls = []

    def prepare(**kwargs):
        assert kwargs["manifest"] is manifest
        calls.append("workspace")
        raise QAWorkspaceError("Workspace preparation unavailable.")

    stage = qa_module.make_qa_stage(
        artifact_loader=lambda *args, **kwargs: (manifest, tmp_path / "staging"),
        workspace_preparer=prepare, runs_dir=tmp_path,
    )
    result = stage({
        "task": TaskRequest(task="Example"),
        "development_result": DevelopmentResult(outcome=StageOutcome.COMPLETED),
    })["qa_result"]

    assert calls == ["workspace"]
    assert result.verdict == QAVerdict.BLOCKED
    assert result.summary == "QA could not complete: Workspace preparation unavailable."
    verified = next(
        event for event in capture_events if event["event"] == "artifact_verification_completed"
    )
    assert "file_count" not in verified["data"]


def make_release(tmp_path, manifest, *, fail_agent=False, block_validation=False):
    task = TaskRequest(task="Implement addition.")
    report = QAReport(
        run_id=task.run_id, revision_count=0, verdict=QAVerdict.PASS,
        manifest_path=str(tmp_path / "manifest.json"),
        checks=[QACheckResult(
            name="unit_test", check_type=QACheckType.UNIT_TEST,
            status=QACheckStatus.PASSED, exit_code=0,
        )],
    )
    artifacts = SimpleNamespace(
        manifest=manifest, qa_report=report, manifest_path=tmp_path / "manifest.json",
        qa_report_path=tmp_path / "test_report.json",
    )
    calls = []

    def loader(*args, **kwargs):
        calls.append("validate")
        if block_validation:
            raise ReleaseArtifactError("Private validation detail")
        return artifacts

    def prepare(**kwargs):
        calls.append("workspace")
        candidate = kwargs["release_dir"]
        candidate.mkdir(parents=True)
        (candidate / "add.py").write_text("def add(a, b): return a + b\n")
        return candidate

    def generate(*args):
        calls.append("agent")
        if fail_agent:
            raise RuntimeError("Private agent detail")
        return ReleaseAssessment(
            summary="Addition implementation", changes=["CREATE: add.py"],
            testing_summary="Unit test check passed.",
        )

    stage = release_module.make_release_stage(
        artifact_loader=loader, workspace_preparer=prepare,
        release_generator=generate, runs_dir=tmp_path,
    )
    return stage, {"task": task, "revision_count": 0}, calls


def test_release_fallback_events_remain_staged(tmp_path, manifest, capture_events):
    stage, state, calls = make_release(tmp_path, manifest, fail_agent=True)
    result = stage(state)["release_result"]

    assert result.status == ReleaseStatus.STAGED
    assert calls == ["validate", "workspace", "agent"]
    names = [event["event"] for event in capture_events]
    assert "documentation_generation_completed" not in names
    assert names.index("documentation_fallback_started") < names.index("documentation_fallback_completed")
    assert names[-1] == "persistence_completed"
    assert "Private agent detail" not in str(capture_events)
    attempt = tmp_path / state["task"].run_id / "release/attempt-0"
    assert result.change_ref == str(attempt / "candidate")
    assert "Fallback: deterministic documentation" in (attempt / "release_report.md").read_text()
    assert capture_events[-1]["data"]["manifest_path"] == str(attempt / "release_manifest.json")


def test_release_block_emits_sanitized_reason(tmp_path, manifest, capture_events):
    stage, state, calls = make_release(tmp_path, manifest, block_validation=True)
    result = stage(state)["release_result"]

    assert result.status == ReleaseStatus.BLOCKED
    assert calls == ["validate"]
    assert [event["event"] for event in capture_events] == ["artifact_validation_started", "blocked"]
    assert capture_events[-1]["data"]["reason"] == "ReleaseArtifactError"
    assert "Private validation detail" not in str(capture_events)


@pytest.mark.parametrize("artifacts", [object(), SimpleNamespace(manifest=object())])
def test_release_progress_accepts_opaque_injected_context(
    tmp_path, capture_events, artifacts,
):
    calls = []

    def prepare(**kwargs):
        assert kwargs["artifacts"] is artifacts
        calls.append("workspace")
        raise ReleaseWorkspaceError("Workspace preparation unavailable.")

    stage = release_module.make_release_stage(
        artifact_loader=lambda *args, **kwargs: artifacts,
        workspace_preparer=prepare, runs_dir=tmp_path,
    )
    result = stage({"task": TaskRequest(task="Example")})["release_result"]

    assert calls == ["workspace"]
    assert result.status == ReleaseStatus.BLOCKED
    assert result.summary == "Release candidate preparation failed."
    assert result.errors == ["Workspace preparation unavailable."]
    validated = next(
        event for event in capture_events if event["event"] == "artifact_validation_completed"
    )
    assert "file_count" not in validated["data"]


def test_streaming_release_fallback_preserves_result_and_artifacts(tmp_path, manifest):
    stage, state, calls = make_release(tmp_path, manifest, fail_agent=True)
    direct_result = stage(state)["release_result"]
    attempt = tmp_path / state["task"].run_id / "release/attempt-0"
    original_files = {
        path.relative_to(attempt): path.read_bytes()
        for path in attempt.rglob("*") if path.is_file()
    }
    shutil.rmtree(attempt)
    calls.clear()

    graph = build_graph(
        research=lambda state: {"research_result": ResearchResult(outcome=StageOutcome.COMPLETED)},
        developer=lambda state: {"development_result": DevelopmentResult(outcome=StageOutcome.COMPLETED)},
        qa=lambda state: {"qa_result": QAResult(verdict=QAVerdict.PASS)},
        release=stage,
    )
    streamed = list(graph.stream(state, stream_mode=["custom", "updates"]))
    updates = [payload for mode, payload in streamed if mode == "updates"]
    result = next(update["release"]["release_result"] for update in updates if "release" in update)

    assert result == direct_result
    assert calls == ["validate", "workspace", "agent"]
    assert any(
        mode == "custom" and payload["event"] == "documentation_fallback_completed"
        for mode, payload in streamed
    )
    assert {
        path.relative_to(attempt): path.read_bytes()
        for path in attempt.rglob("*") if path.is_file()
    } == original_files
