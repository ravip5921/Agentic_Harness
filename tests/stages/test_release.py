
"""Tests for Release stage orchestration and documentation fallback."""

from pathlib import Path
from types import SimpleNamespace
import json
import pytest

from harness.models import (
    ImplementationManifest,
    QACheckResult,
    QACheckStatus,
    QACheckType,
    QAReport,
    QAVerdict,
    ReleaseAssessment,
    ReleaseManifest,
    ReleaseStatus,
    ReleaseDecision,
    TaskRequest,
)

from harness.stages.release import make_release_stage
from harness.tools.release_artifacts import ReleaseArtifactError


def make_stage(tmp_path, generator=None, artifact_error=None,
               approver=lambda **_: ReleaseDecision.KEEP_STAGED, pr_creator=None):
    calls = []

    task = TaskRequest(task="Implement addition.")

    manifest = ImplementationManifest(
        summary="Addition implementation",
        files=[
            {
                "path": "add.py",
                "operation": "create",
                "sha256": "a" * 64,
            }
        ],
    )

    qa_report = QAReport(
        run_id=task.run_id,
        revision_count=0,
        verdict=QAVerdict.PASS,
        checks=[
            QACheckResult(
                name="syntax",
                check_type=QACheckType.SYNTAX,
                status=QACheckStatus.PASSED,
                exit_code=0,
            ),
            QACheckResult(
                name="unit_test",
                check_type=QACheckType.UNIT_TEST,
                status=QACheckStatus.PASSED,
                exit_code=0,
            ),
        ],
        manifest_path=str(tmp_path / "manifest.json"),
    )

    artifacts = SimpleNamespace(
        manifest=manifest,
        qa_report=qa_report,
        manifest_path=tmp_path / "manifest.json",
        qa_report_path=tmp_path / "test_report.json",
    )

    def artifact_loader(state, runs_dir):
        calls.append("validate")

        if artifact_error is not None:
            raise ReleaseArtifactError(artifact_error)

        return artifacts

    def workspace_preparer(artifacts, release_dir, repository_path):
        calls.append("workspace")

        release_dir.mkdir(parents=True)
        (release_dir / "add.py").write_text(
            "def add(a, b): return a + b\n"
        )

        return release_dir

    def release_generator(task, manifest, qa_report):
        calls.append("agent")

        if generator is not None:
            return generator(task, manifest, qa_report)

        return ReleaseAssessment(
            summary="Addition implementation ready for review.",
            changes=["CREATE: add.py"],
            testing_summary="Syntax and pytest checks passed.",
            review_notes=["Review before integration."],
        )

    stage = make_release_stage(
        artifact_loader=artifact_loader,
        workspace_preparer=workspace_preparer,
        release_generator=release_generator,
        human_approver=approver,
        **({"pr_creator": pr_creator} if pr_creator is not None else {}),
        runs_dir=tmp_path,
    )

    state = {
        "task": task,
        "revision_count": 0,
    }

    return stage, state, calls


def test_release_staged_with_agent(tmp_path):
    stage, state, calls = make_stage(tmp_path)

    result = stage(state)["release_result"]

    assert result.status == ReleaseStatus.STAGED
    assert calls == ["validate", "workspace", "agent"]

    attempt = (
        tmp_path / state["task"].run_id
        / "release" / "attempt-0"
    )

    assert (attempt / "candidate" / "add.py").exists()
    assert (attempt / "release_manifest.json").exists()
    assert (attempt / "release_assessment.json").exists()
    assert (attempt / "release_report.md").exists()

    manifest = ReleaseManifest.model_validate_json(
        (attempt / "release_manifest.json").read_text()
    )

    assert manifest.run_id == state["task"].run_id


def test_agent_failure_uses_deterministic_fallback(tmp_path):
    def failing_generator(task, manifest, qa_report):
        raise RuntimeError("Simulated Groq failure")

    stage, state, calls = make_stage(
        tmp_path,
        generator=failing_generator,
    )

    result = stage(state)["release_result"]

    assert result.status == ReleaseStatus.STAGED
    assert result.errors == []

    attempt = (
        tmp_path / state["task"].run_id
        / "release" / "attempt-0"
    )

    assessment = ReleaseAssessment.model_validate_json(
        (attempt / "release_assessment.json").read_text()
    )

    assert any(
        "deterministic" in item.lower()
        for item in assessment.limitations
    )

    report = (attempt / "release_report.md").read_text()

    assert "Fallback: deterministic documentation" in report
    assert (attempt / "candidate" / "add.py").exists()


def test_invalid_qa_artifacts_block_release(tmp_path):
    stage, state, calls = make_stage(
        tmp_path,
        artifact_error="QA did not pass.",
    )

    result = stage(state)["release_result"]

    assert result.status == ReleaseStatus.BLOCKED
    assert calls == ["validate"]
    assert result.change_ref is None


@pytest.mark.parametrize("approver", [None, lambda **_: ReleaseDecision.REJECT,
                                     lambda **_: "create_pr"])
def test_missing_rejected_or_invalid_approval_never_publishes(tmp_path, approver):
    def publish(**kwargs):
        pytest.fail("Unapproved release must never publish")

    stage, state, _ = make_stage(tmp_path, approver=approver, pr_creator=publish)
    result = stage(state)["release_result"]
    assert result.status == ReleaseStatus.BLOCKED
    assert Path(result.change_ref).is_dir()


def test_approval_sees_reviewable_artifacts_and_records_decision_before_pr(tmp_path):
    def approve(*, task, candidate, report_path):
        assert candidate.is_dir()
        assert report_path.is_file()
        assert (candidate.parent / "release_manifest.json").is_file()
        return ReleaseDecision.CREATE_PR

    def publish(*, task, artifacts, candidate, report_path, revision):
        decision = json.loads((candidate.parent / "release_decision.json").read_text())
        assert decision["decision"] == "create_pr"
        assert decision["run_id"] == task.run_id
        assert len(decision["manifest_sha256"]) == 64
        return "https://github.com/example/project/pull/1"

    stage, state, _ = make_stage(tmp_path, approver=approve, pr_creator=publish)
    result = stage(state)["release_result"]
    assert result.status == ReleaseStatus.PR_CREATED
    assert result.change_ref.endswith("/pull/1")
    attempt = tmp_path / state["task"].run_id / "release/attempt-0"
    assert json.loads((attempt / "release_result.json").read_text())["status"] == "pr_created"
    assert "Final status: pr_created" in (attempt / "release_report.md").read_text()


def test_failed_pr_preserves_candidate_and_reports_failure(tmp_path):
    def publish(**kwargs):
        raise RuntimeError("Branch pushed; check GitHub before retrying.")

    stage, state, _ = make_stage(tmp_path,
        approver=lambda **_: ReleaseDecision.CREATE_PR, pr_creator=publish)
    result = stage(state)["release_result"]
    assert result.status == ReleaseStatus.BLOCKED
    assert Path(result.change_ref).is_dir()
    assert "Branch pushed" in result.errors[0]


def test_failed_decision_persistence_never_publishes(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("harness.stages.release")
    original_write = module.write_text_atomic

    def write(path, content):
        if path.name == "release_decision.json":
            raise OSError("Disk full")
        original_write(path, content)

    monkeypatch.setattr(module, "write_text_atomic", write)
    stage, state, _ = make_stage(tmp_path,
        approver=lambda **_: ReleaseDecision.CREATE_PR,
        pr_creator=lambda **_: pytest.fail("Decision must be saved before publication"))
    assert stage(state)["release_result"].status == ReleaseStatus.BLOCKED


def test_audit_failure_after_publication_preserves_pr_url(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("harness.stages.release")
    original_write = module.write_text_atomic

    def write(path, content):
        if path.name == "release_result.json":
            raise OSError("Disk full after publication")
        original_write(path, content)

    monkeypatch.setattr(module, "write_text_atomic", write)
    stage, state, _ = make_stage(tmp_path,
        approver=lambda **_: ReleaseDecision.CREATE_PR,
        pr_creator=lambda **_: "https://github.com/example/project/pull/1")
    result = stage(state)["release_result"]
    assert result.status == ReleaseStatus.PR_CREATED
    assert result.change_ref.endswith("/pull/1")
    assert "metadata could not be saved" in result.errors[0]

