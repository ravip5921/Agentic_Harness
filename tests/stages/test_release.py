
"""Tests for Release stage orchestration and documentation fallback."""

from pathlib import Path
from types import SimpleNamespace

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
    TaskRequest,
)

from harness.stages.release import make_release_stage
from harness.tools.release_artifacts import ReleaseArtifactError


def make_stage(tmp_path, generator=None, artifact_error=None):
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

