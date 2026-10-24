
"""Tests for Release artifact validation.

Uses real manifests, reports, staged files, and QA workspaces.
Does not invoke Groq, Docker, or Git.
"""

import hashlib

import pytest

from harness.models import (
    TaskRequest,
    ResearchResult,
    DevelopmentResult,
    QAResult,
    StageOutcome,
    QAVerdict,
    QACheckType,
    QACheckStatus,
    QACheckResult,
    QAReport,
    EngineeringSpec,
    ImplementationManifest,
    GeneratedFile,
    QARequirementAssessment,
    RequirementFinding,
    RequirementStatus,
)

from harness.tools.filesystem import stage_generated_files
from harness.tools.qa_context import QAImplementationContext, VerifiedSourceFile
from harness.tools.release_artifacts import (
    ReleaseArtifactError,
    load_release_artifacts,
)


SOURCE_CODE = "def add(a, b):\n    return a + b\n"

TEST_CODE = (
    "from add import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)


def make_spec():
    return EngineeringSpec(
        title="Addition utility",
        objective="Implement and test integer addition.",
        requirements=["Implement add(a, b)."],
        technical_approach="Use Python.",
        implementation_steps=[
            "Implement add().",
            "Write pytest tests.",
        ],
        acceptance_criteria=[
            "add(2, 3) returns 5.",
        ],
    )


def make_assessment(spec):
    return QARequirementAssessment(
        findings=[
            RequirementFinding(
                requirement_id="REQ-1",
                requirement=spec.requirements[0],
                status=RequirementStatus.SATISFIED,
                explanation="The implementation performs addition.",
                implementation_files=["add.py"],
            ),
            RequirementFinding(
                requirement_id="AC-1",
                requirement=spec.acceptance_criteria[0],
                status=RequirementStatus.SATISFIED,
                explanation="The behavior is covered by a test.",
                implementation_files=["add.py"],
                test_files=["tests/test_add.py"],
            ),
        ],
        overall_summary="All requirements satisfied.",
    )


def make_checks():
    return [
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
            stdout="1 passed",
        ),
    ]


@pytest.fixture
def release_fixture(tmp_path):
    """Construct a fully valid persisted Development + QA run."""

    task = TaskRequest(
        task="Implement add(a, b) and tests.",
    )

    run_dir = tmp_path / task.run_id
    spec = make_spec()

    # Research artifacts.
    research_dir = run_dir / "research"
    research_dir.mkdir(parents=True)

    spec_path = research_dir / "engineering_spec.json"
    spec_path.write_text(
        spec.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # Developer artifacts.
    development_dir = run_dir / "development" / "attempt-0"
    staging_dir = development_dir / "staging"

    records = stage_generated_files(
        files=[
            GeneratedFile(
                path="add.py",
                operation="create",
                content=SOURCE_CODE,
            ),
            GeneratedFile(
                path="tests/test_add.py",
                operation="create",
                content=TEST_CODE,
            ),
        ],
        staging_root=staging_dir,
    )

    manifest = ImplementationManifest(
        summary="Addition implementation",
        files=records,
    )

    manifest_path = development_dir / "manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # QA artifacts.
    qa_dir = run_dir / "qa" / "attempt-0"
    workspace = qa_dir / "workspace"

    workspace.mkdir(parents=True)
    (workspace / "add.py").write_text(SOURCE_CODE)

    tests_dir = workspace / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_add.py").write_text(TEST_CODE)

    report = QAReport(
        run_id=task.run_id,
        revision_count=0,
        verdict=QAVerdict.PASS,
        summary="QA completed successfully.",
        checks=make_checks(),
        requirements_assessment=make_assessment(spec),
        manifest_path=str(manifest_path),
        workspace_path=str(workspace),
    )

    report_path = qa_dir / "test_report.json"
    report_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    state = {
        "task": task,
        "revision_count": 0,
        "research_result": ResearchResult(
            outcome=StageOutcome.COMPLETED,
            artifacts=[str(spec_path)],
        ),
        "development_result": DevelopmentResult(
            outcome=StageOutcome.COMPLETED,
            artifacts=[
                str(manifest_path),
                str(staging_dir),
            ],
        ),
        "qa_result": QAResult(
            verdict=QAVerdict.PASS,
            report_path=str(report_path),
        ),
    }

    return {
        "state": state,
        "runs_dir": tmp_path,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "staging_dir": staging_dir,
        "workspace": workspace,
    }


def update_report(fixture, change):
    """Modify a persisted QA report for negative testing."""

    path = fixture["report_path"]
    report = QAReport.model_validate_json(path.read_text())

    change(report)

    path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )


def test_valid_qa_pass(release_fixture):
    artifacts = load_release_artifacts(
        release_fixture["state"],
        runs_dir=release_fixture["runs_dir"],
    )

    assert artifacts.qa_report.verdict == QAVerdict.PASS
    assert len(artifacts.manifest.files) == 2
    assert artifacts.staging_dir == release_fixture["staging_dir"]
    assert artifacts.qa_workspace == release_fixture["workspace"]


def add_reviewed_existing_test(fixture):
    workspace = fixture["workspace"]
    (workspace / "tests/test_base.py").write_text("def test_base(): assert True\n")
    manifest = ImplementationManifest.model_validate_json(fixture["manifest_path"].read_text())
    context = QAImplementationContext(
        files=[VerifiedSourceFile(path=record.path, operation=record.operation,
                                  content=(fixture["staging_dir"] / record.path).read_text())
               for record in manifest.files],
        repository_files={"tests/test_base.py": "def test_base(): assert True\n"},
    )
    context_path = workspace.parent / "implementation_context.json"
    context_path.write_text(context.model_dump_json(indent=2))

    def reference_existing_test(report):
        report.requirements_assessment.findings[-1].test_files.append("tests/test_base.py")

    update_report(fixture, reference_existing_test)
    return context_path


def test_release_accepts_unchanged_test_reviewed_by_qa(release_fixture):
    add_reviewed_existing_test(release_fixture)
    assert load_release_artifacts(release_fixture["state"],
        runs_dir=release_fixture["runs_dir"]).qa_report.verdict == QAVerdict.PASS


@pytest.mark.parametrize("mutation", ["changed", "deleted", "symlink", "missing_context", "unsafe_path"])
def test_release_rejects_unverifiable_unchanged_file_evidence(release_fixture, mutation):
    context_path = add_reviewed_existing_test(release_fixture)
    test_file = release_fixture["workspace"] / "tests/test_base.py"
    if mutation == "changed":
        test_file.write_text("changed after QA\n")
    elif mutation == "deleted":
        test_file.unlink()
    elif mutation == "symlink":
        test_file.unlink()
        test_file.symlink_to(release_fixture["workspace"] / "tests/test_add.py")
    elif mutation == "missing_context":
        context_path.unlink()
    else:
        context = QAImplementationContext.model_validate_json(context_path.read_text())
        context.repository_files = {"../outside.py": "outside"}
        context_path.write_text(context.model_dump_json(indent=2))
    with pytest.raises(ReleaseArtifactError):
        load_release_artifacts(release_fixture["state"], runs_dir=release_fixture["runs_dir"])


def test_rejects_failed_qa(release_fixture):
    release_fixture["state"]["qa_result"].verdict = QAVerdict.FAIL

    with pytest.raises(
        ReleaseArtifactError,
        match="QA PASS",
    ):
        load_release_artifacts(
            release_fixture["state"],
            runs_dir=release_fixture["runs_dir"],
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "f" * 32),
        ("revision_count", 1),
    ],
)
def test_rejects_wrong_run_or_attempt(
    release_fixture,
    field,
    value,
):
    update_report(
        release_fixture,
        lambda report: setattr(report, field, value),
    )

    with pytest.raises(ReleaseArtifactError):
        load_release_artifacts(
            release_fixture["state"],
            runs_dir=release_fixture["runs_dir"],
        )


def test_rejects_modified_staged_file(release_fixture):
    source = release_fixture["staging_dir"] / "add.py"

    source.write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ReleaseArtifactError,
        match="hash mismatch",
    ):
        load_release_artifacts(
            release_fixture["state"],
            runs_dir=release_fixture["runs_dir"],
        )


def test_rejects_missing_requirements_assessment(
    release_fixture,
):
    update_report(
        release_fixture,
        lambda report: setattr(
            report,
            "requirements_assessment",
            None,
        ),
    )

    with pytest.raises(
        ReleaseArtifactError,
        match="no requirements assessment",
    ):
        load_release_artifacts(
            release_fixture["state"],
            runs_dir=release_fixture["runs_dir"],
        )


def test_rejects_incomplete_requirements_assessment(
    release_fixture,
):
    def remove_criterion(report):
        assessment = report.requirements_assessment

        assessment.findings = [
            finding
            for finding in assessment.findings
            if finding.requirement_id != "AC-1"
        ]

    update_report(release_fixture, remove_criterion)

    with pytest.raises(
        ReleaseArtifactError,
        match="validation failed",
    ):
        load_release_artifacts(
            release_fixture["state"],
            runs_dir=release_fixture["runs_dir"],
        )
