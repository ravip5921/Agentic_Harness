
"""Unit tests for QA stage orchestration."""

from pathlib import Path

import pytest

from harness.models import (
    DevelopmentResult,
    ImplementationManifest,
    QACheckResult,
    QACheckStatus,
    QACheckType,
    QAVerdict,
    StageOutcome,
    TaskRequest,
    EngineeringSpec,
    FileOperation,
    QARequirementAssessment,
    RequirementFinding,
    RequirementStatus,
)
from harness.stages.qa import (
    combine_qa_verdicts,
    make_qa_stage,
)
from harness.tools.qa_context import (
    QAImplementationContext,
    VerifiedSourceFile,
)


def make_state():
    return {
        "task": TaskRequest(
            task="Build a Python utility",
            max_dev_revisions=0,
        ),
        "revision_count": 0,
        "development_result": DevelopmentResult(
            outcome=StageOutcome.COMPLETED,
        ),
    }


def make_stage(
    tmp_path,
    results,
    requirement_status=RequirementStatus.SATISFIED,
    requirements_assessor=None,
):
    """Create a QA stage using mocked tools and requirements assessment."""

    calls = []
    reports = []

    specification = EngineeringSpec(
        title="Addition Utility",
        objective="Implement integer addition.",
        requirements=["Implement add()."],
        technical_approach="Use Python.",
        implementation_steps=["Implement the addition function."],
        acceptance_criteria=["add(2, 3) returns 5."],
    )

    manifest = ImplementationManifest(
        summary="Example",
        files=[
            {
                "path": "main.py",
                "operation": "create",
                "sha256": "a" * 64,
            }
        ],
    )

    def artifact_loader(state, runs_dir):
        calls.append("artifacts")
        return manifest, tmp_path / "staging"

    def workspace_preparer(
        manifest,
        staging_dir,
        workspace_dir,
        repository_path,
    ):
        calls.append("workspace")
        return workspace_dir

    def test_executor(check_type, workspace):
        calls.append(check_type.value)
        return results[check_type]

    def specification_loader(state, runs_dir):
        calls.append("specification")
        return specification

    def context_loader(state, runs_dir):
        calls.append("context")
        return QAImplementationContext(
            files=[
                VerifiedSourceFile(
                    path="main.py",
                    operation=FileOperation.CREATE,
                    content="def add(a, b):\n    return a + b\n",
                )
            ]
        )

    def default_requirements_assessor(
        task,
        specification,
        manifest,
        implementation,
        checks,
    ):
        calls.append("requirements")

        return QARequirementAssessment(
            findings=[
                RequirementFinding(
                    requirement_id="REQ-1",
                    requirement="Implement add().",
                    status=requirement_status,
                    explanation="Reviewed the generated implementation.",
                    implementation_files=["main.py"],
                ),
                RequirementFinding(
                    requirement_id="AC-1",
                    requirement="add(2, 3) returns 5.",
                    status=requirement_status,
                    explanation="Reviewed the addition behavior.",
                    implementation_files=["main.py"],
                ),
            ],
            overall_summary="Requirements assessment completed.",
        )

    def report_writer(report, report_path):
        calls.append("report")
        reports.append(report)

    stage = make_qa_stage(
        artifact_loader=artifact_loader,
        workspace_preparer=workspace_preparer,
        test_executor=test_executor,
        report_writer=report_writer,
        specification_loader=specification_loader,
        context_loader=context_loader,
        requirements_assessor=(
            requirements_assessor
            if requirements_assessor is not None
            else default_requirements_assessor
        ),
        runs_dir=tmp_path,
    )

    return stage, calls, reports


def check(check_type, status):
    return QACheckResult(
        name=check_type.value,
        check_type=check_type,
        status=status,
        exit_code=0 if status == QACheckStatus.PASSED else 1,
        stdout="Example check output",
    )


def test_qa_pass(tmp_path):
    stage, calls, reports = make_stage(
        tmp_path,
        {
            QACheckType.SYNTAX: check(
                QACheckType.SYNTAX,
                QACheckStatus.PASSED,
            ),
            QACheckType.UNIT_TEST: check(
                QACheckType.UNIT_TEST,
                QACheckStatus.PASSED,
            ),
        },
    )

    result = stage(make_state())["qa_result"]

    assert result.verdict == QAVerdict.PASS
    assert result.feedback is None
    assert result.report_path is not None

    assert calls == [
        "artifacts",
        "workspace",
        "syntax",
        "unit_test",
        "specification",
        "context",
        "requirements",
        "report",
    ]

    assert reports[0].verdict == QAVerdict.PASS


def test_qa_fail_creates_structured_feedback(tmp_path):
    stage, calls, reports = make_stage(
        tmp_path,
        {
            QACheckType.SYNTAX: check(
                QACheckType.SYNTAX,
                QACheckStatus.PASSED,
            ),
            QACheckType.UNIT_TEST: check(
                QACheckType.UNIT_TEST,
                QACheckStatus.FAILED,
            ),
        },
    )

    result = stage(make_state())["qa_result"]

    assert result.verdict == QAVerdict.FAIL
    assert result.feedback is not None
    assert result.feedback.development_attempt == 0
    assert result.feedback.recommendations

    assert reports[0].verdict == QAVerdict.FAIL


def test_qa_blocked_by_executor(tmp_path):
    stage, calls, reports = make_stage(
        tmp_path,
        {
            QACheckType.SYNTAX: check(
                QACheckType.SYNTAX,
                QACheckStatus.BLOCKED,
            ),
        },
    )

    result = stage(make_state())["qa_result"]

    assert result.verdict == QAVerdict.BLOCKED
    assert result.feedback is None

    assert "unit_test" not in calls
    assert reports[0].verdict == QAVerdict.BLOCKED


def test_qa_missing_task(tmp_path):
    stage, calls, reports = make_stage(
        tmp_path,
        {},
    )

    result = stage({})["qa_result"]

    assert result.verdict == QAVerdict.BLOCKED
    assert calls == []
    assert reports == []


def test_qa_rejects_existing_attempt(tmp_path):
    state = make_state()

    attempt_dir = (
        tmp_path
        / state["task"].run_id
        / "qa"
        / "attempt-0"
    )
    attempt_dir.mkdir(parents=True)

    stage, calls, reports = make_stage(
        tmp_path,
        {},
    )

    result = stage(state)["qa_result"]

    assert result.verdict == QAVerdict.BLOCKED
    assert "already exists" in result.summary
    assert calls == []

def make_requirements_assessment(status):
    """Create a complete synthetic requirements assessment."""

    return QARequirementAssessment(
        findings=[
            RequirementFinding(
                requirement_id="REQ-1",
                requirement="Implement add().",
                status=status,
                explanation="Reviewed implementation.",
                implementation_files=["main.py"],
            ),
            RequirementFinding(
                requirement_id="AC-1",
                requirement="add(2, 3) returns 5.",
                status=status,
                explanation="Reviewed test coverage.",
                implementation_files=["main.py"],
            ),
        ],
        overall_summary="Requirements evaluated.",
    )


def test_execution_pass_but_requirement_fails():
    assessment = make_requirements_assessment(
        RequirementStatus.NOT_SATISFIED
    )

    assert combine_qa_verdicts(
        QAVerdict.PASS,
        assessment,
    ) == QAVerdict.FAIL


def test_execution_pass_but_requirement_uncertain():
    assessment = make_requirements_assessment(
        RequirementStatus.UNCERTAIN
    )

    assert combine_qa_verdicts(
        QAVerdict.PASS,
        assessment,
    ) == QAVerdict.BLOCKED


def test_execution_pass_without_assessment_is_blocked():
    assert combine_qa_verdicts(
        QAVerdict.PASS,
        None,
    ) == QAVerdict.BLOCKED


def test_execution_failure_remains_failure():
    assert combine_qa_verdicts(
        QAVerdict.FAIL,
        None,
    ) == QAVerdict.FAIL


def test_execution_and_requirements_both_pass():
    assessment = make_requirements_assessment(
        RequirementStatus.SATISFIED
    )

    assert combine_qa_verdicts(
        QAVerdict.PASS,
        assessment,
    ) == QAVerdict.PASS

def test_qa_fails_when_requirements_not_satisfied(tmp_path):
    stage, calls, reports = make_stage(
        tmp_path,
        {
            QACheckType.SYNTAX: check(
                QACheckType.SYNTAX,
                QACheckStatus.PASSED,
            ),
            QACheckType.UNIT_TEST: check(
                QACheckType.UNIT_TEST,
                QACheckStatus.PASSED,
            ),
        },
        requirement_status=RequirementStatus.NOT_SATISFIED,
    )

    result = stage(make_state())["qa_result"]

    assert result.verdict == QAVerdict.FAIL
    assert result.feedback is not None
    assert result.feedback.recommendations

    assert reports[0].requirements_assessment is not None
    assert all(
        finding.status == RequirementStatus.NOT_SATISFIED
        for finding in reports[0].requirements_assessment.findings
    )

    assert "requirements" in calls