
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
    dependencies=None,
    setup_status=QACheckStatus.PASSED,
    feedback_generator=None,
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
        dependencies=dependencies or [],
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

    def dependency_preparer(dependencies, dependencies_dir):
        calls.append("dependencies")
        assert dependencies == manifest.dependencies
        assert dependencies_dir.name == "dependencies"
        assert dependencies_dir.parent.name == "attempt-0"
        return QACheckResult(
            name="dependency_setup",
            check_type=QACheckType.DEPENDENCY_SETUP,
            status=setup_status,
            message="Declared dependencies could not be installed."
            if setup_status != QACheckStatus.PASSED else "Dependencies ready.",
        )

    def test_executor(check_type, workspace, **kwargs):
        calls.append(check_type.value)
        if check_type == QACheckType.UNIT_TEST and dependencies:
            assert kwargs["dependency_dir"] == workspace.parent / "dependencies"
        else:
            assert kwargs == {}
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
        runtime_preparer=lambda: None,
        artifact_loader=artifact_loader,
        workspace_preparer=workspace_preparer,
        test_executor=test_executor,
        dependency_preparer=dependency_preparer,
        feedback_generator=feedback_generator,
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
        "dependencies",
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


@pytest.mark.parametrize("setup_status", [
    QACheckStatus.BLOCKED, QACheckStatus.FAILED, QACheckStatus.SKIPPED,
])
def test_dependency_setup_failure_stops_tests_and_repair_feedback(tmp_path, setup_status):
    def unexpected(*args):
        raise AssertionError("QA assessment and repair must not run after setup failure")

    stage, calls, reports = make_stage(
        tmp_path, {}, dependencies=["Pillow"], setup_status=setup_status,
        requirements_assessor=unexpected, feedback_generator=unexpected,
    )
    result = stage(make_state())["qa_result"]

    assert result.verdict == QAVerdict.BLOCKED
    assert result.feedback is None
    assert "dependencies could not be installed" in result.summary
    assert calls == ["artifacts", "workspace", "dependencies", "report"]
    assert reports[0].checks[0].check_type == QACheckType.DEPENDENCY_SETUP
    assert reports[0].checks[0].status == QACheckStatus.BLOCKED
    assert len(reports[0].checks) == 1


def test_dependencies_are_provisioned_and_passed_to_unit_tests(tmp_path):
    stage, calls, reports = make_stage(
        tmp_path,
        {kind: check(kind, QACheckStatus.PASSED) for kind in (
            QACheckType.SYNTAX, QACheckType.UNIT_TEST,
        )},
        dependencies=["Pillow>=10"],
    )

    result = stage(make_state())["qa_result"]

    assert result.verdict == QAVerdict.PASS
    assert calls.index("dependencies") < calls.index("syntax") < calls.index("unit_test")
    assert [item.check_type for item in reports[0].checks] == [
        QACheckType.DEPENDENCY_SETUP, QACheckType.SYNTAX, QACheckType.UNIT_TEST,
    ]


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
