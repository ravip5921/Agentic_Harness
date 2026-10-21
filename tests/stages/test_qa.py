
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
)
from harness.stages.qa import make_qa_stage


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


def make_stage(tmp_path, results):
    """Create a QA stage using mocked tools."""

    calls = []
    reports = []

    def artifact_loader(state, runs_dir):
        calls.append("artifacts")
        return (
            ImplementationManifest(
                summary="Example",
                files=[
                    {
                        "path": "main.py",
                        "operation": "create",
                        "sha256": "a" * 64,
                    }
                ],
            ),
            tmp_path / "staging",
        )

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

    def report_writer(report, report_path):
        calls.append("report")
        reports.append(report)

    stage = make_qa_stage(
        artifact_loader=artifact_loader,
        workspace_preparer=workspace_preparer,
        test_executor=test_executor,
        report_writer=report_writer,
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
