"""Docker preflight and image provisioning regressions; no real Docker calls."""

import subprocess
from unittest.mock import Mock

import pytest

from harness.graph import route_qa
from harness.models import DevelopmentResult, QAVerdict, StageOutcome, TaskRequest
from harness.stages.qa import make_qa_stage
from harness.tools.qa_runtime import QARuntimeError, prepare_qa_runtime


def result(code=0, stderr=""):
    return subprocess.CompletedProcess([], code, stdout="", stderr=stderr)


def test_existing_image_is_reused():
    runner = Mock(side_effect=[result(), result()])
    prepare_qa_runtime(runner=runner)
    assert [call.args[0][:3] for call in runner.call_args_list] == [
        ["docker", "info"], ["docker", "image", "inspect"],
    ]


def test_missing_image_builds_only_trusted_dockerfile():
    runner = Mock(side_effect=[result(), result(1, "No such image"), result(), result(), result()])
    prepare_qa_runtime(runner=runner)
    build = runner.call_args_list[3]
    assert build.args[0][:4] == ["docker", "build", "--file", "-"]
    assert "FROM python:3.11-slim" in build.kwargs["input"]
    assert build.kwargs["timeout"] == 600
    assert runner.call_args_list[-1].args[0][:3] == ["docker", "image", "inspect"]


@pytest.mark.parametrize("failure", [
    FileNotFoundError("docker missing"),
    PermissionError("access denied"),
    subprocess.TimeoutExpired("docker info", 30),
    result(1, "Cannot connect to the Docker daemon"),
])
def test_unavailable_docker_fails_before_image_inspection(failure):
    runner = Mock(side_effect=[failure])
    with pytest.raises(QARuntimeError, match="Docker access is unavailable"):
        prepare_qa_runtime(runner=runner)
    assert runner.call_count == 1


def test_inspection_failure_is_not_treated_as_missing_image():
    runner = Mock(side_effect=[result(), result(1, "API version unsupported"), result()])
    with pytest.raises(QARuntimeError, match="API version unsupported"):
        prepare_qa_runtime(runner=runner)
    assert runner.call_count == 3


@pytest.mark.parametrize("failure", [
    result(1, "registry unavailable"),
    subprocess.TimeoutExpired("docker build", 600),
])
def test_build_failure_is_explicit(failure):
    runner = Mock(side_effect=[result(), result(1, "No such image"), result(), failure])
    with pytest.raises(QARuntimeError, match="QA image build failed"):
        prepare_qa_runtime(runner=runner)


def test_custom_missing_image_is_not_overwritten():
    runner = Mock(side_effect=[result(), result(1, "No such image"), result()])
    with pytest.raises(QARuntimeError, match="custom image"):
        prepare_qa_runtime(image="custom:latest", runner=runner)
    assert runner.call_count == 3


def test_runtime_exception_blocks_without_workspace_or_developer_retry(tmp_path):
    runner = Mock(return_value=result(1, "Docker API unavailable"))
    artifacts = Mock()
    stage = make_qa_stage(
        runtime_preparer=lambda: prepare_qa_runtime(runner=runner),
        artifact_loader=artifacts, runs_dir=tmp_path,
    )
    state = {
        "task": TaskRequest(task="Build a utility", max_dev_revisions=3),
        "revision_count": 0,
        "development_result": DevelopmentResult(outcome=StageOutcome.COMPLETED),
    }
    state.update(stage(state))
    assert state["qa_result"].verdict == QAVerdict.BLOCKED
    assert "QARuntimeError" in state["qa_result"].summary
    assert "Docker API unavailable" in state["qa_result"].summary
    assert state["qa_result"].feedback is None
    assert route_qa(state) == "blocked"
    assert state["revision_count"] == 0
    artifacts.assert_not_called()
    assert list(tmp_path.iterdir()) == []
