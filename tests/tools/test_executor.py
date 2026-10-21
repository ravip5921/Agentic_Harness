
"""Tests for the sandbox-backed QA executor.

All Docker calls are mocked.
No generated code is executed by these tests.
"""

import subprocess

from harness.models import QACheckStatus, QACheckType
from harness.tools.executor import execute_check


class FakeRunner:
    """Records subprocess calls and simulates Docker results."""

    def __init__(
        self,
        returncode=0,
        stdout="",
        stderr="",
        exception=None,
    ):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.exception = exception
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))

        if self.exception is not None:
            # Allow cleanup calls to succeed after a timeout.
            if command[:3] != ["docker", "rm", "-f"]:
                raise self.exception

        return subprocess.CompletedProcess(
            args=command,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def make_workspace(tmp_path, with_test=True):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    (workspace / "main.py").write_text(
        "def add(a, b):\n    return a + b\n"
    )

    if with_test:
        tests = workspace / "tests"
        tests.mkdir()

        (tests / "test_main.py").write_text(
            "from main import add\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n"
        )

    return workspace


def test_successful_syntax_check(tmp_path):
    workspace = make_workspace(tmp_path)

    runner = FakeRunner(
        stdout="PASS main.py\nPASS tests/test_main.py\n"
    )

    result = execute_check(
        QACheckType.SYNTAX,
        workspace,
        runner=runner,
    )

    assert result.status == QACheckStatus.PASSED
    assert result.exit_code == 0
    assert "PASS main.py" in result.stdout
    assert result.timed_out is False


def test_successful_unit_tests(tmp_path):
    workspace = make_workspace(tmp_path)

    runner = FakeRunner(
        stdout="1 passed in 0.05s\n"
    )

    result = execute_check(
        QACheckType.UNIT_TEST,
        workspace,
        runner=runner,
    )

    assert result.status == QACheckStatus.PASSED
    assert result.exit_code == 0
    assert "1 passed" in result.stdout


def test_failed_unit_tests(tmp_path):
    workspace = make_workspace(tmp_path)

    runner = FakeRunner(
        returncode=1,
        stdout="1 failed in 0.05s\n",
    )

    result = execute_check(
        QACheckType.UNIT_TEST,
        workspace,
        runner=runner,
    )

    assert result.status == QACheckStatus.FAILED
    assert result.exit_code == 1
    assert "1 failed" in result.stdout


def test_docker_command_has_isolation_flags(tmp_path):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()

    execute_check(
        QACheckType.UNIT_TEST,
        workspace,
        runner=runner,
    )

    command = runner.calls[0][0]

    assert command[:2] == ["docker", "run"]
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"

    assert "--read-only" in command
    assert "--cap-drop" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"

    assert "--memory" in command
    assert "--pids-limit" in command
    assert "--security-opt" in command

    assert "--pull" in command
    assert command[command.index("--pull") + 1] == "never"

    mount = command[command.index("--mount") + 1]

    assert str(workspace.resolve()) in mount
    assert "readonly" in mount


def test_syntax_command_does_not_import_generated_code(tmp_path):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()

    execute_check(
        QACheckType.SYNTAX,
        workspace,
        runner=runner,
    )

    command = runner.calls[0][0]

    assert "ast.parse" in command[-1]
    assert command[-2] == "-c"


def test_missing_tests_returns_blocked(tmp_path):
    workspace = make_workspace(
        tmp_path,
        with_test=False,
    )

    runner = FakeRunner()

    result = execute_check(
        QACheckType.UNIT_TEST,
        workspace,
        runner=runner,
    )

    assert result.status == QACheckStatus.BLOCKED
    assert runner.calls == []


def test_unsupported_check_returns_blocked(tmp_path):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()

    result = execute_check(
        QACheckType.INTEGRATION_TEST,
        workspace,
        runner=runner,
    )

    assert result.status == QACheckStatus.BLOCKED
    assert runner.calls == []


def test_timeout_returns_failure_and_attempts_cleanup(tmp_path):
    workspace = make_workspace(tmp_path)

    runner = FakeRunner(
        exception=subprocess.TimeoutExpired(
            cmd="docker run",
            timeout=2,
        ),
    )

    result = execute_check(
        QACheckType.UNIT_TEST,
        workspace,
        timeout_seconds=2,
        runner=runner,
    )

    assert result.status == QACheckStatus.FAILED
    assert result.timed_out is True
    assert result.exit_code is None

    assert len(runner.calls) == 2
    assert runner.calls[1][0][:3] == [
        "docker",
        "rm",
        "-f",
    ]


def test_missing_docker_returns_blocked(tmp_path):
    workspace = make_workspace(tmp_path)

    runner = FakeRunner(
        exception=FileNotFoundError("docker not installed")
    )

    result = execute_check(
        QACheckType.UNIT_TEST,
        workspace,
        runner=runner,
    )

    assert result.status == QACheckStatus.BLOCKED
    assert "Docker executor unavailable" in result.message


def test_output_is_truncated(tmp_path):
    workspace = make_workspace(tmp_path)

    runner = FakeRunner(stdout="x" * 30_000)

    result = execute_check(
        QACheckType.UNIT_TEST,
        workspace,
        runner=runner,
    )

    assert len(result.stdout) == 20_000
