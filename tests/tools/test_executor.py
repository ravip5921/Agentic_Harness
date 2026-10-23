
"""Tests for the sandbox-backed QA executor.

All Docker calls are mocked.
No generated code is executed by these tests.
"""

import subprocess

import pytest

from harness.models import QACheckStatus, QACheckType
from harness.tools.executor import QAExecutionError, execute_check


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


def test_unit_tests_use_prepared_dependencies_without_network(tmp_path):
    workspace = make_workspace(tmp_path)
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    runner = FakeRunner()

    result = execute_check(
        QACheckType.UNIT_TEST,
        workspace,
        dependency_dir=dependencies,
        runner=runner,
    )

    assert result.status == QACheckStatus.PASSED
    command = runner.calls[0][0]
    assert (
        f"type=bind,src={dependencies.resolve()},dst=/qa-dependencies,readonly"
        in command
    )
    assert "PYTHONPATH=/workspace:/workspace/src:/qa-dependencies" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "pip" not in command


def test_syntax_does_not_use_dependency_directory(tmp_path):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()

    execute_check(
        QACheckType.SYNTAX,
        workspace,
        dependency_dir=tmp_path / "not-created",
        runner=runner,
    )

    command = runner.calls[0][0]
    assert not any("/qa-dependencies" in item for item in command)
    assert "PYTHONPATH=/workspace:/workspace/src" in command


@pytest.mark.parametrize("invalid_path", ["missing", "file", "symlink", "ancestor"])
def test_invalid_dependency_directory_is_rejected(tmp_path, invalid_path):
    workspace = make_workspace(tmp_path)
    dependencies = tmp_path / "dependencies"
    if invalid_path == "file":
        dependencies.write_text("not a directory")
    elif invalid_path in {"symlink", "ancestor"}:
        real_directory = tmp_path / "real-dependencies"
        real_directory.mkdir()
        dependencies.symlink_to(real_directory, target_is_directory=True)
        if invalid_path == "ancestor":
            (real_directory / "nested").mkdir()
            dependencies = dependencies / "nested"
    runner = FakeRunner()

    with pytest.raises(QAExecutionError, match="dependency directory"):
        execute_check(
            QACheckType.UNIT_TEST,
            workspace,
            dependency_dir=dependencies,
            runner=runner,
        )

    assert runner.calls == []


@pytest.mark.parametrize("returncode", [125, 126, 127])
def test_docker_runtime_failure_is_blocked(tmp_path, returncode):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner(returncode=returncode, stderr="docker: runtime unavailable")

    result = execute_check(QACheckType.UNIT_TEST, workspace, runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert result.exit_code == returncode
    assert result.stderr == "docker: runtime unavailable"
    assert "runtime" in result.message


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("prefix", ["", "E   "])
def test_missing_python_dependency_is_blocked(tmp_path, stream, prefix):
    workspace = make_workspace(tmp_path)
    output = f"{prefix}ModuleNotFoundError: No module named 'PIL'\n"
    runner = FakeRunner(returncode=2, **{stream: output})

    result = execute_check(QACheckType.UNIT_TEST, workspace, runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert result.exit_code == 2
    assert getattr(result, stream) == output
    assert "PIL" in result.message
    assert "distribution" in result.message


def test_missing_module_message_from_passing_test_is_not_blocked(tmp_path):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner(stdout="ModuleNotFoundError: No module named 'PIL'\n1 passed\n")

    result = execute_check(QACheckType.UNIT_TEST, workspace, runner=runner)

    assert result.status == QACheckStatus.PASSED


@pytest.mark.parametrize("source_root", ["", "src"])
@pytest.mark.parametrize("module_kind", ["package", "module"])
def test_missing_local_submodule_remains_implementation_failure(
    tmp_path, source_root, module_kind,
):
    workspace = make_workspace(tmp_path)
    source = workspace / source_root
    source.mkdir(exist_ok=True)
    if module_kind == "package":
        (source / "drawing").mkdir()
    else:
        (source / "drawing.py").write_text("# local module\n")
    runner = FakeRunner(
        returncode=2,
        stdout="E   ModuleNotFoundError: No module named 'drawing.utilities'\n",
    )

    result = execute_check(QACheckType.UNIT_TEST, workspace, runner=runner)

    assert result.status == QACheckStatus.FAILED
    assert "local implementation" in result.message


@pytest.mark.parametrize("exception", ["ImportError", "OSError"])
@pytest.mark.parametrize("library", ["libGL.so.1", "libexample.so", "libssl.so.1.1"])
def test_missing_system_shared_library_blocks_qa(tmp_path, exception, library):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner(
        returncode=2,
        stdout=(
            f"E   {exception}: {library}: cannot open shared object file: "
            "No such file or directory\n"
        ),
    )

    result = execute_check(QACheckType.UNIT_TEST, workspace, runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert library in result.message
    assert "trusted QA Docker image" in result.message


@pytest.mark.parametrize("output", [
    "E   ImportError: cannot import name 'draw' from 'drawing'\n",
    "E   OSError: libexample.so: cannot open shared object file: Permission denied\n",
    "E   AssertionError: libGL.so.1: cannot open shared object file: No such file or directory\n",
])
def test_other_import_and_os_errors_remain_failures(tmp_path, output):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner(returncode=1, stderr=output)

    result = execute_check(QACheckType.UNIT_TEST, workspace, runner=runner)

    assert result.status == QACheckStatus.FAILED


def test_assertion_mentioning_missing_module_is_still_a_failure(tmp_path):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner(
        returncode=1,
        stdout=(
            '>       assert output == "ModuleNotFoundError: No module named \'PIL\'"\n'
            "E       AssertionError: output did not match\n"
        ),
    )

    result = execute_check(QACheckType.UNIT_TEST, workspace, runner=runner)

    assert result.status == QACheckStatus.FAILED


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
