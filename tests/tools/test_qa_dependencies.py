"""Dependency setup is isolated from generated code and blocks on failure."""

import stat
import subprocess

import pytest

from harness.models import QACheckStatus, QACheckType
from harness.tools.qa_dependencies import (
    MAX_DEPENDENCIES,
    MAX_OUTPUT_CHARS,
    MAX_REQUIREMENT_CHARS,
    prepare_qa_dependencies,
)


class FakeInstaller:
    def __init__(self, *, returncode=0, stdout="", stderr="", exception=None, populate=None):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.exception = exception
        self.populate = populate
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if self.populate is not None:
            self.populate()
        if self.exception is not None:
            raise self.exception
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, self.stderr)


def test_no_dependencies_need_neither_docker_nor_destination(tmp_path):
    destination = tmp_path / "never-created" / "dependencies"
    runner = FakeInstaller(exception=FileNotFoundError("docker"))

    result = prepare_qa_dependencies([], destination, runner=runner)

    assert result.check_type == QACheckType.DEPENDENCY_SETUP
    assert result.status == QACheckStatus.PASSED
    assert result.exit_code == 0
    assert result.duration_seconds == 0.0
    assert result.command == []
    assert runner.calls == []
    assert not destination.parent.exists()


def test_installs_declared_requirements_into_separate_restricted_container(tmp_path):
    destination = tmp_path / "dependencies"

    def populate():
        # The target must already be writable by the unprivileged container.
        assert stat.S_IMODE(destination.stat().st_mode) == 0o777
        package = destination / "PIL"
        package.mkdir()
        (package / "__init__.py").write_text("")

    runner = FakeInstaller(stdout="Successfully installed Pillow\n", populate=populate)
    requirements = ["Pillow>=10,<12", 'requests[socks]>=2; python_version >= "3.11"']

    result = prepare_qa_dependencies(requirements, destination, runner=runner)

    assert result.status == QACheckStatus.PASSED
    assert result.exit_code == 0
    assert "Successfully installed" in result.stdout
    assert (destination / "PIL/__init__.py").exists()
    assert len(runner.calls) == 1
    command, options = runner.calls[0]
    assert result.command == command
    assert options == {"capture_output": True, "text": True, "timeout": 180, "check": False}
    assert command[:2] == ["docker", "run"]
    assert command[command.index("--network") + 1] == "bridge"
    assert command[command.index("--pull") + 1] == "never"
    assert command[command.index("--user") + 1] == "65534:65534"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges:true"
    assert "--read-only" in command
    assert all(flag in command for flag in ["--pids-limit", "--memory", "--memory-swap", "--cpus", "--tmpfs"])
    mounts = [command[index + 1] for index, option in enumerate(command) if option == "--mount"]
    assert mounts == [f"type=bind,src={destination},dst=/dependencies"]
    assert "/workspace" not in " ".join(command)
    assert "--isolated" in command
    assert "--only-binary=:all:" in command
    assert "--ignore-installed" in command
    assert "--no-cache-dir" in command
    assert "--no-compile" in command
    assert command[command.index("--target") + 1] == "/dependencies"
    assert command[command.index("--index-url") + 1] == "https://pypi.org/simple"
    assert command.index("--isolated") < command.index("install")
    assert command[command.index("--") + 1:] == [
        "Pillow<12,>=10", 'requests[socks]>=2; python_version >= "3.11"',
    ]


@pytest.mark.parametrize("dependency", [
    "", " ", "--extra-index-url=https://example.test/simple", "-r requirements.txt",
    "./local-package", "/tmp/local.whl", "../local-package", "git+https://example.test/repo.git",
    "Pillow @ https://example.test/pillow.whl", "Pillow @ file:///tmp/pillow.whl",
    "Pillow\nrequests", "Pillow\rrequests", "Pillow\x00", "Pillow\t>=10",
    'Pillow; os_name == "a\u2028b"',
    "a" * (MAX_REQUIREMENT_CHARS + 1), 42, None,
])
def test_unsafe_or_invalid_requirements_block_before_filesystem_or_docker(tmp_path, dependency):
    destination = tmp_path / "parent" / "dependencies"
    runner = FakeInstaller()

    result = prepare_qa_dependencies([dependency], destination, runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert result.command == []
    assert runner.calls == []
    assert not destination.parent.exists()


def test_excessive_dependency_count_is_rejected(tmp_path):
    runner = FakeInstaller()

    result = prepare_qa_dependencies(["Pillow"] * (MAX_DEPENDENCIES + 1), tmp_path / "deps", runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert runner.calls == []


@pytest.mark.parametrize("returncode", [1, 2, 125, 126, 127, -9])
def test_installer_failures_block_and_remove_partial_install(tmp_path, returncode):
    destination = tmp_path / "dependencies"

    def populate():
        partial = destination / "partial"
        partial.mkdir()
        (partial / "part.py").write_text("partial")

    runner = FakeInstaller(returncode=returncode, stderr="No matching distribution found", populate=populate)

    result = prepare_qa_dependencies(["missing-package"], destination, runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert result.exit_code == returncode
    assert "No matching distribution" in result.stderr
    assert not destination.exists()


@pytest.mark.parametrize("exception", [FileNotFoundError("docker"), PermissionError("denied"), OSError("unavailable")])
def test_missing_executor_blocks_and_removes_destination(tmp_path, exception):
    destination = tmp_path / "dependencies"

    result = prepare_qa_dependencies(["Pillow"], destination, runner=FakeInstaller(exception=exception))

    assert result.status == QACheckStatus.BLOCKED
    assert result.exit_code is None
    assert not destination.exists()


def test_timeout_removes_only_named_container_and_partial_install(tmp_path):
    destination = tmp_path / "dependencies"
    runner = FakeInstaller(exception=subprocess.TimeoutExpired(
        "docker run", 3, output=b"partial output", stderr=b"timed out\xff",
    ))

    result = prepare_qa_dependencies(["Pillow"], destination, timeout_seconds=3, runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert result.timed_out
    assert result.stdout == "partial output"
    assert result.stderr == "timed out\ufffd"
    assert result.exit_code is None
    run_command = runner.calls[0][0]
    name = run_command[run_command.index("--name") + 1]
    assert runner.calls[1][0] == ["docker", "rm", "-f", name]
    assert runner.calls[1][1]["timeout"] == 10
    assert not destination.exists()


def test_logs_are_bounded(tmp_path):
    runner = FakeInstaller(stdout="x" * 30_000, stderr="y" * 30_000)

    result = prepare_qa_dependencies(["Pillow"], tmp_path / "dependencies", runner=runner)

    assert len(result.stdout) == MAX_OUTPUT_CHARS
    assert len(result.stderr) == MAX_OUTPUT_CHARS


def test_existing_destination_is_not_reused_or_deleted(tmp_path):
    destination = tmp_path / "dependencies"
    destination.mkdir()
    marker = destination / "preserved.txt"
    marker.write_text("existing attempt")
    runner = FakeInstaller()

    result = prepare_qa_dependencies(["Pillow"], destination, runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert marker.read_text() == "existing attempt"
    assert runner.calls == []


def test_symlink_ancestor_is_rejected_without_modifying_target(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    runner = FakeInstaller()

    result = prepare_qa_dependencies(["Pillow"], alias / "deps", runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert not (actual / "deps").exists()
    assert runner.calls == []


def test_symlink_destination_is_not_followed_or_deleted(tmp_path):
    target = tmp_path / "outside"
    target.mkdir()
    destination = tmp_path / "dependencies"
    destination.symlink_to(target, target_is_directory=True)
    runner = FakeInstaller()

    result = prepare_qa_dependencies(["Pillow"], destination, runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert destination.is_symlink()
    assert target.exists()
    assert runner.calls == []


def test_docker_mount_delimiters_in_path_are_rejected(tmp_path):
    runner = FakeInstaller()

    result = prepare_qa_dependencies(["Pillow"], tmp_path / "deps,readonly", runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert runner.calls == []


@pytest.mark.parametrize("kind", ["symlink", "unreadable"])
def test_invalid_installed_files_block_and_are_cleaned(tmp_path, kind):
    destination = tmp_path / "dependencies"
    outside = tmp_path / "important.txt"
    outside.write_text("preserve")

    def populate():
        installed = destination / "package.py"
        if kind == "symlink":
            installed.symlink_to(outside)
        else:
            installed.write_text("pass")
            installed.chmod(0o600)

    result = prepare_qa_dependencies(["Pillow"], destination, runner=FakeInstaller(populate=populate))

    assert result.status == QACheckStatus.BLOCKED
    assert not destination.exists()
    assert outside.read_text() == "preserve"


@pytest.mark.parametrize("timeout", [0, -1, 301, True, 1.5])
def test_invalid_timeout_blocks_without_launching_installer(tmp_path, timeout):
    runner = FakeInstaller()

    result = prepare_qa_dependencies(["Pillow"], tmp_path / "deps", timeout_seconds=timeout, runner=runner)

    assert result.status == QACheckStatus.BLOCKED
    assert runner.calls == []
