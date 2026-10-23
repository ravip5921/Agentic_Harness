"""Install declared Python dependencies before executing generated QA code.

Only named, binary distributions from PyPI are supported. The installer has
network access but sees only a disposable dependency directory; generated
project files, repositories, and host credentials are never mounted. Tests
subsequently consume this directory in their separate, offline container.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from harness.models import QACheckResult, QACheckStatus, QACheckType
from harness.tools.executor import (
    DEFAULT_IMAGE,
    MAX_OUTPUT_CHARS,
    MAX_TIMEOUT_SECONDS,
)


MAX_DEPENDENCIES = 100
MAX_REQUIREMENT_CHARS = 512

# The container's UID differs from the host's UID. Writable directories let
# the host remove even a partial install without privileged cleanup. No shell
# or package build scripts are used; all pip arguments below are fixed.
PIP_INSTALL_SCRIPT = """\
import os
import runpy

os.umask(0)
runpy.run_module("pip", run_name="__main__")
"""


def _output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return (value or "")[:MAX_OUTPUT_CHARS]


def _requirements(dependencies: list[str]) -> list[str]:
    """Accept PEP 508 names/specifiers/extras/markers, never installer options."""
    if not isinstance(dependencies, list) or len(dependencies) > MAX_DEPENDENCIES:
        raise ValueError(
            f"Dependencies must be a list of at most {MAX_DEPENDENCIES} requirements."
        )

    requirements = []
    for dependency in dependencies:
        if (
            not isinstance(dependency, str)
            or not dependency.strip()
            or len(dependency) > MAX_REQUIREMENT_CHARS
            or dependency.splitlines() != [dependency]
            or any(ord(character) < 32 or ord(character) == 127 for character in dependency)
        ):
            raise ValueError("Each dependency must be a bounded, single-line requirement.")

        try:
            requirement = Requirement(dependency)
        except InvalidRequirement as exc:
            raise ValueError("Dependency must be a named PEP 508 requirement.") from exc

        if requirement.url is not None:
            raise ValueError("URL and local-file dependencies are not supported by QA.")

        requirements.append(str(requirement))

    return requirements


def _create_dependency_directory(dependencies_dir: Path) -> Path:
    """Create a new trusted destination; never reuse another attempt's files."""
    destination = Path(dependencies_dir).absolute()
    if (
        ".." in destination.parts
        or "," in str(destination)
        or any(ord(character) < 32 for character in str(destination))
    ):
        raise ValueError("Invalid QA dependency destination.")

    for path in (destination, *destination.parents):
        if path.is_symlink():
            raise ValueError("QA dependency destination contains a symlink.")

    if destination.exists():
        raise ValueError("QA dependency destination already exists.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        destination.chmod(0o777)
    except OSError:
        destination.rmdir()
        raise
    return destination


def _validate_installed_files(destination: Path) -> None:
    """Reject unexpected filesystem objects before a test container sees them."""
    for current, directories, files in os.walk(destination, followlinks=False):
        for path in [Path(current), *(Path(current) / name for name in directories + files)]:
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                if mode & 0o005 != 0o005:
                    raise ValueError("Installed dependency directory is not readable by QA.")
            elif stat.S_ISREG(mode):
                if mode & 0o004 != 0o004:
                    raise ValueError("Installed dependency file is not readable by QA.")
            else:
                raise ValueError("Installed dependencies contain a symlink or special file.")


def prepare_qa_dependencies(
    dependencies: list[str],
    dependencies_dir: Path,
    *,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = 180,
    runner: Callable = subprocess.run,
) -> QACheckResult:
    """Prepare a fresh per-attempt dependency directory in an isolated installer.

    A setup failure always blocks QA rather than blaming generated code. The
    caller must only pass harness-owned destinations, and must mount a successful
    nonempty installation read-only in the later offline test container. With
    no dependencies, no directory or Docker process is needed.
    """
    result = QACheckResult(
        name=QACheckType.DEPENDENCY_SETUP.value,
        check_type=QACheckType.DEPENDENCY_SETUP,
        status=QACheckStatus.BLOCKED,
        duration_seconds=0.0,
    )
    destination = None
    started = None

    try:
        requirements = _requirements(dependencies)
        if not requirements:
            result.status = QACheckStatus.PASSED
            result.exit_code = 0
            result.message = "No declared Python dependencies require installation."
            return result

        started = time.monotonic()
        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS
        ):
            raise ValueError("Invalid QA dependency installation timeout.")
        if (
            not isinstance(image, str)
            or not image
            or image.startswith("-")
            or any(character.isspace() or ord(character) < 32 for character in image)
        ):
            raise ValueError("Invalid QA dependency container image.")

        destination = _create_dependency_directory(dependencies_dir)
        container_name = f"harness-qa-dependencies-{uuid.uuid4().hex}"
        command = [
            "docker", "run", "--rm", "--pull", "never",
            "--name", container_name,
            "--network", "bridge",
            "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--pids-limit", "128",
            "--memory", "512m", "--memory-swap", "512m", "--cpus", "1.0",
            "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,nosuid,noexec,size=256m",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "HOME=/tmp",
            "--mount", f"type=bind,src={destination},dst=/dependencies",
            "--workdir", "/tmp",
            image,
            "python", "-c", PIP_INSTALL_SCRIPT,
            "--isolated", "--disable-pip-version-check",
            "install", "--no-cache-dir", "--only-binary=:all:",
            "--ignore-installed", "--no-compile",
            "--target", "/dependencies",
            "--index-url", "https://pypi.org/simple",
            "--", *requirements,
        ]
        result.command = command

        try:
            process = runner(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # Killing the Docker client does not necessarily stop its container.
            try:
                runner(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            result.timed_out = True
            result.stdout = _output(exc.stdout)
            result.stderr = _output(exc.stderr)
            result.message = "QA dependency installation exceeded its timeout."
            return result

        result.exit_code = process.returncode
        result.stdout = _output(process.stdout)
        result.stderr = _output(process.stderr)
        if process.returncode != 0:
            result.message = (
                f"QA dependency installation failed with exit code {process.returncode}. "
                "Tests were not run; resolve dependency availability or installer errors."
            )
            return result

        _validate_installed_files(destination)
        result.status = QACheckStatus.PASSED
        result.message = "Declared Python dependencies installed for offline QA tests."
        return result

    except (OSError, ValueError) as exc:
        result.message = f"QA dependency setup unavailable: {exc}"
        return result
    finally:
        if started is not None:
            result.duration_seconds = time.monotonic() - started
        if destination is not None and result.status != QACheckStatus.PASSED:
            try:
                shutil.rmtree(destination)
            except OSError as exc:
                result.message += f" Could not remove partial dependency installation: {exc}"
