
"""Execute controlled QA checks in restricted Docker containers.

Supported checks:
    - Python syntax validation using ast.parse()
    - Python unit tests using pytest

Generated code is never executed directly on the host.
The executor does not accept arbitrary LLM-generated commands.
"""

import re
import subprocess
import time
import uuid

from collections.abc import Callable
from pathlib import Path

from harness.models import (
    QACheckResult,
    QACheckStatus,
    QACheckType,
)


DEFAULT_IMAGE = "harness-qa-python:3.11"
MAX_OUTPUT_CHARS = 20_000
MAX_TIMEOUT_SECONDS = 300
DOCKER_RUNTIME_FAILURE_CODES = {125, 126, 127}
MISSING_MODULE_PATTERN = re.compile(
    r"^(?:E[ \t]+)?ModuleNotFoundError: No module named "
    r"(['\"])([^'\"\r\n]+)\1[ \t]*$",
    re.MULTILINE,
)
MISSING_SHARED_LIBRARY_PATTERN = re.compile(
    r"^(?:E[ \t]+)?(?:ImportError|OSError): "
    r"(lib[A-Za-z0-9_.+-]*\.so(?:\.[0-9]+)*): "
    r"cannot open shared object file: No such file or directory[ \t]*$",
    re.MULTILINE,
)


# This script parses Python files without importing or executing them.
SYNTAX_SCRIPT = """
import ast
from pathlib import Path

root = Path("/workspace")

files = sorted(
    path for path in root.rglob("*.py")
    if path.is_file()
)

if not files:
    raise SystemExit("No Python files found")

failures = 0

for path in files:
    try:
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        print(f"PASS {path.relative_to(root)}")
    except (SyntaxError, UnicodeError) as exc:
        failures += 1
        print(f"FAIL {path.relative_to(root)}: {exc}")

raise SystemExit(1 if failures else 0)
"""


class QAExecutionError(Exception):
    """Raised for invalid executor requests."""


def _get_test_files(workspace: Path) -> list[Path]:
    """Discover conventional pytest test filenames."""

    return [
        path
        for path in workspace.rglob("*.py")
        if path.is_file()
        and (
            path.name.startswith("test_")
            or path.name.endswith("_test.py")
        )
    ]


def _is_local_module(workspace: Path, module: str) -> bool:
    """Recognize a workspace import root without importing generated code."""
    root = module.split(".", 1)[0]
    if not root.isidentifier():
        return False
    return any(
        (directory / f"{root}.py").is_file()
        or (directory / root).is_dir()
        for directory in (workspace, workspace / "src")
    )


def _container_command(
    check_type: QACheckType,
) -> list[str]:
    """Return a fixed command for a supported check."""

    if check_type == QACheckType.SYNTAX:
        return [
            "python",
            "-c",
            SYNTAX_SCRIPT,
        ]

    if check_type == QACheckType.UNIT_TEST:
        return [
            "python",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
        ]

    raise QAExecutionError(
        f"Unsupported executable QA check: {check_type}"
    )


def _blocked(
    check_type: QACheckType,
    message: str,
) -> QACheckResult:
    return QACheckResult(
        name=check_type.value,
        check_type=check_type,
        status=QACheckStatus.BLOCKED,
        message=message,
    )


def execute_check(
    check_type: QACheckType,
    workspace: Path,
    *,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = 60,
    dependency_dir: Path | None = None,
    runner: Callable = subprocess.run,
) -> QACheckResult:
    """Execute one predefined QA check inside Docker.

    Args:
        check_type: Supported QA check category.
        workspace: Trusted, prepared QA workspace.
        image: Prebuilt, trusted Docker test image.
        timeout_seconds: Maximum execution time.
        dependency_dir: Prepared Python dependencies for unit tests.
        runner: Injectable subprocess runner for testing.

    Returns:
        QACheckResult with status, command, exit code, and logs.

    Notes:
        Docker images must be built in advance.
        No image pulling, dependency installation, or network
        access is performed by this executor.
    """

    if (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        raise QAExecutionError("Invalid QA timeout.")

    if not isinstance(check_type, QACheckType):
        raise QAExecutionError("Invalid QA check type.")

    workspace = Path(workspace).resolve(strict=True)

    if not workspace.is_dir():
        raise QAExecutionError(
            "QA workspace must be a directory."
        )

    if not image or image.startswith("-"):
        raise QAExecutionError("Invalid QA container image.")

    try:
        container_command = _container_command(check_type)
    except QAExecutionError as exc:
        return _blocked(check_type, str(exc))

    # Refuse to execute checks that cannot meaningfully run.
    if check_type == QACheckType.SYNTAX:
        if not any(workspace.rglob("*.py")):
            return _blocked(
                check_type,
                "No Python files found for syntax verification.",
            )

    if check_type == QACheckType.UNIT_TEST:
        if not _get_test_files(workspace):
            return _blocked(
                check_type,
                "No Python test files found.",
            )

    dependency_mount = []
    python_path = "/workspace:/workspace/src"
    if check_type == QACheckType.UNIT_TEST and dependency_dir is not None:
        dependency_path = Path(dependency_dir).absolute()
        if any(
            path.is_symlink()
            for path in (dependency_path, *dependency_path.parents)
        ):
            raise QAExecutionError(
                "QA dependency directory must not contain symlinks in its path."
            )
        try:
            dependency_path = dependency_path.resolve(strict=True)
        except OSError as exc:
            raise QAExecutionError(
                "QA dependency directory must exist."
            ) from exc
        if not dependency_path.is_dir() or "," in str(dependency_path):
            raise QAExecutionError("Invalid QA dependency directory.")
        dependency_mount = [
            "--mount",
            f"type=bind,src={dependency_path},dst=/qa-dependencies,readonly",
        ]
        python_path += ":/qa-dependencies"

    container_name = f"harness-qa-{uuid.uuid4().hex}"

    command = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        container_name,

        # Isolation and resource limits.
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "128",
        "--memory",
        "512m",
        "--memory-swap",
        "512m",
        "--cpus",
        "1.0",

        # Execute as an unprivileged user.
        "--user",
        "65534:65534",

        # Writable temporary area only.
        "--tmpfs",
        "/tmp:rw,nosuid,noexec,size=64m",

        # Prevent Python bytecode and pytest cache writes.
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "--env",
        f"PYTHONPATH={python_path}",
        "--env",
        "HOME=/tmp",

        # Mount project read-only.
        "--mount",
        f"type=bind,src={workspace},dst=/workspace,readonly",
        *dependency_mount,

        "--workdir",
        "/workspace",

        image,
        *container_command,
    ]

    started = time.monotonic()

    try:
        process = runner(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

        elapsed = time.monotonic() - started

        status = (
            QACheckStatus.PASSED
            if process.returncode == 0
            else QACheckStatus.FAILED
        )
        message = (
            "Check passed."
            if process.returncode == 0
            else f"Check failed with exit code {process.returncode}."
        )
        if process.returncode in DOCKER_RUNTIME_FAILURE_CODES:
            status = QACheckStatus.BLOCKED
            message = (
                "Docker could not run the QA check "
                f"(exit code {process.returncode}). Check the QA image and runtime."
            )
        elif process.returncode != 0 and check_type == QACheckType.UNIT_TEST:
            missing_modules = sorted({
                match.group(2)
                for output in (process.stdout, process.stderr)
                for match in MISSING_MODULE_PATTERN.finditer(output)
            })
            unresolved_modules = [
                module for module in missing_modules
                if not _is_local_module(workspace, module)
            ]
            missing_libraries = sorted({
                match.group(1)
                for output in (process.stdout, process.stderr)
                for match in MISSING_SHARED_LIBRARY_PATTERN.finditer(output)
            })
            if missing_libraries:
                status = QACheckStatus.BLOCKED
                message = (
                    "QA unit tests require unavailable system shared libraries: "
                    + ", ".join(missing_libraries)
                    + ". Provision these libraries in the trusted QA Docker image "
                    "before rerunning tests; Python dependency installation "
                    "does not install system libraries."
                )
            elif unresolved_modules:
                status = QACheckStatus.BLOCKED
                message = (
                    "QA unit tests could not import Python module(s): "
                    + ", ".join(unresolved_modules)
                    + ". These import roots were not found in the workspace. "
                    "Check the implementation and dependency manifest before "
                    "rerunning tests; import names are not necessarily package "
                    "distribution names."
                )
            elif missing_modules:
                message = (
                    "QA unit tests could not import local implementation module(s): "
                    + ", ".join(missing_modules)
                    + ". Check the implementation and test imports."
                )

        return QACheckResult(
            name=check_type.value,
            check_type=check_type,
            status=status,
            command=command,
            exit_code=process.returncode,
            duration_seconds=elapsed,
            stdout=process.stdout[:MAX_OUTPUT_CHARS],
            stderr=process.stderr[:MAX_OUTPUT_CHARS],
            message=message,
        )

    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started

        # A timed-out Docker client may leave its container running.
        # Attempt to force-remove that specific container.
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

        return QACheckResult(
            name=check_type.value,
            check_type=check_type,
            status=QACheckStatus.FAILED,
            command=command,
            exit_code=None,
            duration_seconds=elapsed,
            stdout=(
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )[:MAX_OUTPUT_CHARS],
            stderr=(
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )[:MAX_OUTPUT_CHARS],
            timed_out=True,
            message="QA check exceeded its execution timeout.",
        )

    except (FileNotFoundError, PermissionError) as exc:
        return QACheckResult(
            name=check_type.value,
            check_type=check_type,
            status=QACheckStatus.BLOCKED,
            command=command,
            message=f"Docker executor unavailable: {exc}",
        )

    except OSError as exc:
        return QACheckResult(
            name=check_type.value,
            check_type=check_type,
            status=QACheckStatus.BLOCKED,
            command=command,
            message=f"Could not start QA executor: {exc}",
        )
