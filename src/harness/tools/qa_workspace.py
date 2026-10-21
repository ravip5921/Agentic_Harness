
"""Prepare disposable QA workspaces from Developer artifacts.

Supports:
    - Standalone generated implementations.
    - Existing repositories with CREATE and MODIFY operations.
    - Original-file hash verification.
    - Bounded repository copying.
    - Temporary workspace creation before publication.

Does not execute generated code or provide a security sandbox.
"""

import hashlib
import os
import shutil
import stat
import tempfile

from pathlib import Path

from harness.models import (
    FileOperation,
    ImplementationManifest,
)
from harness.tools.filesystem import (
    FilesystemError,
    validate_relative_path,
)


MAX_WORKSPACE_FILES = 2000
MAX_WORKSPACE_BYTES = 50_000_000
MAX_SINGLE_FILE_BYTES = 5_000_000

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    "runs",
}

EXCLUDED_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
}


class QAWorkspaceError(Exception):
    """Raised when a QA workspace cannot be safely prepared."""


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_repository(
    source: Path,
    destination: Path,
) -> None:
    """Copy bounded, regular repository files to a disposable directory."""

    file_count = 0
    total_bytes = 0

    def walk(directory: Path):
        nonlocal file_count, total_bytes

        for entry in sorted(directory.iterdir()):
            if entry.is_symlink():
                raise QAWorkspaceError(
                    f"Repository symlink not permitted: {entry}"
                )

            info = entry.lstat()
            relative = entry.relative_to(source)

            if stat.S_ISDIR(info.st_mode):
                if entry.name in EXCLUDED_DIRS:
                    continue

                (destination / relative).mkdir(
                    parents=True,
                    exist_ok=True,
                )
                walk(entry)

            elif stat.S_ISREG(info.st_mode):
                if (
                    entry.name in EXCLUDED_FILES
                    or entry.name.startswith(".env.")
                ):
                    continue

                file_count += 1
                total_bytes += info.st_size

                if (
                    file_count > MAX_WORKSPACE_FILES
                    or total_bytes > MAX_WORKSPACE_BYTES
                    or info.st_size > MAX_SINGLE_FILE_BYTES
                ):
                    raise QAWorkspaceError(
                        "Repository exceeds QA snapshot limits."
                    )

                target = destination / relative
                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with entry.open("rb") as src:
                    with target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)

            else:
                raise QAWorkspaceError(
                    f"Unsupported repository file type: {entry}"
                )

    walk(source)


def prepare_qa_workspace(
    manifest: ImplementationManifest,
    staging_dir: Path,
    workspace_dir: Path,
    repository_path: str | None = None,
) -> Path:
    """Prepare a disposable project workspace for QA.

    Args:
        manifest: Validated Developer implementation manifest.
        staging_dir: Directory containing generated files.
        workspace_dir: Harness-controlled destination directory.
        repository_path: Optional original repository path.

    Returns:
        Path to the completed QA workspace.

    Raises:
        QAWorkspaceError: If validation or preparation fails.

    Important:
        Call load_development_artifacts() before this function.
        The original repository is never modified.
    """

    staging = Path(staging_dir).absolute()
    workspace = Path(workspace_dir).absolute()

    repository = (
        Path(repository_path).expanduser().resolve(strict=True)
        if repository_path is not None
        else None
    )

    # Validate inputs.
    if not staging.is_dir():
        raise QAWorkspaceError(
            "Development staging directory does not exist."
        )

    if repository is not None and not repository.is_dir():
        raise QAWorkspaceError(
            "Target repository is not a directory."
        )

    if workspace.exists() or workspace.is_symlink():
        raise QAWorkspaceError(
            "QA workspace already exists."
        )

    if not manifest.files:
        raise QAWorkspaceError(
            "Implementation manifest contains no files."
        )

    # Reject overlapping source/destination paths.
    if (
        workspace.is_relative_to(staging)
        or staging.is_relative_to(workspace)
    ):
        raise QAWorkspaceError(
            "QA workspace must not overlap staging."
        )

    if repository is not None:
        if (
            workspace.is_relative_to(repository)
            or repository.is_relative_to(workspace)
        ):
            raise QAWorkspaceError(
                "QA workspace must not overlap the repository."
            )

    # Reject symlinks in destination ancestors.
    for parent in (workspace.parent, *workspace.parent.parents):
        if parent.is_symlink():
            raise QAWorkspaceError(
                "QA workspace parent contains a symlink."
            )

    workspace.parent.mkdir(parents=True, exist_ok=True)

    temporary = Path(
        tempfile.mkdtemp(
            prefix=".qa-",
            dir=workspace.parent,
        )
    )

    try:
        # 1. Copy the optional original repository.
        if repository is not None:
            copy_repository(
                repository,
                temporary,
            )

        # 2. Overlay the Developer-generated files.
        seen = set()

        for record in manifest.files:
            try:
                relative = validate_relative_path(
                    record.path
                )
            except FilesystemError as exc:
                raise QAWorkspaceError(str(exc)) from exc

            normalized = relative.as_posix()
            path_key = normalized.casefold()

            if path_key in seen:
                raise QAWorkspaceError(
                    f"Duplicate manifest path: {normalized}"
                )

            seen.add(path_key)

            source = staging / relative
            destination = temporary / relative

            if (
                source.is_symlink()
                or not source.is_file()
                or not source.resolve().is_relative_to(staging)
            ):
                raise QAWorkspaceError(
                    f"Invalid staged file: {normalized}"
                )

            if sha256_file(source) != record.sha256:
                raise QAWorkspaceError(
                    f"Staged file hash mismatch: {normalized}"
                )

            if record.operation == FileOperation.CREATE:
                if destination.exists() or destination.is_symlink():
                    raise QAWorkspaceError(
                        f"CREATE target already exists: {normalized}"
                    )

            elif record.operation == FileOperation.MODIFY:
                if (
                    repository is None
                    or not destination.is_file()
                    or destination.is_symlink()
                ):
                    raise QAWorkspaceError(
                        f"MODIFY target missing: {normalized}"
                    )

                if (
                    record.original_sha256 is None
                    or sha256_file(destination)
                    != record.original_sha256
                ):
                    raise QAWorkspaceError(
                        f"Original file hash mismatch: {normalized}"
                    )

            else:
                raise QAWorkspaceError(
                    f"Unsupported operation: {record.operation}"
                )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if record.operation == FileOperation.CREATE:
                with source.open("rb") as src:
                    with destination.open("xb") as dst:
                        shutil.copyfileobj(src, dst)

            else:
                # Only modify the temporary copy.
                shutil.copyfile(source, destination)

        # 3. Publish only after all operations succeed.
        if workspace.exists() or workspace.is_symlink():
            raise QAWorkspaceError(
                "QA workspace appeared during preparation."
            )

        os.rename(temporary, workspace)

        return workspace

    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)

        raise
