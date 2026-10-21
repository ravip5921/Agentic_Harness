"""Load and verify Developer artifacts before QA execution.

Validates the expected development attempt, artifact registration,
manifest structure, generated file paths, content hashes, and
original repository hashes.

This module never executes generated code or modifies the repository.
"""

import hashlib
import os
import re
import stat

from pathlib import Path

from pydantic import ValidationError

from harness.models import (
    FileOperation,
    ImplementationManifest,
    StageOutcome,
)
from harness.state import HarnessState

from harness.tools.filesystem import (
    FilesystemError,
    MAX_FILES,
    MAX_FILE_BYTES,
    MAX_TOTAL_BYTES,
    validate_relative_path,
)


MAX_MANIFEST_BYTES = 1_000_000
SHA256_PATTERN = re.compile(r"[a-f0-9]{64}")


class ArtifactError(Exception):
    """Raised when Developer artifacts cannot be verified."""


def _check_no_symlinks(path: Path) -> None:
    """Reject symlinks in an artifact path or its ancestors."""

    current = path.absolute()

    while True:
        if current.is_symlink():
            raise ArtifactError(
                f"Symlinked artifact path is not allowed: {current}"
            )

        if current == current.parent:
            break

        current = current.parent


def _read_regular_file(path: Path, max_bytes: int) -> bytes:
    """Read a bounded regular file without following its final symlink."""

    _check_no_symlinks(path)

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)

    fd = os.open(path, flags)

    try:
        info = os.fstat(fd)

        if not stat.S_ISREG(info.st_mode):
            raise ArtifactError(
                f"Expected a regular file: {path}"
            )

        if info.st_size > max_bytes:
            raise ArtifactError(
                f"File exceeds size limit: {path}"
            )

        data = os.read(fd, max_bytes + 1)

        if len(data) > max_bytes:
            raise ArtifactError(
                f"File exceeds size limit: {path}"
            )

        return data

    finally:
        os.close(fd)


def _verify_registered_artifacts(
    registered: list[str],
    manifest_path: Path,
    staging_dir: Path,
) -> None:
    """Ensure both expected paths are declared by Developer."""

    registered_paths = {
        Path(path).absolute()
        for path in registered
    }

    if (
        manifest_path not in registered_paths
        or staging_dir not in registered_paths
    ):
        raise ArtifactError(
            "Manifest or staging directory is not registered "
            "in DevelopmentResult.artifacts."
        )


def _verify_manifest_files(
    manifest: ImplementationManifest,
    staging_dir: Path,
    repository_path: str | None,
) -> None:
    """Verify every staged file and its recorded SHA-256 hashes."""

    if not manifest.files:
        raise ArtifactError("Implementation manifest contains no files.")

    if len(manifest.files) > MAX_FILES:
        raise ArtifactError("Manifest exceeds maximum file count.")

    repository_root = None

    if repository_path is not None:
        repository_root = Path(
            repository_path
        ).expanduser().absolute()

        _check_no_symlinks(repository_root)

        if not repository_root.is_dir():
            raise ArtifactError("Target repository is unavailable.")

    seen = set()
    declared_paths = set()
    total_bytes = 0

    for record in manifest.files:
        try:
            relative = validate_relative_path(record.path)
        except FilesystemError as exc:
            raise ArtifactError(str(exc)) from exc

        normalized = relative.as_posix()
        path_key = normalized.casefold()

        if path_key in seen:
            raise ArtifactError(
                f"Duplicate manifest path: {normalized}"
            )

        seen.add(path_key)
        declared_paths.add(normalized)

        if not SHA256_PATTERN.fullmatch(record.sha256):
            raise ArtifactError(
                f"Invalid staged SHA-256: {normalized}"
            )

        staged_path = staging_dir / relative

        content = _read_regular_file(
            staged_path,
            MAX_FILE_BYTES,
        )

        total_bytes += len(content)

        if total_bytes > MAX_TOTAL_BYTES:
            raise ArtifactError(
                "Staged content exceeds total size limit."
            )

        actual_hash = hashlib.sha256(content).hexdigest()

        if actual_hash != record.sha256:
            raise ArtifactError(
                f"Staged file hash mismatch: {normalized}"
            )

        if record.operation == FileOperation.CREATE:
            if record.original_sha256 is not None:
                raise ArtifactError(
                    f"CREATE must not have an original hash: "
                    f"{normalized}"
                )

            if repository_root is not None:
                original = repository_root / relative

                _check_no_symlinks(original)

                if original.exists():
                    raise ArtifactError(
                        f"CREATE conflicts with existing file: "
                        f"{normalized}"
                    )

        elif record.operation == FileOperation.MODIFY:
            if repository_root is None:
                raise ArtifactError(
                    f"MODIFY requires a repository: {normalized}"
                )

            if (
                record.original_sha256 is None
                or not SHA256_PATTERN.fullmatch(
                    record.original_sha256
                )
            ):
                raise ArtifactError(
                    f"Missing or invalid original hash: "
                    f"{normalized}"
                )

            original = repository_root / relative

            original_content = _read_regular_file(
                original,
                MAX_FILE_BYTES,
            )

            original_hash = hashlib.sha256(
                original_content
            ).hexdigest()

            if original_hash != record.original_sha256:
                raise ArtifactError(
                    f"Original repository file changed: "
                    f"{normalized}"
                )

        else:
            raise ArtifactError(
                f"Unsupported operation: {record.operation}"
            )

    # Detect staged files that the manifest does not declare.
    actual_paths = set()

    for current_dir, directories, filenames in os.walk(
        staging_dir,
        followlinks=False,
    ):
        current = Path(current_dir)

        for dirname in directories:
            directory = current / dirname

            if directory.is_symlink():
                raise ArtifactError(
                    f"Symlinked staging directory: {directory}"
                )

        for filename in filenames:
            file_path = current / filename

            _check_no_symlinks(file_path)

            relative_name = file_path.relative_to(
                staging_dir
            ).as_posix()

            actual_paths.add(relative_name)

    if actual_paths != declared_paths:
        missing = declared_paths - actual_paths
        unexpected = actual_paths - declared_paths

        raise ArtifactError(
            "Staging contents do not match manifest. "
            f"Missing: {sorted(missing)}; "
            f"Unexpected: {sorted(unexpected)}"
        )


def load_development_artifacts(
    state: HarnessState,
    runs_dir: Path = Path("runs"),
) -> tuple[ImplementationManifest, Path]:
    """Load and verify the current Developer attempt.

    Returns:
        (validated ImplementationManifest, staging directory)

    Raises:
        ArtifactError: If prerequisites or artifacts are invalid.
    """

    task = state.get("task")
    development = state.get("development_result")

    if task is None:
        raise ArtifactError("Task is missing from workflow state.")

    if development is None:
        raise ArtifactError("Development result is missing.")

    if development.outcome != StageOutcome.COMPLETED:
        raise ArtifactError(
            "Cannot run QA without completed Development."
        )

    if not re.fullmatch(r"[a-f0-9]{32}", task.run_id):
        raise ArtifactError("Invalid run ID.")

    revision = state.get("revision_count", 0)

    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
        or revision > task.max_dev_revisions
    ):
        raise ArtifactError("Invalid development revision counter.")

    runs_root = Path(runs_dir).absolute()

    attempt_dir = (
        runs_root
        / task.run_id
        / "development"
        / f"attempt-{revision}"
    )

    manifest_path = attempt_dir / "manifest.json"
    staging_dir = attempt_dir / "staging"

    try:
        _check_no_symlinks(manifest_path)
        _check_no_symlinks(staging_dir)

        _verify_registered_artifacts(
            development.artifacts,
            manifest_path,
            staging_dir,
        )

        if not staging_dir.is_dir():
            raise ArtifactError(
                "Development staging directory is missing."
            )

        manifest_data = _read_regular_file(
            manifest_path,
            MAX_MANIFEST_BYTES,
        )

        manifest = ImplementationManifest.model_validate_json(
            manifest_data
        )

        _verify_manifest_files(
            manifest,
            staging_dir,
            task.repository_path,
        )

        return manifest, staging_dir

    except ArtifactError:
        raise

    except (OSError, ValidationError, ValueError) as exc:
        raise ArtifactError(
            f"Could not validate development artifacts: {exc}"
        ) from exc