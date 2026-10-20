
"""Validate and stage Developer-generated files safely.

Generated files are written to an isolated staging workspace.
The target repository is never modified by this module.

Supported operations:
    CREATE - Generate a new file.
    MODIFY - Generate a complete replacement for an existing file.

Patch-based modifications are not supported in this version.
"""

import hashlib
import os
import re
import shutil
import tempfile

from pathlib import Path, PurePosixPath

from harness.models import (
    FileOperation,
    GeneratedFile,
    StagedFile,
)


MAX_FILES = 30
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 10_000_000


class FilesystemError(Exception):
    """Raised when generated files cannot be safely staged."""


def validate_relative_path(path: str) -> Path:
    """Validate a generated file path relative to the staging root."""

    if not path or "\x00" in path:
        raise FilesystemError("Invalid or empty file path.")

    # Require platform-independent POSIX-style paths.
    if "\\" in path:
        raise FilesystemError(
            f"Backslashes are not allowed in paths: {path}"
        )

    if re.match(r"^[a-zA-Z]:", path):
        raise FilesystemError(
            f"Drive-qualified paths are not allowed: {path}"
        )

    pure_path = PurePosixPath(path)

    if pure_path.is_absolute():
        raise FilesystemError(
            f"Absolute paths are not allowed: {path}"
        )

    parts = path.split("/")

    if any(part in ("", ".", "..") for part in parts):
        raise FilesystemError(
            f"Invalid relative path: {path}"
        )

    if any(":" in part for part in parts):
        raise FilesystemError(
            f"Colons are not allowed in paths: {path}"
        )

    return Path(*parts)


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 hash of a byte sequence."""

    return hashlib.sha256(content).hexdigest()


def validate_repository_file(
    repository_root: Path,
    relative_path: Path,
) -> Path:
    """Ensure a repository file is safe to inspect."""

    target = repository_root / relative_path

    # Reject symbolic links in the repository path.
    current = repository_root

    for part in relative_path.parts:
        current = current / part

        if current.is_symlink():
            raise FilesystemError(
                f"Symlinked repository path is not allowed: "
                f"{relative_path}"
            )

    if not target.resolve().is_relative_to(repository_root):
        raise FilesystemError(
            f"Path escapes repository: {relative_path}"
        )

    return target


def prepare_generated_files(
    files: list[GeneratedFile],
    repository_path: str | None,
) -> tuple[list[tuple[Path, bytes]], list[StagedFile]]:
    """Validate all generated files and prepare staging metadata.

    This function does not write files.
    """

    if not files:
        raise FilesystemError("No generated files were provided.")

    if len(files) > MAX_FILES:
        raise FilesystemError(
            f"Too many generated files. Maximum: {MAX_FILES}"
        )

    repository_root = None

    if repository_path is not None:
        repository_root = Path(
            repository_path
        ).expanduser().resolve(strict=True)

        if not repository_root.is_dir():
            raise FilesystemError(
                "Target repository path is not a directory."
            )

    prepared_files = []
    staged_records = []

    seen_paths = set()
    total_bytes = 0

    for generated in files:
        relative_path = validate_relative_path(
            generated.path
        )

        normalized_path = relative_path.as_posix()
        path_key = normalized_path.casefold()

        if path_key in seen_paths:
            raise FilesystemError(
                f"Duplicate generated file: {normalized_path}"
            )

        seen_paths.add(path_key)

        content = generated.content.encode("utf-8")

        if len(content) > MAX_FILE_BYTES:
            raise FilesystemError(
                f"Generated file exceeds size limit: "
                f"{normalized_path}"
            )

        total_bytes += len(content)

        if total_bytes > MAX_TOTAL_BYTES:
            raise FilesystemError(
                "Generated files exceed total size limit."
            )

        original_hash = None

        if repository_root is not None:
            target = validate_repository_file(
                repository_root,
                relative_path,
            )

            exists = target.exists()

        else:
            target = None
            exists = False

        if generated.operation == FileOperation.CREATE:
            if exists:
                raise FilesystemError(
                    f"CREATE target already exists: "
                    f"{normalized_path}"
                )

        elif generated.operation == FileOperation.MODIFY:
            if repository_root is None:
                raise FilesystemError(
                    "MODIFY requires an existing repository."
                )

            if not exists or not target.is_file():
                raise FilesystemError(
                    f"MODIFY target is not an existing file: "
                    f"{normalized_path}"
                )

            if target.stat().st_size > MAX_FILE_BYTES:
                raise FilesystemError(
                    f"Original file exceeds size limit: "
                    f"{normalized_path}"
                )

            try:
                original_content = target.read_bytes()
            except OSError as exc:
                raise FilesystemError(
                    f"Could not read original file: "
                    f"{normalized_path}"
                ) from exc

            original_hash = sha256_bytes(
                original_content
            )

        else:
            raise FilesystemError(
                f"Unsupported file operation: "
                f"{generated.operation}"
            )

        prepared_files.append(
            (relative_path, content)
        )

        staged_records.append(
            StagedFile(
                path=normalized_path,
                operation=generated.operation,
                sha256=sha256_bytes(content),
                original_sha256=original_hash,
            )
        )

    # Prevent file/directory collisions, such as generating
    # both "src/module.py" and "src/module.py/child.py".
    paths = {path.as_posix() for path, _ in prepared_files}

    for path, _ in prepared_files:
        for parent in path.parents:
            if parent == Path("."):
                break

            if parent.as_posix() in paths:
                raise FilesystemError(
                    f"File/directory path collision: {path}"
                )

    return prepared_files, staged_records


def stage_generated_files(
    files: list[GeneratedFile],
    staging_root: Path,
    repository_path: str | None = None,
) -> list[StagedFile]:
    """Validate and stage a complete batch of generated files.

    Args:
        files: Developer-generated file objects.
        staging_root: Trusted destination for generated files.
        repository_path: Optional existing target repository.

    Returns:
        Validated metadata describing each staged file.

    Raises:
        FilesystemError: If validation or staging fails.

    Notes:
        - The original repository is never modified.
        - All files are validated before staging.
        - Staging occurs in a temporary directory.
        - The completed directory is published only after success.
        - An existing staging directory is never intentionally overwritten.
    """

    # Validate the entire batch before making changes.
    try:
        prepared, records = prepare_generated_files(
            files=files,
            repository_path=repository_path,
        )
    except OSError as exc:
        raise FilesystemError(
            f"File validation failed: {exc}"
        ) from exc

    staging_root = Path(staging_root).absolute()

    # The staging location must be controlled by the harness,
    # not selected by the LLM.
    parent = staging_root.parent

    temporary_root = None

    try:
        if staging_root.exists() or staging_root.is_symlink():
            raise FilesystemError(
                f"Staging directory already exists: "
                f"{staging_root}"
            )

        parent.mkdir(parents=True, exist_ok=True)

        # Fail if the workspace parent resolves through symlinks.
        if parent.resolve() != parent:
            raise FilesystemError(
                "Staging parent must not contain symlinks."
            )

        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=".staging-",
                dir=parent,
            )
        )

        for relative_path, content in prepared:
            destination = temporary_root / relative_path

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Exclusive creation prevents accidental overwrites.
            with destination.open("xb") as output:
                output.write(content)

        # Publish the completed staging workspace.
        os.rename(temporary_root, staging_root)

        temporary_root = None

    except (OSError, FilesystemError) as exc:
        raise FilesystemError(
            f"Could not stage generated files: {exc}"
        ) from exc

    finally:
        # Remove incomplete temporary files on failure.
        if temporary_root is not None:
            shutil.rmtree(
                temporary_root,
                ignore_errors=True,
            )

    return records
