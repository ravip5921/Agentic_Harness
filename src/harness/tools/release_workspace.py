
"""Prepare a reviewable Release candidate from QA-approved artifacts.

This module:
- Requires validated ReleaseArtifacts.
- Reuses the existing safe workspace preparation tool.
- Supports standalone tasks and optional existing repositories.
- Verifies the generated files in the resulting candidate.
- Never modifies the original repository.

It does not create commits, merge branches, or deploy code.
"""

import hashlib
from pathlib import Path

from harness.tools.release_artifacts import ReleaseArtifacts
from harness.tools.qa_workspace import (
    QAWorkspaceError,
    prepare_qa_workspace,
)
from harness.tools.filesystem import (
    MAX_FILE_BYTES,
    validate_relative_path,
)


class ReleaseWorkspaceError(Exception):
    """Raised when Release candidate preparation fails."""


def verify_release_candidate(
    artifacts: ReleaseArtifacts,
    candidate_dir: Path,
) -> None:
    """Verify the candidate contains the exact QA-approved files."""

    candidate_dir = Path(candidate_dir).absolute()

    for record in artifacts.manifest.files:
        relative = validate_relative_path(record.path)

        candidate_file = candidate_dir / relative

        if (
            candidate_file.is_symlink()
            or not candidate_file.is_file()
            or not candidate_file.resolve().is_relative_to(candidate_dir)
        ):
            raise ReleaseWorkspaceError(
                f"Missing or unsafe release file: {record.path}"
            )

        if candidate_file.stat().st_size > MAX_FILE_BYTES:
            raise ReleaseWorkspaceError(
                f"Release file exceeds size limit: {record.path}"
            )

        actual_hash = hashlib.sha256(
            candidate_file.read_bytes()
        ).hexdigest()

        if actual_hash != record.sha256:
            raise ReleaseWorkspaceError(
                f"Release file hash mismatch: {record.path}"
            )


def prepare_release_workspace(
    artifacts: ReleaseArtifacts,
    release_dir: Path,
    repository_path: str | None = None,
) -> Path:
    """Create a reviewable Release candidate.

    Args:
        artifacts:
            Previously validated Developer and QA artifacts.
        release_dir:
            Harness-controlled directory for the candidate.
        repository_path:
            Optional original target repository.

    Returns:
        Path to the candidate directory.

    Raises:
        ReleaseWorkspaceError:
            If preparation or verification fails.
    """

    candidate_dir = Path(release_dir).absolute()

    if candidate_dir.exists() or candidate_dir.is_symlink():
        raise ReleaseWorkspaceError(
            f"Release candidate already exists: {candidate_dir}"
        )

    try:
        prepared = prepare_qa_workspace(
            manifest=artifacts.manifest,
            staging_dir=artifacts.staging_dir,
            workspace_dir=candidate_dir,
            repository_path=repository_path,
        )

        verify_release_candidate(
            artifacts,
            prepared,
        )

        return prepared

    except (QAWorkspaceError, OSError, ValueError) as exc:
        raise ReleaseWorkspaceError(
            f"Could not prepare Release candidate: {exc}"
        ) from exc
