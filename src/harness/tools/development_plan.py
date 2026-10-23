
"""Trusted validation of Developer file plans."""

from harness.models import DevelopmentPlan, FileOperation
from harness.tools.filesystem import validate_relative_path
from harness.tools.repository import RepositoryContext


class DevelopmentPlanError(ValueError):
    """Raised when a Developer plan violates filesystem rules."""


def validate_development_plan(
    plan: DevelopmentPlan,
    repository: RepositoryContext | None = None,
) -> None:
    """Validate paths and CREATE/MODIFY semantics.

    All operations are interpreted relative to the original
    repository, never to earlier development attempts.
    """

    if plan.status == "blocked":
        return

    seen_paths: set[str] = set()

    repository_files = set()
    repository_excerpts = {}

    if repository is not None:
        repository_files = {
            validate_relative_path(path).as_posix()
            for path in repository.files
        }

        repository_excerpts = {
            validate_relative_path(path).as_posix(): content
            for path, content in repository.excerpts.items()
        }

    for planned_file in plan.files:
        relative = validate_relative_path(planned_file.path)
        path = relative.as_posix()

        # Case-insensitive duplicate detection.
        key = path.casefold()

        if key in seen_paths:
            raise DevelopmentPlanError(
                f"Duplicate planned file: {path}"
            )

        seen_paths.add(key)

        exists_in_repository = path in repository_files

        if planned_file.operation == FileOperation.CREATE:
            if exists_in_repository:
                raise DevelopmentPlanError(
                    f"Cannot CREATE existing repository file: {path}"
                )

        elif planned_file.operation == FileOperation.MODIFY:
            if not exists_in_repository:
                raise DevelopmentPlanError(
                    f"Cannot MODIFY file absent from original "
                    f"repository: {path}"
                )

            if path not in repository_excerpts:
                raise DevelopmentPlanError(
                    f"Cannot MODIFY file without original contents: "
                    f"{path}"
                )

            # Conservative policy: a truncated repository context
            # cannot establish that the complete file was supplied.
            if repository.truncated:
                raise DevelopmentPlanError(
                    f"Cannot MODIFY file with truncated repository "
                    f"context: {path}"
                )

        else:
            raise DevelopmentPlanError(
                f"Unsupported file operation for {path}: "
                f"{planned_file.operation}"
            )
