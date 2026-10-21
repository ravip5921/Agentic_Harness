
"""Developer stage: generate, validate, and stage implementation files.

Consumes the engineering specification produced by Research.

The Developer Agent proposes implementation files, while trusted tools
validate file operations and write artifacts into an isolated workspace.

This stage never modifies the target repository or executes generated code.
"""

import os
import re
import shutil
import tempfile

from collections.abc import Callable
from pathlib import Path

from harness.models import (
    DevelopmentAssessment,
    DevelopmentResult,
    FileOperation,
    ImplementationManifest,
    StageOutcome,
)

from harness.state import HarnessState

from harness.agents.developer_agent import (
    GroqDeveloperAgent,
    DevelopmentGenerationError,
)

from harness.tools.specifications import (
    load_engineering_spec,
    SpecificationError,
)

from harness.tools.repository import (
    inspect_repository,
    RepositoryError,
)

from harness.tools.filesystem import (
    stage_generated_files,
    validate_relative_path,
    validate_repository_file,
    FilesystemError,
)


class DevelopmentValidationError(Exception):
    """Raised when a proposed implementation is not safe to stage."""


def validate_modifications(
    assessment: DevelopmentAssessment,
    repository,
) -> None:
    """Ensure MODIFY targets were fully available to the Developer Agent.

    Repository inspection may omit or truncate files. We must not
    replace an existing file unless the agent received its complete
    original contents.
    """

    modifications = [
        file
        for file in assessment.files
        if file.operation == FileOperation.MODIFY
    ]

    if not modifications:
        return

    if repository is None:
        raise DevelopmentValidationError(
            "MODIFY requires an existing repository."
        )

    if repository.truncated:
        raise DevelopmentValidationError(
            "Repository inspection was truncated; "
            "MODIFY cannot be safely validated."
        )

    repository_root = Path(repository.path).resolve()

    for generated in modifications:
        relative_path = validate_relative_path(
            generated.path
        )

        relative_name = relative_path.as_posix()

        if relative_name not in repository.excerpts:
            raise DevelopmentValidationError(
                f"Original file was not available to the agent: "
                f"{relative_name}"
            )

        original_path = validate_repository_file(
            repository_root,
            relative_path,
        )

        if not original_path.is_file():
            raise DevelopmentValidationError(
                f"Original file no longer exists: {relative_name}"
            )

        # Compare the actual original bytes with the inspected excerpt.
        # This also rejects files that were decoded with replacement
        # characters, since their original bytes won't match.
        actual_content = original_path.read_bytes()

        try:
            actual_text = actual_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DevelopmentValidationError(
                f"Original file is not valid UTF-8: {relative_name}"
            ) from exc

        inspected_text = repository.excerpts[relative_name]

        # Normalize newline representations for comparison.
        def normalize_newlines(text: str) -> str:
            return text.replace("\r\n", "\n").replace("\r", "\n")

        if normalize_newlines(actual_text) != normalize_newlines(inspected_text):
            raise DevelopmentValidationError(
                f"Original file changed or was not fully inspected: "
                f"{relative_name}"
            )

def write_manifest(
    manifest: ImplementationManifest,
    manifest_path: Path,
) -> None:
    """Write a validated manifest using a temporary file."""

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=".manifest-",
            suffix=".json",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(manifest.model_dump_json(indent=2))

        os.replace(temporary_path, manifest_path)
        temporary_path = None

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def make_developer_stage(
    generator: Callable,
    specification_loader: Callable = load_engineering_spec,
    repository_reader: Callable = inspect_repository,
    file_stager: Callable = stage_generated_files,
    runs_dir: Path = Path("runs"),
):
    """Create a Developer stage with injectable dependencies.

    Args:
        generator:
            Callable accepting (specification, repository, qa_feedback)
            and returning DevelopmentAssessment.
        specification_loader:
            Loads the validated Research engineering specification.
        repository_reader:
            Collects optional repository context.
        file_stager:
            Validates and stages generated files.
        runs_dir:
            Root directory for run artifacts.

    Returns:
        A LangGraph-compatible Developer node.
    """

    def developer(state: HarnessState) -> dict:
        task = state.get("task")

        if task is None:
            return {
                "development_result": DevelopmentResult(
                    outcome=StageOutcome.BLOCKED,
                    summary="Development cannot start without a task.",
                    errors=["TaskRequest is missing from workflow state."],
                )
            }
        # revision_count represents QA-triggered retries.
        # The initial Development attempt is attempt-0.
        revision_count = state.get("revision_count", 0)

        if not isinstance(revision_count, int) or revision_count < 0:
            return {
                "development_result": DevelopmentResult(
                    outcome=StageOutcome.FAILED,
                    summary="Invalid development revision counter.",
                    errors=["revision_count must be non-negative."],
                )
            }

        if not re.fullmatch(r"[a-f0-9]{32}", task.run_id):
            return {
                "development_result": DevelopmentResult(
                    outcome=StageOutcome.FAILED,
                    summary="Invalid run identifier.",
                    errors=["Invalid run_id format."],
                )
            }

        runs_root = Path(runs_dir).resolve()

        attempt_dir = (
            runs_root
            / task.run_id
            / "development"
            / f"attempt-{revision_count}"
        )

        staging_dir = attempt_dir / "staging"
        manifest_path = attempt_dir / "manifest.json"

        # Prevent accidental reuse or replacement of previous artifacts.
        if attempt_dir.exists() or attempt_dir.is_symlink():
            return {
                "development_result": DevelopmentResult(
                    outcome=StageOutcome.BLOCKED,
                    summary="Development attempt already exists.",
                    errors=[
                        f"Existing attempt workspace: {attempt_dir}"
                    ],
                )
            }

        staged = False

        try:
            # 1. Load Research's validated specification.
            specification = specification_loader(
                state,
                runs_dir=runs_root,
            )

            # 2. Inspect optional repository context.
            repository = repository_reader(
                task.repository_path
            )

            if task.repository_path is not None and repository is None:
                raise DevelopmentValidationError(
                    "Repository context could not be collected."
                )

            # 3. Collect previous QA feedback, only on retry.
            qa_feedback = []

            if revision_count > 0:
                qa_result = state.get("qa_result")

                if qa_result is None or not qa_result.feedback:
                    raise DevelopmentValidationError(
                        "QA retry requires actionable feedback."
                    )

                qa_feedback = qa_result.feedback

            # 4. Generate an implementation proposal.
            assessment = generator(
                specification,
                repository,
                qa_feedback,
            )

            if not isinstance(assessment, DevelopmentAssessment):
                raise DevelopmentValidationError(
                    "Generator must return DevelopmentAssessment."
                )

            if assessment.status == "blocked":
                return {
                    "development_result": DevelopmentResult(
                        outcome=StageOutcome.BLOCKED,
                        summary=assessment.reason,
                        errors=[assessment.reason],
                    )
                }

            # 5. Validate MODIFY operations against the repository
            # context actually supplied to the agent.
            validate_modifications(
                assessment,
                repository,
            )

            # 6. Validate and stage generated files.
            records = file_stager(
                files=assessment.files,
                staging_root=staging_dir,
                repository_path=task.repository_path,
            )

            staged = True

            # 7. Build and save the implementation manifest.
            manifest = ImplementationManifest(
                summary=assessment.summary or (
                    "Generated implementation files."
                ),
                files=records,
                dependencies=assessment.dependencies,
                notes=assessment.notes,
            )

            write_manifest(
                manifest,
                manifest_path,
            )

            # 8. Return successful Development result.
            return {
                "development_result": DevelopmentResult(
                    outcome=StageOutcome.COMPLETED,
                    summary=manifest.summary,
                    artifacts=[
                        str(manifest_path),
                        str(staging_dir),
                    ],
                )
            }

        except (
            SpecificationError,
            RepositoryError,
            FileNotFoundError,
        ) as exc:
            result = DevelopmentResult(
                outcome=StageOutcome.BLOCKED,
                summary="Development prerequisites are unavailable.",
                errors=[str(exc)],
            )

        except DevelopmentValidationError as exc:
            result = DevelopmentResult(
                outcome=StageOutcome.BLOCKED,
                summary="Development validation blocked.",
                errors=[str(exc)],
            )

        except FilesystemError as exc:
            result = DevelopmentResult(
                outcome=StageOutcome.FAILED,
                summary="Generated files failed staging validation.",
                errors=[str(exc)],
            )

        except (DevelopmentGenerationError, OSError, ValueError) as exc:
            result = DevelopmentResult(
                outcome=StageOutcome.FAILED,
                summary="Development execution failed.",
                errors=[str(exc)],
            )

        # Remove staged files if a later step (such as manifest
        # creation) fails. Never remove a pre-existing attempt.
        if staged and attempt_dir.exists():
            shutil.rmtree(attempt_dir)

        return {"development_result": result}

    return developer


def developer(state: HarnessState) -> dict:
    """Default LangGraph entry point for Development."""

    agent = GroqDeveloperAgent()

    node = make_developer_stage(
        generator=agent.generate,
    )

    return node(state)
