
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
import logging

logger = logging.getLogger(__name__)
from collections.abc import Callable
from pathlib import Path

from harness.models import (
    DevelopmentAssessment,
    DevelopmentResult,
    FileOperation,
    ImplementationManifest,
    StageOutcome,
    PreviousFile,
    PreviousImplementation,
    QAFeedback,
    QAVerdict,
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
from harness.tools.artifacts import (
    ArtifactError,
    load_development_artifacts,
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


MAX_PREVIOUS_CONTEXT_BYTES = 100_000


def load_previous_implementation(
    state: HarnessState,
    runs_dir: Path = Path("runs"),
) -> PreviousImplementation:
    """Load verified files from the previous Development attempt.

    The previous attempt's manifest and staged files are validated
    before their contents are included in the Developer Agent context.
    """

    revision_count = state.get("revision_count", 0)

    if revision_count < 1:
        raise DevelopmentValidationError(
            "Previous implementation is only available on retries."
        )

    previous_attempt = revision_count - 1

    # Reuse the artifact loader, which identifies the attempt
    # using revision_count.
    previous_state = {
        **state,
        "revision_count": previous_attempt,
    }

    manifest, staging_dir = load_development_artifacts(
        previous_state,
        runs_dir=runs_dir,
    )

    files = []
    total_bytes = 0

    for record in manifest.files:
        relative_path = validate_relative_path(record.path)
        file_path = staging_dir / relative_path

        content_bytes = file_path.read_bytes()

        total_bytes += len(content_bytes)

        if total_bytes > MAX_PREVIOUS_CONTEXT_BYTES:
            raise DevelopmentValidationError(
                "Previous implementation exceeds the "
                "Developer Agent context size limit."
            )

        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DevelopmentValidationError(
                f"Previous generated file is not UTF-8: {record.path}"
            ) from exc

        files.append(
            PreviousFile(
                path=record.path,
                operation=record.operation,
                content=content,
            )
        )

    return PreviousImplementation(
        development_attempt=previous_attempt,
        files=files,
    )
def make_developer_stage(
    generator: Callable,
    specification_loader: Callable = load_engineering_spec,
    repository_reader: Callable = inspect_repository,
    file_stager: Callable = stage_generated_files,
    previous_implementation_loader: Callable = load_previous_implementation,
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
            qa_feedback = None
            previous_implementation = None

            if revision_count > 0:
                qa_result = state.get("qa_result")

                if (
                    qa_result is None
                    or qa_result.verdict != QAVerdict.FAIL
                    or qa_result.feedback is None
                ):
                    raise DevelopmentValidationError(
                        "Developer retry requires failed QA "
                        "with structured feedback."
                    )

                qa_feedback = qa_result.feedback

                if not isinstance(qa_feedback, QAFeedback):
                    raise DevelopmentValidationError(
                        "QA feedback must be a QAFeedback object."
                    )

                if qa_feedback.development_attempt != revision_count - 1:
                    raise DevelopmentValidationError(
                        "QA feedback does not match the previous "
                        "development attempt."
                    )

                previous_implementation = previous_implementation_loader(
                    state,
                    runs_dir=runs_root,
                )

            # 4. Generate an implementation proposal.
            assessment = generator(
                specification,
                repository,
                qa_feedback,
                previous_implementation,
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
            ArtifactError,
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
            logger.exception(
                "Development failed for run %s, attempt %d",
                task.run_id,
                revision_count,
            )
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
