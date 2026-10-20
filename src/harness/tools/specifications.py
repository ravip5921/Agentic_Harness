
"""Load and validate engineering specifications produced by Research."""

import re
from pathlib import Path

from pydantic import ValidationError

from harness.models import EngineeringSpec, StageOutcome
from harness.state import HarnessState


MAX_SPEC_BYTES = 1_000_000


class SpecificationError(Exception):
    """Raised when the Research specification cannot be consumed."""


def load_engineering_spec(
    state: HarnessState,
    runs_dir: Path = Path("runs"),
) -> EngineeringSpec:
    """Load the validated engineering specification from Research.

    Args:
        state: Shared LangGraph state containing the task and
            Research result.
        runs_dir: Root directory containing per-run artifacts.

    Returns:
        A validated EngineeringSpec instance.

    Raises:
        SpecificationError: If Research has not completed or
            the specification is missing, unsafe, or invalid.
    """

    task = state.get("task")
    research_result = state.get("research_result")

    if task is None:
        raise SpecificationError("Task is missing from workflow state.")

    if research_result is None:
        raise SpecificationError("Research result is missing.")

    if research_result.outcome != StageOutcome.COMPLETED:
        raise SpecificationError(
            "Cannot develop without completed Research."
        )

    # Match the format used by the Research artifact writer.
    if not re.fullmatch(r"[a-f0-9]{32}", task.run_id):
        raise SpecificationError("Invalid run ID.")

    runs_root = runs_dir.resolve()

    expected_path = (
        runs_root
        / task.run_id
        / "research"
        / "engineering_spec.json"
    )

    # Only accept the specification registered by Research.
    # Paths are resolved before comparison to prevent traversal.
    registered_paths = {
        Path(path).resolve()
        for path in research_result.artifacts
    }

    if expected_path.resolve() not in registered_paths:
        raise SpecificationError(
            "Engineering specification is not registered "
            "in Research artifacts."
        )

    # Do not accept symlinked artifact directories or files.
    relative_path = expected_path.relative_to(runs_root)
    current = runs_root

    for part in relative_path.parts:
        current = current / part

        if current.is_symlink():
            raise SpecificationError(
                "Symlinks are not allowed in specification paths."
            )

    try:
        if not expected_path.is_file():
            raise SpecificationError(
                "Engineering specification file is missing."
            )

        if expected_path.stat().st_size > MAX_SPEC_BYTES:
            raise SpecificationError(
                "Engineering specification exceeds size limit."
            )

        content = expected_path.read_text(encoding="utf-8")

        return EngineeringSpec.model_validate_json(content)

    except (OSError, UnicodeError) as exc:
        raise SpecificationError(
            f"Could not read engineering specification: {exc}"
        ) from exc

    except ValidationError as exc:
        raise SpecificationError(
            f"Invalid engineering specification: {exc}"
        ) from exc
