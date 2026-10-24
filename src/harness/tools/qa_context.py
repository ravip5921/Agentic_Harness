
"""Load complete, verified implementation context for QA review."""

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from harness.models import FileOperation, ImplementationManifest
from harness.state import HarnessState
from harness.tools.artifacts import load_development_artifacts
from harness.tools.filesystem import validate_relative_path, validate_repository_file
from harness.tools.repository import inspect_repository


MAX_QA_CONTEXT_BYTES = 100_000


class QAContextError(Exception):
    """Implementation context cannot be safely or fully loaded."""


class VerifiedSourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    operation: FileOperation
    content: str


class QAImplementationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[VerifiedSourceFile] = Field(min_length=1)
    repository_files: dict[str, str] = Field(default_factory=dict)
    repository_context_truncated: bool = False


def _read_verified_file(
    path: Path,
    expected_sha256: str,
    max_bytes: int,
) -> str:
    """Read a bounded file and verify its contents again."""

    if path.is_symlink() or not path.is_file():
        raise QAContextError(f"Invalid staged file: {path}")

    # All parent paths are already checked by the artifact loader.
    # Open the final component without following a symlink.
    import os

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)

    try:
        fd = os.open(path, flags)
        try:
            data = os.read(fd, max_bytes + 1)
        finally:
            os.close(fd)
    except OSError as exc:
        raise QAContextError(f"Cannot read {path}: {exc}") from exc

    if len(data) > max_bytes:
        raise QAContextError(
            f"Generated file exceeds QA context limit: {path}"
        )

    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise QAContextError(f"Staged file changed: {path}")

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QAContextError(
            f"Generated file is not UTF-8: {path}"
        ) from exc


def load_qa_implementation_context(
    state: HarnessState,
    runs_dir: Path = Path("runs"),
    max_total_bytes: int = MAX_QA_CONTEXT_BYTES,
) -> QAImplementationContext:
    """Load generated files and bounded unchanged files from the tested snapshot.

    Rejects oversized or non-text implementations instead of silently
    producing incomplete review context.
    """

    if max_total_bytes < 1:
        raise ValueError("max_total_bytes must be positive.")

    manifest: ImplementationManifest
    manifest, staging_dir = load_development_artifacts(
        state, runs_dir=runs_dir
    )

    remaining = max_total_bytes
    result = []

    for record in manifest.files:
        relative = validate_relative_path(record.path)
        content = _read_verified_file(
            staging_dir / relative,
            record.sha256,
            remaining,
        )

        remaining -= len(content.encode("utf-8"))

        result.append(
            VerifiedSourceFile(
                path=record.path,
                operation=record.operation,
                content=content,
            )
        )

    repository_files = {}
    truncated = False
    task = state["task"]
    if task.repository_path is not None:
        # Use the snapshot actually tested, rather than rereading the live repo.
        relative_workspace = (
            Path(task.run_id) / "qa"
            / f"attempt-{state.get('revision_count', 0)}" / "workspace"
        )
        workspace = validate_repository_file(Path(runs_dir).resolve(), relative_workspace)
        repository = inspect_repository(str(workspace))
        generated_paths = {record.path for record in manifest.files}
        truncated = repository.truncated
        for path, content in repository.excerpts.items():
            if path in generated_paths:
                continue
            size = len(content.encode("utf-8"))
            if size > remaining:
                truncated = True
                continue
            repository_files[path] = content
            remaining -= size

    return QAImplementationContext(files=result, repository_files=repository_files,
                                   repository_context_truncated=truncated)
