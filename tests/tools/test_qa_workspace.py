
"""Unit tests for disposable QA workspace preparation.

Tests are filesystem-only:
- No LangGraph execution
- No Groq API calls
- No generated code execution
"""

import hashlib

import pytest

from harness.models import (
    FileOperation,
    ImplementationManifest,
    StagedFile,
)
from harness.tools.qa_workspace import (
    QAWorkspaceError,
    prepare_qa_workspace,
)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def make_manifest(
    files: list[StagedFile],
) -> ImplementationManifest:
    return ImplementationManifest(
        summary="Test implementation",
        files=files,
    )


def make_staged_file(
    staging_dir,
    path: str,
    content: str,
    operation: FileOperation = FileOperation.CREATE,
    original_content: str | None = None,
) -> StagedFile:
    """Write one fixture file and create its manifest record."""

    destination = staging_dir / path
    destination.parent.mkdir(parents=True, exist_ok=True)

    data = content.encode("utf-8")
    destination.write_bytes(data)

    return StagedFile(
        path=path,
        operation=operation,
        sha256=sha256(data),
        original_sha256=(
            sha256(original_content.encode("utf-8"))
            if original_content is not None
            else None
        ),
    )


# ---------------------------------------------------------
# Standalone projects
# ---------------------------------------------------------

def test_prepare_standalone_workspace(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "src/example/main.py",
        "VALUE = 42\n",
    )

    manifest = make_manifest([record])
    workspace = tmp_path / "qa" / "workspace"

    result = prepare_qa_workspace(
        manifest=manifest,
        staging_dir=staging,
        workspace_dir=workspace,
    )

    assert result == workspace
    assert (workspace / "src/example/main.py").read_text() == (
        "VALUE = 42\n"
    )

    # Original staging directory is preserved.
    assert (staging / "src/example/main.py").exists()


def test_prepare_multiple_generated_files(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    records = [
        make_staged_file(
            staging,
            "src/package/__init__.py",
            "",
        ),
        make_staged_file(
            staging,
            "src/package/main.py",
            "def main(): return 10\n",
        ),
        make_staged_file(
            staging,
            "tests/test_main.py",
            "def test_main(): assert True\n",
        ),
    ]

    workspace = tmp_path / "workspace"

    prepare_qa_workspace(
        make_manifest(records),
        staging,
        workspace,
    )

    assert (workspace / "src/package/__init__.py").exists()
    assert (workspace / "src/package/main.py").exists()
    assert (workspace / "tests/test_main.py").exists()


def test_modify_without_repository_is_rejected(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "main.py",
        "VALUE = 2\n",
        operation=FileOperation.MODIFY,
        original_content="VALUE = 1\n",
    )

    workspace = tmp_path / "workspace"

    with pytest.raises(QAWorkspaceError, match="MODIFY target missing"):
        prepare_qa_workspace(
            make_manifest([record]),
            staging,
            workspace,
        )

    assert not workspace.exists()


# ---------------------------------------------------------
# Existing repositories
# ---------------------------------------------------------

def test_repository_create_preserves_original(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    (repository / "existing.py").write_text("ORIGINAL = True\n")

    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "new.py",
        "NEW = True\n",
    )

    workspace = tmp_path / "workspace"

    prepare_qa_workspace(
        manifest=make_manifest([record]),
        staging_dir=staging,
        workspace_dir=workspace,
        repository_path=str(repository),
    )

    assert (workspace / "existing.py").read_text() == "ORIGINAL = True\n"
    assert (workspace / "new.py").read_text() == "NEW = True\n"

    assert (repository / "existing.py").read_text() == "ORIGINAL = True\n"
    assert not (repository / "new.py").exists()


def test_repository_modify_preserves_original(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    original = "VALUE = 1\n"
    updated = "VALUE = 2\n"

    (repository / "config.py").write_text(original)

    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "config.py",
        updated,
        operation=FileOperation.MODIFY,
        original_content=original,
    )

    workspace = tmp_path / "workspace"

    prepare_qa_workspace(
        make_manifest([record]),
        staging,
        workspace,
        repository_path=str(repository),
    )

    assert (workspace / "config.py").read_text() == updated
    assert (repository / "config.py").read_text() == original


def test_repository_create_existing_file_rejected(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    (repository / "main.py").write_text("OLD = True\n")

    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "main.py",
        "NEW = True\n",
    )

    workspace = tmp_path / "workspace"

    with pytest.raises(QAWorkspaceError, match="CREATE target already exists"):
        prepare_qa_workspace(
            make_manifest([record]),
            staging,
            workspace,
            repository_path=str(repository),
        )

    assert not workspace.exists()
    assert (repository / "main.py").read_text() == "OLD = True\n"


def test_repository_original_hash_mismatch(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    (repository / "main.py").write_text("VALUE = 99\n")

    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "main.py",
        "VALUE = 2\n",
        operation=FileOperation.MODIFY,
        original_content="VALUE = 1\n",
    )

    workspace = tmp_path / "workspace"

    with pytest.raises(QAWorkspaceError, match="Original file hash mismatch"):
        prepare_qa_workspace(
            make_manifest([record]),
            staging,
            workspace,
            repository_path=str(repository),
        )

    assert not workspace.exists()


# ---------------------------------------------------------
# Artifact integrity
# ---------------------------------------------------------

def test_rejects_staged_hash_mismatch(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "main.py",
        "VALUE = 1\n",
    )

    # Modify file after generating its manifest hash.
    (staging / "main.py").write_text("VALUE = 999\n")

    workspace = tmp_path / "workspace"

    with pytest.raises(QAWorkspaceError, match="hash mismatch"):
        prepare_qa_workspace(
            make_manifest([record]),
            staging,
            workspace,
        )

    assert not workspace.exists()


def test_rejects_missing_staged_file(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "missing.py",
        "VALUE = 1\n",
    )

    (staging / "missing.py").unlink()

    workspace = tmp_path / "workspace"

    with pytest.raises(QAWorkspaceError, match="Invalid staged file"):
        prepare_qa_workspace(
            make_manifest([record]),
            staging,
            workspace,
        )

    assert not workspace.exists()


def test_rejects_duplicate_manifest_paths(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "main.py",
        "VALUE = 1\n",
    )

    workspace = tmp_path / "workspace"

    with pytest.raises(QAWorkspaceError, match="Duplicate manifest path"):
        prepare_qa_workspace(
            make_manifest([record, record]),
            staging,
            workspace,
        )

    assert not workspace.exists()


# ---------------------------------------------------------
# Workspace path safety
# ---------------------------------------------------------

def test_rejects_existing_workspace(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "main.py",
        "VALUE = 1\n",
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    important = workspace / "important.txt"
    important.write_text("Do not overwrite")

    with pytest.raises(QAWorkspaceError, match="already exists"):
        prepare_qa_workspace(
            make_manifest([record]),
            staging,
            workspace,
        )

    assert important.read_text() == "Do not overwrite"


def test_rejects_workspace_inside_staging(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "main.py",
        "VALUE = 1\n",
    )

    workspace = staging / "qa_workspace"

    with pytest.raises(QAWorkspaceError, match="overlap"):
        prepare_qa_workspace(
            make_manifest([record]),
            staging,
            workspace,
        )


def test_rejects_workspace_inside_repository(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "new.py",
        "VALUE = 1\n",
    )

    workspace = repository / "qa_workspace"

    with pytest.raises(QAWorkspaceError, match="overlap"):
        prepare_qa_workspace(
            make_manifest([record]),
            staging,
            workspace,
            repository_path=str(repository),
        )


# ---------------------------------------------------------
# Symlink protection
# ---------------------------------------------------------

def test_rejects_staged_symlink(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = True\n")

    symlink = staging / "linked.py"
    symlink.symlink_to(outside)

    record = StagedFile(
        path="linked.py",
        operation=FileOperation.CREATE,
        sha256=sha256(outside.read_bytes()),
    )

    workspace = tmp_path / "workspace"

    with pytest.raises(QAWorkspaceError, match="Invalid staged file"):
        prepare_qa_workspace(
            make_manifest([record]),
            staging,
            workspace,
        )

    assert not workspace.exists()


def test_rejects_repository_symlink(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = True\n")

    (repository / "linked.py").symlink_to(outside)

    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "new.py",
        "VALUE = 1\n",
    )

    workspace = tmp_path / "workspace"

    with pytest.raises(QAWorkspaceError, match="symlink"):
        prepare_qa_workspace(
            make_manifest([record]),
            staging,
            workspace,
            repository_path=str(repository),
        )

    assert not workspace.exists()


# ---------------------------------------------------------
# Snapshot resource limits
# ---------------------------------------------------------

def test_rejects_oversized_repository_file(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    # Exceeds MAX_SINGLE_FILE_BYTES (5 MB).
    (repository / "large.bin").write_bytes(
        b"x" * 5_000_001
    )

    staging = tmp_path / "staging"
    staging.mkdir()

    record = make_staged_file(
        staging,
        "new.py",
        "VALUE = 1\n",
    )

    workspace = tmp_path / "workspace"

    with pytest.raises(QAWorkspaceError, match="snapshot limits"):
        prepare_qa_workspace(
            make_manifest([record]),
            staging,
            workspace,
            repository_path=str(repository),
        )

    assert not workspace.exists()