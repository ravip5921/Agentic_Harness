
"""Unit tests for Developer filesystem staging tools."""

import hashlib

import pytest

from harness.models import FileOperation, GeneratedFile
from harness.tools.filesystem import (
    FilesystemError,
    stage_generated_files,
    validate_relative_path,
)


def make_file(
    path: str,
    content: str = "print('hello')\n",
    operation: FileOperation = FileOperation.CREATE,
) -> GeneratedFile:
    return GeneratedFile(
        path=path,
        operation=operation,
        content=content,
    )


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------
# Path validation
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    [
        "../outside.py",
        "src/../outside.py",
        "/tmp/outside.py",
        "./module.py",
        "src//module.py",
        "src/module.py/",
        "C:/temp/file.py",
        r"src\module.py",
        "",
        "src/\x00file.py",
        "src/file:name.py",
    ],
)
def test_rejects_unsafe_paths(path):
    with pytest.raises(FilesystemError):
        validate_relative_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "main.py",
        "src/package/module.py",
        "tests/test_module.py",
    ],
)
def test_accepts_safe_relative_paths(path):
    assert validate_relative_path(path).as_posix() == path


# ---------------------------------------------------------
# CREATE operations
# ---------------------------------------------------------

def test_stage_single_file(tmp_path):
    staging = tmp_path / "staging"
    content = "print('hello')\n"

    records = stage_generated_files(
        files=[make_file("main.py", content)],
        staging_root=staging,
    )

    assert (staging / "main.py").read_text() == content
    assert len(records) == 1

    record = records[0]

    assert record.path == "main.py"
    assert record.operation == FileOperation.CREATE
    assert record.sha256 == sha256(content.encode())
    assert record.original_sha256 is None


def test_stage_multiple_files(tmp_path):
    staging = tmp_path / "staging"

    files = [
        make_file("src/package/__init__.py", ""),
        make_file("src/package/main.py", "VALUE = 10\n"),
        make_file("tests/test_main.py", "def test_example(): pass\n"),
    ]

    records = stage_generated_files(
        files=files,
        staging_root=staging,
    )

    assert len(records) == 3

    for file in files:
        assert (staging / file.path).read_text() == file.content


def test_create_rejects_existing_repository_file(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    (repository / "existing.py").write_text("OLD = True\n")

    staging = tmp_path / "staging"

    with pytest.raises(FilesystemError, match="already exists"):
        stage_generated_files(
            files=[make_file("existing.py")],
            staging_root=staging,
            repository_path=str(repository),
        )

    assert not staging.exists()
    assert (repository / "existing.py").read_text() == "OLD = True\n"


# ---------------------------------------------------------
# MODIFY operations
# ---------------------------------------------------------

def test_modify_existing_file(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    original_content = "VALUE = 1\n"
    updated_content = "VALUE = 2\n"

    (repository / "config.py").write_text(original_content)

    staging = tmp_path / "staging"

    records = stage_generated_files(
        files=[
            make_file(
                "config.py",
                updated_content,
                FileOperation.MODIFY,
            )
        ],
        staging_root=staging,
        repository_path=str(repository),
    )

    assert (staging / "config.py").read_text() == updated_content

    # Original repository must remain unchanged.
    assert (repository / "config.py").read_text() == original_content

    assert records[0].operation == FileOperation.MODIFY
    assert records[0].sha256 == sha256(updated_content.encode())
    assert records[0].original_sha256 == sha256(
        original_content.encode()
    )


def test_modify_without_repository_is_rejected(tmp_path):
    staging = tmp_path / "staging"

    with pytest.raises(FilesystemError, match="requires an existing repository"):
        stage_generated_files(
            files=[
                make_file(
                    "config.py",
                    "VALUE = 2\n",
                    FileOperation.MODIFY,
                )
            ],
            staging_root=staging,
        )

    assert not staging.exists()


def test_modify_nonexistent_file_is_rejected(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    staging = tmp_path / "staging"

    with pytest.raises(FilesystemError, match="not an existing file"):
        stage_generated_files(
            files=[
                make_file(
                    "missing.py",
                    operation=FileOperation.MODIFY,
                )
            ],
            staging_root=staging,
            repository_path=str(repository),
        )

    assert not staging.exists()


# ---------------------------------------------------------
# Batch validation
# ---------------------------------------------------------

def test_duplicate_paths_are_rejected(tmp_path):
    staging = tmp_path / "staging"

    with pytest.raises(FilesystemError, match="Duplicate"):
        stage_generated_files(
            files=[
                make_file("src/main.py"),
                make_file("src/main.py"),
            ],
            staging_root=staging,
        )

    assert not staging.exists()


def test_case_insensitive_duplicates_are_rejected(tmp_path):
    staging = tmp_path / "staging"

    with pytest.raises(FilesystemError, match="Duplicate"):
        stage_generated_files(
            files=[
                make_file("src/Main.py"),
                make_file("src/main.py"),
            ],
            staging_root=staging,
        )

    assert not staging.exists()


def test_file_directory_collision_is_rejected(tmp_path):
    staging = tmp_path / "staging"

    with pytest.raises(FilesystemError, match="collision"):
        stage_generated_files(
            files=[
                make_file("src/module.py"),
                make_file("src/module.py/child.py"),
            ],
            staging_root=staging,
        )

    assert not staging.exists()


def test_batch_validation_prevents_partial_staging(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    staging = tmp_path / "staging"

    files = [
        make_file("new.py"),
        make_file(
            "missing.py",
            operation=FileOperation.MODIFY,
        ),
    ]

    with pytest.raises(FilesystemError):
        stage_generated_files(
            files=files,
            staging_root=staging,
            repository_path=str(repository),
        )

    assert not staging.exists()


def test_empty_batch_is_rejected(tmp_path):
    with pytest.raises(FilesystemError, match="No generated files"):
        stage_generated_files(
            files=[],
            staging_root=tmp_path / "staging",
        )


# ---------------------------------------------------------
# Resource limits
# ---------------------------------------------------------

def test_rejects_too_many_files(tmp_path):
    files = [
        make_file(f"file_{i}.py")
        for i in range(31)
    ]

    with pytest.raises(FilesystemError, match="Too many"):
        stage_generated_files(
            files=files,
            staging_root=tmp_path / "staging",
        )


def test_rejects_oversized_file(tmp_path):
    large_content = "x" * 1_000_001

    with pytest.raises(FilesystemError, match="size limit"):
        stage_generated_files(
            files=[make_file("large.py", large_content)],
            staging_root=tmp_path / "staging",
        )


# ---------------------------------------------------------
# Existing staging directory
# ---------------------------------------------------------

def test_existing_staging_directory_is_not_overwritten(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    existing = staging / "important.txt"
    existing.write_text("preserve me")

    with pytest.raises(FilesystemError, match="already exists"):
        stage_generated_files(
            files=[make_file("main.py")],
            staging_root=staging,
        )

    assert existing.read_text() == "preserve me"
    assert not (staging / "main.py").exists()


# ---------------------------------------------------------
# Symlink protection
# ---------------------------------------------------------

def test_rejects_symlinked_repository_file(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = True\n")

    (repository / "linked.py").symlink_to(outside)

    staging = tmp_path / "staging"

    with pytest.raises(FilesystemError, match="Symlinked"):
        stage_generated_files(
            files=[
                make_file(
                    "linked.py",
                    operation=FileOperation.MODIFY,
                )
            ],
            staging_root=staging,
            repository_path=str(repository),
        )

    assert not staging.exists()


def test_rejects_symlinked_staging_parent(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()

    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(FilesystemError):
        stage_generated_files(
            files=[make_file("main.py")],
            staging_root=linked / "staging",
        )

    assert not (actual / "staging").exists()
