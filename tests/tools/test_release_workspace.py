
"""Tests for preparing isolated Release candidates."""

import hashlib

import pytest

from harness.models import (
    GeneratedFile,
    ImplementationManifest,
)
from harness.tools.filesystem import stage_generated_files
from harness.tools.release_artifacts import ReleaseArtifacts
from harness.tools.release_workspace import (
    ReleaseWorkspaceError,
    prepare_release_workspace,
)


def make_artifacts(tmp_path):
    staging = tmp_path / "development" / "staging"

    records = stage_generated_files(
        files=[
            GeneratedFile(
                path="src/add.py",
                operation="create",
                content="def add(a, b): return a + b\n",
            )
        ],
        staging_root=staging,
    )

    manifest = ImplementationManifest(
        summary="Addition module",
        files=records,
    )

    # The workspace-preparation function uses manifest and staging.
    # Other fields are metadata and aren't accessed here.
    return ReleaseArtifacts(
        manifest=manifest,
        staging_dir=staging,
        qa_report=None,
        qa_workspace=tmp_path / "qa",
        manifest_path=tmp_path / "manifest.json",
        qa_report_path=tmp_path / "test_report.json",
    )


def test_standalone_release_candidate(tmp_path):
    artifacts = make_artifacts(tmp_path)

    candidate = prepare_release_workspace(
        artifacts,
        tmp_path / "release" / "candidate",
    )

    assert candidate.exists()
    assert (candidate / "src/add.py").read_text() == (
        "def add(a, b): return a + b\n"
    )


def test_repository_release_preserves_original(tmp_path):
    artifacts = make_artifacts(tmp_path)

    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "existing.py").write_text("VALUE = 10\n")

    candidate = prepare_release_workspace(
        artifacts,
        tmp_path / "release" / "candidate",
        repository_path=str(repository),
    )

    assert (candidate / "existing.py").exists()
    assert (candidate / "src/add.py").exists()

    # Original target repository remains unchanged.
    assert not (repository / "src/add.py").exists()
    assert (repository / "existing.py").read_text() == "VALUE = 10\n"


def test_existing_candidate_is_rejected(tmp_path):
    artifacts = make_artifacts(tmp_path)

    candidate = tmp_path / "release" / "candidate"
    candidate.mkdir(parents=True)

    with pytest.raises(
        ReleaseWorkspaceError,
        match="already exists",
    ):
        prepare_release_workspace(
            artifacts,
            candidate,
        )
