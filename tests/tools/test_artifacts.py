
"""Tests for loading and validating Developer artifacts."""

import hashlib

import pytest

from harness.models import (
    DevelopmentResult,
    FileOperation,
    ImplementationManifest,
    StagedFile,
    StageOutcome,
    TaskRequest,
)
from harness.tools.artifacts import (
    ArtifactError,
    load_development_artifacts,
)


def make_artifacts(tmp_path):
    task = TaskRequest(task="Example task")

    attempt_dir = (
        tmp_path
        / task.run_id
        / "development"
        / "attempt-0"
    )

    staging = attempt_dir / "staging"
    staging.mkdir(parents=True)

    content = b"VALUE = 42\n"
    (staging / "main.py").write_bytes(content)

    manifest = ImplementationManifest(
        summary="Generated example module.",
        files=[
            StagedFile(
                path="main.py",
                operation=FileOperation.CREATE,
                sha256=hashlib.sha256(content).hexdigest(),
            )
        ],
    )

    manifest_path = attempt_dir / "manifest.json"

    manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    state = {
        "task": task,
        "revision_count": 0,
        "development_result": DevelopmentResult(
            outcome=StageOutcome.COMPLETED,
            artifacts=[
                str(manifest_path),
                str(staging),
            ],
        ),
    }

    return state, manifest_path, staging


def test_load_valid_artifacts(tmp_path):
    state, _, staging = make_artifacts(tmp_path)

    manifest, loaded_staging = load_development_artifacts(
        state,
        runs_dir=tmp_path,
    )

    assert manifest.summary == "Generated example module."
    assert len(manifest.files) == 1
    assert loaded_staging == staging


def test_rejects_modified_staged_file(tmp_path):
    state, _, staging = make_artifacts(tmp_path)

    (staging / "main.py").write_text("VALUE = 999\n")

    with pytest.raises(ArtifactError, match="hash mismatch"):
        load_development_artifacts(state, runs_dir=tmp_path)


def test_rejects_missing_file(tmp_path):
    state, _, staging = make_artifacts(tmp_path)

    (staging / "main.py").unlink()

    with pytest.raises(ArtifactError):
        load_development_artifacts(state, runs_dir=tmp_path)


def test_rejects_unexpected_file(tmp_path):
    state, _, staging = make_artifacts(tmp_path)

    (staging / "unexpected.py").write_text("print('extra')")

    with pytest.raises(ArtifactError, match="do not match"):
        load_development_artifacts(state, runs_dir=tmp_path)


def test_rejects_unregistered_artifacts(tmp_path):
    state, _, _ = make_artifacts(tmp_path)

    state["development_result"].artifacts = []

    with pytest.raises(ArtifactError, match="not registered"):
        load_development_artifacts(state, runs_dir=tmp_path)


def test_rejects_incomplete_development(tmp_path):
    state, _, _ = make_artifacts(tmp_path)

    state["development_result"].outcome = StageOutcome.BLOCKED

    with pytest.raises(ArtifactError, match="completed Development"):
        load_development_artifacts(state, runs_dir=tmp_path)
