
import pytest

from harness.models import (
    DevelopmentResult,
    GeneratedFile,
    StageOutcome,
    TaskRequest,
)
from harness.tools.filesystem import stage_generated_files
from harness.tools.qa_context import (
    load_qa_implementation_context,
    QAContextError,
)
from harness.models import ImplementationManifest


def make_state(tmp_path):
    task = TaskRequest(task="Build a calculator.")

    attempt = (
        tmp_path / task.run_id / "development" / "attempt-0"
    )

    staging = attempt / "staging"

    files = [
        GeneratedFile(
            path="calculator.py",
            operation="create",
            content="def add(a, b): return a + b\n",
        ),
        GeneratedFile(
            path="tests/test_calculator.py",
            operation="create",
            content="def test_add(): assert 2 + 3 == 5\n",
        ),
    ]

    records = stage_generated_files(
        files,
        staging_root=staging,
    )

    manifest = ImplementationManifest(
        summary="Calculator implementation",
        files=records,
    )

    manifest_path = attempt / "manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    state = {
        "task": task,
        "revision_count": 0,
        "development_result": DevelopmentResult(
            outcome=StageOutcome.COMPLETED,
            artifacts=[str(manifest_path), str(staging)],
        ),
    }

    return state


def test_loads_generated_source_and_tests(tmp_path):
    state = make_state(tmp_path)

    context = load_qa_implementation_context(
        state,
        runs_dir=tmp_path,
    )

    assert len(context.files) == 2
    assert {item.path for item in context.files} == {
        "calculator.py",
        "tests/test_calculator.py",
    }

    assert "def add" in context.files[0].content


def test_rejects_incomplete_context(tmp_path):
    state = make_state(tmp_path)

    with pytest.raises(QAContextError, match="context limit"):
        load_qa_implementation_context(
            state,
            runs_dir=tmp_path,
            max_total_bytes=5,
        )


def prepare_repository_snapshot(tmp_path):
    state = make_state(tmp_path)
    repository = tmp_path / "repository"
    repository.mkdir()
    state["task"].repository_path = str(repository)
    workspace = tmp_path / state["task"].run_id / "qa/attempt-0/workspace"
    (workspace / "tests").mkdir(parents=True)
    (workspace / "tests/test_base.py").write_text("def test_base(): assert True\n")
    (workspace / "calculator.py").write_text("This file is already supplied from staging")
    return state, repository, workspace


def test_unchanged_context_comes_from_tested_snapshot(tmp_path):
    state, repository, workspace = prepare_repository_snapshot(tmp_path)
    (repository / "tests").mkdir()
    (repository / "tests/test_base.py").write_text("different live file\n")
    context = load_qa_implementation_context(state, runs_dir=tmp_path)
    assert context.repository_files == {"tests/test_base.py": "def test_base(): assert True\n"}
    assert "calculator.py" not in context.repository_files


def test_unchanged_context_respects_remaining_budget(tmp_path):
    state, _, _ = prepare_repository_snapshot(tmp_path)
    generated_size = sum(len(file.content.encode()) for file in
        load_qa_implementation_context(state, runs_dir=tmp_path).files)
    context = load_qa_implementation_context(state, runs_dir=tmp_path,
                                             max_total_bytes=generated_size)
    assert context.repository_files == {}
    assert context.repository_context_truncated


def test_unchanged_context_excludes_symlinks(tmp_path):
    state, _, workspace = prepare_repository_snapshot(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n")
    (workspace / "linked.py").symlink_to(outside)
    context = load_qa_implementation_context(state, runs_dir=tmp_path)
    assert "linked.py" not in context.repository_files
