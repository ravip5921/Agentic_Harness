
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
