
"""Tests for engineering specification loading."""

import pytest

from harness.models import (
    EngineeringSpec,
    ResearchResult,
    StageOutcome,
    TaskRequest,
)

from harness.tools.specifications import (
    SpecificationError,
    load_engineering_spec,
)


def make_spec():
    return EngineeringSpec(
        title="Example implementation",
        objective="Build a small reusable Python module.",
        requirements=[
            "Implement the requested functionality."
        ],
        technical_approach="Use modular Python components.",
        implementation_steps=[
            "Implement the module.",
            "Add tests.",
        ],
        acceptance_criteria=[
            "All required unit tests pass."
        ],
    )


def make_state(tmp_path, outcome=StageOutcome.COMPLETED):
    task = TaskRequest(task="Build an example module")

    research_dir = tmp_path / task.run_id / "research"
    research_dir.mkdir(parents=True)

    spec_path = research_dir / "engineering_spec.json"

    spec_path.write_text(
        make_spec().model_dump_json(indent=2),
        encoding="utf-8",
    )

    state = {
        "task": task,
        "research_result": ResearchResult(
            outcome=outcome,
            artifacts=[str(spec_path)],
        ),
    }

    return state, spec_path


def test_load_valid_specification(tmp_path):
    state, _ = make_state(tmp_path)

    spec = load_engineering_spec(
        state,
        runs_dir=tmp_path,
    )

    assert isinstance(spec, EngineeringSpec)
    assert spec.title == "Example implementation"
    assert spec.requirements


def test_research_must_be_completed(tmp_path):
    state, _ = make_state(
        tmp_path,
        outcome=StageOutcome.BLOCKED,
    )

    with pytest.raises(SpecificationError):
        load_engineering_spec(state, runs_dir=tmp_path)


def test_specification_must_exist(tmp_path):
    state, spec_path = make_state(tmp_path)

    spec_path.unlink()

    with pytest.raises(
        SpecificationError,
        match="missing",
    ):
        load_engineering_spec(state, runs_dir=tmp_path)


def test_invalid_specification_is_rejected(tmp_path):
    state, spec_path = make_state(tmp_path)

    spec_path.write_text(
        '{"title": "Invalid"}',
        encoding="utf-8",
    )

    with pytest.raises(
        SpecificationError,
        match="Invalid engineering specification",
    ):
        load_engineering_spec(state, runs_dir=tmp_path)


def test_unregistered_specification_is_rejected(tmp_path):
    state, _ = make_state(tmp_path)

    state["research_result"].artifacts = []

    with pytest.raises(
        SpecificationError,
        match="not registered",
    ):
        load_engineering_spec(state, runs_dir=tmp_path)


def test_specification_size_limit(tmp_path):
    state, spec_path = make_state(tmp_path)

    spec_path.write_text(
        "x" * 1_000_001,
        encoding="utf-8",
    )

    with pytest.raises(
        SpecificationError,
        match="size limit",
    ):
        load_engineering_spec(state, runs_dir=tmp_path)
