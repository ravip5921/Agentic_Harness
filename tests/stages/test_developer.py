
"""Tests for Developer stage orchestration.

Uses real specification loading and filesystem staging,
but replaces the Groq Developer Agent with deterministic generators.
"""

import pytest

from harness.graph import build_graph
from harness.models import (
    DevelopmentAssessment,
    EngineeringSpec,
    FileOperation,
    GeneratedFile,
    ImplementationManifest,
    PreviousFile,
    PreviousImplementation,
    QAFeedback,
    QAResult,
    QAVerdict,
    ResearchResult,
    RunStatus,
    StageOutcome,
    TaskRequest,
)
from harness.stages.developer import make_developer_stage
from harness.tools.repository import RepositoryContext


def sample_spec():
    return EngineeringSpec(
        title="Example Python module",
        objective="Implement a small Python utility.",
        requirements=["Provide a function named greet."],
        technical_approach="Implement a standalone Python module.",
        implementation_steps=[
            "Create greet.py.",
            "Create a unit test.",
        ],
        acceptance_criteria=[
            "greet('Ada') returns 'Hello, Ada!'"
        ],
    )


def make_state(tmp_path, repository_path=None):
    """Create valid Research artifacts for Developer to consume."""

    task = TaskRequest(
        task="Implement a greeting utility",
        repository_path=(
            str(repository_path)
            if repository_path is not None
            else None
        ),
    )

    research_dir = tmp_path / task.run_id / "research"
    research_dir.mkdir(parents=True)

    spec_path = research_dir / "engineering_spec.json"
    spec_path.write_text(
        sample_spec().model_dump_json(indent=2),
        encoding="utf-8",
    )

    return {
        "task": task,
        "revision_count": 0,
        "research_result": ResearchResult(
            outcome=StageOutcome.COMPLETED,
            artifacts=[str(spec_path)],
        ),
    }


def successful_generator(
    specification, repository, qa_feedback, previous_implementation,
):
    """Simulate successful generation of multiple files."""

    return DevelopmentAssessment(
        status="ready",
        summary="Generated greeting utility and tests.",
        files=[
            GeneratedFile(
                path="greet.py",
                operation=FileOperation.CREATE,
                content=(
                    "def greet(name: str) -> str:\n"
                    "    return f'Hello, {name}!'\n"
                ),
            ),
            GeneratedFile(
                path="tests/test_greet.py",
                operation=FileOperation.CREATE,
                content=(
                    "from greet import greet\n\n"
                    "def test_greet():\n"
                    "    assert greet('Ada') == 'Hello, Ada!'\n"
                ),
            ),
        ],
    )


def create_stage(tmp_path, generator=successful_generator, **kwargs):
    return make_developer_stage(
        generator=generator,
        runs_dir=tmp_path,
        **kwargs,
    )


# ---------------------------------------------------------
# Successful generation
# ---------------------------------------------------------

def test_successful_standalone_development(tmp_path):
    """Test the full Developer flow without Groq."""

    state = make_state(tmp_path)
    developer = create_stage(tmp_path)

    result = developer(state)["development_result"]

    assert result.outcome == StageOutcome.COMPLETED
    assert len(result.artifacts) == 2

    attempt_dir = (
        tmp_path
        / state["task"].run_id
        / "development"
        / "attempt-0"
    )

    assert (attempt_dir / "manifest.json").is_file()
    assert (attempt_dir / "staging/greet.py").is_file()
    assert (attempt_dir / "staging/tests/test_greet.py").is_file()

    manifest = ImplementationManifest.model_validate_json(
        (attempt_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert len(manifest.files) == 2

    assert all(
        file.operation == FileOperation.CREATE
        for file in manifest.files
    )

    assert all(
        file.original_sha256 is None
        for file in manifest.files
    )


# ---------------------------------------------------------
# Existing repository modifications
# ---------------------------------------------------------

def test_repository_modification_preserves_original(tmp_path):
    repository = tmp_path / "target_repository"
    repository.mkdir()

    original = "VALUE = 1\n"
    (repository / "config.py").write_text(original)

    state = make_state(tmp_path, repository)

    def generator(
        specification, repository_context, qa_feedback, previous_implementation,
    ):
        assert repository_context is not None
        assert repository_context.excerpts["config.py"] == original

        return DevelopmentAssessment(
            status="ready",
            summary="Updated configuration module.",
            files=[
                GeneratedFile(
                    path="config.py",
                    operation=FileOperation.MODIFY,
                    content="VALUE = 2\n",
                )
            ],
        )

    developer = create_stage(tmp_path, generator)

    result = developer(state)["development_result"]

    assert result.outcome == StageOutcome.COMPLETED

    attempt_dir = (
        tmp_path
        / state["task"].run_id
        / "development"
        / "attempt-0"
    )

    assert (
        attempt_dir / "staging/config.py"
    ).read_text() == "VALUE = 2\n"

    # The target repository must not be modified.
    assert (repository / "config.py").read_text() == original

    manifest = ImplementationManifest.model_validate_json(
        (attempt_dir / "manifest.json").read_text()
    )

    assert manifest.files[0].operation == FileOperation.MODIFY
    assert manifest.files[0].original_sha256 is not None
    assert (
        manifest.files[0].sha256
        != manifest.files[0].original_sha256
    )


# ---------------------------------------------------------
# Missing prerequisites
# ---------------------------------------------------------

def test_missing_task_blocks_development(tmp_path):
    developer = create_stage(tmp_path)

    result = developer({})["development_result"]

    assert result.outcome == StageOutcome.BLOCKED
    assert result.errors


def test_missing_specification_blocks_development(tmp_path):
    state = make_state(tmp_path)

    spec_path = (
        tmp_path
        / state["task"].run_id
        / "research"
        / "engineering_spec.json"
    )

    spec_path.unlink()

    developer = create_stage(tmp_path)

    result = developer(state)["development_result"]

    assert result.outcome == StageOutcome.BLOCKED
    assert result.errors


def test_blocked_agent_does_not_stage_files(tmp_path):
    state = make_state(tmp_path)

    def blocked_generator(
        specification, repository, qa_feedback, previous_implementation,
    ):
        return DevelopmentAssessment(
            status="blocked",
            reason="Implementation requirements are insufficient.",
        )

    developer = create_stage(tmp_path, blocked_generator)

    result = developer(state)["development_result"]

    assert result.outcome == StageOutcome.BLOCKED
    assert result.artifacts == []

    development_dir = (
        tmp_path / state["task"].run_id / "development"
    )

    assert not development_dir.exists()


# ---------------------------------------------------------
# Validation and filesystem safety
# ---------------------------------------------------------

def test_unsafe_generated_path_is_rejected(tmp_path):
    state = make_state(tmp_path)

    def unsafe_generator(
        specification, repository, qa_feedback, previous_implementation,
    ):
        return DevelopmentAssessment(
            status="ready",
            summary="Unsafe implementation.",
            files=[
                GeneratedFile(
                    path="../outside.py",
                    operation=FileOperation.CREATE,
                    content="print('unsafe')\n",
                )
            ],
        )

    developer = create_stage(tmp_path, unsafe_generator)

    result = developer(state)["development_result"]

    assert result.outcome == StageOutcome.FAILED
    assert result.errors

    assert not (tmp_path / "outside.py").exists()


def test_incomplete_repository_context_blocks_modify(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    (repository / "existing.py").write_text("VALUE = 1\n")

    state = make_state(tmp_path, repository)

    def incomplete_reader(path):
        return RepositoryContext(
            path=str(repository.resolve()),
            files=["existing.py"],
            excerpts={
                "existing.py": "VALUE = 1\n",
            },
            truncated=True,
        )

    def modify_generator(
        specification, repository_context, qa_feedback, previous_implementation,
    ):
        return DevelopmentAssessment(
            status="ready",
            summary="Modify an existing file.",
            files=[
                GeneratedFile(
                    path="existing.py",
                    operation=FileOperation.MODIFY,
                    content="VALUE = 2\n",
                )
            ],
        )

    developer = create_stage(
        tmp_path,
        generator=modify_generator,
        repository_reader=incomplete_reader,
    )

    result = developer(state)["development_result"]

    assert result.outcome == StageOutcome.BLOCKED
    assert (repository / "existing.py").read_text() == "VALUE = 1\n"


def test_invalid_generator_response_is_rejected(tmp_path):
    state = make_state(tmp_path)

    def invalid_generator(
        specification, repository, qa_feedback, previous_implementation,
    ):
        return {"status": "ready"}

    developer = create_stage(
        tmp_path,
        generator=invalid_generator,
    )

    result = developer(state)["development_result"]

    assert result.outcome == StageOutcome.BLOCKED
    assert result.errors


# ---------------------------------------------------------
# QA retries
# ---------------------------------------------------------

def test_qa_retry_uses_separate_attempt(tmp_path):
    state = make_state(tmp_path)

    received_feedback = []
    received_previous = []

    def generator(
        specification, repository, qa_feedback, previous_implementation,
    ):
        received_feedback.append(qa_feedback)
        received_previous.append(previous_implementation)

        revision = len(received_feedback) - 1

        return DevelopmentAssessment(
            status="ready",
            summary=f"Development attempt {revision}",
            files=[
                GeneratedFile(
                    path="main.py",
                    operation=FileOperation.CREATE,
                    content=f"VERSION = {revision}\n",
                )
            ],
        )

    developer = create_stage(tmp_path, generator=generator)

    first = developer(state)["development_result"]

    assert first.outcome == StageOutcome.COMPLETED

    state["development_result"] = first
    state["revision_count"] = 1
    state["qa_result"] = QAResult(
        verdict=QAVerdict.FAIL,
        feedback=QAFeedback(
            summary="Generated code failed a test.",
            development_attempt=0,
        ),
    )

    second = developer(state)["development_result"]

    assert second.outcome == StageOutcome.COMPLETED

    development_dir = (
        tmp_path / state["task"].run_id / "development"
    )

    assert (
        development_dir / "attempt-0/staging/main.py"
    ).read_text() == "VERSION = 0\n"

    assert (
        development_dir / "attempt-1/staging/main.py"
    ).read_text() == "VERSION = 1\n"

    assert received_feedback == [
        None,
        QAFeedback(
            summary="Generated code failed a test.",
            development_attempt=0,
        ),
    ]
    assert received_previous == [
        None,
        PreviousImplementation(
            development_attempt=0,
            files=[PreviousFile(
                path="main.py",
                operation=FileOperation.CREATE,
                content="VERSION = 0\n",
            )],
        ),
    ]


def test_retry_receives_feedback_and_previous_files(tmp_path):
    state = make_state(tmp_path)
    feedback = QAFeedback(summary="Fix the greeting.", development_attempt=0)
    previous = PreviousImplementation(
        development_attempt=0,
        files=[PreviousFile(
            path="greet.py",
            operation=FileOperation.CREATE,
            content="def greet(name):\n    return name\n",
        )],
    )
    captured = {}

    def generator(specification, repository, qa_feedback, previous_implementation):
        captured["feedback"] = qa_feedback
        captured["previous"] = previous_implementation
        return DevelopmentAssessment(status="blocked", reason="Captured retry context.")

    def previous_loader(current_state, runs_dir):
        assert current_state is state
        assert runs_dir == tmp_path.resolve()
        return previous

    state.update(
        revision_count=1,
        qa_result=QAResult(verdict=QAVerdict.FAIL, feedback=feedback),
    )
    stage = create_stage(
        tmp_path,
        generator=generator,
        previous_implementation_loader=previous_loader,
    )

    result = stage(state)["development_result"]

    assert result.outcome == StageOutcome.BLOCKED
    assert result.summary == "Captured retry context."
    assert captured["feedback"] is feedback
    assert captured["previous"] is previous


@pytest.mark.parametrize("feedback_attempt", [0, 2])
def test_retry_rejects_feedback_for_wrong_attempt(tmp_path, feedback_attempt):
    state = make_state(tmp_path)
    state.update(
        revision_count=2,
        qa_result=QAResult(
            verdict=QAVerdict.FAIL,
            feedback=QAFeedback(
                summary="Feedback from the wrong attempt.",
                development_attempt=feedback_attempt,
            ),
        ),
    )

    def unexpected_call(*args, **kwargs):
        pytest.fail("Mismatched feedback must block before loading or generation.")

    stage = create_stage(
        tmp_path,
        generator=unexpected_call,
        previous_implementation_loader=unexpected_call,
    )

    result = stage(state)["development_result"]

    assert result.outcome == StageOutcome.BLOCKED
    assert "does not match the previous development attempt" in result.errors[0]
    assert result.artifacts == []


def test_retry_requires_qa_feedback(tmp_path):
    state = make_state(tmp_path)
    state["revision_count"] = 1

    developer = create_stage(tmp_path)

    result = developer(state)["development_result"]

    assert result.outcome == StageOutcome.BLOCKED


def test_existing_attempt_cannot_be_overwritten(tmp_path):
    state = make_state(tmp_path)
    developer = create_stage(tmp_path)

    first = developer(state)["development_result"]
    second = developer(state)["development_result"]

    assert first.outcome == StageOutcome.COMPLETED
    assert second.outcome == StageOutcome.BLOCKED


# ---------------------------------------------------------
# Manifest failure and cleanup
# ---------------------------------------------------------

def test_manifest_failure_removes_staged_artifacts(
    tmp_path,
    monkeypatch,
):
    import importlib

    developer_module = importlib.import_module(
        "harness.stages.developer"
    )

    state = make_state(tmp_path)

    def failing_write_manifest(manifest, path):
        raise OSError("Simulated manifest write failure.")

    monkeypatch.setattr(
        developer_module,
        "write_manifest",
        failing_write_manifest,
    )

    developer = create_stage(tmp_path)

    result = developer(state)["development_result"]

    assert result.outcome == StageOutcome.FAILED

    attempt_dir = (
        tmp_path
        / state["task"].run_id
        / "development"
        / "attempt-0"
    )

    assert not attempt_dir.exists()


# ---------------------------------------------------------
# LangGraph integration
# ---------------------------------------------------------

def test_developer_integrates_with_graph(tmp_path):
    """Research -> Developer -> QA BLOCKED -> END."""

    state = make_state(tmp_path)
    executed = []

    def completed_research(current_state):
        executed.append("research")
        return {
            "research_result": current_state["research_result"]
        }

    developer_stage = create_stage(tmp_path)

    def traced_developer(current_state):
        executed.append("developer")
        return developer_stage(current_state)

    def traced_qa(current_state):
        executed.append("qa")
        assert current_state["development_result"].outcome == StageOutcome.COMPLETED
        return {
            "qa_result": QAResult(
                verdict=QAVerdict.BLOCKED,
                summary="QA execution is unavailable in this graph-routing test.",
            )
        }

    def unexpected_release(current_state):
        raise AssertionError(
            "Release must not execute when QA is blocked."
        )

    graph = build_graph(
        research=completed_research,
        developer=traced_developer,
        qa=traced_qa,
        release=unexpected_release,
    )

    result = graph.invoke(state)

    assert executed == [
        "research",
        "developer",
        "qa",
    ]

    assert (
        result["research_result"].outcome
        == StageOutcome.COMPLETED
    )

    assert (
        result["development_result"].outcome
        == StageOutcome.COMPLETED
    )

    assert result["qa_result"].verdict == QAVerdict.BLOCKED
    assert result["status"] == RunStatus.BLOCKED

    assert "release_result" not in result
