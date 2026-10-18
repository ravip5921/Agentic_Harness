"""Tests for the Research stage using injected dependencies."""

from harness.models import (
    EngineeringSpec,
    ResearchAssessment,
    StageOutcome,
    TaskRequest,
)

from harness.stages.research import make_research_stage


def fake_repository_reader(path):
    return None


def fake_evidence_reader(urls):
    return []


def successful_generator(task, repository, evidence):
    return ResearchAssessment(
        status="ready",
        specification=EngineeringSpec(
            title="Example implementation",
            objective="Implement the requested behavior.",
            requirements=["Implement the requested feature."],
            technical_approach="Use a modular implementation.",
            implementation_steps=[
                "Create the implementation module.",
                "Add tests.",
            ],
            acceptance_criteria=[
                "The implementation passes its unit tests."
            ],
        ),
    )


def test_research_completed(tmp_path):
    research = make_research_stage(
        generator=successful_generator,
        repository_reader=fake_repository_reader,
        evidence_reader=fake_evidence_reader,
        runs_dir=tmp_path,
    )

    task = TaskRequest(task="Implement a sample feature")

    result = research({"task": task})["research_result"]

    assert result.outcome == StageOutcome.COMPLETED
    assert len(result.artifacts) == 3

    research_dir = tmp_path / task.run_id / "research"

    assert (research_dir / "engineering_spec.md").exists()
    assert (research_dir / "engineering_spec.json").exists()
    assert (research_dir / "evidence.json").exists()


def test_research_blocked(tmp_path):
    def blocked_generator(task, repository, evidence):
        return ResearchAssessment(
            status="blocked",
            reason="Missing essential requirements.",
        )

    research = make_research_stage(
        generator=blocked_generator,
        repository_reader=fake_repository_reader,
        evidence_reader=fake_evidence_reader,
        runs_dir=tmp_path,
    )

    task = TaskRequest(task="Unclear feature")

    result = research({"task": task})["research_result"]

    assert result.outcome == StageOutcome.BLOCKED
    assert result.errors
    assert result.artifacts == []

    assert not (tmp_path / task.run_id / "research").exists()


def test_research_rejects_unverified_source(tmp_path):
    def invalid_generator(task, repository, evidence):
        spec = successful_generator(
            task, repository, evidence
        ).specification

        spec.source_ids = ["invented-source"]

        return ResearchAssessment(
            status="ready",
            specification=spec,
        )

    research = make_research_stage(
        generator=invalid_generator,
        repository_reader=fake_repository_reader,
        evidence_reader=fake_evidence_reader,
        runs_dir=tmp_path,
    )

    result = research({
        "task": TaskRequest(task="Example task")
    })["research_result"]

    assert result.outcome == StageOutcome.BLOCKED
    assert result.artifacts == []


def test_research_model_failure(tmp_path):
    from harness.agents.research_agent import (
        ResearchGenerationError,
    )

    def failing_generator(task, repository, evidence):
        raise ResearchGenerationError("Model unavailable.")

    research = make_research_stage(
        generator=failing_generator,
        repository_reader=fake_repository_reader,
        evidence_reader=fake_evidence_reader,
        runs_dir=tmp_path,
    )

    result = research({
        "task": TaskRequest(task="Example task")
    })["research_result"]

    assert result.outcome == StageOutcome.FAILED