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


def test_research_uses_discovered_documentation(tmp_path):
    """Discovered URLs should reach the evidence retriever."""

    from datetime import datetime, timezone

    from harness.models import (
        TaskRequest,
        ResearchAssessment,
        EngineeringSpec,
        EvidenceSource,
        StageOutcome,
    )
    from harness.stages.research import make_research_stage

    seen = {}

    def fake_discoverer(task_description, domains):
        seen["task"] = task_description
        seen["domains"] = domains

        return [
            "https://docs.example.com/api"
        ]

    def fake_evidence_reader(urls):
        seen["urls"] = urls

        return [
            EvidenceSource(
                source_id="source-1",
                url=urls[0],
                title="Example API",
                retrieved_at=datetime.now(
                    timezone.utc
                ).isoformat(),
                content_sha256="a" * 64,
                excerpt="Example API reference documentation.",
            )
        ]

    def fake_generator(task, repository, evidence):
        seen["evidence"] = evidence

        return ResearchAssessment(
            status="ready",
            specification=EngineeringSpec(
                title="Example API client",
                objective="Implement an API integration.",
                requirements=["Retrieve API data."],
                technical_approach="Use the documented API.",
                implementation_steps=["Implement client."],
                acceptance_criteria=[
                    "The client retrieves requested data."
                ],
                source_ids=["source-1"],
            ),
        )

    task = TaskRequest(
        task="Build an Example API client.",
        parameters={
            "discover_documentation": True,
            "documentation_domains": ["docs.example.com"],
        },
    )

    stage = make_research_stage(
        generator=fake_generator,
        repository_reader=lambda path: None,
        evidence_reader=fake_evidence_reader,
        documentation_discoverer=fake_discoverer,
        runs_dir=tmp_path,
    )

    result = stage({"task": task})["research_result"]

    assert result.outcome == StageOutcome.COMPLETED
    assert seen["urls"] == [
        "https://docs.example.com/api"
    ]
    assert len(seen["evidence"]) == 1

    evidence_path = (
        tmp_path
        / task.run_id
        / "research"
        / "evidence.json"
    )

    assert evidence_path.exists()
