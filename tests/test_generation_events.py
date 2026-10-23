"""Progress must describe validated Research and Developer operations only."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.graph import END, START, StateGraph

from harness.agents.developer_agent import (
    DevelopmentGenerationError,
    GroqDeveloperAgent,
)
from harness.agents.research_agent import ResearchGenerationError
from harness.models import (
    DevelopmentAssessment,
    EngineeringSpec,
    EvidenceSource,
    GeneratedFile,
    ImplementationManifest,
    QAFeedback,
    QAResult,
    QAVerdict,
    ResearchAssessment,
    StageOutcome,
    TaskRequest,
)
from harness.stages.developer import make_developer_stage
from harness.stages.research import make_research_stage
from harness.state import HarnessState


@pytest.fixture
def captured_events(monkeypatch):
    events = []
    monkeypatch.setattr("harness.events.get_stream_writer", lambda: events.append)
    return events


def specification(**changes):
    values = {
        "title": "Addition utility",
        "objective": "Implement integer addition.",
        "requirements": ["Provide add(a, b).", "Use integer arithmetic."],
        "technical_approach": "Use a standalone Python module.",
        "implementation_steps": ["Implement the function and tests."],
        "acceptance_criteria": ["add(2, 3) returns 5."],
    }
    values.update(changes)
    return EngineeringSpec(**values)


def test_research_stream_reports_actual_counts_and_preserves_artifacts(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "README.md").write_text("Example repository\n")
    task = TaskRequest(
        task="Implement addition",
        repository_path=str(repository),
        parameters={"documentation_urls": ["https://example.com/one", "https://example.com/two"]},
    )
    evidence = [EvidenceSource(
        source_id="source-1", url="https://example.com/one",
        retrieved_at="2026-10-22T00:00:00+00:00", content_sha256="a" * 64,
    )]
    spec = specification(source_ids=["source-1"])

    def retrieve(urls):
        assert urls == task.parameters["documentation_urls"]
        return evidence

    stage = make_research_stage(
        generator=lambda *args: ResearchAssessment(status="ready", specification=spec),
        evidence_reader=retrieve, runs_dir=tmp_path / "runs",
    )
    builder = StateGraph(HarnessState)
    builder.add_node("research", stage)
    builder.add_edge(START, "research")
    builder.add_edge("research", END)
    chunks = list(builder.compile().stream({"task": task}, stream_mode=["custom", "updates"]))

    events = [payload for mode, payload in chunks if mode == "custom"]
    assert [event["event"] for event in events] == [
        "repository_inspection_started", "repository_inspection_completed",
        "evidence_retrieval_started", "evidence_retrieval_completed",
        "specification_generation_started", "specification_generated",
    ]
    assert events[1]["data"] == {"file_count": 1}
    assert events[2]["data"] == {"supplied_source_count": 2}
    assert events[3]["data"] == {"source_count": 1}
    assert events[-1]["data"] == {"requirement_count": 2, "acceptance_criteria_count": 1}
    updates = [payload for mode, payload in chunks if mode == "updates"]
    assert len(updates) == 1
    assert set(updates[0]["research"]) == {"research_result"}
    result = updates[0]["research"]["research_result"]
    research_dir = tmp_path / "runs" / task.run_id / "research"
    assert result.outcome == StageOutcome.COMPLETED
    assert result.artifacts == [str(research_dir / name) for name in (
        "engineering_spec.md", "engineering_spec.json", "evidence.json",
    )]
    assert json.loads(Path(result.artifacts[1]).read_text()) == spec.model_dump(mode="json")
    assert json.loads(Path(result.artifacts[2]).read_text()) == [item.model_dump(mode="json") for item in evidence]


@pytest.mark.parametrize("failure", ["unverified_source", "blocked", "generation_error"])
def test_research_never_announces_unvalidated_specification(tmp_path, captured_events, failure):
    private_detail = "private prompt or exception contents"

    def generate(*args):
        if failure == "generation_error":
            raise ResearchGenerationError(private_detail)
        if failure == "blocked":
            return ResearchAssessment(status="blocked", reason=private_detail)
        return ResearchAssessment(
            status="ready", specification=specification(source_ids=["invented-source"]),
        )

    stage = make_research_stage(
        generator=generate, evidence_reader=lambda urls: [], runs_dir=tmp_path,
    )
    result = stage({"task": TaskRequest(task="Implement addition")})["research_result"]

    assert result.outcome == (StageOutcome.FAILED if failure == "generation_error" else StageOutcome.BLOCKED)
    assert result.artifacts == []
    assert "specification_generated" not in [event["event"] for event in captured_events]
    assert captured_events[-1]["level"] == ("failure" if failure == "generation_error" else "warning")
    assert private_detail not in json.dumps(captured_events)


def test_research_persistence_failure_does_not_announce_success(tmp_path, captured_events, monkeypatch):
    def cannot_persist(*args):
        raise OSError("private storage details")

    monkeypatch.setattr("harness.stages.research.write_research_artifacts", cannot_persist)
    stage = make_research_stage(
        generator=lambda *args: ResearchAssessment(status="ready", specification=specification()),
        evidence_reader=lambda urls: [], runs_dir=tmp_path,
    )

    result = stage({"task": TaskRequest(task="Implement addition")})["research_result"]

    assert result.outcome == StageOutcome.FAILED
    assert result.artifacts == []
    assert "specification_generated" not in [event["event"] for event in captured_events]
    assert "private storage details" not in json.dumps(captured_events)


def test_progress_accepts_opaque_injected_research_contexts(tmp_path, captured_events):
    repository, evidence = object(), object()

    def blocked_generator(task, received_repository, received_evidence):
        assert received_repository is repository
        assert received_evidence is evidence
        return ResearchAssessment(status="blocked", reason="Need clarification.")

    stage = make_research_stage(
        generator=blocked_generator,
        repository_reader=lambda _: repository,
        evidence_reader=lambda _: evidence,
        runs_dir=tmp_path,
    )
    result = stage({"task": TaskRequest(task="Example", repository_path=str(tmp_path))})
    assert result["research_result"].outcome == StageOutcome.BLOCKED
    assert result["research_result"].summary == "Need clarification."
    assert result["research_result"].artifacts == []


def developer_client(responses, captured_events):
    calls = []
    pending = iter(responses)

    def create(**kwargs):
        calls.append(kwargs)
        # A start event must already be available when each request begins.
        expected = "implementation_planning_started" if len(calls) == 1 else "file_generation_started"
        operations = [event for event in captured_events if not event["event"].startswith("llm_")]
        assert operations[-1]["event"] == expected
        assert captured_events[-1]["event"] == "llm_request_started"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(pending)))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def file_plan(paths):
    return json.dumps({
        "status": "ready", "summary": "Implement addition and tests.",
        "files": [{"path": path, "operation": "create", "description": "Planned file"} for path in paths],
    })


def test_developer_events_follow_two_phase_validation_without_source_output(captured_events):
    paths = ["add.py", "tests/test_add.py"]
    contents = ["def add(a, b): return a + b\n", "from add import add\ndef test_add(): assert add(2, 3) == 5\n"]
    client, calls = developer_client([file_plan(paths), *contents], captured_events)

    assessment = GroqDeveloperAgent(client=client).generate(specification())

    captured_events = [event for event in captured_events if not event["event"].startswith("llm_")]
    assert [event["event"] for event in captured_events] == [
        "implementation_planning_started", "file_plan_validated",
        "file_generation_started", "file_generation_completed",
        "file_generation_started", "file_generation_completed",
    ]
    assert captured_events[1]["data"] == {"file_count": 2}
    assert [event["data"]["path"] for event in captured_events[2:]] == [paths[0], paths[0], paths[1], paths[1]]
    assert [file.content for file in assessment.files] == contents
    assert len(calls) == 3
    assert not any(content in json.dumps(captured_events) for content in contents)


@pytest.mark.parametrize("response", ["", "```python\nVALUE = 1\n```"])
def test_invalid_file_response_does_not_emit_completion(captured_events, response):
    client, calls = developer_client([file_plan(["add.py"]), response], captured_events)

    with pytest.raises(DevelopmentGenerationError):
        GroqDeveloperAgent(client=client).generate(specification())

    assert len(calls) == 2
    operations = [event for event in captured_events if not event["event"].startswith("llm_")]
    assert operations[-1]["event"] == "file_generation_started"
    assert not any(event["event"] == "file_generation_completed" for event in captured_events)


def test_invalid_plan_does_not_emit_validation_or_generate_files(captured_events):
    client, calls = developer_client([file_plan(["../outside.py"])], captured_events)

    with pytest.raises(DevelopmentGenerationError):
        GroqDeveloperAgent(client=client).generate(specification())

    assert len(calls) == 1
    assert [event["event"] for event in captured_events if not event["event"].startswith("llm_")] == ["implementation_planning_started"]


def test_developer_retry_metadata_and_staging_match_real_artifacts(tmp_path, captured_events):
    task = TaskRequest(task="Implement addition", max_dev_revisions=2)
    state = {"task": task, "revision_count": 0}
    received_previous = []

    def generate(spec, repository, feedback, previous):
        received_previous.append(previous)
        return DevelopmentAssessment(
            status="ready", summary="Addition implementation",
            files=[GeneratedFile(path="add.py", operation="create", content="def add(a, b): return a + b\n")],
        )

    stage = make_developer_stage(
        generator=generate, specification_loader=lambda *args, **kwargs: specification(),
        runs_dir=tmp_path,
    )
    first = stage(state)["development_result"]
    state.update(
        development_result=first, revision_count=1,
        qa_result=QAResult(verdict=QAVerdict.FAIL, feedback=QAFeedback(summary="Improve coverage", development_attempt=0)),
    )
    second = stage(state)["development_result"]

    starts = [event for event in captured_events if event["event"] == "implementation_generation_started"]
    assert [event["data"] for event in starts] == [
        {"revision": 0, "max_revisions": 2}, {"revision": 1, "max_revisions": 2},
    ]
    assert received_previous[0] is None
    assert received_previous[1].development_attempt == 0
    assert not any(event["event"] in {"implementation_planning_started", "file_generation_started"} for event in captured_events)
    completed = [event for event in captured_events if event["event"] == "implementation_staged"]
    assert len(completed) == 2
    for revision, (event, result) in enumerate(zip(completed, [first, second])):
        attempt_dir = tmp_path / task.run_id / "development" / f"attempt-{revision}"
        assert result.outcome == StageOutcome.COMPLETED
        assert result.artifacts == [str(attempt_dir / "manifest.json"), str(attempt_dir / "staging")]
        assert event["data"] == {
            "file_count": 1, "manifest_path": result.artifacts[0], "staging_path": result.artifacts[1],
        }
        manifest = ImplementationManifest.model_validate_json(Path(result.artifacts[0]).read_text())
        assert [file.path for file in manifest.files] == ["add.py"]
        assert (Path(result.artifacts[1]) / "add.py").read_text() == "def add(a, b): return a + b\n"


def test_manifest_failure_does_not_announce_staging_success(tmp_path, captured_events, monkeypatch):
    def cannot_persist(*args):
        raise OSError("private storage details")

    monkeypatch.setattr("harness.stages.developer.write_manifest", cannot_persist)
    task = TaskRequest(task="Implement addition")
    stage = make_developer_stage(
        generator=lambda *args: DevelopmentAssessment(
            status="ready", summary="Addition implementation",
            files=[GeneratedFile(path="add.py", operation="create", content="def add(a, b): return a + b\n")],
        ),
        specification_loader=lambda *args, **kwargs: specification(), runs_dir=tmp_path,
    )

    result = stage({"task": task})["development_result"]

    assert result.outcome == StageOutcome.FAILED
    assert result.artifacts == []
    assert not (tmp_path / task.run_id / "development" / "attempt-0").exists()
    assert "implementation_staged" not in [event["event"] for event in captured_events]
    assert captured_events[-1]["level"] == "failure"
    assert "private storage details" not in json.dumps(captured_events)
