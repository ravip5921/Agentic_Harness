"""Exercise streaming with real stages and artifact tools, without Groq or Docker."""

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from harness.agents.developer_agent import GroqDeveloperAgent
from harness.graph import build_graph
from harness.models import (
    EngineeringSpec,
    FileOperation,
    QACheckResult,
    QACheckStatus,
    QACheckType,
    QAReport,
    QARequirementAssessment,
    QAVerdict,
    ReleaseAssessment,
    ReleaseManifest,
    ReleaseStatus,
    RequirementFinding,
    RequirementStatus,
    ResearchAssessment,
    RunStatus,
    StageOutcome,
    TaskRequest,
)
from harness.stages.developer import make_developer_stage
from harness.stages.qa import make_qa_stage
from harness.stages.release import make_release_stage
from harness.stages.research import make_research_stage
from harness.state import HarnessState


GOOD_CODE = "def add(a, b):\n    return a + b\n"
BAD_CODE = "def add(a, b):\n    return a - b\n"
TEST_CODE = (
    "from calculator import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)


class RecordingProgress:
    """Record only the public progress consumer interface."""

    def __init__(self):
        self.events = []
        self.updates = []

    def event(self, payload):
        self.events.append(payload)

    def update(self, payload, current_state):
        self.updates.append((payload, dict(current_state)))


def make_workflow(runs_dir, *, repair=False, always_bad=False,
                  release_failure=False, observe=lambda _stage: None,
                  rate_limit_failures=0, requested_tokens=6489, rate_limit_status=429):
    """Mock generation/controlled execution while retaining every trusted tool."""
    calls = {"planning": [], "files": [], "checks": [], "release": 0, "rate_limit_errors": 0}
    spec = EngineeringSpec(
        title="Addition utility",
        objective="Implement integer addition.",
        requirements=["Implement add(a, b)."],
        technical_approach="Use Python.",
        implementation_steps=["Implement addition and a pytest test."],
        acceptance_criteria=["add(2, 3) returns 5."],
    )

    def research_generator(task, repository, evidence):
        observe("research")
        assert evidence == []
        return ResearchAssessment(status="ready", specification=spec)

    def create_completion(**kwargs):
        nonlocal rate_limit_failures
        if rate_limit_failures:
            import httpx
            from groq import APIStatusError, RateLimitError

            rate_limit_failures -= 1
            calls["rate_limit_errors"] += 1
            message = (
                "tokens per minute (TPM): Limit 8000, Used 2318, "
                f"Requested {requested_tokens}. Please try again in 6.0525s."
            )
            error_type = RateLimitError if rate_limit_status == 429 else APIStatusError
            raise error_type(message, response=httpx.Response(
                rate_limit_status, request=httpx.Request("POST", "https://api.groq.com/test"),
            ), body={"message": message})

        context = json.loads(kwargs["messages"][-1]["content"])
        observe("developer")

        if "current_file" not in context:
            calls["planning"].append(context)
            content = json.dumps({
                "status": "ready",
                "summary": "Implement addition and pytest tests.",
                "files": [
                    {"path": "calculator.py", "operation": "create",
                     "description": "Integer addition."},
                    {"path": "tests/test_calculator.py", "operation": "create",
                     "description": "Test addition."},
                ],
            })
        else:
            calls["files"].append(context)
            path = context["current_file"]["path"]
            bad = always_bad or (repair and context["qa_feedback"] is None)
            content = (BAD_CODE if bad else GOOD_CODE) if path == "calculator.py" else TEST_CODE

        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
        )])

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create_completion),
    ))
    developer_agent = GroqDeveloperAgent(client=client)

    def execute_check(check_type, workspace):
        calls["checks"].append(check_type)
        # Read actual verified staged bytes; never execute generated code here.
        code = (workspace / "calculator.py").read_text(encoding="utf-8")
        passed = check_type == QACheckType.SYNTAX or code == GOOD_CODE
        return QACheckResult(
            name=check_type.value,
            check_type=check_type,
            status=QACheckStatus.PASSED if passed else QACheckStatus.FAILED,
            exit_code=0 if passed else 1,
            stdout="1 passed" if passed else "AssertionError: expected 5",
        )

    def assess_requirements(task, specification, manifest, implementation, checks):
        observe("qa")
        code = next(file.content for file in implementation.files
                    if file.path == "calculator.py")
        status = (RequirementStatus.SATISFIED if code == GOOD_CODE
                  else RequirementStatus.NOT_SATISFIED)
        return QARequirementAssessment(
            findings=[
                RequirementFinding(
                    requirement_id=requirement_id,
                    requirement=requirement,
                    status=status,
                    explanation="Reviewed the actual staged addition implementation.",
                    implementation_files=["calculator.py"],
                    test_files=["tests/test_calculator.py"],
                )
                for requirement_id, requirement in (
                    ("REQ-1", specification.requirements[0]),
                    ("AC-1", specification.acceptance_criteria[0]),
                )
            ],
            overall_summary="Reviewed all requirements and acceptance criteria.",
        )

    def generate_release(task, manifest, qa_report):
        observe("release")
        calls["release"] += 1
        assert qa_report.verdict == QAVerdict.PASS
        if release_failure:
            raise RuntimeError("Mock release documentation service unavailable")
        return ReleaseAssessment(
            summary="Addition candidate prepared for review.",
            changes=[f"CREATE {file.path}" for file in manifest.files],
            testing_summary="Syntax, unit tests, and requirement assessment passed.",
        )

    workflow = build_graph(
        research=make_research_stage(generator=research_generator, runs_dir=runs_dir),
        developer=make_developer_stage(generator=developer_agent.generate, runs_dir=runs_dir),
        qa=make_qa_stage(test_executor=execute_check,
                         requirements_assessor=assess_requirements, runs_dir=runs_dir),
        release=make_release_stage(release_generator=generate_release, runs_dir=runs_dir),
    )
    return workflow, calls


def normalized(value, runs_dir):
    """Compare domain data while allowing independent artifact roots."""
    return json.dumps(value, default=lambda obj: obj.model_dump(mode="json"),
                      sort_keys=True).replace(str(runs_dir), "<runs>")


def artifact_contents(runs_dir):
    return {
        str(path.relative_to(runs_dir)): path.read_text(encoding="utf-8").replace(
            str(runs_dir), "<runs>",
        )
        for path in runs_dir.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize(
    "options,expected_status,revision",
    [
        ({}, RunStatus.STAGED, 0),
        ({"repair": True}, RunStatus.STAGED, 1),
        ({"always_bad": True}, RunStatus.FAILED, 1),
        ({"release_failure": True}, RunStatus.STAGED, 0),
    ],
    ids=["success", "qa-repair", "retry-limit", "release-fallback"],
)
def test_streaming_preserves_full_state_and_all_artifacts(tmp_path, options,
                                                         expected_status, revision):
    from harness.cli import stream_workflow

    repository = tmp_path / "repository"
    repository.mkdir()
    readme = repository / "README.md"
    readme.write_text("Original repository contents.\n", encoding="utf-8")
    task = TaskRequest(task="Implement addition with pytest tests.",
                       repository_path=str(repository), max_dev_revisions=1)
    initial = {"task": task, "revision_count": 0, "status": RunStatus.RUNNING}
    invoked_root, streamed_root = tmp_path / "invoked", tmp_path / "streamed"
    invoked_graph, invoked_calls = make_workflow(invoked_root, **options)
    streamed_graph, streamed_calls = make_workflow(streamed_root, **options)
    progress = RecordingProgress()

    invoked = invoked_graph.invoke(initial)
    streamed = stream_workflow(streamed_graph, initial, progress)

    assert normalized(streamed, streamed_root) == normalized(invoked, invoked_root)
    assert artifact_contents(streamed_root) == artifact_contents(invoked_root)
    assert streamed["status"] == expected_status
    assert streamed["revision_count"] == revision
    assert set(streamed) <= HarnessState.__annotations__.keys()
    assert {event["stage"] for event in progress.events} >= {"research", "developer", "qa"}
    assert all({"stage", "event", "message", "data"} <= event.keys()
               for event in progress.events)
    assert any("research" in update for update, _state in progress.updates)
    assert any("qa" in update for update, _state in progress.updates)
    assert streamed["research_result"].outcome == StageOutcome.COMPLETED
    assert streamed["development_result"].outcome == StageOutcome.COMPLETED
    assert len(streamed_calls["planning"]) == revision + 1
    assert len(streamed_calls["files"]) == 2 * (revision + 1)
    assert len(streamed_calls["checks"]) == 2 * (revision + 1)
    assert streamed_calls == invoked_calls
    assert readme.read_text(encoding="utf-8") == "Original repository contents.\n"
    assert sorted(path.name for path in repository.iterdir()) == ["README.md"]

    if revision:
        first, retry = streamed_calls["planning"]
        assert first["qa_feedback"] is None
        assert first["previous_implementation"] is None
        assert retry["qa_feedback"]["development_attempt"] == 0
        assert retry["previous_implementation"]["development_attempt"] == 0
        assert next(file["content"] for file in retry["previous_implementation"]["files"]
                    if file["path"] == "calculator.py") == BAD_CODE
        assert all(file["operation"] == FileOperation.CREATE
                   for file in retry["previous_implementation"]["files"])
        for attempt in (0, 1):
            report_path = streamed_root / task.run_id / "qa" / f"attempt-{attempt}" / "test_report.json"
            report = QAReport.model_validate_json(report_path.read_text())
            expected = QAVerdict.PASS if attempt and not options.get("always_bad") else QAVerdict.FAIL
            assert report.verdict == expected

    assert not (streamed_root / task.run_id / "development" / "attempt-2").exists()
    if expected_status == RunStatus.FAILED:
        assert streamed_calls["release"] == 0
        assert "release_result" not in streamed
        return

    assert streamed_calls["release"] == 1
    assert streamed["qa_result"].verdict == QAVerdict.PASS
    assert streamed["release_result"].status == ReleaseStatus.STAGED
    release_dir = streamed_root / task.run_id / "release" / f"attempt-{revision}"
    candidate = release_dir / "candidate"
    assert Path(streamed["release_result"].change_ref) == candidate
    assert (candidate / "calculator.py").read_text() == GOOD_CODE
    assert (candidate / "tests/test_calculator.py").read_text() == TEST_CODE
    manifest = ReleaseManifest.model_validate_json((release_dir / "release_manifest.json").read_text())
    assert manifest.run_id == task.run_id
    assert manifest.revision_count == revision
    assert manifest.release_directory == str(candidate)
    assert len(manifest.files) == 2
    if options.get("release_failure"):
        report = (release_dir / "release_report.md").read_text()
        assert "Fallback: deterministic documentation" in report
        assessment = ReleaseAssessment.model_validate_json((release_dir / "release_assessment.json").read_text())
        assert any("deterministic" in item for item in assessment.limitations)
        assert streamed["release_result"].errors == []


def test_custom_events_are_consumed_before_generation_returns(tmp_path):
    from harness.cli import stream_workflow

    ready = {stage: Event() for stage in ("research", "developer", "qa", "release")}

    class ObservedProgress(RecordingProgress):
        def event(self, payload):
            super().event(payload)
            if payload["event"].endswith("started"):
                ready[payload["stage"]].set()

    progress = ObservedProgress()
    snapshots = []

    def observe(stage):
        # LangGraph runs nodes on a worker. Hold the mocked request open until
        # its start event reaches the consumer, as a real LLM request would.
        assert ready[stage].wait(timeout=5), f"No live progress while {stage} is running"
        snapshots.append((stage, list(progress.events)))

    workflow, _calls = make_workflow(tmp_path, observe=observe)
    task = TaskRequest(task="Implement addition.")
    result = stream_workflow(workflow, {"task": task, "revision_count": 0}, progress)

    assert result["status"] == RunStatus.STAGED
    assert {stage for stage, _events in snapshots} == {"research", "developer", "qa", "release"}
    for stage, events in snapshots:
        assert any(event["stage"] == stage and event["event"].endswith("started")
                   for event in events), f"No live operation start before {stage} generation"


def test_cli_displays_live_progress_and_preserves_final_summary(tmp_path, monkeypatch, capsys):
    import harness.cli as cli

    output_during_generation = []
    ready = {stage: Event() for stage in ("research", "developer", "qa", "release")}

    class ObservedRenderer(cli.ProgressRenderer):
        def event(self, payload):
            super().event(payload)
            if payload["event"].endswith("started"):
                ready[payload["stage"]].set()

    def observe(stage):
        assert ready[stage].wait(timeout=5), f"No live CLI progress during {stage}"
        output_during_generation.append((stage, capsys.readouterr().out))

    task = TaskRequest(task="Implement addition.")
    workflow, _calls = make_workflow(tmp_path, observe=observe)
    monkeypatch.setattr(cli, "create_task", lambda _args: task)
    monkeypatch.setattr(cli, "create_workflow", lambda _runs_dir: workflow)
    monkeypatch.setattr(cli, "ProgressRenderer", ObservedRenderer)

    exit_code = cli.harness_cli([task.task, "--runs-dir", str(tmp_path)])
    final_output = capsys.readouterr().out
    live_output = "".join(output for _stage, output in output_during_generation)

    assert exit_code == 0
    assert "RESEARCH" in output_during_generation[0][1]
    assert "=== Agentic Harness Result ===" not in live_output
    assert all(stage.upper() in live_output for stage in ("research", "developer", "qa", "release"))
    assert "=== Agentic Harness Result ===" in final_output
    assert f"Run ID: {task.run_id}" in final_output
    assert "Status: staged" in final_output
    assert "Development revisions: 0" in final_output
    assert "engineering_spec.json" in final_output
    assert "manifest.json" in final_output
    assert "test_report.json" in final_output
    assert str(tmp_path / task.run_id / "release" / "attempt-0" / "candidate") in final_output
    assert "\x1b[" not in live_output + final_output


def test_groq_429_wait_is_streamed_and_request_recovers(tmp_path, monkeypatch):
    from io import StringIO

    from rich.console import Console

    from harness.cli import stream_workflow
    from harness.tools.progress import ProgressRenderer

    received = Event()
    waits = []
    output = StringIO()

    class RateLimitRenderer(ProgressRenderer):
        def event(self, payload):
            super().event(payload)
            if payload["event"] == "groq_rate_limit_wait_started":
                received.set()

    def fake_sleep(seconds):
        # Hold the mocked retry wait until the CLI has displayed its wait event.
        assert received.wait(timeout=5), "429 retry wait was not streamed while waiting"
        received.clear()
        waits.append(seconds)

    monkeypatch.setattr("harness.tools.groq_rate_limit.sleep", fake_sleep)
    graph, calls = make_workflow(tmp_path, rate_limit_failures=2)
    task = TaskRequest(task="Implement addition.")
    with RateLimitRenderer(Console(file=output, force_terminal=False, width=120)) as renderer:
        result = stream_workflow(graph, {"task": task, "revision_count": 0}, renderer)
        renderer.finish(result["status"])

    assert waits == [pytest.approx(8.0525), pytest.approx(8.0525)]
    assert calls["rate_limit_errors"] == 2
    assert len(calls["planning"]) == 1
    assert len(calls["files"]) == 2
    assert result["status"] == RunStatus.STAGED
    assert result["release_result"].change_ref == str(tmp_path / task.run_id / "release/attempt-0/candidate")
    assert set(result) <= HarnessState.__annotations__.keys()
    text = output.getvalue()
    assert text.count("waiting 8.0525s before retry") == 2
    assert text.count("Groq rate limit cleared") == 1
    assert "Workflow completed: STAGED" in text


@pytest.mark.parametrize("status_code", [429, 413])
def test_oversized_request_exhaustion_fails_cli_gracefully(tmp_path, monkeypatch, capsys, status_code):
    import harness.cli as cli

    waits = []
    monkeypatch.setattr("harness.tools.groq_rate_limit.sleep", waits.append)
    task = TaskRequest(task="Implement addition.")
    workflow, calls = make_workflow(
        tmp_path, rate_limit_failures=3, requested_tokens=9000, rate_limit_status=status_code,
    )
    monkeypatch.setattr(cli, "create_task", lambda _args: task)
    monkeypatch.setattr(cli, "create_workflow", lambda _runs_dir: workflow)

    assert cli.harness_cli([task.task, "--runs-dir", str(tmp_path)]) == 1

    output = capsys.readouterr().out
    assert calls["rate_limit_errors"] == 3
    assert waits == [pytest.approx(8.0525), pytest.approx(8.0525)]
    assert "stopping after 2 retries" in output
    assert "Requested 9000 tokens exceeds the TPM limit 8000" in output
    assert "Workflow completed: FAILED" in output
    assert "=== Agentic Harness Result ===" in output
    assert "Status: failed" in output
    assert "QA: Not executed" in output
    assert "Release: Not executed" in output
    assert not (tmp_path / task.run_id / "development/attempt-0").exists()
