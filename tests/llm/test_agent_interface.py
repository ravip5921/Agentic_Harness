"""Every agent operation can run through Gemini or a custom adapter offline."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from harness.agents.developer_agent import DeveloperAgent, DevelopmentGenerationError
from harness.agents.qa_agent import QAAgent
from harness.agents.release_agent import ReleaseAgent
from harness.agents.research_agent import ResearchAgent
from harness.llm import Completion
from harness.models import (
    DevelopmentPlan, EngineeringSpec, ImplementationManifest, PlannedFile, QAReport, TaskRequest,
)
from harness.tools.qa_context import QAImplementationContext


def inputs():
    spec = EngineeringSpec(
        title="Addition", objective="Add integers", requirements=["Implement add"],
        technical_approach="Python", implementation_steps=["Write code"],
        acceptance_criteria=["add(1, 1) returns 2"],
    )
    task = TaskRequest(task="Implement addition")
    manifest = ImplementationManifest(summary="Addition", files=[{
        "path": "add.py", "operation": "create", "sha256": "a" * 64,
    }])
    planned = PlannedFile(path="add.py", operation="create", description="Addition")
    plan = DevelopmentPlan(status="ready", summary="Addition", files=[planned])
    implementation = QAImplementationContext(files=[{
        "path": "add.py", "operation": "create", "content": "def add(a, b): return a+b",
    }])
    report = QAReport(run_id=task.run_id, revision_count=0, verdict="pass", manifest_path="manifest.json", checks=[{
        "name": "unit tests", "check_type": "unit_test", "status": "passed", "exit_code": 0,
    }])
    return spec, task, manifest, planned, plan, implementation, report


@pytest.mark.parametrize("operation", ["research", "planning", "file", "requirements", "feedback", "release"])
def test_all_agent_operations_use_gemini_without_groq_credentials(monkeypatch, operation):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    spec, task, manifest, planned, plan, implementation, report = inputs()
    payloads = {
        "research": {"status": "ready", "reason": "", "specification": spec.model_dump()},
        "planning": plan.model_dump(mode="json"),
        "file": "def add(a, b): return a + b\n",
        "feedback": {"summary": "Review", "development_attempt": 0, "file_feedback": [],
                     "test_feedback": [], "recommendations": ["Review addition"]},
        "requirements": {"overall_summary": "Satisfied", "findings": [{
            "requirement_id": item_id, "requirement": requirement, "status": "satisfied",
            "explanation": "Reviewed", "implementation_files": ["add.py"], "test_files": [],
            "missing_coverage": [],
        } for item_id, requirement in [("REQ-1", spec.requirements[0]), ("AC-1", spec.acceptance_criteria[0])]]},
        "release": {"summary": "Addition", "testing_summary": "Unit tests passed", "changes": [],
                    "limitations": [], "review_notes": []},
    }
    text = payloads[operation] if operation == "file" else json.dumps(payloads[operation])
    body = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}
    received = SimpleNamespace(status_code=200, json=lambda: body, close=Mock())
    post = Mock(return_value=received)
    monkeypatch.setattr("harness.llm.http.requests.post", post)
    operations = {
        "research": lambda: ResearchAgent().generate(task, None, []),
        "planning": lambda: DeveloperAgent().plan(spec),
        "file": lambda: DeveloperAgent().generate_file(planned, plan, spec),
        "requirements": lambda: QAAgent().assess_requirements(task, spec, manifest, implementation, []),
        "feedback": lambda: QAAgent().generate(spec, manifest, [], 0),
        "release": lambda: ReleaseAgent().generate(task, manifest, report),
    }
    assert operations[operation]() is not None
    post.assert_called_once()
    assert post.call_args.args[0].startswith("https://generativelanguage.googleapis.com/")
    config = post.call_args.kwargs["json"]["generationConfig"]
    assert ("responseJsonSchema" in config) == (operation != "file")


def test_custom_adapter_requires_no_provider_configuration(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unconfigured-provider")
    custom = Mock(complete=Mock(return_value=Completion('{"status":"blocked","reason":"Missing information"}')))
    result = ResearchAgent(generator=custom).generate(TaskRequest(task="Example"), None, [])
    assert result.status == "blocked"
    assert custom.complete.call_args.kwargs["schema_name"] == "research_assessment"


def test_developer_does_not_accept_or_complete_a_truncated_file(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    received = SimpleNamespace(status_code=200, close=Mock(), json=lambda: {"candidates": [{
        "finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "def add("}]},
    }]})
    monkeypatch.setattr("harness.llm.http.requests.post", Mock(return_value=received))
    events = []
    monkeypatch.setattr("harness.events.get_stream_writer", lambda: events.append)
    spec, task, manifest, planned, plan, implementation, report = inputs()
    with pytest.raises(DevelopmentGenerationError, match="MAX_TOKENS"):
        DeveloperAgent().generate_file(planned, plan, spec)
    assert "file_generation_completed" not in [event["event"] for event in events]
