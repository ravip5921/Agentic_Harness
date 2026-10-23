
"""Tests for GroqReleaseAgent using a mocked Groq client."""

import json
from types import SimpleNamespace

import pytest

from harness.agents.release_agent import (
    GroqReleaseAgent,
    ReleaseGenerationError,
)
from harness.models import (
    TaskRequest,
    ImplementationManifest,
    StagedFile,
    FileOperation,
    QAVerdict,
    QAReport,
    QACheckResult,
    QACheckType,
    QACheckStatus,
    ReleaseAssessment,
)


class FakeGroqClient:
    def __init__(self, content):
        self.content = content
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.content
                    )
                )
            ]
        )


def sample_inputs():
    task = TaskRequest(
        task="Implement add(a, b) and pytest tests.",
    )

    manifest = ImplementationManifest(
        summary="Addition implementation.",
        files=[
            StagedFile(
                path="add.py",
                operation=FileOperation.CREATE,
                sha256="a" * 64,
            )
        ],
    )

    report = QAReport(
        run_id=task.run_id,
        revision_count=0,
        verdict=QAVerdict.PASS,
        summary="QA passed.",
        checks=[
            QACheckResult(
                name="syntax",
                check_type=QACheckType.SYNTAX,
                status=QACheckStatus.PASSED,
                exit_code=0,
            ),
            QACheckResult(
                name="unit_test",
                check_type=QACheckType.UNIT_TEST,
                status=QACheckStatus.PASSED,
                exit_code=0,
            ),
        ],
        manifest_path="/example/manifest.json",
    )

    return task, manifest, report


def test_release_agent_generates_assessment():
    payload = {
        "summary": "Prepared an addition utility for review.",
        "changes": [
            "Created add.py with an addition function."
        ],
        "testing_summary": (
            "Python syntax and unit tests passed."
        ),
        "limitations": [
            "Not deployed or integrated into a target repository."
        ],
        "review_notes": [
            "Review the generated implementation before integration."
        ],
    }

    client = FakeGroqClient(json.dumps(payload))
    agent = GroqReleaseAgent(
        model="test-model",
        client=client,
    )

    task, manifest, report = sample_inputs()

    result = agent.generate(task, manifest, report)

    assert isinstance(result, ReleaseAssessment)
    assert result.summary == payload["summary"]
    assert len(result.changes) == 1

    request = client.calls[0]

    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["name"] == "release_assessment"
    assert request["model"] == "test-model"
    assert "Required JSON Schema" in (
        request["messages"][0]["content"]
    )

    context = json.loads(
        request["messages"][1]["content"]
    )

    assert context["original_task"]["task"] == task.task
    assert len(context["implementation_manifest"]["files"]) == 1
    assert context["qa_report"]["verdict"] == "pass"


def test_release_agent_rejects_invalid_response():
    client = FakeGroqClient(
        json.dumps({
            "summary": "Release candidate prepared."
            # Missing required testing_summary.
        })
    )

    agent = GroqReleaseAgent(client=client)
    task, manifest, report = sample_inputs()

    with pytest.raises(ReleaseGenerationError):
        agent.generate(task, manifest, report)
