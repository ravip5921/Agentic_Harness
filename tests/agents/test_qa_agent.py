
"""Unit tests for structured Groq QA feedback generation."""

import json
from types import SimpleNamespace

import pytest

from harness.agents.qa_agent import (
    GroqQAAgent,
    QAGenerationError,
)
from harness.models import (
    EngineeringSpec,
    ImplementationManifest,
    QACheckResult,
    QACheckStatus,
    QACheckType,
    StagedFile,
    FileOperation,
)


class FakeGroqClient:
    def __init__(self, content):
        self.content = content
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
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
    spec = EngineeringSpec(
        title="Calculator",
        objective="Implement addition.",
        requirements=["Add two integers."],
        technical_approach="Python function.",
        implementation_steps=["Implement add()."],
        acceptance_criteria=["Addition tests pass."],
    )

    manifest = ImplementationManifest(
        summary="Calculator implementation",
        files=[
            StagedFile(
                path="calculator.py",
                operation=FileOperation.CREATE,
                sha256="a" * 64,
            )
        ],
    )

    checks = [
        QACheckResult(
            name="pytest",
            check_type=QACheckType.UNIT_TEST,
            status=QACheckStatus.FAILED,
            exit_code=1,
            stdout="Expected 5, got -1",
        )
    ]

    return spec, manifest, checks


def test_generate_structured_qa_feedback():
    payload = {
        "summary": "Addition test failed.",
        "development_attempt": 0,
        "file_feedback": [
            {
                "path": "calculator.py",
                "category": "test_failure",
                "message": "Addition returns an incorrect result.",
                "line": None,
                "related_test": "test_add"
            }
        ],
        "test_feedback": [
            {
                "test_name": "test_add",
                "message": "Expected 5, got -1.",
                "related_files": ["calculator.py"],
                "check_name": "pytest"
            }
        ],
        "recommendations": [
            "Inspect calculator.py and correct the addition logic."
        ]
    }

    client = FakeGroqClient(json.dumps(payload))
    agent = GroqQAAgent(client=client)

    spec, manifest, checks = sample_inputs()

    feedback = agent.generate(
        specification=spec,
        manifest=manifest,
        checks=checks,
        development_attempt=0,
        file_excerpts={
            "calculator.py": "def add(a, b): return a - b"
        },
    )

    assert feedback.development_attempt == 0
    assert len(feedback.file_feedback) == 1
    assert feedback.file_feedback[0].path == "calculator.py"
    assert feedback.test_feedback[0].test_name == "test_add"

    assert len(client.calls) == 1

    request = client.calls[0]

    assert "Required JSON Schema" in request["messages"][0]["content"]

    context = json.loads(
        request["messages"][1]["content"]
    )

    assert context["development_attempt"] == 0
    assert "calculator.py" in context["file_excerpts"]


def test_rejects_wrong_development_attempt():
    payload = {
        "summary": "Failure",
        "development_attempt": 3,
        "file_feedback": [],
        "test_feedback": [],
        "recommendations": [],
    }

    agent = GroqQAAgent(
        client=FakeGroqClient(json.dumps(payload))
    )

    spec, manifest, checks = sample_inputs()

    with pytest.raises(
        QAGenerationError,
        match="wrong development attempt"
    ):
        agent.generate(spec, manifest, checks, 0)


def test_rejects_unknown_file_reference():
    payload = {
        "summary": "Failure",
        "development_attempt": 0,
        "file_feedback": [
            {
                "path": "unknown.py",
                "category": "implementation",
                "message": "Potential defect."
            }
        ],
        "test_feedback": [],
        "recommendations": [],
    }

    agent = GroqQAAgent(
        client=FakeGroqClient(json.dumps(payload))
    )

    spec, manifest, checks = sample_inputs()

    with pytest.raises(
        QAGenerationError,
        match="undeclared file"
    ):
        agent.generate(spec, manifest, checks, 0)
