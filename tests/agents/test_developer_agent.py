
"""Tests for GroqDeveloperAgent using a fake Groq API client.

No live Groq calls, repository writes, or code execution occur.
"""

import json
from types import SimpleNamespace

import pytest

from harness.agents.developer_agent import (
    DevelopmentGenerationError,
    GroqDeveloperAgent,
)
from harness.models import (
    DevelopmentAssessment,
    EngineeringSpec,
    FileOperation,
)
from harness.tools.repository import RepositoryContext


class FakeCompletions:
    """Capture the completion request and return a controlled response."""

    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content)
                )
            ]
        )


class FakeGroqClient:
    def __init__(self, content=None, error=None):
        self.completions = FakeCompletions(
            content=content,
            error=error,
        )
        self.chat = SimpleNamespace(completions=self.completions)


def sample_spec():
    return EngineeringSpec(
        title="Config validator",
        objective="Validate JSON configuration using a schema.",
        requirements=["Load JSON", "Validate required fields"],
        technical_approach="Use a schema model and file loader.",
        implementation_steps=["Create the loader", "Add unit tests"],
        acceptance_criteria=[
            "Valid JSON loads",
            "Invalid JSON is rejected",
        ],
    )


def ready_response():
    return {
        "status": "ready",
        "reason": "",
        "summary": "Generated a config loader with tests.",
        "files": [
            {
                "path": "src/config_validator/loader.py",
                "operation": "create",
                "content": "def load_config(path):\n    return path\n",
                "description": "Configuration loading API",
            },
            {
                "path": "tests/test_loader.py",
                "operation": "create",
                "content": "def test_placeholder():\n    assert True\n",
                "description": "Initial test scaffold",
            },
        ],
        "dependencies": ["pydantic"],
        "notes": ["Generated code has not been executed."],
    }


def test_successful_generation_full_flow():
    """TRACE THIS: specification -> Groq request -> JSON -> validated files."""

    fake_client = FakeGroqClient(
        content=json.dumps(ready_response())
    )
    agent = GroqDeveloperAgent(
        model="test-model",
        client=fake_client,
    )

    spec = sample_spec()

    assessment = agent.generate(
        specification=spec,
        repository=None,
        qa_feedback=None,
    )

    # Inspect the outgoing Groq request.
    request = fake_client.completions.last_request

    assert request["model"] == "test-model"
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][0]["role"] == "system"

    sent_context = json.loads(
        request["messages"][1]["content"]
    )

    assert sent_context["specification"]["title"] == spec.title
    assert sent_context["repository"] is None
    assert sent_context["qa_feedback"] == []

    # Inspect the validated result.
    assert isinstance(assessment, DevelopmentAssessment)
    assert assessment.status == "ready"
    assert len(assessment.files) == 2

    assert assessment.files[0].operation == FileOperation.CREATE
    assert assessment.files[0].path == "src/config_validator/loader.py"
    assert assessment.files[1].path == "tests/test_loader.py"


def test_repository_context_and_qa_feedback_are_forwarded():
    response = ready_response()

    response["files"] = [
        {
            "path": "src/service.py",
            "operation": "modify",
            "content": (
                "def get_users():\n"
                "    return ['updated']\n"
            ),
            "description": "Modify existing functionality",
        }
    ]

    fake_client = FakeGroqClient(
        content=json.dumps(response)
    )
    agent = GroqDeveloperAgent(client=fake_client)

    repository = RepositoryContext(
        path="/example/repository",
        files=["src/service.py"],
        excerpts={
            "src/service.py": "def get_users():\n    return []\n"
        },
    )

    assessment = agent.generate(
        specification=sample_spec(),
        repository=repository,
        qa_feedback=["Test failed: empty result"],
    )

    context = json.loads(
        fake_client.completions.last_request["messages"][1]["content"]
    )

    assert context["repository"]["excerpts"][
        "src/service.py"
    ].startswith("def get_users")

    assert context["qa_feedback"] == [
        "Test failed: empty result"
    ]

    assert assessment.files[0].operation == FileOperation.MODIFY


def test_blocked_assessment():
    response = {
        "status": "blocked",
        "reason": "Missing API contract",
        "files": [],
    }

    agent = GroqDeveloperAgent(
        client=FakeGroqClient(content=json.dumps(response))
    )

    assessment = agent.generate(sample_spec())

    assert assessment.status == "blocked"
    assert assessment.reason == "Missing API contract"
    assert assessment.files == []


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        "{",
        '{"status":"ready","files":[]}',
    ],
)
def test_invalid_model_response_is_rejected(content):
    agent = GroqDeveloperAgent(
        client=FakeGroqClient(content=content)
    )

    with pytest.raises(DevelopmentGenerationError):
        agent.generate(sample_spec())


def test_invalid_file_operation_is_rejected():
    response = ready_response()
    response["files"][0]["operation"] = "delete"

    agent = GroqDeveloperAgent(
        client=FakeGroqClient(content=json.dumps(response))
    )

    with pytest.raises(DevelopmentGenerationError):
        agent.generate(sample_spec())


def test_empty_model_response_is_rejected():
    agent = GroqDeveloperAgent(
        client=FakeGroqClient(content="")
    )

    with pytest.raises(
        DevelopmentGenerationError,
        match="empty response",
    ):
        agent.generate(sample_spec())


def test_groq_api_failure_is_wrapped():
    agent = GroqDeveloperAgent(
        client=FakeGroqClient(
            error=RuntimeError("Simulated API outage")
        )
    )

    with pytest.raises(
        DevelopmentGenerationError,
        match="generation failed",
    ):
        agent.generate(sample_spec())
