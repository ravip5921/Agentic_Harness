
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
    PreviousFile,
    PreviousImplementation,
    QAFeedback,
    DevelopmentPlan,
    GeneratedFile,
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


class FakeSequentialGroqClient:
    """Return different responses for successive Groq requests."""

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        if not self.responses:
            raise AssertionError(
                "Unexpected additional Groq request."
            )

        content = self.responses.pop(0)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content
                    )
                )
            ]
        )

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
    """TRACE THIS: plan -> generate files -> DevelopmentAssessment."""

    plan_response = {
        "status": "ready",
        "reason": "",
        "summary": "Implement JSON config validation with tests.",
        "files": [
            {
                "path": "config_validator.py",
                "operation": "create",
                "description": "Configuration validation module",
            },
            {
                "path": "tests/test_config_validator.py",
                "operation": "create",
                "description": "Pytest tests for config validation",
            },
        ],
        "dependencies": ["pydantic", "pytest"],
        "notes": [],
    }

    implementation_code = (
        "import json\n"
        "from pathlib import Path\n\n"
        "def load_config(path):\n"
        "    return json.loads(Path(path).read_text())\n"
    )

    test_code = (
        "from config_validator import load_config\n\n"
        "def test_load_config(tmp_path):\n"
        "    path = tmp_path / 'config.json'\n"
        "    path.write_text('{\"name\": \"test\"}')\n"
        "    assert load_config(path)['name'] == 'test'\n"
    )

    client = FakeSequentialGroqClient(
        responses=[
            json.dumps(plan_response),
            implementation_code,
            test_code,
        ]
    )

    agent = GroqDeveloperAgent(
        model="test-model",
        client=client,
    )

    assessment = agent.generate(
        specification=sample_spec(),
        repository=None,
        qa_feedback=None,
    )

    assert assessment.status == "ready"
    assert len(assessment.files) == 2

    assert assessment.files[0].path == "config_validator.py"
    assert assessment.files[0].operation == FileOperation.CREATE
    assert assessment.files[0].content == implementation_code

    assert assessment.files[1].path == "tests/test_config_validator.py"
    assert assessment.files[1].content == test_code

    assert assessment.dependencies == ["pydantic", "pytest"]

    # One planning request plus two file-generation requests.
    assert len(client.calls) == 3
    assert len(client.responses) == 0

    # Planning uses strict schema-constrained JSON.
    response_format = client.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"] == "development_plan"
    assert "files" in response_format["json_schema"]["schema"]["properties"]

    # File generation uses raw text.
    assert "response_format" not in client.calls[1]
    assert "response_format" not in client.calls[2]

    # The second file generator receives the first generated file.
    second_file_context = json.loads(
        client.calls[2]["messages"][1]["content"]
    )

    assert len(second_file_context["generated_files"]) == 1
    assert (
        second_file_context["generated_files"][0]["content"]
        == implementation_code
    )


def test_repository_context_and_qa_feedback_are_forwarded():
    """Verify repository and retry context reach both Developer phases."""

    plan_response = {
        "status": "ready",
        "reason": "",
        "summary": "Update existing service implementation.",
        "files": [
            {
                "path": "src/service.py",
                "operation": "modify",
                "description": "Correct the existing service behavior",
            }
        ],
        "dependencies": [],
        "notes": [],
    }

    updated_code = (
        "def get_users():\n"
        "    return ['updated']\n"
    )

    client = FakeSequentialGroqClient(
        responses=[
            json.dumps(plan_response),
            updated_code,
        ]
    )

    agent = GroqDeveloperAgent(client=client)

    repository = RepositoryContext(
        path="/example/repository",
        files=["src/service.py"],
        excerpts={
            "src/service.py": (
                "def get_users():\n"
                "    return []\n"
            )
        },
    )

    feedback = QAFeedback(
        summary="Test failed: empty result",
        development_attempt=0,
        recommendations=[
            "Correct get_users() to return the expected result."
        ],
    )

    previous = PreviousImplementation(
        development_attempt=0,
        files=[
            PreviousFile(
                path="src/service.py",
                operation=FileOperation.MODIFY,
                content=(
                    "def get_users():\n"
                    "    return []\n"
                ),
            )
        ],
    )

    assessment = agent.generate(
        specification=sample_spec(),
        repository=repository,
        qa_feedback=feedback,
        previous_implementation=previous,
    )

    assert assessment.status == "ready"
    assert len(assessment.files) == 1

    generated = assessment.files[0]

    assert generated.path == "src/service.py"
    assert generated.operation == FileOperation.MODIFY
    assert generated.content == updated_code

    # One planning call plus one file-generation call.
    assert len(client.calls) == 2

    planning_context = json.loads(
        client.calls[0]["messages"][1]["content"]
    )

    file_context = json.loads(
        client.calls[1]["messages"][1]["content"]
    )

    for context in (planning_context, file_context):
        assert context["repository"]["path"] == "/example/repository"

        assert (
            context["qa_feedback"]["summary"]
            == "Test failed: empty result"
        )

        assert (
            context["previous_implementation"]["development_attempt"]
            == 0
        )

        assert (
            context["previous_implementation"]["files"][0]["path"]
            == "src/service.py"
        )

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
        client=FakeSequentialGroqClient(
            error=RuntimeError("Simulated API outage")
        )
    )

    with pytest.raises(
        DevelopmentGenerationError,
        match="Developer planning failed",
    ) as exc_info:
        agent.generate(specification=sample_spec())

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )

def test_two_phase_generation(monkeypatch):
    """Ensure generation plans first and produces each file individually."""

    from harness.agents.developer_agent import GroqDeveloperAgent
    from harness.models import (
        DevelopmentPlan,
        EngineeringSpec,
        GeneratedFile,
    )

    agent = GroqDeveloperAgent(client=object())

    spec = EngineeringSpec(
        title="Addition utility",
        objective="Implement integer addition.",
        requirements=["Implement add(a, b)."],
        technical_approach="Use Python.",
        implementation_steps=["Implement function and tests."],
        acceptance_criteria=["add(2, 3) returns 5."],
    )

    plan = DevelopmentPlan(
        summary="Addition implementation",
        files=[
            {
                "path": "add.py",
                "operation": "create",
                "description": "Addition function",
            },
            {
                "path": "tests/test_add.py",
                "operation": "create",
                "description": "Pytest coverage",
            },
        ],
        status="ready",
    )

    calls = []

    def fake_plan(**kwargs):
        calls.append("plan")
        return plan

    def fake_generate_file(**kwargs):
        planned = kwargs["planned_file"]
        previous = kwargs["generated_files"]

        calls.append(planned.path)

        if planned.path == "add.py":
            assert previous == []
            content = "def add(a, b): return a + b\n"
        else:
            assert len(previous) == 1
            assert previous[0].path == "add.py"
            content = (
                "from add import add\n"
                "def test_add(): assert add(2, 3) == 5\n"
            )

        return GeneratedFile(
            path=planned.path,
            operation=planned.operation,
            content=content,
            description=planned.description,
        )

    monkeypatch.setattr(agent, "plan", fake_plan)
    monkeypatch.setattr(agent, "generate_file", fake_generate_file)

    result = agent.generate(specification=spec)

    assert result.status == "ready"
    assert len(result.files) == 2
    assert calls == [
        "plan",
        "add.py",
        "tests/test_add.py",
    ]


def test_blocked_plan_is_valid():
    plan = DevelopmentPlan(
        status="blocked",
        reason="Required API documentation is unavailable.",
        files=[],
    )

    assert plan.status == "blocked"
    assert plan.files == []


def test_blocked_plan_does_not_generate_files(monkeypatch):
    agent = GroqDeveloperAgent(client=object())

    specification = EngineeringSpec(
        title="API integration",
        objective="Implement an external API client.",
        requirements=["Retrieve remote resources."],
        technical_approach="Use the documented API.",
        implementation_steps=["Implement client."],
        acceptance_criteria=["Client retrieves requested data."],
    )

    def fake_plan(**kwargs):
        return DevelopmentPlan(
            status="blocked",
            reason="API endpoint and access requirements are unknown.",
        )

    def fail_if_called(**kwargs):
        raise AssertionError(
            "File generation must not run for blocked plans."
        )

    monkeypatch.setattr(agent, "plan", fake_plan)
    monkeypatch.setattr(agent, "generate_file", fail_if_called)

    result = agent.generate(specification)

    assert result.status == "blocked"
    assert "API endpoint" in result.reason
    assert result.files == []


def test_retry_generation_does_not_restore_omitted_files(monkeypatch):
    agent = GroqDeveloperAgent(client=object())

    previous = PreviousImplementation(
        development_attempt=0,
        files=[
            PreviousFile(
                path="calculator.py",
                operation=FileOperation.CREATE,
                content="def add(a, b): return a - b\n",
            ),
            PreviousFile(
                path="unused.py",
                operation=FileOperation.CREATE,
                content="VALUE = 1\n",
            ),
        ],
    )

    new_plan = DevelopmentPlan(
        status="ready",
        summary="Consolidate implementation",
        files=[
            {
                "path": "calculator.py",
                "operation": "create",
                "description": "Correct addition logic",
            }
        ],
    )

    monkeypatch.setattr(
        agent,
        "plan",
        lambda **kwargs: new_plan,
    )

    def fake_generate_file(**kwargs):
        planned = kwargs["planned_file"]

        return GeneratedFile(
            path=planned.path,
            operation=planned.operation,
            description=planned.description,
            content="def add(a, b): return a + b\n",
        )

    monkeypatch.setattr(
        agent,
        "generate_file",
        fake_generate_file,
    )

    result = agent.generate(
        specification=sample_spec(),
        previous_implementation=previous,
    )

    assert result.status == "ready"
    assert [file.path for file in result.files] == [
        "calculator.py",
    ]
