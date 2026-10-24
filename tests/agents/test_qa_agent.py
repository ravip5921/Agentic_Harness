
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
    TaskRequest,
    QARequirementAssessment,
    RequirementStatus,
)

from harness.tools.qa_context import (
    QAImplementationContext,
    VerifiedSourceFile,
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

    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["name"] == "qa_feedback"
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

def make_requirements_inputs():
    """Create a task, specification, and generated implementation."""

    task = TaskRequest(
        task=(
            "Implement a Python function that adds two integers "
            "and include pytest tests."
        ),
    )

    specification = EngineeringSpec(
        title="Addition utility",
        objective="Implement and test integer addition.",
        requirements=[
            "Implement a function add(a, b).",
            "Support negative integers.",
        ],
        technical_approach="Use a Python function.",
        implementation_steps=[
            "Implement add().",
            "Write pytest tests.",
        ],
        acceptance_criteria=[
            "add(2, 3) returns 5.",
            "All pytest tests pass.",
        ],
    )

    implementation = QAImplementationContext(
        files=[
            VerifiedSourceFile(
                path="calculator.py",
                operation=FileOperation.CREATE,
                content="def add(a, b):\n    return a + b\n",
            ),
            VerifiedSourceFile(
                path="tests/test_calculator.py",
                operation=FileOperation.CREATE,
                content=(
                    "from calculator import add\n\n"
                    "def test_add():\n"
                    "    assert add(2, 3) == 5\n"
                ),
            ),
        ]
    )

    manifest = ImplementationManifest(
        summary="Addition utility",
        files=[
            StagedFile(
                path="calculator.py",
                operation=FileOperation.CREATE,
                sha256="a" * 64,
            ),
            StagedFile(
                path="tests/test_calculator.py",
                operation=FileOperation.CREATE,
                sha256="b" * 64,
            ),
        ],
    )

    checks = [
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
            stdout="1 passed",
        ),
    ]

    return task, specification, manifest, implementation, checks


def make_assessment_response(
    specification: EngineeringSpec,
    *,
    omit_id: str | None = None,
) -> str:
    """Build a synthetic LLM response using expected requirement IDs."""

    findings = []

    for prefix, requirements in (
        ("REQ", specification.requirements),
        ("AC", specification.acceptance_criteria),
    ):
        for index, requirement in enumerate(requirements, start=1):
            requirement_id = f"{prefix}-{index}"

            if requirement_id == omit_id:
                continue

            findings.append(
                {
                    "requirement_id": requirement_id,
                    "requirement": requirement,
                    "status": "satisfied",
                    "explanation": (
                        "The implementation and tests were reviewed."
                    ),
                    "implementation_files": ["calculator.py"],
                    "test_files": ["tests/test_calculator.py"],
                    "missing_coverage": [],
                }
            )

    return json.dumps(
        {
            "findings": findings,
            "overall_summary": "Requirements review completed.",
        }
    )


def test_assess_requirements_complete():
    (
        task,
        specification,
        manifest,
        implementation,
        checks,
    ) = make_requirements_inputs()

    client = FakeGroqClient(
        make_assessment_response(specification)
    )

    agent = GroqQAAgent(client=client)

    assessment = agent.assess_requirements(
        task=task,
        specification=specification,
        manifest=manifest,
        implementation=implementation,
        checks=checks,
    )

    assert isinstance(assessment, QARequirementAssessment)
    assert len(assessment.findings) == 4

    assert {
        finding.requirement_id
        for finding in assessment.findings
    } == {"REQ-1", "REQ-2", "AC-1", "AC-2"}

    assert all(
        finding.status == RequirementStatus.SATISFIED
        for finding in assessment.findings
    )

    assert len(client.calls) == 1

    request = client.calls[0]
    system_prompt = request["messages"][0]["content"]

    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["name"] == "qa_requirement_assessment"
    assert "Required JSON Schema" in system_prompt

    context = json.loads(
        request["messages"][1]["content"]
    )

    assert "original_task" in context
    assert len(context["requirement_checklist"]) == 4
    assert len(context["generated_files"]) == 2
    assert len(context["execution_results"]) == 2


def test_assess_requirements_accepts_reviewed_unchanged_test():
    task, specification, manifest, implementation, checks = make_requirements_inputs()
    implementation.repository_files = {"tests/test_base.py": "def test_base(): assert True\n"}
    payload = json.loads(make_assessment_response(specification))
    payload["findings"][-1]["test_files"] = ["tests/test_base.py"]
    client = FakeGroqClient(json.dumps(payload))
    assessment = GroqQAAgent(client=client).assess_requirements(
        task, specification, manifest, implementation, checks)
    assert assessment.findings[-1].test_files == ["tests/test_base.py"]
    context = json.loads(client.calls[0]["messages"][1]["content"])
    assert context["repository_files"] == implementation.repository_files


def test_assess_requirements_rejects_missing_criterion():
    (
        task,
        specification,
        manifest,
        implementation,
        checks,
    ) = make_requirements_inputs()

    # The response is valid JSON and valid QARequirementAssessment,
    # but it omits acceptance criterion AC-2.
    client = FakeGroqClient(
        make_assessment_response(
            specification,
            omit_id="AC-2",
        )
    )

    agent = GroqQAAgent(client=client)

    with pytest.raises(
        QAGenerationError,
        match="invalid or incomplete requirements assessment",
    ):
        agent.assess_requirements(
            task=task,
            specification=specification,
            manifest=manifest,
            implementation=implementation,
            checks=checks,
        )

    assert len(client.calls) == 1
