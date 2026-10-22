
"""Integration tests for the Developer -> QA -> Developer repair cycle.

Real components:
- LangGraph orchestration and retry routing
- Developer staging and manifest generation
- Previous implementation artifact loading
- QA artifact validation and workspace preparation
- QA verdict, feedback, and report generation

Mocked components:
- Research specification generation
- Developer LLM
- QA requirements-assessment LLM
- Docker check execution
- Release stage
"""

import os

import pytest
from pathlib import Path

from harness.graph import build_graph

from harness.models import (
    DevelopmentAssessment,
    FileOperation,
    QARequirementAssessment,
    QACheckResult,
    QACheckStatus,
    QACheckType,
    QAVerdict,
    ReleaseResult,
    ReleaseStatus,
    RequirementFinding,
    RequirementStatus,
    ResearchResult,
    RunStatus,
    StageOutcome,
    TaskRequest,
    EngineeringSpec,
)

from harness.stages.developer import make_developer_stage
from harness.stages.qa import (
    make_qa_stage,
    build_feedback,
    combine_qa_verdicts,
)


from harness.agents.developer_agent import GroqDeveloperAgent
from harness.agents.qa_agent import GroqQAAgent

RUN_LIVE_GROQ = os.getenv("RUN_LIVE_GROQ") == "1"

live_groq = pytest.mark.skipif(
    not RUN_LIVE_GROQ,
    reason="Set RUN_LIVE_GROQ=1 to run live Groq integration tests.",
)

# ---------------------------------------------------------
# Shared fixtures and fake components
# ---------------------------------------------------------

BAD_CODE = "def add(a, b):\n    return a - b\n"

GOOD_CODE = "def add(a, b):\n    return a + b\n"

TEST_CODE = (
    "from calculator import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)


def make_spec():
    return EngineeringSpec(
        title="Addition utility",
        objective="Implement integer addition.",
        requirements=["Implement add(a, b)."],
        technical_approach="Use Python.",
        implementation_steps=[
            "Implement add().",
            "Write pytest tests.",
        ],
        acceptance_criteria=[
            "add(2, 3) returns 5.",
        ],
    )


def make_research_stage(runs_dir):
    """Mock Research generation, but write real Research artifacts."""

    def research(state):
        task = state["task"]
        spec = make_spec()

        directory = runs_dir / task.run_id / "research"
        directory.mkdir(parents=True, exist_ok=True)

        spec_path = directory / "engineering_spec.json"
        spec_path.write_text(
            spec.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return {
            "research_result": ResearchResult(
                outcome=StageOutcome.COMPLETED,
                summary="Engineering specification generated.",
                artifacts=[str(spec_path)],
            )
        }

    return research


def make_generated_files(code):
    return [
        {
            "path": "calculator.py",
            "operation": FileOperation.CREATE,
            "content": code,
        },
        {
            "path": "tests/test_calculator.py",
            "operation": FileOperation.CREATE,
            "content": TEST_CODE,
        },
    ]


def make_developer_generator(calls, always_bad=False):
    """Mock LLM that repairs the implementation on the second attempt."""

    def generator(
        specification,
        repository,
        qa_feedback,
        previous_implementation,
    ):
        calls.append(
            {
                "feedback": qa_feedback,
                "previous": previous_implementation,
            }
        )

        attempt = len(calls) - 1

        if attempt == 0 or always_bad:
            code = BAD_CODE
        else:
            code = GOOD_CODE

        return DevelopmentAssessment(
            status="ready",
            summary=f"Generated implementation attempt {attempt}.",
            files=make_generated_files(code),
        )

    return generator


def fake_test_executor(check_type, workspace):
    """Simulate test results based on the actual staged implementation."""

    code = (workspace / "calculator.py").read_text(
        encoding="utf-8"
    )

    if check_type == QACheckType.SYNTAX:
        return QACheckResult(
            name="syntax",
            check_type=check_type,
            status=QACheckStatus.PASSED,
            exit_code=0,
            stdout="Syntax valid.",
        )

    if check_type == QACheckType.UNIT_TEST:
        passed = "return a + b" in code

        return QACheckResult(
            name="unit_test",
            check_type=check_type,
            status=(
                QACheckStatus.PASSED
                if passed
                else QACheckStatus.FAILED
            ),
            exit_code=0 if passed else 1,
            stdout=(
                "1 passed"
                if passed
                else "AssertionError: expected 5, received -1"
            ),
        )

    raise AssertionError(f"Unexpected check: {check_type}")


def fake_requirements_assessor(
    task,
    specification,
    manifest,
    implementation,
    checks,
):
    """Mock QA reviewer assessing the actual implementation contents."""

    code = next(
        file.content
        for file in implementation.files
        if file.path == "calculator.py"
    )

    status = (
        RequirementStatus.SATISFIED
        if "return a + b" in code
        else RequirementStatus.NOT_SATISFIED
    )

    return QARequirementAssessment(
        findings=[
            RequirementFinding(
                requirement_id="REQ-1",
                requirement=specification.requirements[0],
                status=status,
                explanation="Reviewed the addition implementation.",
                implementation_files=["calculator.py"],
            ),
            RequirementFinding(
                requirement_id="AC-1",
                requirement=specification.acceptance_criteria[0],
                status=status,
                explanation="Reviewed addition behavior and tests.",
                implementation_files=["calculator.py"],
                test_files=["tests/test_calculator.py"],
            ),
        ],
        overall_summary="Requirements assessment completed.",
    )


def make_release_stage(calls):
    """Placeholder Release that records whether it was reached."""

    def release(state):
        calls.append(state["qa_result"].verdict)

        return {
            "release_result": ReleaseResult(
                status=ReleaseStatus.BLOCKED,
                summary="Release not implemented yet.",
            )
        }

    return release


def make_workflow(
    tmp_path,
    developer_calls,
    release_calls,
    *,
    always_bad=False,
    qa_wrapper=None,
):
    """Build a real graph with mocked agents and executor."""

    research = make_research_stage(tmp_path)

    developer = make_developer_stage(
        generator=make_developer_generator(
            developer_calls,
            always_bad=always_bad,
        ),
        runs_dir=tmp_path,
    )

    qa = make_qa_stage(
        test_executor=fake_test_executor,
        requirements_assessor=fake_requirements_assessor,
        runs_dir=tmp_path,
    )

    if qa_wrapper is not None:
        original_qa = qa

        def wrapped_qa(state):
            result = original_qa(state)
            qa_wrapper(state, result)
            return result

        qa = wrapped_qa

    release = make_release_stage(release_calls)

    return build_graph(
        research=research,
        developer=developer,
        qa=qa,
        release=release,
    )


def attempt_dir(tmp_path, task, revision):
    return (
        tmp_path
        / task.run_id
        / "development"
        / f"attempt-{revision}"
    )


# ---------------------------------------------------------
# Test 1: Complete successful repair cycle
# ---------------------------------------------------------

def test_repair_cycle_succeeds(tmp_path):
    developer_calls = []
    release_calls = []

    graph = make_workflow(
        tmp_path,
        developer_calls,
        release_calls,
    )

    task = TaskRequest(
        task="Implement add(a, b) with pytest tests.",
        max_dev_revisions=1,
    )

    result = graph.invoke(
        {"task": task, "revision_count": 0}
    )

    # Initial attempt + one QA-triggered retry.
    assert result["revision_count"] == 1
    assert len(developer_calls) == 2

    # Initial Developer call has no retry context.
    assert developer_calls[0]["feedback"] is None
    assert developer_calls[0]["previous"] is None

    # Second Developer call receives verified previous code.
    feedback = developer_calls[1]["feedback"]
    previous = developer_calls[1]["previous"]

    assert feedback is not None
    assert feedback.development_attempt == 0
    assert previous is not None
    assert previous.development_attempt == 0

    previous_files = {
        file.path: file.content
        for file in previous.files
    }

    assert previous_files["calculator.py"] == BAD_CODE
    assert "tests/test_calculator.py" in previous_files

    # Both Developer attempts remain independently available.
    attempt_0 = attempt_dir(tmp_path, task, 0)
    attempt_1 = attempt_dir(tmp_path, task, 1)

    assert (attempt_0 / "manifest.json").exists()
    assert (attempt_1 / "manifest.json").exists()

    assert (
        attempt_0 / "staging" / "calculator.py"
    ).read_text() == BAD_CODE

    assert (
        attempt_1 / "staging" / "calculator.py"
    ).read_text() == GOOD_CODE

    # QA reports exist for both attempts.
    assert (
        tmp_path / task.run_id / "qa"
        / "attempt-0" / "test_report.json"
    ).exists()

    assert (
        tmp_path / task.run_id / "qa"
        / "attempt-1" / "test_report.json"
    ).exists()

    # Repaired implementation passes QA and reaches Release.
    assert result["qa_result"].verdict == QAVerdict.PASS
    assert release_calls == [QAVerdict.PASS]

    # Release is intentionally still a placeholder.
    assert result["status"] == RunStatus.BLOCKED


# ---------------------------------------------------------
# Test 2: Retry limit enforcement
# ---------------------------------------------------------

def test_retry_limit_is_enforced(tmp_path):
    developer_calls = []
    release_calls = []

    graph = make_workflow(
        tmp_path,
        developer_calls,
        release_calls,
        always_bad=True,
    )

    task = TaskRequest(
        task="Implement add(a, b) with pytest tests.",
        max_dev_revisions=1,
    )

    result = graph.invoke(
        {"task": task, "revision_count": 0}
    )

    # Initial attempt and exactly one retry.
    assert len(developer_calls) == 2
    assert result["revision_count"] == 1

    assert result["qa_result"].verdict == QAVerdict.FAIL
    assert result["status"] == RunStatus.FAILED

    # Release must never run after failing QA.
    assert release_calls == []

    # No third attempt is generated.
    assert not attempt_dir(tmp_path, task, 2).exists()


# ---------------------------------------------------------
# Test 3: Corrupted previous artifacts block the retry
# ---------------------------------------------------------

def test_retry_blocks_on_corrupted_previous_artifact(tmp_path):
    developer_calls = []
    release_calls = []

    def corrupt_previous_artifact(state, qa_update):
        """Simulate corruption after QA finishes attempt 0."""

        if state.get("revision_count", 0) != 0:
            return

        assert qa_update["qa_result"].verdict == QAVerdict.FAIL

        task = state["task"]

        file_path = (
            attempt_dir(tmp_path, task, 0)
            / "staging"
            / "calculator.py"
        )

        file_path.write_text(
            "def add(a, b): return 999\n",
            encoding="utf-8",
        )

    graph = make_workflow(
        tmp_path,
        developer_calls,
        release_calls,
        qa_wrapper=corrupt_previous_artifact,
    )

    task = TaskRequest(
        task="Implement add(a, b) with pytest tests.",
        max_dev_revisions=1,
    )

    result = graph.invoke(
        {"task": task, "revision_count": 0}
    )

    # Retry is attempted but cannot load verified previous files.
    assert result["revision_count"] == 1

    # Developer LLM is not called again.
    assert len(developer_calls) == 1

    assert result["development_result"].outcome == StageOutcome.BLOCKED
    assert result["status"] == RunStatus.BLOCKED

    # No second implementation should be published.
    assert not attempt_dir(tmp_path, task, 1).exists()

    assert release_calls == []


@live_groq
def test_live_groq_detects_unmet_requirements():
    """Real QA LLM should detect an implementation violating the task."""

    from harness.tools.qa_context import (
        QAImplementationContext,
        VerifiedSourceFile,
    )

    from harness.models import (
        ImplementationManifest,
        StagedFile,
        QARequirementAssessment,
        RequirementStatus,
    )

    task = TaskRequest(
        task=(
            "Implement is_prime(n), returning True exactly when "
            "an integer n is prime, including handling 2, odd "
            "composites, zero, one, and negative integers."
        )
    )

    specification = EngineeringSpec(
        title="Prime number checker",
        objective="Implement correct primality checking.",
        requirements=[
            "Return True exactly for prime integers.",
            "Return False for integers less than 2.",
        ],
        technical_approach="Use Python.",
        implementation_steps=[
            "Implement is_prime().",
            "Write pytest tests.",
        ],
        acceptance_criteria=[
            "is_prime(2) is True.",
            "is_prime(9) is False.",
            "is_prime(1) is False.",
        ],
    )

    # Deliberately incorrect: identifies positive odd numbers,
    # not prime numbers.
    code = (
        "def is_prime(n):\n"
        "    return n > 1 and n % 2 != 0\n"
    )

    # Tests pass but omit cases that expose the defect.
    test_code = (
        "from prime import is_prime\n\n"
        "def test_prime():\n"
        "    assert is_prime(3)\n"
        "    assert is_prime(5)\n"
        "    assert not is_prime(4)\n"
    )

    manifest = ImplementationManifest(
        summary="Prime number utility",
        files=[
            StagedFile(
                path="prime.py",
                operation=FileOperation.CREATE,
                sha256="a" * 64,
            ),
            StagedFile(
                path="tests/test_prime.py",
                operation=FileOperation.CREATE,
                sha256="b" * 64,
            ),
        ],
    )

    implementation = QAImplementationContext(
        files=[
            VerifiedSourceFile(
                path="prime.py",
                operation=FileOperation.CREATE,
                content=code,
            ),
            VerifiedSourceFile(
                path="tests/test_prime.py",
                operation=FileOperation.CREATE,
                content=test_code,
            ),
        ]
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

    agent = GroqQAAgent()

    assessment = agent.assess_requirements(
        task=task,
        specification=specification,
        manifest=manifest,
        implementation=implementation,
        checks=checks,
    )

    assert isinstance(assessment, QARequirementAssessment)

    findings = {
        finding.requirement_id: finding
        for finding in assessment.findings
    }

    # Complete requirements coverage is enforced by the agent.
    assert len(findings) == 5

    # The implementation demonstrably violates both criteria.
    assert findings["AC-1"].status == RequirementStatus.NOT_SATISFIED
    assert findings["AC-2"].status == RequirementStatus.NOT_SATISFIED

    # Passing developer tests must not override the QA assessment.
    verdict = combine_qa_verdicts(
        QAVerdict.PASS,
        assessment,
    )

    assert verdict == QAVerdict.FAIL


@live_groq
def test_live_groq_developer_repairs_previous_implementation():
    """Real Developer LLM should use QA feedback and previous code."""

    from harness.models import (
        QAFeedback,
        PreviousFile,
        PreviousImplementation,
    )

    specification = EngineeringSpec(
        title="Addition utility",
        objective="Implement correct integer addition.",
        requirements=[
            "Implement add(a, b) for integers.",
        ],
        technical_approach="Use Python.",
        implementation_steps=[
            "Implement the addition function.",
            "Write pytest tests.",
        ],
        acceptance_criteria=[
            "add(2, 3) returns 5.",
            "add(-2, 3) returns 1.",
        ],
    )

    previous = PreviousImplementation(
        development_attempt=0,
        files=[
            PreviousFile(
                path="calculator.py",
                operation=FileOperation.CREATE,
                content=BAD_CODE,
            ),
            PreviousFile(
                path="tests/test_calculator.py",
                operation=FileOperation.CREATE,
                content=TEST_CODE,
            ),
        ],
    )

    feedback = QAFeedback(
        summary="Addition implementation is incorrect.",
        development_attempt=0,
        recommendations=[
            "The add(a, b) function currently subtracts b from a.",
            "Fix the implementation so add(2, 3) returns 5.",
            "Preserve the pytest test file in the complete revision.",
        ],
    )

    agent = GroqDeveloperAgent()

    assessment = agent.generate(
        specification=specification,
        repository=None,
        qa_feedback=feedback,
        previous_implementation=previous,
    )

    assert assessment.status == "ready"

    generated_files = {
        file.path: file
        for file in assessment.files
    }

    assert "calculator.py" in generated_files
    assert "tests/test_calculator.py" in generated_files

    # The implementation should actually perform addition.
    namespace = {}

    # Inspect rather than execute generated code on the host.
    import ast

    source = generated_files["calculator.py"].content
    tree = ast.parse(source)

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "add"
    ]

    assert functions, "Developer did not implement add()."

    # Verify the revision differs from the known broken version.
    assert source != BAD_CODE
