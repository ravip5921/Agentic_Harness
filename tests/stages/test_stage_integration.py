
"""Integration tests for the four-stage LangGraph workflow.

The live test uses real Groq calls for Research and Developer.
QA and Release remain placeholder stages.

PowerShell:
    $env:RUN_LIVE_GROQ = "1"
    pytest tests/stages/test_stage_integration.py -v -s

Without RUN_LIVE_GROQ=1, the live test is skipped.
"""

import os
from pathlib import Path

import pytest

from harness.config import load_environment
from harness.graph import build_graph

from harness.models import (
    ImplementationManifest,
    ResearchResult,
    RunStatus,
    StageOutcome,
    TaskRequest,
    QAVerdict,
)

from harness.agents.research_agent import GroqResearchAgent
from harness.agents.developer_agent import GroqDeveloperAgent

from harness.stages.research import make_research_stage
from harness.stages.developer import make_developer_stage
from harness.stages.qa import qa
from harness.stages.release import release


load_environment()


# ---------------------------------------------------------
# Offline: Research blocks the workflow
# ---------------------------------------------------------

def test_blocked_research_stops_workflow():
    """No downstream stages execute when Research is blocked."""

    def blocked_research(state):
        return {
            "research_result": ResearchResult(
                outcome=StageOutcome.BLOCKED,
                summary="Research could not proceed.",
            )
        }

    def unexpected_developer(state):
        raise AssertionError(
            "Developer should not execute after blocked Research."
        )

    def unexpected_qa(state):
        raise AssertionError(
            "QA should not execute after blocked Research."
        )

    def unexpected_release(state):
        raise AssertionError(
            "Release should not execute after blocked Research."
        )

    graph = build_graph(
        research=blocked_research,
        developer=unexpected_developer,
        qa=unexpected_qa,
        release=unexpected_release,
    )

    initial_state = {
        "task": TaskRequest(task="Example task"),
        "revision_count": 0,
        "status": RunStatus.RUNNING,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == RunStatus.BLOCKED

    assert (
        final_state["research_result"].outcome
        == StageOutcome.BLOCKED
    )

    assert "development_result" not in final_state
    assert "qa_result" not in final_state
    assert "release_result" not in final_state


# ---------------------------------------------------------
# Live: Research -> Developer -> QA (blocked)
# ---------------------------------------------------------

@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GROQ") != "1",
    reason="Live Groq integration tests are opt-in.",
)
def test_live_research_to_developer_workflow(tmp_path):
    """Run real Research and Developer through LangGraph."""

    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not configured.")

    # 1. Create the input task. No pre-created specification.
    task = TaskRequest(
        task=(
            "Build a small standalone Python utility that implements "
            "a function named is_even(value: int) -> bool. "
            "Return True when the integer is even and False otherwise. "
            "Support positive numbers, negative numbers, and zero. "
            "Use only the Python standard library for the implementation. "
            "Create even_utils.py containing the implementation and "
            "tests/test_even_utils.py containing pytest tests. "
            "Include tests covering positive even, positive odd, zero, "
            "negative even, and negative odd integers. "
            "The generated test file must import is_even from even_utils. "
            "The utility must be a standalone project without requiring "
            "an existing repository or external service."
        ),
        repository_path=None,
        max_dev_revisions=2,
    )

    initial_state = {
        "task": task,
        "revision_count": 0,
        "status": RunStatus.RUNNING,
    }

    # 2. Initialize real agents.
    research_agent = GroqResearchAgent()
    developer_agent = GroqDeveloperAgent()

    # Both stages MUST use the same artifact root.
    research_stage = make_research_stage(
        generator=research_agent.generate,
        runs_dir=tmp_path,
    )

    developer_stage = make_developer_stage(
        generator=developer_agent.generate,
        runs_dir=tmp_path,
    )

    executed_stages = []

    def traced_research(state):
        executed_stages.append("research")
        result = research_stage(state)

        research_result = result["research_result"]

        print("\n=== RESEARCH STAGE ===")
        print("Outcome:", research_result.outcome)
        print("Summary:", research_result.summary)
        print("Errors:", research_result.errors)
        print("Artifacts:", research_result.artifacts)

        return result

    def traced_developer(state):
        executed_stages.append("developer")
        result = developer_stage(state)

        development_result = result["development_result"]

        print("\n=== DEVELOPER STAGE ===")
        print("Outcome:", development_result.outcome)
        print("Summary:", development_result.summary)
        print("Errors:", development_result.errors)
        print("Artifacts:", development_result.artifacts)

        return result

    def traced_qa(state):
        executed_stages.append("qa")
        result = qa(state)

        print("\n=== QA STAGE ===")
        print("Verdict:", result["qa_result"].verdict)

        return result

    def unexpected_release(state):
        raise AssertionError(
            "Release must not run while QA is blocked."
        )

    # 3. Build the real LangGraph workflow.
    graph = build_graph(
        research=traced_research,
        developer=traced_developer,
        qa=traced_qa,
        release=unexpected_release,
    )

    # 4. Execute Research -> Developer -> QA.
    final_state = graph.invoke(initial_state)

    # 5. Verify stage routing.
    assert executed_stages == [
        "research",
        "developer",
        "qa",
    ], f"Unexpected stage sequence: {executed_stages}"

    assert (
        final_state["research_result"].outcome
        == StageOutcome.COMPLETED
    ), final_state["research_result"].errors

    assert (
        final_state["development_result"].outcome
        == StageOutcome.COMPLETED
    ), final_state["development_result"].errors

    assert final_state["qa_result"].verdict == QAVerdict.BLOCKED
    assert final_state["status"] == RunStatus.BLOCKED

    assert "release_result" not in final_state

    # 6. Inspect Research artifacts.
    research_dir = tmp_path / task.run_id / "research"

    specification_path = (
        research_dir / "engineering_spec.json"
    )

    assert specification_path.is_file()

    print("\n=== GENERATED ENGINEERING SPEC ===")
    print(
        specification_path.read_text(encoding="utf-8")
    )

    # 7. Inspect Developer artifacts.
    attempt_dir = (
        tmp_path
        / task.run_id
        / "development"
        / "attempt-0"
    )

    manifest_path = attempt_dir / "manifest.json"
    staging_dir = attempt_dir / "staging"

    assert manifest_path.is_file()
    assert staging_dir.is_dir()

    manifest = ImplementationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest.files, "No implementation files generated."

    print("\n=== IMPLEMENTATION MANIFEST ===")
    print(manifest.model_dump_json(indent=2))

    print("\n=== GENERATED IMPLEMENTATION FILES ===")

    for file in manifest.files:
        staged_file = staging_dir / file.path

        assert staged_file.is_file()

        print(f"\n--- {file.path} ---")
        print(staged_file.read_text(encoding="utf-8"))

    # 8. Check that Research's explicit filenames were followed.
    assert (staging_dir / "even_utils.py").is_file()
    assert (
        staging_dir / "tests" / "test_even_utils.py"
    ).is_file()

    print("\n=== FINAL WORKFLOW RESULT ===")
    print("Executed stages:", executed_stages)
    print("Workflow status:", final_state["status"])
    print("Run ID:", task.run_id)
    print("Artifacts root:", tmp_path / task.run_id)
