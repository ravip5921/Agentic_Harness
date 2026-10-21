
"""Live integration test for the Developer stage.

This test makes a real Groq API call.

PowerShell:
    $env:RUN_LIVE_GROQ = "1"
    pytest tests/stages/test_developer_live.py -v -s

Live tests are skipped unless RUN_LIVE_GROQ=1.
"""

import os

import pytest
from dotenv import load_dotenv

from harness.agents.developer_agent import GroqDeveloperAgent
from harness.models import (
    EngineeringSpec,
    ImplementationManifest,
    ResearchResult,
    StageOutcome,
    TaskRequest,
)
from harness.stages.developer import make_developer_stage


load_dotenv()


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GROQ") != "1",
    reason="Live Groq tests are opt-in.",
)
def test_live_developer_generates_standalone_project(tmp_path):
    """Test the complete Developer stage using the real Groq API."""

    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not configured.")

    # ---------------------------------------------------------
    # 1. Create a realistic engineering specification
    # ---------------------------------------------------------

    specification = EngineeringSpec(
        title="Even Number Utility",
        objective=(
            "Create a standalone Python utility that determines "
            "whether an integer is even."
        ),
        requirements=[
            "Implement is_even(value: int) -> bool.",
            "Return True for even integers and False for odd integers.",
            "Support positive integers, negative integers, and zero.",
            "Use only the Python standard library.",
            "Include pytest-compatible unit tests.",
            "Create exactly two files: even_utils.py and "
            "tests/test_even_utils.py.",
        ],
        technical_approach=(
            "Implement is_even using the modulo operator. "
            "Use pytest for unit testing."
        ),
        implementation_steps=[
            "Create even_utils.py containing is_even.",
            "Create tests/test_even_utils.py.",
            "Add tests covering positive, negative, zero, "
            "even, and odd integers.",
        ],
        acceptance_criteria=[
            "is_even(2) returns True.",
            "is_even(3) returns False.",
            "is_even(0) returns True.",
            "is_even(-4) returns True.",
            "is_even(-5) returns False.",
            "The generated test file imports is_even from even_utils.",
        ],
    )

    # ---------------------------------------------------------
    # 2. Construct the task
    # ---------------------------------------------------------

    task = TaskRequest(
        task="Implement the Even Number Utility",
        max_dev_revisions=2,
    )

    # ---------------------------------------------------------
    # 3. Simulate completed Research artifacts
    # ---------------------------------------------------------

    research_dir = tmp_path / task.run_id / "research"
    research_dir.mkdir(parents=True)

    specification_path = (
        research_dir / "engineering_spec.json"
    )

    specification_path.write_text(
        specification.model_dump_json(indent=2),
        encoding="utf-8",
    )

    state = {
        "task": task,
        "revision_count": 0,
        "research_result": ResearchResult(
            outcome=StageOutcome.COMPLETED,
            summary="Engineering specification generated.",
            artifacts=[str(specification_path)],
        ),
    }

    # ---------------------------------------------------------
    # 4. Initialize the actual Groq Developer Agent
    # ---------------------------------------------------------

    agent = GroqDeveloperAgent()

    developer = make_developer_stage(
        generator=agent.generate,
        runs_dir=tmp_path,
    )

    # ---------------------------------------------------------
    # 5. Execute the Developer stage
    # ---------------------------------------------------------

    result = developer(state)["development_result"]

    print("\n=== Developer Result ===")
    print("Outcome:", result.outcome)
    print("Summary:", result.summary)
    print("Errors:", result.errors)
    print("Artifacts:", result.artifacts)

    assert result.outcome == StageOutcome.COMPLETED, result.errors

    # ---------------------------------------------------------
    # 6. Locate generated artifacts
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 7. Validate the implementation manifest
    # ---------------------------------------------------------

    manifest = ImplementationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    print("\n=== Implementation Manifest ===")
    print(manifest.model_dump_json(indent=2))

    assert len(manifest.files) >= 2

    # ---------------------------------------------------------
    # 8. Inspect the actual generated files
    # ---------------------------------------------------------

    print("\n=== Generated Files ===")

    for file in manifest.files:
        staged_path = staging_dir / file.path

        assert staged_path.is_file()

        print(f"\n--- {file.path} ---")
        print(
            staged_path.read_text(encoding="utf-8")
        )

    # ---------------------------------------------------------
    # 9. Check the expected project structure
    # ---------------------------------------------------------

    assert (staging_dir / "even_utils.py").is_file()

    assert (
        staging_dir / "tests" / "test_even_utils.py"
    ).is_file()

    print("\nLive Developer integration test completed successfully.")
