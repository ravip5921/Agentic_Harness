
"""Command-line interface for the generic agentic harness.

Accepts natural-language tasks, constructs a validated TaskRequest,
and executes the four-stage LangGraph workflow.
"""

import argparse
import json
import sys

from pathlib import Path

from pydantic import ValidationError

from harness.config import load_environment
from harness.graph import build_graph
from harness.models import RunStatus, TaskRequest
from harness.tools.progress import ProgressRenderer

from harness.stages.research import make_research_stage
from harness.stages.developer import make_developer_stage
from harness.stages.qa import (
    qa,
    make_qa_stage,
)
from harness.stages.release import (
    release,
    make_release_stage,
)

from harness.agents.research_agent import ResearchAgent
from harness.agents.developer_agent import DeveloperAgent
from harness.agents.qa_agent import QAAgent
from harness.agents.release_agent import ReleaseAgent


def create_parser() -> argparse.ArgumentParser:
    """Configure command-line arguments."""

    parser = argparse.ArgumentParser(
        prog="harness",
        description=(
            "Generic agentic software-development harness "
            "with Research, Developer, QA, and Release stages."
        ),
    )

    parser.add_argument(
        "task",
        type=str,
        help="Natural-language description of the requested task.",
    )

    parser.add_argument(
        "--repository",
        type=Path,
        default=None,
        help="Optional path to an existing target repository.",
    )

    parser.add_argument(
        "--documentation-url",
        action="append",
        default=[],
        help=(
            "Optional documentation URL. "
            "Can be specified multiple times."
        ),
    )

    parser.add_argument(
        "--max-dev-revisions",
        type=int,
        default=2,
        help="Maximum QA-triggered Developer retries (default: 2).",
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory for generated run artifacts.",
    )

    return parser


def create_task(args: argparse.Namespace) -> TaskRequest:
    """Convert CLI input into a validated TaskRequest."""

    repository_path = None

    if args.repository is not None:
        repository = args.repository.expanduser().resolve()

        if not repository.is_dir():
            raise ValueError(
                f"Repository directory does not exist: {repository}"
            )

        repository_path = str(repository)

    parameters = {}

    if args.documentation_url:
        parameters["documentation_urls"] = args.documentation_url

    return TaskRequest(
        task=args.task,
        repository_path=repository_path,
        parameters=parameters,
        max_dev_revisions=args.max_dev_revisions,
    )


def create_workflow(runs_dir: Path):
    """Initialize agents and assemble the LangGraph workflow."""

    research_agent = ResearchAgent()
    developer_agent = DeveloperAgent()

    research_stage = make_research_stage(
        generator=research_agent.generate,
        runs_dir=runs_dir,
    )

    developer_stage = make_developer_stage(
        generator=developer_agent.generate,
        runs_dir=runs_dir,
    )

    qa_agent = QAAgent()

    qa_stage = make_qa_stage(
        requirements_assessor=qa_agent.assess_requirements,
        feedback_generator=qa_agent.generate,
        runs_dir=runs_dir,
    )

    def generate_release_notes(task, manifest, qa_report):
        return ReleaseAgent().generate(
            task, manifest, qa_report
        )

    release_stage = make_release_stage(
        release_generator=generate_release_notes,
        runs_dir=runs_dir,
    )
    return build_graph(
        research=research_stage,
        developer=developer_stage,
        qa=qa_stage,
        release=release_stage,
    )


def display_result(state: dict) -> None:
    """Print the final workflow status and available artifacts."""

    task = state["task"]

    print("\n=== Agentic Harness Result ===")
    print(f"Run ID: {task.run_id}")
    print(f"Status: {state['status'].value}")
    print(f"Development revisions: {state.get('revision_count', 0)}")

    stages = [
        ("Research", "research_result"),
        ("Developer", "development_result"),
        ("QA", "qa_result"),
        ("Release", "release_result"),
    ]

    for stage_name, key in stages:
        result = state.get(key)

        if result is None:
            print(f"\n{stage_name}: Not executed")
            continue

        print(f"\n{stage_name}:")
        print(json.dumps(result.model_dump(mode="json"), indent=2))


def stream_workflow(workflow, initial_state: dict, progress: ProgressRenderer) -> dict:
    """Render live events and collect the same complete state as invoke().

    `updates` contains partial node results. `values` provides LangGraph's
    canonical accumulated state, including retry counters and terminal status.
    Neither custom events nor presentation data are added to workflow state.
    """
    final_state = dict(initial_state)
    for mode, payload in workflow.stream(
        initial_state,
        stream_mode=["custom", "updates", "values"],
    ):
        if mode == "values":
            final_state = payload
        elif mode == "custom":
            progress.event(payload)
        elif mode == "updates":
            progress.update(payload, final_state)
    return final_state


def harness_cli(argv: list[str] | None = None) -> int:
    """Parse CLI input, execute the graph, and return an exit code."""

    load_environment()

    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        task = create_task(args)

        runs_dir = args.runs_dir.expanduser().resolve()
        runs_dir.mkdir(parents=True, exist_ok=True)

        workflow = create_workflow(runs_dir)

        initial_state = {
            "task": task,
            "revision_count": 0,
            "status": RunStatus.RUNNING,
        }

        print(f"Starting harness run: {task.run_id}")

        with ProgressRenderer() as progress:
            final_state = stream_workflow(workflow, initial_state, progress)
            progress.finish(final_state["status"])

        display_result(final_state)

        status = final_state["status"]

        if status == RunStatus.STAGED:
            return 0

        if status == RunStatus.BLOCKED:
            return 2

        return 1

    except (ValidationError, ValueError) as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 1

    except Exception as exc:
        print(f"Harness execution failed: {exc}", file=sys.stderr)
        return 1
