
"""Command-line interface for the generic agentic harness.

Accepts natural-language tasks, constructs a validated TaskRequest,
and executes the four-stage LangGraph workflow.
"""

import argparse
import json
import sys

from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from harness.graph import build_graph
from harness.models import RunStatus, TaskRequest

from harness.stages.research import make_research_stage
from harness.stages.developer import make_developer_stage
from harness.stages.qa import qa
from harness.stages.release import release

from harness.agents.research_agent import GroqResearchAgent
from harness.agents.developer_agent import GroqDeveloperAgent


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

    research_agent = GroqResearchAgent()
    developer_agent = GroqDeveloperAgent()

    research_stage = make_research_stage(
        generator=research_agent.generate,
        runs_dir=runs_dir,
    )

    developer_stage = make_developer_stage(
        generator=developer_agent.generate,
        runs_dir=runs_dir,
    )

    return build_graph(
        research=research_stage,
        developer=developer_stage,
        qa=qa,
        release=release,
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


def harness_cli(argv: list[str] | None = None) -> int:
    """Parse CLI input, execute the graph, and return an exit code."""

    load_dotenv()

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

        final_state = workflow.invoke(initial_state)

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