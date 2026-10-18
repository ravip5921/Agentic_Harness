
"""Provider-independent four-stage workflow with deterministic gates.

The harness coordinates Research, Developer, QA, and Release.
Stage callables are injected, keeping orchestration task-independent.

QA success routes directly to Release. The Release implementation is
responsible for enforcing configured release policies and permissions.

No repository modifications are performed by this module.
"""

from collections.abc import Callable
from typing import Literal

from langgraph.graph import StateGraph, START, END

from harness.models import (
    QAVerdict,
    RunStatus,
    StageOutcome,
    ReleaseStatus,
)
from harness.state import HarnessState


Stage = Callable[[HarnessState], dict]


def route_research(
    state: HarnessState,
) -> Literal["developer", "failed", "blocked"]:
    """Route based on the Research stage outcome."""
    outcome = state["research_result"].outcome

    if outcome == StageOutcome.COMPLETED:
        return "developer"

    return "blocked" if outcome == StageOutcome.BLOCKED else "failed"


def route_development(
    state: HarnessState,
) -> Literal["qa", "failed", "blocked"]:
    """Route based on the Developer stage outcome."""
    outcome = state["development_result"].outcome

    if outcome == StageOutcome.COMPLETED:
        return "qa"

    return "blocked" if outcome == StageOutcome.BLOCKED else "failed"


def route_qa(
    state: HarnessState,
) -> Literal["release", "retry", "failed", "blocked"]:
    """Route QA results, allowing bounded development retries."""
    verdict = state["qa_result"].verdict

    if verdict == QAVerdict.PASS:
        return "release"

    if verdict == QAVerdict.BLOCKED:
        return "blocked"

    if state.get("revision_count", 0) < state["task"].max_dev_revisions:
        return "retry"

    return "failed"


def route_release(
    state: HarnessState,
) -> Literal["staged", "failed", "blocked"]:
    """Route based on the Release stage result."""
    status = state["release_result"].status

    if status == ReleaseStatus.STAGED:
        return "staged"

    return "blocked" if status == ReleaseStatus.BLOCKED else "failed"


def retry(state: HarnessState) -> dict:
    """Increment the revision counter before returning to Developer."""
    return {
        "revision_count": state.get("revision_count", 0) + 1
    }


def end_with(status: RunStatus) -> Stage:
    """Create a terminal node that records the final run status."""
    return lambda _state: {"status": status}


def build_graph(
    *,
    research: Stage,
    developer: Stage,
    qa: Stage,
    release: Stage,
):
    """Build the generic four-stage workflow using injected callables."""

    graph = StateGraph(HarnessState)

    # Main workflow stages
    graph.add_node("research", research)
    graph.add_node("developer", developer)
    graph.add_node("qa", qa)
    graph.add_node("release", release)

    # Control node for bounded QA retries
    graph.add_node("retry", retry)

    # Terminal status nodes
    terminal_statuses = [
        ("failed", RunStatus.FAILED),
        ("blocked", RunStatus.BLOCKED),
        ("staged", RunStatus.STAGED),
    ]

    for name, status in terminal_statuses:
        graph.add_node(name, end_with(status))
        graph.add_edge(name, END)

    # Entry point
    graph.add_edge(START, "research")

    # Conditional stage transitions
    graph.add_conditional_edges("research", route_research)
    graph.add_conditional_edges("developer", route_development)
    graph.add_conditional_edges("qa", route_qa)
    graph.add_conditional_edges("release", route_release)

    # Retry failed QA through Developer
    graph.add_edge("retry", "developer")
    return graph.compile()
