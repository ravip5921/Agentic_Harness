"""Provider-independent workflow and deterministic gates.

Stage callables are injected; real agents are implemented separately.
No repo writes happen in this module. Approval is only a boolean gate at I0;
a trusted, resumable approval mechanism must be added before real release.
"""
from collections.abc import Callable
from typing import Literal

from langgraph.graph import StateGraph, START, END

from harness.models import QAVerdict, RunStatus, StageOutcome, ReleaseStatus
from harness.state import HarnessState

Stage = Callable[[HarnessState], dict]


def route_research(state: HarnessState) -> Literal["developer", "failed", "blocked"]:
    outcome = state["research_result"].outcome
    if outcome == StageOutcome.COMPLETED:
        return "developer"
    return "blocked" if outcome == StageOutcome.BLOCKED else "failed"


def route_development(state: HarnessState) -> Literal["qa", "failed", "blocked"]:
    outcome = state["development_result"].outcome
    if outcome == StageOutcome.COMPLETED:
        return "qa"
    return "blocked" if outcome == StageOutcome.BLOCKED else "failed"


def route_qa(state: HarnessState) -> Literal["approval", "retry", "failed", "blocked"]:
    verdict = state["qa_result"].verdict
    if verdict == QAVerdict.PASS:
        return "approval"
    if verdict == QAVerdict.BLOCKED:
        return "blocked"
    if state.get("revision_count", 0) < state["task"].max_dev_revisions:
        return "retry"
    return "failed"


def route_approval(state: HarnessState) -> Literal["release", "pending"]:
    return "release" if state.get("release_approved") is True else "pending"


def route_release(state: HarnessState) -> Literal["staged", "failed", "blocked"]:
    status = state["release_result"].status
    if status == ReleaseStatus.STAGED:
        return "staged"
    return "blocked" if status == ReleaseStatus.BLOCKED else "failed"


def retry(state: HarnessState) -> dict:
    return {"revision_count": state.get("revision_count", 0) + 1}


def end_with(status: RunStatus) -> Stage:
    return lambda _state: {"status": status}


def build_graph(*, research: Stage, developer: Stage, qa: Stage, release: Stage):
    """Build a graph with injected stage functions (nothing task-specific)."""
    graph = StateGraph(HarnessState)
    graph.add_node("research", research)
    graph.add_node("developer", developer)
    graph.add_node("qa", qa)
    graph.add_node("retry", retry)
    # Dedicated gate nodes do not perform side effects.
    graph.add_node("approval", lambda _state: {})
    graph.add_node("release", release)
    for name, status in [
        ("failed", RunStatus.FAILED),
        ("blocked", RunStatus.BLOCKED),
        ("pending", RunStatus.AWAITING_APPROVAL),
        ("staged", RunStatus.STAGED),
    ]:
        graph.add_node(name, end_with(status))
        graph.add_edge(name, END)

    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", route_research)
    graph.add_conditional_edges("developer", route_development)
    graph.add_conditional_edges("qa", route_qa)
    graph.add_edge("retry", "developer")
    graph.add_conditional_edges("approval", route_approval)
    graph.add_conditional_edges("release", route_release)
    return graph.compile()
