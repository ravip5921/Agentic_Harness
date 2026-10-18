
"""Research stage of the generic agentic harness.

Responsibilities:
    - Understand the requested task and its requirements.
    - Inspect the target repository when provided.
    - Research relevant documentation and technical constraints.
    - Produce an engineering design specification.
    - Return a validated ResearchResult.
"""

from harness.models import ResearchResult, StageOutcome
from harness.state import HarnessState


def research(state: HarnessState) -> dict:
    """Execute the Research stage.

    Args:
        state: Current shared workflow state containing the task.

    Returns:
        Partial state update with a ResearchResult.
    """
    task = state["task"]

    result = ResearchResult(
        outcome=StageOutcome.BLOCKED,
        summary=f"Research not yet implemented for task: {task.task}",
        artifacts=[],
        errors=["Research implementation is not yet available."],
    )

    return {"research_result": result}
