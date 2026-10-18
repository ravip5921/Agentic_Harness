"""LangGraph state schema. Nodes return partial updates to this state."""
from typing import TypedDict

from harness.models import (
    DevelopmentResult, QAResult, ReleaseResult, ResearchResult,
    RunStatus, TaskRequest,
)


class HarnessState(TypedDict, total=False):
    task: TaskRequest
    research_result: ResearchResult
    development_result: DevelopmentResult
    qa_result: QAResult
    release_result: ReleaseResult
    revision_count: int  # Number of QA-triggered development retries, not initial attempts
    status: RunStatus
