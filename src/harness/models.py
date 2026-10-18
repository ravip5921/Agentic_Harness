"""Task-independent contracts for the four-stage harness."""
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StageOutcome(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class QAVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"


class ReleaseStatus(str, Enum):
    STAGED = "staged"
    FAILED = "failed"
    BLOCKED = "blocked"


class RunStatus(str, Enum):
    RUNNING = "running"
    FAILED = "failed"
    BLOCKED = "blocked"
    STAGED = "staged"


class TaskRequest(StrictModel):
    """Domain-neutral task; params are data, not executable instructions."""
    task: str = Field(min_length=1)
    repository_path: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    max_dev_revisions: int = Field(default=2, ge=0, le=10)
    run_id: str = Field(default_factory=lambda: uuid4().hex)


class StageResult(StrictModel):
    outcome: StageOutcome
    summary: str = ""
    artifacts: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ResearchResult(StageResult):
    pass


class DevelopmentResult(StageResult):
    pass


class QAResult(StrictModel):
    verdict: QAVerdict
    summary: str = ""
    report_path: str | None = None
    feedback: list[str] = Field(default_factory=list)


class ReleaseResult(StrictModel):
    status: ReleaseStatus
    summary: str = ""
    change_ref: str | None = None
    errors: list[str] = Field(default_factory=list)
