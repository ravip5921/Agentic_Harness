"""Task-independent contracts for the four-stage harness."""
from enum import Enum
from uuid import uuid4

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


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


# Models for Research Stage

class EvidenceSource(StrictModel):
    """One retrieved, traceable source of research evidence."""

    source_id: str = Field(min_length=1)
    url: str
    title: str = ""
    retrieved_at: str
    content_sha256: str
    excerpt: str = ""


class EngineeringSpec(StrictModel):
    """Task-independent implementation specification."""

    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)

    requirements: list[str] = Field(min_length=1)
    repository_context: str = ""

    technical_approach: str = Field(min_length=1)
    implementation_steps: list[str] = Field(min_length=1)
    acceptance_criteria: list[str] = Field(min_length=1)

    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class ResearchAssessment(StrictModel):
    """Model assessment before accepting a research specification."""

    status: Literal["ready", "blocked"]
    reason: str = ""
    specification: EngineeringSpec | None = None

    @model_validator(mode="after")
    def validate_assessment(self):
        if self.status == "ready" and self.specification is None:
            raise ValueError(
                "A ready assessment must contain a specification."
            )

        if self.status == "blocked" and not self.reason.strip():
            raise ValueError(
                "A blocked assessment must explain why."
            )

        return self

# Models for Developer Stage


class FileOperation(str, Enum):
    CREATE = "create"
    MODIFY = "modify"


class GeneratedFile(StrictModel):
    """A single file proposed by the Developer Agent."""

    path: str = Field(min_length=1)
    operation: FileOperation
    content: str
    description: str = ""


class StagedFile(StrictModel):
    path: str
    operation: FileOperation
    sha256: str
    original_sha256: str | None = None


class ImplementationManifest(StrictModel):
    summary: str = Field(min_length=1)
    files: list[StagedFile] = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DevelopmentAssessment(StrictModel):
    """Developer Agent's structured generation result."""

    status: Literal["ready", "blocked"]

    reason: str = ""
    files: list[GeneratedFile] = Field(default_factory=list)

    summary: str = ""
    dependencies: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_assessment(self):
        if self.status == "ready" and not self.files:
            raise ValueError(
                "A ready assessment must contain generated files."
            )

        if self.status == "blocked" and not self.reason.strip():
            raise ValueError(
                "A blocked assessment must explain why."
            )

        return self
