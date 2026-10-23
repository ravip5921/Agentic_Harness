"""Task-independent contracts for the four-stage harness."""
from __future__ import annotations

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

    feedback: QAFeedback | None = None

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


class PlannedFile(StrictModel):
    """Metadata describing one file to generate."""

    path: str = Field(min_length=1)
    operation: FileOperation
    description: str = Field(min_length=1)


class DevelopmentPlan(StrictModel):
    """Developer's file-generation plan and readiness assessment."""

    status: Literal["ready", "blocked"]
    reason: str = ""

    summary: str = ""
    files: list[PlannedFile] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_plan(self):
        if self.status == "ready":
            if not self.summary.strip():
                raise ValueError(
                    "A ready plan must contain a summary."
                )
            if not self.files:
                raise ValueError(
                    "A ready plan must contain planned files."
                )

        if self.status == "blocked":
            if not self.reason.strip():
                raise ValueError(
                    "A blocked plan must explain why."
                )
            if self.files:
                raise ValueError(
                    "A blocked plan must not propose files."
                )

        return self

        
class GeneratedFileContent(StrictModel):
    """Response from an individual file-generation request."""

    content: str = Field(min_length=1)


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


# Models for QA Stage


class QACheckType(str, Enum):
    """Category of verification performed by QA."""

    INTEGRITY = "integrity"
    DEPENDENCY_SETUP = "dependency_setup"
    SYNTAX = "syntax"
    UNIT_TEST = "unit_test"
    CONTRACT_TEST = "contract_test"
    INTEGRATION_TEST = "integration_test"


class QACheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class QACheckResult(StrictModel):
    """Result of one independently recorded QA check."""

    name: str = Field(min_length=1)
    check_type: QACheckType
    status: QACheckStatus

    command: list[str] = Field(default_factory=list)
    exit_code: int | None = None

    duration_seconds: float | None = Field(
        default=None,
        ge=0,
    )

    stdout: str = ""
    stderr: str = ""

    timed_out: bool = False
    message: str = ""


class QAReport(StrictModel):
    """Complete verification report for one development attempt."""

    run_id: str = Field(min_length=1)
    revision_count: int = Field(ge=0)

    verdict: QAVerdict
    summary: str = ""

    checks: list[QACheckResult] = Field(
        default_factory=list
    )

    feedback: QAFeedback | None = None

    manifest_path: str
    workspace_path: str | None = None
    requirements_assessment: QARequirementAssessment | None = None
    
    @model_validator(mode="after")
    def validate_verdict(self):
        # A passing report must contain executed, passing checks.
        if self.verdict == QAVerdict.PASS:
            if not self.checks:
                raise ValueError(
                    "QA cannot pass without verification checks."
                )

            if not any(
                check.check_type in {
                    QACheckType.UNIT_TEST,
                    QACheckType.CONTRACT_TEST,
                    QACheckType.INTEGRATION_TEST,
                }
                and check.status == QACheckStatus.PASSED
                for check in self.checks
            ):
                raise ValueError(
                    "QA PASS requires at least one passing test check."
                )

            if any(
                check.status != QACheckStatus.PASSED
                for check in self.checks
            ):
                raise ValueError(
                    "QA PASS cannot contain failed, skipped, or blocked checks."
                )

        return self

# Structured QA feedback models


class QAFeedbackCategory(str, Enum):
    SYNTAX = "syntax"
    TEST_FAILURE = "test_failure"
    CONTRACT = "contract"
    IMPLEMENTATION = "implementation"


class FileFeedback(StrictModel):
    """Actionable feedback associated with one implementation file."""

    path: str = Field(min_length=1)
    category: QAFeedbackCategory
    message: str = Field(min_length=1)

    line: int | None = Field(default=None, ge=1)
    related_test: str | None = None


class TestFeedback(StrictModel):
    """Details of a failing test."""

    test_name: str = Field(min_length=1)
    message: str = Field(min_length=1)

    related_files: list[str] = Field(default_factory=list)
    check_name: str | None = None


class QAFeedback(StrictModel):
    """Structured feedback for a development revision."""

    summary: str = Field(min_length=1)

    file_feedback: list[FileFeedback] = Field(
        default_factory=list
    )

    test_feedback: list[TestFeedback] = Field(
        default_factory=list
    )

    recommendations: list[str] = Field(
        default_factory=list
    )

    # Identifies the implementation the feedback refers to.
    development_attempt: int = Field(ge=0)


# Models for Developer revision context


class PreviousFile(StrictModel):
    """A previously generated implementation file."""

    path: str = Field(min_length=1)
    operation: FileOperation
    content: str


class PreviousImplementation(StrictModel):
    """Validated implementation from an earlier development attempt."""

    development_attempt: int = Field(ge=0)
    files: list[PreviousFile] = Field(min_length=1)


class RequirementStatus(str, Enum):
    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    UNCERTAIN = "uncertain"


class RequirementFinding(StrictModel):
    """Assessment of one requirement or acceptance criterion."""
    requirement_id: str = Field(min_length=1)

    requirement: str = Field(min_length=1)
    status: RequirementStatus

    explanation: str = Field(min_length=1)

    implementation_files: list[str] = Field(
        default_factory=list
    )

    test_files: list[str] = Field(
        default_factory=list
    )

    missing_coverage: list[str] = Field(
        default_factory=list
    )


class QARequirementAssessment(StrictModel):
    """LLM assessment of implementation and test completeness."""

    findings: list[RequirementFinding] = Field(
        min_length=1
    )

    overall_summary: str = Field(min_length=1)


# Models for Release Stage


class ReleaseManifest(StrictModel):
    """Description of a staged release."""

    run_id: str
    revision_count: int

    summary: str

    files: list[StagedFile]

    source_manifest_path: str
    qa_report_path: str

    repository_path: str | None = None
    release_directory: str


class ReleaseAssessment(StrictModel):
    """Advisory release documentation generated by the Release Agent."""

    summary: str = Field(min_length=1)
    changes: list[str] = Field(default_factory=list)
    testing_summary: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)
