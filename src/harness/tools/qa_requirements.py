
"""Deterministic requirement coverage validation for QA."""

from pydantic import BaseModel, ConfigDict

from harness.models import (
    EngineeringSpec,
    ImplementationManifest,
    QARequirementAssessment,
)


class RequirementCoverageError(Exception):
    """The QA assessment does not cover the original specification."""


class RequirementItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    requirement: str


def build_requirement_items(
    specification: EngineeringSpec,
) -> list[RequirementItem]:
    """Create stable IDs for requirements and acceptance criteria."""

    items = []

    for index, requirement in enumerate(
        specification.requirements, start=1
    ):
        items.append(
            RequirementItem(
                requirement_id=f"REQ-{index}",
                requirement=requirement,
            )
        )

    for index, criterion in enumerate(
        specification.acceptance_criteria, start=1
    ):
        items.append(
            RequirementItem(
                requirement_id=f"AC-{index}",
                requirement=criterion,
            )
        )

    return items


def validate_requirement_coverage(
    assessment: QARequirementAssessment,
    specification: EngineeringSpec,
    manifest: ImplementationManifest,
) -> None:
    """Reject omitted, duplicated, invented, or altered requirements.

    File references must point to files listed in the Developer manifest.
    This checks completeness and consistency, not whether the LLM's
    judgments are factually correct.
    """

    expected = {
        item.requirement_id: item.requirement
        for item in build_requirement_items(specification)
    }

    valid_paths = {record.path for record in manifest.files}
    found = set()

    for finding in assessment.findings:
        requirement_id = finding.requirement_id

        if requirement_id not in expected:
            raise RequirementCoverageError(
                f"Unknown requirement ID: {requirement_id}"
            )

        if requirement_id in found:
            raise RequirementCoverageError(
                f"Duplicate assessment: {requirement_id}"
            )

        found.add(requirement_id)

        if finding.requirement != expected[requirement_id]:
            raise RequirementCoverageError(
                f"Requirement text mismatch: {requirement_id}"
            )

        referenced_files = (
            finding.implementation_files + finding.test_files
        )

        unknown = set(referenced_files) - valid_paths

        if unknown:
            raise RequirementCoverageError(
                f"Unknown file references in {requirement_id}: "
                f"{sorted(unknown)}"
            )

    missing = set(expected) - found

    if missing:
        raise RequirementCoverageError(
            f"Missing requirement assessments: {sorted(missing)}"
        )
