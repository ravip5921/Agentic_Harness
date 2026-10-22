
import pytest

from harness.models import (
    EngineeringSpec,
    ImplementationManifest,
    FileOperation,
    StagedFile,
    QARequirementAssessment,
    RequirementFinding,
    RequirementStatus,
)
from harness.tools.qa_requirements import (
    build_requirement_items,
    validate_requirement_coverage,
    RequirementCoverageError,
)


def make_spec():
    return EngineeringSpec(
        title="Calculator",
        objective="Implement addition.",
        requirements=["Add integers.", "Support negative numbers."],
        technical_approach="Use Python.",
        implementation_steps=["Implement add()."],
        acceptance_criteria=["Return the correct sum."],
    )


def make_manifest():
    return ImplementationManifest(
        summary="Calculator implementation",
        files=[
            StagedFile(
                path="calculator.py",
                operation=FileOperation.CREATE,
                sha256="a" * 64,
            ),
        ],
    )


def make_assessment(spec, ids=None):
    items = build_requirement_items(spec)
    selected = [
        item for item in items
        if ids is None or item.requirement_id in ids
    ]

    return QARequirementAssessment(
        findings=[
            RequirementFinding(
                requirement_id=item.requirement_id,
                requirement=item.requirement,
                status=RequirementStatus.SATISFIED,
                explanation="Reviewed generated implementation.",
                implementation_files=["calculator.py"],
            )
            for item in selected
        ],
        overall_summary="Review completed.",
    )


def test_builds_requirement_and_criterion_ids():
    items = build_requirement_items(make_spec())

    assert [item.requirement_id for item in items] == [
        "REQ-1",
        "REQ-2",
        "AC-1",
    ]


def test_complete_assessment_is_accepted():
    spec = make_spec()
    assessment = make_assessment(spec)

    validate_requirement_coverage(
        assessment, spec, make_manifest()
    )


def test_missing_criterion_is_rejected():
    spec = make_spec()
    assessment = make_assessment(spec, {"REQ-1", "REQ-2"})

    with pytest.raises(
        RequirementCoverageError,
        match="Missing requirement assessments",
    ):
        validate_requirement_coverage(
            assessment, spec, make_manifest()
        )


def test_duplicate_finding_is_rejected():
    spec = make_spec()
    assessment = make_assessment(spec)
    assessment.findings.append(
        assessment.findings[0].model_copy(deep=True)
    )

    with pytest.raises(
        RequirementCoverageError, match="Duplicate"
    ):
        validate_requirement_coverage(
            assessment, spec, make_manifest()
        )


def test_modified_requirement_text_is_rejected():
    spec = make_spec()
    assessment = make_assessment(spec)
    assessment.findings[0].requirement = "Different requirement."

    with pytest.raises(
        RequirementCoverageError,
        match="text mismatch",
    ):
        validate_requirement_coverage(
            assessment, spec, make_manifest()
        )


def test_unknown_file_reference_is_rejected():
    spec = make_spec()
    assessment = make_assessment(spec)
    assessment.findings[0].implementation_files = ["unknown.py"]

    with pytest.raises(
        RequirementCoverageError,
        match="Unknown file references",
    ):
        validate_requirement_coverage(
            assessment, spec, make_manifest()
        )
