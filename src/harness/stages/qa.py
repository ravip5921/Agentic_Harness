
"""QA stage: verify, test, and report on Developer implementations.

The QA stage:
- Validates Developer artifacts and manifest integrity.
- Prepares a disposable test workspace.
- Executes supported checks through the sandbox executor.
- Writes a structured QA report.
- Returns a QAResult for LangGraph routing.

Generated code is never executed directly by this module.
"""

import os
import re
import tempfile

from collections.abc import Callable
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from harness.models import (
    QAFeedback,
    QACheckResult,
    QACheckStatus,
    QACheckType,
    QAReport,
    QAResult,
    QAVerdict,
    QARequirementAssessment,
    RequirementStatus,
    ImplementationManifest,
)
from harness.state import HarnessState

from harness.tools.artifacts import (
    ArtifactError,
    load_development_artifacts,
)
from harness.tools.qa_workspace import (
    QAWorkspaceError,
    prepare_qa_workspace,
)
from harness.tools.qa_context import (
    QAContextError,
    load_qa_implementation_context,
)

from harness.tools.qa_requirements import (
    validate_requirement_coverage,
    RequirementCoverageError,
)
from harness.tools.executor import (
    QAExecutionError,
    execute_check,
)
from harness.tools.specifications import (
    SpecificationError,
    load_engineering_spec,
)
from harness.agents.qa_agent import (
    GroqQAAgent,
    QAGenerationError,
)


class QAStageError(Exception):
    """Raised when QA orchestration cannot complete."""

def collect_file_excerpts(
    manifest: ImplementationManifest,
    workspace: Path,
    max_files: int = 10,
    max_chars_per_file: int = 8000,
) -> dict[str, str]:
    """Collect bounded implementation excerpts for QA analysis."""

    excerpts = {}

    for record in manifest.files[:max_files]:
        path = workspace / record.path

        if not path.is_file() or path.is_symlink():
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue

        excerpts[record.path] = content[:max_chars_per_file]

    return excerpts

def determine_verdict(
    checks: list[QACheckResult],
) -> QAVerdict:
    """Derive an overall verdict from executed checks."""

    if not checks:
        return QAVerdict.BLOCKED

    # Infrastructure or unavailable verification prevents QA passing.
    if any(
        check.status == QACheckStatus.BLOCKED
        for check in checks
    ):
        return QAVerdict.BLOCKED

    # Actual check failures are implementation failures.
    if any(
        check.status == QACheckStatus.FAILED
        for check in checks
    ):
        return QAVerdict.FAIL

    if any(
        check.status == QACheckStatus.SKIPPED
        for check in checks
    ):
        return QAVerdict.BLOCKED

    required = {
        QACheckType.SYNTAX,
        QACheckType.UNIT_TEST,
    }

    passed_types = {
        check.check_type
        for check in checks
        if check.status == QACheckStatus.PASSED
    }

    if not required.issubset(passed_types):
        return QAVerdict.BLOCKED

    return QAVerdict.PASS


def build_feedback(
    checks: list[QACheckResult],
    revision_count: int,
    requirements_assessment: QARequirementAssessment | None = None,
) -> QAFeedback | None:
    """Build deterministic feedback from tests and requirements."""

    recommendations = []

    failed_checks = [
        check
        for check in checks
        if check.status == QACheckStatus.FAILED
    ]

    for check in failed_checks:
        output = "\n".join(
            part.strip()
            for part in (check.stdout, check.stderr)
            if part and part.strip()
        )

        recommendations.append(
            f"{check.check_type.value} failed "
            f"({check.name}):\n"
            f"{output[-4000:] if output else check.message}"
        )

    if requirements_assessment is not None:
        for finding in requirements_assessment.findings:
            if finding.status != RequirementStatus.NOT_SATISFIED:
                continue

            recommendations.append(
                f"{finding.requirement_id}: "
                f"{finding.requirement}\n"
                f"Finding: {finding.explanation}"
            )

            for gap in finding.missing_coverage:
                recommendations.append(
                    f"{finding.requirement_id} missing coverage: {gap}"
                )

    if not recommendations:
        return None

    return QAFeedback(
        summary=(
            f"{len(failed_checks)} execution check(s) failed; "
            f"{len(recommendations)} feedback item(s)."
        ),
        development_attempt=revision_count,
        recommendations=recommendations,
    )


def combine_qa_verdicts(
    execution_verdict: QAVerdict,
    requirements_assessment: QARequirementAssessment | None,
) -> QAVerdict:
    """Combine test results and requirement findings.

    Execution failures remain FAIL.
    Infrastructure blocks remain BLOCKED.
    Requirements assessment is mandatory for PASS.
    """

    if execution_verdict == QAVerdict.BLOCKED:
        return QAVerdict.BLOCKED

    if execution_verdict == QAVerdict.FAIL:
        return QAVerdict.FAIL

    if requirements_assessment is None:
        return QAVerdict.BLOCKED

    statuses = {
        finding.status
        for finding in requirements_assessment.findings
    }

    if RequirementStatus.NOT_SATISFIED in statuses:
        return QAVerdict.FAIL

    if RequirementStatus.UNCERTAIN in statuses:
        return QAVerdict.BLOCKED

    if statuses == {RequirementStatus.SATISFIED}:
        return QAVerdict.PASS

    return QAVerdict.BLOCKED

def write_qa_report(
    report: QAReport,
    report_path: Path,
) -> None:
    """Atomically write the QA report."""

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".qa-report-",
            suffix=".json",
            dir=report_path.parent,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(
                report.model_dump_json(indent=2)
            )

        os.replace(
            temporary_path,
            report_path,
        )
        temporary_path = None

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def make_qa_stage(
    artifact_loader=load_development_artifacts,
    workspace_preparer=prepare_qa_workspace,
    test_executor=execute_check,
    report_writer=write_qa_report,
    specification_loader=load_engineering_spec,
    context_loader=load_qa_implementation_context,
    requirements_assessor=None,
    feedback_generator=None,
    runs_dir: Path = Path("runs"),
):
    """Construct a LangGraph-compatible QA stage.

    Dependencies are injectable for isolated unit testing.
    """

    def qa(state: HarnessState) -> dict:
        task = state.get("task")
        development = state.get("development_result")

        if task is None or development is None:
            return {
                "qa_result": QAResult(
                    verdict=QAVerdict.BLOCKED,
                    summary="QA prerequisites are missing.",
                )
            }

        revision_count = state.get("revision_count", 0)

        if (
            isinstance(revision_count, bool)
            or not isinstance(revision_count, int)
            or revision_count < 0
            or revision_count > task.max_dev_revisions
        ):
            return {
                "qa_result": QAResult(
                    verdict=QAVerdict.BLOCKED,
                    summary="Invalid development revision counter.",
                )
            }

        if not re.fullmatch(r"[a-f0-9]{32}", task.run_id):
            return {
                "qa_result": QAResult(
                    verdict=QAVerdict.BLOCKED,
                    summary="Invalid run identifier.",
                )
            }

        runs_root = Path(runs_dir).resolve()

        attempt_dir = (
            runs_root
            / task.run_id
            / "qa"
            / f"attempt-{revision_count}"
        )

        workspace_dir = attempt_dir / "workspace"
        report_path = attempt_dir / "test_report.json"

        # Never silently overwrite an earlier QA attempt.
        if attempt_dir.exists() or attempt_dir.is_symlink():
            return {
                "qa_result": QAResult(
                    verdict=QAVerdict.BLOCKED,
                    summary="QA attempt already exists.",
                )
            }

        try:
            # 1. Verify Developer's manifest and staged files.
            manifest, staging_dir = artifact_loader(
                state,
                runs_dir=runs_root,
            )

            # 2. Prepare a disposable implementation workspace.
            workspace = workspace_preparer(
                manifest=manifest,
                staging_dir=staging_dir,
                workspace_dir=workspace_dir,
                repository_path=task.repository_path,
            )

            # 3. Execute controlled checks.
            checks = []

            for check_type in (
                QACheckType.SYNTAX,
                QACheckType.UNIT_TEST,
            ):
                result = test_executor(
                    check_type,
                    workspace,
                )

                if not isinstance(result, QACheckResult):
                    raise QAStageError(
                        "Executor returned an invalid result."
                    )

                checks.append(result)

                # Do not continue after an infrastructure block.
                if result.status == QACheckStatus.BLOCKED:
                    break

            # 4. Determine execution verdict.
            execution_verdict = determine_verdict(checks)

            requirements_assessment = None
            assessment_error = None

            # Requirements evaluation is necessary for PASS.
            # We also run it on FAIL when possible to identify additional defects.
            if (
                requirements_assessor is not None
                and execution_verdict != QAVerdict.BLOCKED
            ):
                try:
                    specification = specification_loader(
                        state,
                        runs_dir=runs_root,
                    )

                    implementation = context_loader(
                        state,
                        runs_dir=runs_root,
                    )

                    requirements_assessment = requirements_assessor(
                        task,
                        specification,
                        manifest,
                        implementation,
                        checks,
                    )

                    if not isinstance(
                        requirements_assessment,
                        QARequirementAssessment,
                    ):
                        raise QAStageError(
                            "Requirements assessor returned an invalid type."
                        )

                    # Validate even injected assessors, not just GroqQAAgent.
                    validate_requirement_coverage(
                        requirements_assessment,
                        specification,
                        manifest,
                    )

                except Exception as exc:
                    logger.exception(
                        "Requirements assessment failed for run %s, attempt %d",
                        task.run_id,
                        revision_count,
                    )
                    assessment_error = str(exc)
                    requirements_assessment = None

            # 5. Combine execution and requirements findings.
            verdict = combine_qa_verdicts(
                execution_verdict,
                requirements_assessment,
            )

            # 6. Build deterministic fallback feedback.
            feedback = (
                build_feedback(
                    checks,
                    revision_count,
                    requirements_assessment,
                )
                if verdict == QAVerdict.FAIL
                else None
            )

            # Optional LLM-assisted repair explanation.
            if verdict == QAVerdict.FAIL and feedback_generator is not None:
                try:
                    # Reuse the specification if the assessment loaded it.
                    # Otherwise obtain it here for failure analysis.
                    specification = specification_loader(
                        state,
                        runs_dir=runs_root,
                    )

                    excerpts = collect_file_excerpts(
                        manifest,
                        workspace,
                    )

                    generated_feedback = feedback_generator(
                        specification,
                        manifest,
                        checks,
                        revision_count,
                        excerpts,
                    )

                    if not isinstance(generated_feedback, QAFeedback):
                        raise QAGenerationError(
                            "QA Agent must return QAFeedback."
                        )

                    if generated_feedback.development_attempt != revision_count:
                        raise QAGenerationError(
                            "Feedback references the wrong development attempt."
                        )

                    # Preserve deterministic findings even when the LLM provides
                    # additional analysis or overlooks a reported failure.
                    if feedback is not None:
                        generated_feedback.recommendations = list(
                            dict.fromkeys(
                                feedback.recommendations
                                + generated_feedback.recommendations
                            )
                        )

                    feedback = generated_feedback

                except Exception:
                    logger.exception(
                        "Feedback generation failed for run %s, attempt %d. "
                        "Using deterministic feedback.",
                        task.run_id,
                        revision_count,
                    )

            # 7. Produce summary.
            if verdict == QAVerdict.PASS:
                summary = (
                    "Execution checks passed and all requirements "
                    "were assessed as satisfied."
                )

            elif verdict == QAVerdict.FAIL:
                summary = (
                    "Implementation failed execution or requirements QA."
                )

            else:
                summary = "QA verification was blocked."

                if assessment_error:
                    summary += (
                        f" Requirements assessment unavailable: "
                        f"{assessment_error}"
                    )


            # 5. Persist a structured QA report.
            report = QAReport(
                run_id=task.run_id,
                revision_count=revision_count,
                verdict=verdict,
                summary=summary,
                checks=checks,
                requirements_assessment=requirements_assessment,
                feedback=feedback,
                manifest_path=str(
                    runs_root
                    / task.run_id
                    / "development"
                    / f"attempt-{revision_count}"
                    / "manifest.json"
                ),
                workspace_path=str(workspace),
            )

            report_writer(
                report,
                report_path,
            )

            # 6. Return the result for LangGraph routing.
            return {
                "qa_result": QAResult(
                    verdict=verdict,
                    summary=summary,
                    report_path=str(report_path),
                    feedback=feedback,
                )
            }

        except (
            ArtifactError,
            QAWorkspaceError,
            QAExecutionError,
            QAStageError,
            OSError,
            ValueError,
        ) as exc:
            return {
                "qa_result": QAResult(
                    verdict=QAVerdict.BLOCKED,
                    summary=f"QA could not complete: {exc}",
                )
            }

    return qa


def qa(state: HarnessState) -> dict:
    """Default Groq-assisted QA stage."""

    agent = GroqQAAgent()

    node = make_qa_stage(
        requirements_assessor=agent.assess_requirements,
        feedback_generator=agent.generate,
    )

    return node(state)