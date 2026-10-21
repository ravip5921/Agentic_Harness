
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


from harness.models import (
    QAFeedback,
    QACheckResult,
    QACheckStatus,
    QACheckType,
    QAReport,
    QAResult,
    QAVerdict,
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
from harness.tools.executor import (
    QAExecutionError,
    execute_check,
)
from harness.tools.specifications import (
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
) -> QAFeedback | None:
    """Generate initial structured feedback from failing checks.

    Detailed file and test attribution will be added when we
    implement richer test-result parsing.
    """

    failed = [
        check
        for check in checks
        if check.status == QACheckStatus.FAILED
    ]

    if not failed:
        return None

    recommendations = []

    for check in failed:
        # Preserve relevant output for the next Developer attempt.
        output = "\n".join(
            part.strip()
            for part in (check.stdout, check.stderr)
            if part and part.strip()
        )

        details = output[-4000:] if output else check.message

        recommendations.append(
            f"{check.check_type.value} check failed "
            f"({check.name}):\n{details}"
        )

    return QAFeedback(
        summary=f"{len(failed)} QA check(s) failed.",
        development_attempt=revision_count,
        recommendations=recommendations,
    )


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

            # 4. Determine verdict and construct feedback.
            verdict = determine_verdict(checks)

            feedback = None

            if verdict == QAVerdict.FAIL:
                # Deterministic fallback is always available.
                feedback = build_feedback(checks, revision_count)

                if feedback_generator is not None:
                    try:
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
                                "Feedback references the wrong attempt."
                            )

                        feedback = generated_feedback

                    except (Exception):
                        logger = logging.getLogger(__name__)

                        # Preserve test findings if the LLM fails.
                        # Groq failure must not change a genuine QA FAIL
                        # to BLOCKED or PASS.
                        logger.exception(
                            "QA feedback generation failed for run %s, attempt %d. "
                            "Using deterministic feedback instead.",
                            task.run_id,
                            revision_count,
                        )
                        pass
            if verdict == QAVerdict.PASS:
                summary = "All required QA checks passed."
            elif verdict == QAVerdict.FAIL:
                summary = "Implementation failed QA checks."
            else:
                summary = "QA verification was blocked."

            # 5. Persist a structured QA report.
            report = QAReport(
                run_id=task.run_id,
                revision_count=revision_count,
                verdict=verdict,
                summary=summary,
                checks=checks,
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
        feedback_generator=agent.generate,
    )

    return node(state)