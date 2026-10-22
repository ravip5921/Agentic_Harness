
"""Validate QA-approved artifacts before Release.

Release must not trust an in-memory PASS verdict alone.

This module verifies:
- QA verdict and persisted report.
- Current run and development attempt.
- Required execution checks.
- Complete requirements assessment.
- Developer manifest and staged-file integrity.
- QA workspace consistency with the Developer manifest.

It does not modify repositories or publish release artifacts.
"""

import hashlib
import os
import re
import stat

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from harness.models import (
    ImplementationManifest,
    QACheckStatus,
    QACheckType,
    QAReport,
    QAVerdict,
    RequirementStatus,
    StageOutcome,
)
from harness.state import HarnessState

from harness.tools.artifacts import (
    ArtifactError,
    load_development_artifacts,
)
from harness.tools.filesystem import (
    MAX_FILE_BYTES,
    validate_relative_path,
)
from harness.tools.qa_requirements import (
    RequirementCoverageError,
    validate_requirement_coverage,
)
from harness.tools.specifications import (
    load_engineering_spec,
    SpecificationError,
)


MAX_QA_REPORT_BYTES = 2_000_000


class ReleaseArtifactError(Exception):
    """Release prerequisites or artifacts could not be verified."""


@dataclass(frozen=True)
class ReleaseArtifacts:
    """Validated inputs for preparing a release candidate."""

    manifest: ImplementationManifest
    staging_dir: Path
    qa_report: QAReport
    qa_workspace: Path
    manifest_path: Path
    qa_report_path: Path


def _check_no_symlinks(path: Path) -> None:
    """Reject symlinks in a path and its ancestors."""

    current = path.absolute()

    while True:
        if current.is_symlink():
            raise ReleaseArtifactError(
                f"Symlinked release artifact path: {current}"
            )

        if current == current.parent:
            break

        current = current.parent


def _read_regular_file(
    path: Path,
    max_bytes: int,
) -> bytes:
    """Read a bounded regular file without following final symlinks."""

    _check_no_symlinks(path)

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)

    fd = os.open(path, flags)

    try:
        info = os.fstat(fd)

        if not stat.S_ISREG(info.st_mode):
            raise ReleaseArtifactError(
                f"Not a regular file: {path}"
            )

        if info.st_size > max_bytes:
            raise ReleaseArtifactError(
                f"File exceeds size limit: {path}"
            )

        data = os.read(fd, max_bytes + 1)

        if len(data) > max_bytes:
            raise ReleaseArtifactError(
                f"File exceeds size limit: {path}"
            )

        return data

    finally:
        os.close(fd)


def _validate_execution_checks(
    report: QAReport,
) -> None:
    """Verify that mandatory execution checks passed."""

    required = {
        QACheckType.SYNTAX,
        QACheckType.UNIT_TEST,
    }

    passed = {
        check.check_type
        for check in report.checks
        if (
            check.status == QACheckStatus.PASSED
            and check.exit_code == 0
            and not check.timed_out
        )
    }

    if not required.issubset(passed):
        missing = required - passed

        raise ReleaseArtifactError(
            "QA report lacks successful required checks: "
            f"{sorted(item.value for item in missing)}"
        )

    if any(
        check.status != QACheckStatus.PASSED
        or check.exit_code != 0
        or check.timed_out
        for check in report.checks
    ):
        raise ReleaseArtifactError(
            "QA report contains unsuccessful execution checks."
        )


def _validate_requirements(
    report: QAReport,
    state: HarnessState,
    manifest: ImplementationManifest,
    runs_root: Path,
) -> None:
    """Verify complete, satisfied requirements assessment."""

    assessment = report.requirements_assessment

    if assessment is None:
        raise ReleaseArtifactError(
            "QA report has no requirements assessment."
        )

    specification = load_engineering_spec(
        state,
        runs_dir=runs_root,
    )

    validate_requirement_coverage(
        assessment,
        specification,
        manifest,
    )

    if any(
        finding.status != RequirementStatus.SATISFIED
        for finding in assessment.findings
    ):
        raise ReleaseArtifactError(
            "Not all requirements were assessed as satisfied."
        )


def _validate_workspace(
    workspace: Path,
    manifest: ImplementationManifest,
) -> None:
    """Verify that QA's tested workspace still contains staged files.

    This validates generated-file content, including MODIFY replacements.
    Unchanged repository files are not independently attested here.
    """

    _check_no_symlinks(workspace)

    if not workspace.is_dir():
        raise ReleaseArtifactError(
            "QA workspace is missing."
        )

    for record in manifest.files:
        relative = validate_relative_path(record.path)

        candidate = workspace / relative

        data = _read_regular_file(
            candidate,
            MAX_FILE_BYTES,
        )

        actual_hash = hashlib.sha256(data).hexdigest()

        if actual_hash != record.sha256:
            raise ReleaseArtifactError(
                "QA workspace differs from approved "
                f"implementation: {record.path}"
            )


def load_release_artifacts(
    state: HarnessState,
    runs_dir: Path = Path("runs"),
) -> ReleaseArtifacts:
    """Load and verify all artifacts necessary for Release.

    Returns:
        ReleaseArtifacts containing validated Developer and QA inputs.

    Raises:
        ReleaseArtifactError when any prerequisite is invalid.
    """

    task = state.get("task")
    development = state.get("development_result")
    qa_result = state.get("qa_result")

    if task is None:
        raise ReleaseArtifactError(
            "Task is missing from workflow state."
        )

    if development is None:
        raise ReleaseArtifactError(
            "Development result is missing."
        )

    if development.outcome != StageOutcome.COMPLETED:
        raise ReleaseArtifactError(
            "Release requires completed Development."
        )

    if qa_result is None or qa_result.verdict != QAVerdict.PASS:
        raise ReleaseArtifactError(
            "Release requires a QA PASS verdict."
        )

    if not re.fullmatch(r"[a-f0-9]{32}", task.run_id):
        raise ReleaseArtifactError("Invalid run ID.")

    revision = state.get("revision_count", 0)

    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
        or revision > task.max_dev_revisions
    ):
        raise ReleaseArtifactError(
            "Invalid development revision counter."
        )

    runs_root = Path(runs_dir).absolute()

    development_dir = (
        runs_root
        / task.run_id
        / "development"
        / f"attempt-{revision}"
    )

    qa_dir = (
        runs_root
        / task.run_id
        / "qa"
        / f"attempt-{revision}"
    )

    manifest_path = development_dir / "manifest.json"
    expected_workspace = qa_dir / "workspace"
    report_path = qa_dir / "test_report.json"

    try:
        # 1. Validate the exact persisted QA report path.
        if qa_result.report_path is None:
            raise ReleaseArtifactError(
                "QA report path is missing."
            )

        if Path(qa_result.report_path).absolute() != report_path:
            raise ReleaseArtifactError(
                "QA result references an unexpected report."
            )

        report_data = _read_regular_file(
            report_path,
            MAX_QA_REPORT_BYTES,
        )

        report = QAReport.model_validate_json(
            report_data
        )

        # 2. Check run and attempt identity.
        if report.run_id != task.run_id:
            raise ReleaseArtifactError(
                "QA report belongs to a different run."
            )

        if report.revision_count != revision:
            raise ReleaseArtifactError(
                "QA report belongs to a different development attempt."
            )

        if report.verdict != QAVerdict.PASS:
            raise ReleaseArtifactError(
                "Persisted QA report does not indicate PASS."
            )

        if (
            report.manifest_path is None
            or Path(report.manifest_path).absolute()
            != manifest_path
        ):
            raise ReleaseArtifactError(
                "QA report references an unexpected Developer manifest."
            )

        if (
            report.workspace_path is None
            or Path(report.workspace_path).absolute()
            != expected_workspace
        ):
            raise ReleaseArtifactError(
                "QA report references an unexpected workspace."
            )

        # 3. Verify all required checks.
        _validate_execution_checks(report)

        # 4. Revalidate Developer artifacts.
        manifest, staging_dir = load_development_artifacts(
            state,
            runs_dir=runs_root,
        )

        # 5. Verify requirements assessment.
        _validate_requirements(
            report,
            state,
            manifest,
            runs_root,
        )

        # 6. Verify QA workspace matches approved generated files.
        _validate_workspace(
            expected_workspace,
            manifest,
        )

        return ReleaseArtifacts(
            manifest=manifest,
            staging_dir=staging_dir,
            qa_report=report,
            qa_workspace=expected_workspace,
            manifest_path=manifest_path,
            qa_report_path=report_path,
        )

    except ReleaseArtifactError:
        raise

    except (
        ArtifactError,
        SpecificationError,
        RequirementCoverageError,
        ValidationError,
        OSError,
        ValueError,
    ) as exc:
        raise ReleaseArtifactError(
            f"Release artifact validation failed: {exc}"
        ) from exc
