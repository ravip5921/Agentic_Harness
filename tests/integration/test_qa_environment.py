"""Dependency provisioning regressions using real staging and graph routing.

Generated Pillow code is copied and inspected, never imported on the host.
Only agent generation, dependency installation, and checks are faked.
"""

from pathlib import Path

import pytest

from harness.graph import build_graph
from harness.models import (
    DevelopmentAssessment,
    EngineeringSpec,
    FileOperation,
    ImplementationManifest,
    QACheckResult,
    QACheckStatus,
    QACheckType,
    QAReport,
    QARequirementAssessment,
    QAVerdict,
    RequirementFinding,
    RequirementStatus,
    ResearchResult,
    RunStatus,
    StageOutcome,
    TaskRequest,
)
from harness.stages.developer import make_developer_stage
from harness.stages.qa import make_qa_stage
from harness.stages.release import make_release_stage


LINE_DRAWER_SOURCE = (
    "from PIL import Image, ImageDraw\n\n"
    "def draw_line(start, end):\n"
    "    image = Image.new('RGB', (100, 100), 'white')\n"
    "    ImageDraw.Draw(image).line([start, end], fill='black')\n"
    "    return image\n"
)
LINE_DRAWER_TEST = (
    "from line_drawer import draw_line\n\n"
    "def test_line_endpoints():\n"
    "    image = draw_line((1, 1), (10, 10))\n"
    "    assert image.getpixel((1, 1)) == (0, 0, 0)\n"
    "    assert image.getpixel((10, 10)) == (0, 0, 0)\n"
)
REQUESTED_FILES = {"line_drawer.py", "test_line_drawer.py"}


def run_line_drawer_graph(
    runs_dir: Path,
    *,
    setup_status=QACheckStatus.PASSED,
    setup_timed_out=False,
    unit_status=QACheckStatus.PASSED,
):
    """Exercise real Developer artifacts, QA preparation, reports, and routing."""

    calls = {
        "order": [],
        "developer": [],
        "setup": [],
        "checks": [],
        "assessment": [],
        "feedback": [],
        "release": [],
    }
    specification = EngineeringSpec(
        title="Straight line graphics library",
        objective="Draw a straight line between two points using Pillow.",
        requirements=[
            "Create exactly line_drawer.py and test_line_drawer.py."
        ],
        technical_approach="Use Pillow and declare it in manifest dependencies.",
        implementation_steps=["Implement draw_line and test its endpoints."],
        acceptance_criteria=["Both endpoints of the line are black pixels."],
    )

    def research(state):
        research_dir = runs_dir / state["task"].run_id / "research"
        research_dir.mkdir(parents=True)
        spec_path = research_dir / "engineering_spec.json"
        spec_path.write_text(specification.model_dump_json(), encoding="utf-8")
        return {
            "research_result": ResearchResult(
                outcome=StageOutcome.COMPLETED,
                artifacts=[str(spec_path)],
            )
        }

    def generate(specification, repository, qa_feedback, previous_implementation):
        calls["developer"].append(qa_feedback)
        return DevelopmentAssessment(
            status="ready",
            summary="Pillow line drawing implementation and tests.",
            dependencies=["Pillow"],
            files=[
                {
                    "path": "line_drawer.py",
                    "operation": FileOperation.CREATE,
                    "content": LINE_DRAWER_SOURCE,
                },
                {
                    "path": "test_line_drawer.py",
                    "operation": FileOperation.CREATE,
                    "content": LINE_DRAWER_TEST,
                },
            ],
        )

    def prepare_dependencies(dependencies, dependencies_dir):
        calls["order"].append("dependency_setup")
        calls["setup"].append((list(dependencies), dependencies_dir))
        assert dependencies == ["Pillow"]
        if setup_status == QACheckStatus.PASSED:
            dependencies_dir.mkdir()
            installed_package = dependencies_dir / "PIL"
            installed_package.mkdir()
            (installed_package / "__init__.py").write_text(
                "# Mock installation artifact; never imported.\n"
            )
        return QACheckResult(
            name="dependency_setup",
            check_type=QACheckType.DEPENDENCY_SETUP,
            status=setup_status,
            command=["docker", "run", "qa-image", "python", "-m", "pip", "install", "Pillow"],
            exit_code=0 if setup_status == QACheckStatus.PASSED else 1,
            timed_out=setup_timed_out,
            stdout="Successfully installed Pillow" if setup_status == QACheckStatus.PASSED else "Resolving Pillow",
            stderr="" if setup_status == QACheckStatus.PASSED else "Dependency provisioning unavailable",
            message="Dependency setup completed." if setup_status == QACheckStatus.PASSED else "Dependency setup blocked.",
        )

    def execute(check_type, workspace, *, dependency_dir=None):
        calls["order"].append(check_type.value)
        calls["checks"].append((check_type, workspace, dependency_dir))
        assert setup_status == QACheckStatus.PASSED
        assert calls["order"][0] == "dependency_setup"
        assert {path.name for path in workspace.iterdir()} == REQUESTED_FILES
        if check_type == QACheckType.SYNTAX:
            assert dependency_dir is None
            status = QACheckStatus.PASSED
        else:
            assert check_type == QACheckType.UNIT_TEST
            assert dependency_dir == calls["setup"][0][1]
            assert dependency_dir.parent == workspace.parent
            assert dependency_dir != workspace
            assert dependency_dir.is_dir()
            status = unit_status
        return QACheckResult(
            name=check_type.value,
            check_type=check_type,
            status=status,
            exit_code=0 if status == QACheckStatus.PASSED else 2,
            stdout="1 passed" if status == QACheckStatus.PASSED else "E   ModuleNotFoundError: No module named 'PIL'",
            message="Check passed." if status == QACheckStatus.PASSED else "QA environment cannot import PIL.",
        )

    def assess(task, specification, manifest, implementation, checks):
        calls["assessment"].append(manifest.dependencies)
        assert {file.path for file in implementation.files} == REQUESTED_FILES
        assert all(check.status == QACheckStatus.PASSED for check in checks)
        return QARequirementAssessment(
            overall_summary="The exact requested files and line endpoint tests are present.",
            findings=[
                RequirementFinding(
                    requirement_id="REQ-1",
                    requirement=specification.requirements[0],
                    status=RequirementStatus.SATISFIED,
                    explanation="Exactly the two requested files were generated.",
                    implementation_files=["line_drawer.py"],
                    test_files=["test_line_drawer.py"],
                ),
                RequirementFinding(
                    requirement_id="AC-1",
                    requirement=specification.acceptance_criteria[0],
                    status=RequirementStatus.SATISFIED,
                    explanation="Endpoint assertions cover the requested behavior.",
                    implementation_files=["line_drawer.py"],
                    test_files=["test_line_drawer.py"],
                ),
            ],
        )

    def feedback(*args):
        calls["feedback"].append(args)
        raise AssertionError("Environment blocks must not generate repair feedback.")

    real_release = make_release_stage(runs_dir=runs_dir)

    def release(state):
        calls["release"].append(state["qa_result"].verdict)
        return real_release(state)

    graph = build_graph(
        research=research,
        developer=make_developer_stage(generator=generate, runs_dir=runs_dir),
        qa=make_qa_stage(
            dependency_preparer=prepare_dependencies,
            test_executor=execute,
            requirements_assessor=assess,
            feedback_generator=feedback,
            runs_dir=runs_dir,
        ),
        release=release,
    )
    task = TaskRequest(
        task=(
            "Build a Pillow graphics library to draw a straight line. "
            "Create only line_drawer.py and test_line_drawer.py."
        ),
        max_dev_revisions=3,
    )
    result = graph.invoke({"task": task, "revision_count": 0})
    report = QAReport.model_validate_json(
        Path(result["qa_result"].report_path).read_text(encoding="utf-8")
    )
    return result, calls, report


def test_manifest_dependencies_are_prepared_before_unit_tests(tmp_path):
    result, calls, report = run_line_drawer_graph(tmp_path)

    assert result["status"] == RunStatus.STAGED
    assert result["qa_result"].verdict == QAVerdict.PASS
    assert result["revision_count"] == 0
    assert calls["developer"] == [None]
    assert calls["order"] == ["dependency_setup", "syntax", "unit_test"]
    assert calls["assessment"] == [["Pillow"]]
    assert calls["feedback"] == []
    assert calls["release"] == [QAVerdict.PASS]

    workspace = Path(report.workspace_path)
    assert {path.name for path in workspace.iterdir()} == REQUESTED_FILES
    assert (workspace / "line_drawer.py").read_text() == LINE_DRAWER_SOURCE
    assert not (workspace / "requirements.txt").exists()
    manifest = ImplementationManifest.model_validate_json(
        (
            tmp_path / result["task"].run_id / "development"
            / "attempt-0" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest.dependencies == ["Pillow"]
    assert {file.path for file in manifest.files} == REQUESTED_FILES
    assert report.checks[0].check_type == QACheckType.DEPENDENCY_SETUP
    assert report.checks[0].exit_code == 0
    assert report.checks[0].stdout == "Successfully installed Pillow"
    assert (calls["setup"][0][1] / "PIL" / "__init__.py").is_file()
    candidate = Path(result["release_result"].change_ref)
    assert {path.name for path in candidate.iterdir()} == REQUESTED_FILES
    assert not (candidate / "requirements.txt").exists()
    assert not (candidate / "PIL").exists()


@pytest.mark.parametrize(
    ("setup_status", "setup_timed_out"),
    [
        (QACheckStatus.BLOCKED, False),
        (QACheckStatus.FAILED, False),
        (QACheckStatus.BLOCKED, True),
    ],
    ids=["environment-blocked", "installer-nonzero", "installer-timeout"],
)
def test_dependency_setup_failure_stops_without_developer_retry(
    tmp_path, setup_status, setup_timed_out
):
    result, calls, report = run_line_drawer_graph(
        tmp_path,
        setup_status=setup_status,
        setup_timed_out=setup_timed_out,
    )

    assert result["status"] == RunStatus.BLOCKED
    assert result["qa_result"].verdict == QAVerdict.BLOCKED
    assert result["revision_count"] == 0
    assert calls["developer"] == [None]
    assert calls["checks"] == []
    assert calls["assessment"] == []
    assert calls["feedback"] == []
    assert calls["release"] == []
    assert result["qa_result"].feedback is None
    assert report.feedback is None
    assert report.requirements_assessment is None
    assert len(report.checks) == 1
    assert report.checks[0].check_type == QACheckType.DEPENDENCY_SETUP
    assert report.checks[0].status == QACheckStatus.BLOCKED
    assert report.checks[0].stdout == "Resolving Pillow"
    assert report.checks[0].stderr == "Dependency provisioning unavailable"
    assert report.checks[0].timed_out == setup_timed_out
    assert {path.name for path in Path(report.workspace_path).iterdir()} == REQUESTED_FILES


def test_missing_import_environment_block_does_not_trigger_repair(tmp_path):
    result, calls, report = run_line_drawer_graph(
        tmp_path, unit_status=QACheckStatus.BLOCKED
    )

    assert result["status"] == RunStatus.BLOCKED
    assert result["revision_count"] == 0
    assert calls["developer"] == [None]
    assert calls["assessment"] == []
    assert calls["feedback"] == []
    assert calls["release"] == []
    assert report.feedback is None
    assert report.checks[-1].check_type == QACheckType.UNIT_TEST
    assert report.checks[-1].status == QACheckStatus.BLOCKED
    assert "No module named 'PIL'" in report.checks[-1].stdout
