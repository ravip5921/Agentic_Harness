
import pytest

from harness.models import (
    DevelopmentPlan,
    GeneratedFile,
    FileOperation,
)
from harness.tools.development_plan import (
    DevelopmentPlanError,
    validate_development_plan,
)
from harness.tools.repository import RepositoryContext
from harness.tools.filesystem import FilesystemError
from harness.agents.developer_agent import normalize_plan_operations

def make_plan(files):
    return DevelopmentPlan(
        status="ready",
        reason="",
        summary="Revised implementation",
        files=files,
    )


def make_file(path, operation="create"):
    return {
        "path": path,
        "operation": operation,
        "description": f"Generate {path}",
    }


def test_retry_can_omit_previous_file():
    """A retry can replace a three-file implementation with two files."""

    # The previous attempt had calculator.py, utils.py, and tests.
    # Only the new plan determines the current candidate.
    plan = make_plan([
        make_file("calculator.py"),
        make_file("tests/test_calculator.py"),
    ])

    validate_development_plan(plan, repository=None)

    assert "utils.py" not in {
        file.path for file in plan.files
    }


def test_retry_can_introduce_new_file():
    """A revised implementation can add new files."""

    plan = make_plan([
        make_file("calculator.py"),
        make_file("helpers.py"),
        make_file("tests/test_calculator.py"),
    ])

    validate_development_plan(plan, repository=None)

    assert "helpers.py" in {
        file.path for file in plan.files
    }


def test_retry_cannot_modify_file_absent_from_original_repo():
    """Previous CREATE files do not become repository MODIFY files."""

    plan = make_plan([
        make_file("calculator.py", "modify"),
    ])

    with pytest.raises(
        DevelopmentPlanError,
        match="absent from original repository",
    ):
        validate_development_plan(plan, repository=None)


@pytest.mark.parametrize(
    "files",
    [
        [
            make_file("calculator.py"),
            make_file("calculator.py"),
        ],
        [
            make_file("calculator.py"),
            make_file("Calculator.py"),
        ],
    ],
)
def test_duplicate_paths_are_rejected(files):
    plan = make_plan(files)

    with pytest.raises(
        DevelopmentPlanError,
        match="Duplicate planned file",
    ):
        validate_development_plan(plan)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.py",
        "/tmp/outside.py",
    ],
)
def test_unsafe_paths_are_rejected(unsafe_path):
    plan = make_plan([
        make_file(unsafe_path),
    ])

    with pytest.raises(FilesystemError):
        validate_development_plan(plan)

def test_create_cannot_overwrite_existing_repository_file():
    repository = RepositoryContext(
        path="/example/repository",
        files=["src/service.py"],
        excerpts={
            "src/service.py": "def service(): pass\n",
        },
    )

    plan = make_plan([
        make_file("src/service.py", "create"),
    ])

    with pytest.raises(
        DevelopmentPlanError,
        match="Cannot CREATE existing repository file",
    ):
        validate_development_plan(plan, repository)


def test_modify_existing_repository_file():
    repository = RepositoryContext(
        path="/example/repository",
        files=["src/service.py"],
        excerpts={
            "src/service.py": "def service(): pass\n",
        },
    )

    plan = make_plan([
        make_file("src/service.py", "modify"),
    ])

    validate_development_plan(plan, repository)

def test_retry_normalizes_modify_to_create_without_repository():
    plan = make_plan([
        make_file("calculator.py", "modify"),
    ])

    normalized = normalize_plan_operations(
        plan,
        repository=None,
    )

    assert normalized.files[0].operation == FileOperation.CREATE