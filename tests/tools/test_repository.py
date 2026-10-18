"""Tests for safe repository-context collection."""

from harness.tools.repository import inspect_repository


def test_inspects_supported_files(tmp_path):
    (tmp_path / "README.md").write_text(
        "# Example repository"
    )

    (tmp_path / "main.py").write_text(
        "def main(): pass"
    )

    context = inspect_repository(str(tmp_path))

    assert "README.md" in context.files
    assert "main.py" in context.files
    assert "Example repository" in context.excerpts["README.md"]


def test_skips_sensitive_files(tmp_path):
    (tmp_path / ".env").write_text(
        "SECRET=do-not-read"
    )

    (tmp_path / "main.py").write_text(
        "print('safe')"
    )

    context = inspect_repository(str(tmp_path))

    assert ".env" not in context.files
    assert "main.py" in context.files
    assert "do-not-read" not in str(context.excerpts)


def test_repository_is_optional():
    assert inspect_repository(None) is None