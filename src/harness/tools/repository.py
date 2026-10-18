
"""Read-only, bounded repository context collection."""

from pathlib import Path

from pydantic import BaseModel, Field


SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__",
    "node_modules", "dist", "build", ".next",
    ".pytest_cache", "runs",
}

SAFE_NAMES = {
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "Makefile",
}

SAFE_SUFFIXES = {".py", ".js", ".ts", ".java", ".go", ".rs"}

MAX_FILES = 30
MAX_FILE_BYTES = 20_000


class RepositoryError(Exception):
    pass


class RepositoryContext(BaseModel):
    path: str
    files: list[str] = Field(default_factory=list)
    excerpts: dict[str, str] = Field(default_factory=dict)
    truncated: bool = False


def inspect_repository(
    repository_path: str | None,
) -> RepositoryContext | None:
    """Inspect an explicitly supplied repository without executing code."""

    if repository_path is None:
        return None

    root = Path(repository_path).expanduser().resolve(strict=True)

    if not root.is_dir():
        raise RepositoryError("Repository path is not a directory.")

    result = RepositoryContext(path=str(root))

    count = 0

    for path in sorted(root.rglob("*")):
        if count >= MAX_FILES:
            result.truncated = True
            break

        relative = path.relative_to(root)

        if any(part in SKIP_DIRS for part in relative.parts):
            continue

        if path.is_symlink() or not path.is_file():
            continue

        # Do not follow paths that escape the repository.
        if not path.resolve().is_relative_to(root):
            continue

        if path.name not in SAFE_NAMES and path.suffix not in SAFE_SUFFIXES:
            continue

        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                result.truncated = True
                continue

            content = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        rel = relative.as_posix()
        result.files.append(rel)
        result.excerpts[rel] = content
        count += 1

    return result
