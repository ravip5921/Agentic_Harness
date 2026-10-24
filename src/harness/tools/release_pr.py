"""Publish only approved manifest files from an isolated clone to a draft PR."""

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

from harness.models import FileOperation, TaskRequest
from harness.tools.filesystem import validate_relative_path, validate_repository_file
from harness.tools.release_workspace import verify_release_candidate


class ReleasePRError(Exception):
    """A PR could not be created; the local candidate remains available."""


def _run(args: list[str], cwd: Path) -> str:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GH_PROMPT_DISABLED="1",
               GIT_LITERAL_PATHSPECS="1")
    try:
        result = subprocess.run(
            args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=120, check=True,
        )
    except FileNotFoundError as exc:
        raise ReleasePRError(
            f"{args[0]} executable was not found, or its working directory is unavailable. "
            + ("Install GitHub CLI (gh) and restart the terminal/application so it picks up PATH. "
               "Git SSH access alone does not provide GitHub CLI access."
               if args[0] == "gh" else "Check installation and PATH.")
        ) from exc
    except PermissionError as exc:
        raise ReleasePRError(f"Permission denied starting {args[0]}; check executable and working-directory access.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ReleasePRError(f"{args[0]} operation timed out after 120 seconds; check network and service availability.") from exc
    except subprocess.CalledProcessError as exc:
        if args[:3] == ["gh", "auth", "status"]:
            message = (
                "GitHub CLI authentication check failed "
                f"(exit code {exc.returncode}). Run 'gh auth status --hostname github.com' "
                "in the same environment as the harness, then 'gh auth login --hostname github.com' "
                "if needed. Git SSH credentials are separate from GitHub CLI authentication."
            )
        elif args[:3] == ["gh", "pr", "create"]:
            message = (
                f"GitHub CLI PR creation failed (exit code {exc.returncode}). "
                "Check the active GitHub CLI account, repository permissions, and GitHub availability."
            )
        else:
            message = f"{args[0]} operation failed (exit code {exc.returncode}); check authentication and repository access."
        raise ReleasePRError(message) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        # Command output and arguments can include credential-bearing URLs.
        raise ReleasePRError(f"{args[0]} operation failed; check installation, authentication and repository access.") from exc
    return result.stdout.strip()


def _git(cwd: Path, *args: str) -> str:
    return _run(["git", "-c", "core.hooksPath=/dev/null", *args], cwd)


def create_release_pr(*, task: TaskRequest, artifacts, candidate: Path,
                      report_path: Path, revision: int) -> str:
    """Push a fresh branch to origin and create a draft PR, without merging.

    The target must be a clean Git repository at the same commit as its current
    branch on GitHub origin. This avoids publishing unrelated local changes.
    """
    if task.repository_path is None:
        raise ReleasePRError("Creating a PR requires --repository.")
    repository = Path(task.repository_path).resolve(strict=True)
    if Path(_git(repository, "rev-parse", "--show-toplevel")).resolve() != repository:
        raise ReleasePRError("The target must be the Git repository root.")
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ReleasePRError("Commit or stash target repository changes before creating a PR.")
    base = _git(repository, "symbolic-ref", "--quiet", "--short", "HEAD")
    head = _git(repository, "rev-parse", "HEAD")
    if "origin" not in _git(repository, "remote").splitlines():
        raise ReleasePRError(
            "No origin remote is configured. A local Git repository alone cannot create a GitHub PR. "
            "Create a GitHub repository, add it as origin, and push the current branch first; "
            "or choose 'staged' to keep this release local."
        )
    remote = _git(repository, "remote", "get-url", "origin")
    match = re.fullmatch(
        r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
        r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?", remote,
    )
    if match is None:
        raise ReleasePRError("PR creation currently supports GitHub origin URLs only.")
    github_repo = match.group(1)
    author_name = _git(repository, "config", "user.name")
    author_email = _git(repository, "config", "user.email")
    _run(["gh", "auth", "status", "--hostname", "github.com"], repository)
    branch = f"harness/{task.run_id}/attempt-{revision}"
    pushed = False
    try:
        with tempfile.TemporaryDirectory(prefix="harness-pr-") as temporary:
            clone = Path(temporary) / "repository"
            _git(Path(temporary), "clone", "--single-branch", "--branch", base, "--", remote, str(clone))
            if _git(clone, "rev-parse", "HEAD") != head:
                raise ReleasePRError("Local HEAD and origin branch differ; synchronize and rerun QA first.")
            verify_release_candidate(artifacts, candidate)
            _git(clone, "checkout", "-b", branch)
            paths = []
            for record in artifacts.manifest.files:
                relative = validate_relative_path(record.path)
                if any(part.casefold() == ".git" for part in relative.parts):
                    raise ReleasePRError("Git metadata cannot be included in a release.")
                target = validate_repository_file(clone, relative)
                if record.operation == FileOperation.CREATE:
                    if target.exists():
                        raise ReleasePRError(f"CREATE target already exists: {record.path}")
                elif (not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()
                      != record.original_sha256):
                    raise ReleasePRError(f"Original file changed: {record.path}; rerun QA.")
                data = (candidate / relative).read_bytes()
                if hashlib.sha256(data).hexdigest() != record.sha256:
                    raise ReleasePRError(f"Candidate changed after approval: {record.path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                paths.append(relative.as_posix())
            _git(clone, "add", "--", *paths)
            changed = set(_git(clone, "diff", "--cached", "--name-only", "-z").split("\0")) - {""}
            if not changed or not changed.issubset(set(paths)):
                raise ReleasePRError("PR must contain only changes listed in the approved manifest.")
            title = f"Harness release {task.run_id[:12]} (attempt {revision})"
            _git(clone, "-c", f"user.name={author_name}", "-c", f"user.email={author_email}",
                 "-c", "commit.gpgsign=false", "commit", "-m", title)
            # Read back committed blobs: filters must not change QA-approved bytes.
            for record in artifacts.manifest.files:
                blob = subprocess.run(
                    ["git", "show", f"HEAD:{record.path}"], cwd=clone,
                    capture_output=True, check=True, timeout=30,
                ).stdout
                if hashlib.sha256(blob).hexdigest() != record.sha256:
                    raise ReleasePRError(f"Git filters changed approved content: {record.path}")
            body = Path(temporary) / "pr-body.md"
            body.write_text(
                f"Human-approved harness run `{task.run_id}`, attempt {revision}.\n\n"
                + report_path.read_text(encoding="utf-8"), encoding="utf-8",
            )
            _git(clone, "push", "origin", f"HEAD:refs/heads/{branch}")
            pushed = True
            url = _run([
                "gh", "pr", "create", "--repo", github_repo, "--draft",
                "--base", base, "--head", branch, "--title", title,
                "--body-file", str(body),
            ], clone)
            if not re.fullmatch(r"https://github\.com/[^/]+/[^/]+/pull/[0-9]+", url):
                raise ReleasePRError("GitHub CLI did not return a PR URL; check GitHub before retrying.")
            return url
    except Exception as exc:
        recovery = f" Branch {branch} was pushed; check GitHub before retrying." if pushed else ""
        if isinstance(exc, ReleasePRError):
            raise ReleasePRError(str(exc) + recovery) from exc
        raise ReleasePRError("PR publication failed; the staged candidate is preserved." + recovery) from exc
