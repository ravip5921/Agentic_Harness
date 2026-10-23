"""Real Git repositories and pushes; only GitHub transport and gh are replaced."""

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.models import ImplementationManifest, TaskRequest
from harness.tools import release_pr


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.fixture
def publication(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", str(remote))
    source = tmp_path / "source"
    git(tmp_path, "init", "-b", "main", str(source))
    git(source, "config", "user.name", "Harness Test")
    git(source, "config", "user.email", "test@example.com")
    (source / "existing.py").write_text("old = True\n")
    (source / "untouched.txt").write_text("preserved\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "Initial")
    git(source, "remote", "add", "origin", str(remote))
    git(source, "push", "-u", "origin", "main")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "existing.py").write_text("old = False\n")
    (candidate / "new.py").write_text("new = True\n")
    (candidate / "not-in-manifest.txt").write_text("must not publish\n")
    manifest = ImplementationManifest(summary="Example change", files=[
        dict(path="existing.py", operation="modify",
             sha256=hashlib.sha256((candidate / "existing.py").read_bytes()).hexdigest(),
             original_sha256=hashlib.sha256((source / "existing.py").read_bytes()).hexdigest()),
        dict(path="new.py", operation="create",
             sha256=hashlib.sha256((candidate / "new.py").read_bytes()).hexdigest()),
    ])
    report = tmp_path / "report.md"
    report.write_text("# Release Report\n\nQA passed.\n")
    task = TaskRequest(task="Example change", repository_path=str(source))
    real_git, real_run = release_pr._git, release_pr._run
    gh_calls = []

    def local_git(cwd, *args):
        if cwd == source and args == ("remote", "get-url", "origin"):
            return "https://github.com/example/project.git"
        if args[0] == "clone":
            args = (*args[:-2], str(remote), args[-1])
        return real_git(cwd, *args)

    def fake_gh(args, cwd):
        if args[0] != "gh":
            return real_run(args, cwd)
        gh_calls.append(args)
        if args[1:3] == ["pr", "create"]:
            assert "--draft" in args
            assert args[args.index("--base") + 1] == "main"
            assert args[args.index("--repo") + 1] == "example/project"
            assert "QA passed" in Path(args[args.index("--body-file") + 1]).read_text()
            return "https://github.com/example/project/pull/123"
        return ""

    monkeypatch.setattr(release_pr, "_git", local_git)
    monkeypatch.setattr(release_pr, "_run", fake_gh)
    return SimpleNamespace(
        kwargs=dict(task=task, artifacts=SimpleNamespace(manifest=manifest),
                    candidate=candidate, report_path=report, revision=0),
        source=source, remote=remote, candidate=candidate, gh_calls=gh_calls,
        branch=f"harness/{task.run_id}/attempt-0",
    )


def test_pr_contains_only_manifest_files_and_leaves_source_unchanged(publication):
    p = publication
    original_head = git(p.source, "rev-parse", "HEAD")
    assert release_pr.create_release_pr(**p.kwargs).endswith("/pull/123")
    assert git(p.source, "rev-parse", "HEAD") == original_head
    assert git(p.source, "status", "--porcelain") == ""
    assert (p.source / "existing.py").read_text() == "old = True\n"
    changed = git(p.remote, "diff", "--name-only", f"main..{p.branch}").splitlines()
    assert set(changed) == {"existing.py", "new.py"}
    assert git(p.remote, "show", f"{p.branch}:existing.py") == "old = False"


@pytest.mark.parametrize("mutation", ["dirty", "tampered", "stale_original", "diverged"])
def test_invalid_release_never_pushes(publication, mutation):
    p = publication
    if mutation == "dirty":
        (p.source / "untracked.txt").write_text("local work")
    elif mutation == "tampered":
        (p.candidate / "existing.py").write_text("tampered\n")
    elif mutation == "stale_original":
        p.kwargs["artifacts"].manifest.files[0].original_sha256 = "0" * 64
    else:
        (p.source / "untouched.txt").write_text("local commit\n")
        git(p.source, "add", ".")
        git(p.source, "commit", "-m", "Unpushed")
    with pytest.raises(release_pr.ReleasePRError):
        release_pr.create_release_pr(**p.kwargs)
    assert git(p.remote, "for-each-ref", "--format=%(refname)", "refs/heads/harness/") == ""
    assert not any(call[1:3] == ["pr", "create"] for call in p.gh_calls)


def test_failed_pr_creation_reports_pushed_branch(publication, monkeypatch):
    p = publication
    real_run = release_pr._run

    def fail_create(args, cwd):
        if args[:3] == ["gh", "pr", "create"]:
            raise release_pr.ReleasePRError("GitHub API unavailable")
        return real_run(args, cwd)

    monkeypatch.setattr(release_pr, "_run", fail_create)
    with pytest.raises(release_pr.ReleasePRError, match=f"Branch {p.branch} was pushed"):
        release_pr.create_release_pr(**p.kwargs)
    assert git(p.remote, "rev-parse", p.branch)
    assert (p.candidate / "existing.py").is_file()


def test_standalone_cannot_create_pr(tmp_path):
    with pytest.raises(release_pr.ReleasePRError, match="requires --repository"):
        release_pr.create_release_pr(task=TaskRequest(task="Standalone"), artifacts=None,
                                     candidate=tmp_path, report_path=tmp_path, revision=0)


def test_local_only_repository_explains_missing_origin(publication):
    p = publication
    git(p.source, "remote", "remove", "origin")
    with pytest.raises(release_pr.ReleasePRError, match="No origin remote is configured"):
        release_pr.create_release_pr(**p.kwargs)
    assert p.gh_calls == []
    assert (p.candidate / "existing.py").is_file()
