from pathlib import Path

import pytest

from harness.models import ReleaseDecision, TaskRequest
from harness.tools.release_approval import prompt_release_approval


@pytest.mark.parametrize("answers,repository,expected", [
    (["no"], "/repo", ReleaseDecision.REJECT),
    ([""], "/repo", ReleaseDecision.REJECT),
    (["yes", ""], "/repo", ReleaseDecision.KEEP_STAGED),
    (["y", "invalid", "pr"], "/repo", ReleaseDecision.CREATE_PR),
    (["y"], None, ReleaseDecision.KEEP_STAGED),
])
def test_human_choices(monkeypatch, answers, repository, expected):
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert prompt_release_approval(
        task=TaskRequest(task="Task", repository_path=repository),
        candidate=Path("candidate"), report_path=Path("report.md"),
    ) == expected


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt, OSError])
@pytest.mark.parametrize("approved", [False, True])
def test_input_loss_does_not_authorize_release(monkeypatch, error, approved):
    calls = 0

    def read(_):
        nonlocal calls
        calls += 1
        if approved and calls == 1:
            return "yes"
        raise error()

    monkeypatch.setattr("builtins.input", read)
    assert prompt_release_approval(
        task=TaskRequest(task="Task", repository_path="/repo"),
        candidate=Path("candidate"), report_path=Path("report.md"),
    ) == ReleaseDecision.REJECT
