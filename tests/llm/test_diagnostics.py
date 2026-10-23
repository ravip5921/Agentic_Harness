"""Truncation evidence must survive rejection without exposing model content."""

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console

from harness.llm import GenerationError, LLMSettings
from harness.llm.diagnostics import diagnostics_to
from harness.llm.groq import GroqGenerator
from harness.tools.progress import ProgressRenderer


def generator(text, reason):
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=reason)],
        usage=SimpleNamespace(prompt_tokens=700, completion_tokens=4096, total_tokens=4796,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=3000)),
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=Mock(return_value=completion),
    )))
    return GroqGenerator(LLMSettings("groq", "test-model", "secret-key", max_output_tokens=4096), client)


def messages():
    return [{"role": "user", "content": json.dumps({
        "current_file": {"path": "tests/test_client.py"},
        "repository": "private repository context",
    })}]


@pytest.mark.parametrize("reason,diagnosis,accepted", [
    ("length", "output_limit_reached", False),
    ("stop", "normal_stop_not_completeness_proof", True),
    (None, "finish_reason_unreported", True),
])
def test_response_diagnostics_persist_before_validation_and_print_useful_fields(
    tmp_path, monkeypatch, reason, diagnosis, accepted,
):
    path = tmp_path / "diagnostics" / "development-attempt-0.jsonl"
    text = 'def test_client():\n    assert "private generated content" == ('
    adapter = generator(text, reason)
    events = []
    monkeypatch.setattr("harness.events.get_stream_writer", lambda: events.append)
    with diagnostics_to(path):
        if accepted:
            assert adapter.complete(stage="developer", messages=messages()).text == text
        else:
            with pytest.raises(GenerationError, match="finish_reason=length"):
                adapter.complete(stage="developer", messages=messages())

    records = [json.loads(line) for line in path.read_text().splitlines()]
    request, response = records[:2]
    assert request["event"] == "request_started"
    assert request["path"] == "tests/test_client.py"
    assert request["model"] == "test-model"
    assert request["max_output_tokens"] == 4096
    assert request["timeout_seconds"] == 90
    assert response["request_id"] == request["request_id"]
    assert response["finish_reason"] == reason
    assert response["diagnosis"] == diagnosis
    assert response["usage"]["reasoning_tokens"] == 3000
    assert response["usage"]["completion_tokens"] == 4096
    assert response["output_bytes"] == len(text.encode())
    assert response["output_lines"] == 2
    assert response["python_syntax_valid"] is False
    assert response["syntax_error_line"] == 2
    if not accepted:
        assert records[-1]["event"] == "request_failed"

    output = StringIO()
    with ProgressRenderer(Console(file=output, force_terminal=False, width=240)) as progress:
        for event in events:
            progress.event(event)
    printed = output.getvalue()
    assert "output limit=4096 tokens" in printed
    assert "finish=" in printed
    assert "reasoning=3000" in printed
    assert "Python syntax=invalid" in printed
    for private in ("secret-key", "private generated content", "private repository context"):
        assert private not in path.read_text()
        assert private not in printed


def test_request_failure_is_recorded_without_exception_contents(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    adapter = generator("unused", "stop")
    adapter.client.chat.completions.create.side_effect = TimeoutError("private exception detail")
    with diagnostics_to(path):
        with pytest.raises(TimeoutError):
            adapter.complete(stage="developer", messages=messages())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["request_started", "request_failed"]
    assert records[-1]["error_type"] == "TimeoutError"
    assert "private exception detail" not in path.read_text()


def test_diagnostic_write_failure_does_not_change_completion(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.write_text("regular file")
    adapter = generator("print('complete')\n", "stop")
    with diagnostics_to(occupied / "diagnostics.jsonl"):
        assert adapter.complete(stage="developer", messages=messages()).text == "print('complete')\n"


def test_diagnostic_destination_is_reset_after_failure(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    adapter = generator("partial", "length")
    with pytest.raises(GenerationError):
        with diagnostics_to(path):
            adapter.complete(stage="developer", messages=messages())
    original = path.read_text()
    generator("complete", "stop").complete(stage="developer", messages=messages())
    assert path.read_text() == original
