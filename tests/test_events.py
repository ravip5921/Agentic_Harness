from unittest.mock import Mock

import pytest

from harness.events import emit_event


def test_emitter_forwards_structured_event(monkeypatch):
    writer = Mock()
    monkeypatch.setattr("harness.events.get_stream_writer", lambda: writer)
    data = {"path": "tests/test_client.py"}

    emit_event("developer", "file_generation_started", "Generating tests/test_client.py",
               data=data, level="start")

    writer.assert_called_once_with({
        "stage": "developer",
        "event": "file_generation_started",
        "message": "Generating tests/test_client.py",
        "data": data,
        "level": "start",
    })
    assert writer.call_args.args[0]["data"] is not data


def test_emitter_is_safe_outside_graph():
    emit_event("research", "inspection_started", "Inspecting repository", level="start")


@pytest.mark.parametrize("failure_at", ["lookup", "write"])
def test_observability_failure_is_ignored(monkeypatch, failure_at):
    failure = Mock(side_effect=RuntimeError("stream unavailable"))
    monkeypatch.setattr("harness.events.get_stream_writer",
                        failure if failure_at == "lookup" else lambda: failure)
    emit_event("qa", "check_completed", "Syntax checks passed", level="success")
    failure.assert_called_once()
