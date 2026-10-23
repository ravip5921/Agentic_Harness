"""Exercise 429 recovery without real waits or network requests."""

from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest
from groq import APIStatusError, InternalServerError, RateLimitError

from harness.tools import groq_rate_limit
from harness.llm import retry as llm_retry


MESSAGE = (
    "Rate limit reached for model test in organization org_private on tokens per minute (TPM): "
    "Limit 8000, Used 2318, Requested 6489. Please try again in 6.0525s."
)


def rate_limit(message=MESSAGE, headers=None):
    return RateLimitError(message, response=httpx.Response(
        429, headers=headers, request=httpx.Request("POST", "https://api.groq.com/test"),
    ), body={"message": message})


def oversized_413(headers=None):
    message = (
        "Request too large for model `openai/gpt-oss-20b` in organization "
        "`org_private` service tier `on_demand` on tokens per minute (TPM): "
        "Limit 8000, Requested 8406, please reduce your message size and try again."
    )
    return APIStatusError(message, response=httpx.Response(
        413, headers=headers, request=httpx.Request("POST", "https://api.groq.com/test"),
    ), body={"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}})


@pytest.fixture
def runtime(monkeypatch):
    sleeper = Mock()
    events = []
    monkeypatch.setattr(llm_retry, "sleep", sleeper)
    monkeypatch.setattr("harness.events.get_stream_writer", lambda: events.append)
    return sleeper, events


def test_success_is_immediate_and_arguments_are_unchanged(runtime, monkeypatch):
    sleeper, events = runtime
    # The removed fixed-cooldown setting must have no effect.
    monkeypatch.setenv("GROQ_COOLDOWN_SECONDS", "60")
    response = object()
    request = Mock(return_value=response)
    messages = [{"role": "user", "content": "Private input"}]
    assert groq_rate_limit.call_groq("developer", request, model="model", messages=messages) is response
    request.assert_called_once_with(model="model", messages=messages)
    sleeper.assert_not_called()
    assert events == []


@pytest.mark.parametrize("error", [RuntimeError("Private service error"), InternalServerError(
    "Server error", response=httpx.Response(503, request=httpx.Request("POST", "https://api.groq.com/test")),
    body=None,
)])
def test_non_429_errors_are_not_retried_or_delayed(runtime, error):
    sleeper, events = runtime
    request = Mock(side_effect=error)
    with pytest.raises(type(error)) as caught:
        groq_rate_limit.call_groq("qa", request)
    assert caught.value is error
    request.assert_called_once()
    sleeper.assert_not_called()
    assert events == []
    assert groq_rate_limit.call_groq("release", lambda: "ok") == "ok"


def test_user_error_waits_requested_duration_plus_padding_then_recovers(runtime):
    sleeper, events = runtime
    response = object()
    request = Mock(side_effect=[rate_limit(), response])
    assert groq_rate_limit.call_groq("developer", request, model="model") is response
    sleeper.assert_called_once_with(pytest.approx(8.0525))
    assert request.call_args_list == [call(model="model"), call(model="model")]
    assert [event["event"] for event in events] == [
        "groq_rate_limit_wait_started", "groq_rate_limit_retrying", "groq_rate_limit_recovered",
    ]
    assert events[0]["data"] == {
        "seconds": pytest.approx(8.0525), "retry": 1, "max_retries": 5,
        "tpm_limit": 8000, "requested_tokens": 6489, "oversized": False,
    }
    assert "org_private" not in str(events)


@pytest.mark.parametrize("message,headers,expected", [
    ("Rate limited", {"retry-after": "6.0525"}, 8.0525),
    ("Rate limited", {"retry-after-ms": "1250"}, 3.25),
    ("Please try again in 6.0525s.", {"retry-after": "3"}, 8.0525),
    ("Please try again in 2s.", {"retry-after": "8"}, 10),
    ("Please try again in 1m2.5s.", {}, 64.5),
    ("Please try again in 250ms.", {}, 2.25),
    ("Please try again in 1h2m3s.", {}, 3725),
    ("Please try again in 0s.", {}, 2),
    ("Rate limited", {"retry-after": "3", "retry-after-ms": "4500"}, 6.5),
    ("Rate limited", {"retry-after": "garbage"}, 7),
    ("Rate limited", {"retry-after": "nan", "retry-after-ms": "inf"}, 7),
    ("Rate limited", {"retry-after": "-1"}, 7),
])
def test_server_hints_and_fallback(message, headers, expected, runtime):
    sleeper, _events = runtime
    request = Mock(side_effect=[rate_limit(message, headers), "ok"])
    assert groq_rate_limit.call_groq("research", request) == "ok"
    sleeper.assert_called_once_with(pytest.approx(expected))


def test_retry_after_http_date_is_supported(runtime, monkeypatch):
    sleeper, _events = runtime
    monkeypatch.setattr(llm_retry, "time", lambda: 1000)
    request = Mock(side_effect=[rate_limit("Limited", {"retry-after": formatdate(1010, usegmt=True)}), "ok"])
    assert groq_rate_limit.call_groq("research", request) == "ok"
    sleeper.assert_called_once_with(12)


def test_repeated_429s_stop_after_five_retries_with_bounded_fallback(runtime):
    sleeper, events = runtime
    error = rate_limit("No retry hint")
    request = Mock(side_effect=error)
    with pytest.raises(RateLimitError) as caught:
        groq_rate_limit.call_groq("qa", request)
    assert caught.value is error
    assert request.call_count == 6
    assert sleeper.call_args_list == [call(7), call(12), call(22), call(42), call(62)]
    assert events[-1]["event"] == "groq_rate_limit_exhausted"
    assert events[-1]["data"]["retries"] == 5


@pytest.mark.parametrize("counts", ["Limit 8000, Used 0, Requested 9000", "Limit 8,000, Used 0, Requested 9,000"])
def test_individually_oversized_requests_retry_exactly_twice(runtime, counts):
    sleeper, events = runtime
    error = rate_limit(f"tokens per minute (TPM): {counts}. Please try again in 6.0525s.")
    request = Mock(side_effect=error)
    with pytest.raises(RateLimitError) as caught:
        groq_rate_limit.call_groq("developer", request)
    assert caught.value is error
    assert request.call_count == 3
    assert sleeper.call_count == 2
    assert all(args.args[0] == pytest.approx(8.0525) for args in sleeper.call_args_list)
    waits = [event for event in events if event["event"] == "groq_rate_limit_wait_started"]
    assert [event["data"]["retry"] for event in waits] == [1, 2]
    assert all(event["data"]["max_retries"] == 2 and event["data"]["oversized"] for event in waits)
    assert events[-1]["level"] == "failure"
    assert "9000" in events[-1]["message"] and "8000" in events[-1]["message"]
    # The failure releases the global gate for subsequent requests.
    assert groq_rate_limit.call_groq("research", lambda: "next") == "next"


@pytest.mark.parametrize("message", [
    "tokens per minute (TPM): Limit 8000, Used 100, Requested 8000.",
    "requests per day (RPD): Limit 1000, Used 1000, Requested 1001.",
])
def test_equal_budget_and_non_tpm_errors_are_not_marked_oversized(runtime, message):
    _sleeper, events = runtime
    assert groq_rate_limit.call_groq("qa", Mock(side_effect=[rate_limit(message), "ok"])) == "ok"
    assert events[0]["data"]["oversized"] is False
    assert events[0]["data"]["max_retries"] == 5


def test_oversized_request_can_recover_on_a_retry(runtime):
    sleeper, events = runtime
    error = rate_limit("tokens per minute (TPM): Limit 8000, Used 0, Requested 9000.")
    request = Mock(side_effect=[error, error, "ok"])
    assert groq_rate_limit.call_groq("developer", request) == "ok"
    assert request.call_count == 3
    assert sleeper.call_count == 2
    assert events[-1]["event"] == "groq_rate_limit_recovered"


def test_other_agents_cannot_call_during_rate_limit_wait(monkeypatch, runtime):
    first_waiting, release_first, second_attempting, second_called = (Event() for _ in range(4))
    first_request = Mock(side_effect=[rate_limit(), "first"])

    def sleep(seconds):
        first_waiting.set()
        assert release_first.wait(timeout=5)

    def second():
        second_attempting.set()
        return groq_rate_limit.call_groq("release", lambda: second_called.set())

    monkeypatch.setattr(llm_retry, "sleep", sleep)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(groq_rate_limit.call_groq, "developer", first_request)
        try:
            assert first_waiting.wait(timeout=5)
            next_call = pool.submit(second)
            assert second_attempting.wait(timeout=5)
            assert not second_called.wait(timeout=0.05)
        finally:
            release_first.set()
        assert first.result(timeout=5) == "first"
        next_call.result(timeout=5)
    assert second_called.is_set()
    assert first_request.call_count == 2


@pytest.mark.parametrize("operation,stage", [
    ("research", "research"), ("planning", "developer"), ("file", "developer"),
    ("requirements", "qa"), ("feedback", "qa"), ("release", "release"),
])
@pytest.mark.parametrize("status_code", [429, 413])
def test_every_agent_handles_oversized_exhaustion(monkeypatch, runtime, operation, stage, status_code):
    from harness.agents.developer_agent import GroqDeveloperAgent, DevelopmentGenerationError
    from harness.agents.qa_agent import GroqQAAgent, QAGenerationError
    from harness.agents.release_agent import GroqReleaseAgent, ReleaseGenerationError
    from harness.agents.research_agent import GroqResearchAgent, ResearchGenerationError
    from harness.models import (
        DevelopmentPlan, EngineeringSpec, ImplementationManifest, PlannedFile,
        QAReport, QAVerdict, TaskRequest,
    )
    from harness.tools.qa_context import QAImplementationContext

    sleeper, events = runtime
    error = (oversized_413() if status_code == 413 else
             rate_limit("tokens per minute (TPM): Limit 8000, Used 0, Requested 9000."))
    request = Mock(side_effect=error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request)))
    monkeypatch.setattr("harness.llm.groq.Groq", lambda **kwargs: client)
    spec = EngineeringSpec(
        title="Addition", objective="Add integers", requirements=["Implement add"],
        technical_approach="Python", implementation_steps=["Write code"],
        acceptance_criteria=["add(1, 1) returns 2"],
    )
    task = TaskRequest(task="Implement addition")
    manifest = ImplementationManifest(
        summary="Addition", files=[{"path": "add.py", "operation": "create", "sha256": "a" * 64}],
    )
    planned = PlannedFile(path="add.py", operation="create", description="Addition")
    plan = DevelopmentPlan(status="ready", summary="Addition", files=[planned])
    implementation = QAImplementationContext(files=[{
        "path": "add.py", "operation": "create", "content": "def add(a, b): return a+b",
    }])
    report = QAReport(run_id=task.run_id, revision_count=0, verdict=QAVerdict.PASS,
                      manifest_path="manifest.json", checks=[{
                          "name": "unit tests", "check_type": "unit_test", "status": "passed", "exit_code": 0,
                      }])
    calls = {
        "research": lambda: GroqResearchAgent(client=client).generate(task, None, []),
        "planning": lambda: GroqDeveloperAgent(client=client).plan(spec),
        "file": lambda: GroqDeveloperAgent(client=client).generate_file(planned, plan, spec),
        "requirements": lambda: GroqQAAgent(client=client).assess_requirements(task, spec, manifest, implementation, []),
        "feedback": lambda: GroqQAAgent(client=client).generate(spec, manifest, [], 0),
        "release": lambda: GroqReleaseAgent(client=client).generate(task, manifest, report),
    }
    with pytest.raises((ResearchGenerationError, DevelopmentGenerationError, QAGenerationError, ReleaseGenerationError)) as caught:
        calls[operation]()
    assert caught.value.__cause__ is error
    assert request.call_count == 3
    assert sleeper.call_count == 2
    waits = [event for event in events if event["event"] == "groq_rate_limit_wait_started"]
    assert len(waits) == 2
    assert all(event["stage"] == stage for event in waits)


def test_413_oversized_request_stops_after_two_retries(runtime):
    sleeper, events = runtime
    error = oversized_413()
    request = Mock(side_effect=error)
    with pytest.raises(APIStatusError) as caught:
        groq_rate_limit.call_groq("developer", request, model="test-model")
    assert caught.value is error
    assert request.call_args_list == [call(model="test-model")] * 3
    assert sleeper.call_args_list == [call(7), call(12)]
    assert events[-1]["event"] == "groq_rate_limit_exhausted"
    assert events[-1]["data"] == {
        "tpm_limit": 8000, "requested_tokens": 8406, "retries": 2, "oversized": True,
    }
    assert "org_private" not in str(events)
    assert groq_rate_limit.call_groq("research", lambda: "next") == "next"


def test_413_recovery_respects_server_retry_hint(runtime):
    sleeper, events = runtime
    request = Mock(side_effect=[oversized_413({"retry-after": "3"}), "ok"])
    assert groq_rate_limit.call_groq("developer", request) == "ok"
    sleeper.assert_called_once_with(5)
    assert events[0]["data"]["max_retries"] == 2
    assert events[-1]["event"] == "groq_rate_limit_recovered"


@pytest.mark.parametrize("status_code,message", [
    (413, "Request body too large"),
    (413, "tokens per minute (TPM): Limit 8000, Requested 8000."),
    (413, "tokens per minute (TPM): Limit 8000, Requested 7000."),
    (400, "tokens per minute (TPM): Limit 8000, Requested 8406."),
])
def test_unrelated_status_errors_are_not_retried(runtime, status_code, message):
    sleeper, events = runtime
    error = APIStatusError(message, response=httpx.Response(
        status_code, request=httpx.Request("POST", "https://api.groq.com/test"),
    ), body=None)
    request = Mock(side_effect=error)
    with pytest.raises(APIStatusError) as caught:
        groq_rate_limit.call_groq("developer", request)
    assert caught.value is error
    request.assert_called_once()
    sleeper.assert_not_called()
    assert events == []


@pytest.mark.parametrize("module_name,agent_name", [
    ("research_agent", "GroqResearchAgent"), ("developer_agent", "GroqDeveloperAgent"),
    ("qa_agent", "GroqQAAgent"), ("release_agent", "GroqReleaseAgent"),
])
def test_sdk_internal_retries_are_disabled(monkeypatch, module_name, agent_name):
    import importlib

    module = importlib.import_module(f"harness.agents.{module_name}")
    adapter = importlib.import_module("harness.llm.groq")
    constructor = Mock()
    monkeypatch.setattr(adapter, "Groq", constructor)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    getattr(module, agent_name)()
    assert constructor.call_args.kwargs["max_retries"] == 0
