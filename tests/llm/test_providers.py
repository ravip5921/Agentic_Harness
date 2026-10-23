"""Provider contracts exercised with mocked SDK/HTTP transports only."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from harness.llm import GenerationError, LLMSettings, create_generator
from harness.llm.groq import GroqGenerator
from harness.llm.http import (
    CompatibleGenerator, GeminiGenerator, ProviderHTTPError, gemini_json_schema,
)
from harness.models import (
    DevelopmentPlan, QAFeedback, QARequirementAssessment, ReleaseAssessment, ResearchAssessment,
)


@pytest.fixture(autouse=True)
def clean_provider_environment(monkeypatch):
    for name in (
        "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MAX_OUTPUT_TOKENS",
        "LLM_TIMEOUT_SECONDS", "GROQ_API_KEY", "GROQ_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def response(body, status=200, headers=None):
    return SimpleNamespace(
        status_code=status, headers=headers or {}, json=lambda: body,
        text=str(body), close=Mock(),
    )


def gemini_response(text="complete", reason="STOP"):
    return response({
        "candidates": [{"content": {"parts": [{"text": "private reasoning", "thought": True},
                                               {"text": text}]}, "finishReason": reason}],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10,
                          "totalTokenCount": 35, "thoughtsTokenCount": 5},
    })


def test_environment_defaults_and_legacy_groq_settings(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("GROQ_MODEL", "legacy-model")
    settings = LLMSettings.from_environment()
    assert (settings.provider, settings.model, settings.api_key) == ("groq", "legacy-model", "groq-secret")
    assert "groq-secret" not in repr(settings)
    monkeypatch.setenv("LLM_MODEL", "override-model")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    settings = LLMSettings.from_environment()
    assert (settings.model, settings.max_output_tokens, settings.timeout) == ("override-model", 2048, 30)


def test_switching_to_gemini_does_not_use_groq_credentials_or_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("GROQ_MODEL", "groq-model")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_generator()
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    generator = create_generator()
    assert isinstance(generator, GeminiGenerator)
    assert generator.settings.api_key == "gemini-secret"
    assert generator.settings.model == "gemini-test"


@pytest.mark.parametrize("name,value", [
    ("LLM_PROVIDER", "unknown"), ("LLM_MAX_OUTPUT_TOKENS", "0"),
    ("LLM_MAX_OUTPUT_TOKENS", "bad"), ("LLM_TIMEOUT_SECONDS", "nan"),
    ("LLM_TIMEOUT_SECONDS", "-1"), ("LLM_TIMEOUT_SECONDS", "infinity"),
])
def test_invalid_configuration_fails_before_any_request(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        create_generator(client=Mock())


def test_compatible_endpoint_requires_model_and_valid_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    with pytest.raises(ValueError, match="LLM_MODEL"):
        LLMSettings.from_environment()
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        LLMSettings.from_environment()
    monkeypatch.setenv("LLM_BASE_URL", "https://user:secret@example.com/v1")
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        LLMSettings.from_environment()
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    assert isinstance(create_generator(client=Mock()), CompatibleGenerator)


@pytest.mark.parametrize("model", [
    ResearchAssessment, DevelopmentPlan, QAFeedback, QARequirementAssessment, ReleaseAssessment,
])
def test_gemini_schema_translation_resolves_all_agent_definitions(model):
    original = model.model_json_schema()
    snapshot = deepcopy(original)
    converted = gemini_json_schema(original)

    def check(node):
        if isinstance(node, dict):
            assert not {"$ref", "$defs", "default", "minLength"} & node.keys()
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(converted)
    assert original == snapshot
    assert converted["type"] == "object"
    assert set(converted["properties"]) == set(original["properties"])


@pytest.mark.parametrize("structured", [False, True])
def test_gemini_translates_requests_and_normalizes_results(structured):
    received = gemini_response()
    transport = Mock(post=Mock(return_value=received))
    settings = LLMSettings("gemini", "models/gemini-test", "secret", max_output_tokens=7000, timeout=12)
    generator = GeminiGenerator(settings, transport)
    result = generator.complete(
        stage="research", schema=ResearchAssessment.model_json_schema() if structured else None,
        messages=[{"role": "system", "content": "System"},
                  {"role": "user", "content": "Task"}],
    )
    assert result.text == "complete"
    assert result.finish_reason == "STOP"
    assert result.usage == {"prompt_tokens": 20, "completion_tokens": 10,
                            "total_tokens": 35, "reasoning_tokens": 5}
    url = transport.post.call_args.args[0]
    assert url.endswith("/models/gemini-test:generateContent")
    assert "secret" not in url
    kwargs = transport.post.call_args.kwargs
    assert kwargs["headers"]["x-goog-api-key"] == "secret"
    assert kwargs["timeout"] == 12 and kwargs["allow_redirects"] is False
    body = kwargs["json"]
    assert body["systemInstruction"] == {"parts": [{"text": "System"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Task"}]}]
    assert body["generationConfig"]["maxOutputTokens"] == 7000
    if structured:
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["generationConfig"]["responseJsonSchema"]["properties"]["specification"]["anyOf"][1] == {"type": "null"}
    else:
        assert "responseMimeType" not in body["generationConfig"]
        assert "responseJsonSchema" not in body["generationConfig"]
    received.close.assert_called_once()


@pytest.mark.parametrize("provider,reason", [
    ("gemini", "MAX_TOKENS"), ("gemini", "SAFETY"),
    ("groq", "length"), ("groq", "content_filter"),
    ("openai-compatible", "length"),
])
def test_incomplete_or_blocked_output_is_rejected(provider, reason):
    settings = LLMSettings(provider, "test-model", "secret", "https://example.com/v1")
    if provider == "gemini":
        adapter = GeminiGenerator(settings, Mock(post=Mock(return_value=gemini_response(reason=reason))))
    elif provider == "groq":
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="partial file"), finish_reason=reason,
            )]),
        ))))
        adapter = GroqGenerator(settings, client)
    else:
        adapter = CompatibleGenerator(settings, Mock(post=Mock(return_value=response({"choices": [{
            "message": {"content": "partial file"}, "finish_reason": reason,
        }]}))))
    with pytest.raises(GenerationError, match="incomplete or blocked"):
        adapter.complete(stage="developer", messages=[{"role": "user", "content": "Write code"}])


def test_compatible_request_has_strict_schema_and_custom_endpoint():
    transport = Mock(post=Mock(return_value=response({
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    })))
    adapter = CompatibleGenerator(LLMSettings("openai-compatible", "custom", "secret", "https://example.com/v1/"), transport)
    result = adapter.complete(stage="qa", messages=[], schema=QAFeedback.model_json_schema(), schema_name="feedback")
    assert result.text == "{}" and result.usage["total_tokens"] == 15
    assert transport.post.call_args.args == ("https://example.com/v1/chat/completions",)
    kwargs = transport.post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["json"]["model"] == "custom"
    assert kwargs["json"]["response_format"]["json_schema"]["strict"] is True


def test_gemini_native_retry_delay_is_honored_without_real_waits(monkeypatch):
    sleeper = Mock()
    monkeypatch.setattr("harness.llm.retry.sleep", sleeper)
    transport = Mock(post=Mock(side_effect=[response({"error": {
        "message": "Quota exceeded", "details": [{"retryDelay": "3s"}],
    }}, 429), gemini_response()]))
    adapter = GeminiGenerator(LLMSettings("gemini", "model", "secret"), transport)
    assert adapter.complete(stage="research", messages=[]).text == "complete"
    sleeper.assert_called_once_with(5)
    assert transport.post.call_count == 2


@pytest.mark.parametrize("provider,status,requested,attempts", [
    ("gemini", 429, 1, 6), ("gemini", 413, 8406, 3),
    ("openai-compatible", 413, 8406, 3), ("gemini", 413, 8000, 1),
    ("gemini", 401, 8406, 1), ("gemini", 400, 8406, 1),
    ("gemini", 302, 8406, 1),
])
def test_http_errors_have_bounded_retries_and_sanitized_messages(
    monkeypatch, provider, status, requested, attempts,
):
    sleeper = Mock()
    events = []
    monkeypatch.setattr("harness.llm.retry.sleep", sleeper)
    monkeypatch.setattr("harness.events.get_stream_writer", lambda: events.append)
    received = response({"error": {"message": f"secret tokens per minute (TPM): Limit 8000, Requested {requested}."}}, status)
    transport = Mock(post=Mock(return_value=received))
    settings = LLMSettings(provider, "model", "secret", "https://example.com/v1")
    adapter = GeminiGenerator(settings, transport) if provider == "gemini" else CompatibleGenerator(settings, transport)
    with pytest.raises(ProviderHTTPError) as caught:
        adapter.complete(stage="developer", messages=[])
    assert "secret" not in str(caught.value)
    assert "secret" not in str(events)
    assert transport.post.call_count == attempts
    assert sleeper.call_count == attempts - 1
    assert received.close.call_count == attempts
    if attempts > 1:
        assert events[-2]["event"] == "llm_rate_limit_exhausted"
    assert events[-1]["event"] == "llm_request_failed"


@pytest.mark.parametrize("body", [{}, {"candidates": []}, {"candidates": [{"finishReason": "STOP"}]}])
def test_missing_gemini_content_fails_gracefully(body):
    adapter = GeminiGenerator(LLMSettings("gemini", "model", "secret"), Mock(post=Mock(return_value=response(body))))
    with pytest.raises(GenerationError):
        adapter.complete(stage="developer", messages=[])
