"""Native Gemini REST and Chat Completions-compatible provider adapters.

Uses the existing requests dependency. No SDK-specific types reach agents.
"""

from urllib.parse import quote

import requests

from harness.llm.base import Completion, GenerationError, check_completion
from harness.llm.diagnostics import diagnose_generation
from harness.llm.retry import RetryInfo, call_with_retry
from harness.tools.structured_output import strict_response_format


class ProviderHTTPError(GenerationError):
    def __init__(self, provider, status_code, message, headers):
        # Raw provider text is used only for retry classification, never displayed.
        super().__init__(f"{provider} request failed (HTTP {status_code}).")
        self.retry_info = RetryInfo(status_code, message, headers)


def gemini_json_schema(schema: dict) -> dict:
    """Inline Pydantic references and use Gemini's documented schema subset.

    Application validators retain constraints unsupported by Gemini's decoder.
    """
    definitions = schema.get("$defs", {})
    allowed = {
        "type", "title", "description", "properties", "required", "additionalProperties",
        "items", "prefixItems", "enum", "format", "minimum", "maximum", "minItems",
        "maxItems", "anyOf",
    }

    def convert(node, seen=frozenset()):
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = node["$ref"]
            name = ref.removeprefix("#/$defs/")
            if name not in definitions or ref in seen:
                raise ValueError("Gemini adapter requires a nonrecursive local JSON schema.")
            return convert({**definitions[name], **{k: v for k, v in node.items() if k != "$ref"}}, seen | {ref})
        result = {}
        for key, value in node.items():
            if key not in allowed:
                continue
            if key == "properties":
                result[key] = {name: convert(prop, seen) for name, prop in value.items()}
            elif isinstance(value, dict):
                result[key] = convert(value, seen)
            elif isinstance(value, list) and key in {"anyOf", "prefixItems"}:
                result[key] = [convert(item, seen) for item in value]
            else:
                result[key] = value
        return result

    return convert(schema)


class _HTTPGenerator:
    def __init__(self, settings, client=None):
        self.settings = settings
        if client is None:
            settings.require_key()
        # requests has no automatic status retries by default.
        self.client = client if client is not None else requests

    def _post(self, stage, url, headers, payload):
        def request():
            response = self.client.post(
                url, headers=headers, json=payload, timeout=self.settings.timeout,
                allow_redirects=False,
            )
            try:
                if not 200 <= response.status_code < 300:
                    # Redirects are errors too; never forward credentials elsewhere.
                    try:
                        message = response.text
                        body = response.json()
                        for detail in body.get("error", {}).get("details", []):
                            if "retryDelay" in detail:
                                message += f" Please try again in {detail['retryDelay']}."
                    except (ValueError, AttributeError, TypeError):
                        message = response.text
                    raise ProviderHTTPError(
                        self.settings.provider, response.status_code, message, response.headers,
                    )
                return response.json()
            finally:
                response.close()

        def classify(error):
            return error.retry_info if isinstance(error, ProviderHTTPError) else None

        return call_with_retry(stage, self.settings.provider, request, classify)


class GeminiGenerator(_HTTPGenerator):
    @diagnose_generation
    def complete(self, *, stage, messages, schema=None, schema_name="response"):
        system = []
        contents = []
        for message in messages:
            if message["role"] == "system":
                system.append({"text": message["content"]})
            elif message["role"] in {"user", "assistant"}:
                contents.append({
                    "role": "model" if message["role"] == "assistant" else "user",
                    "parts": [{"text": message["content"]}],
                })
            else:
                raise ValueError("Gemini text adapter supports system, user, and assistant messages.")
        config = {"temperature": 0, "maxOutputTokens": self.settings.max_output_tokens}
        if schema is not None:
            config.update(responseMimeType="application/json", responseJsonSchema=gemini_json_schema(schema))
        payload = {"contents": contents, "generationConfig": config}
        if system:
            payload["systemInstruction"] = {"parts": system}
        model = quote(self.settings.model.removeprefix("models/"), safe="")
        body = self._post(
            stage, f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": self.settings.api_key, "Content-Type": "application/json"}, payload,
        )
        candidates = body.get("candidates", [])
        if not candidates:
            raise GenerationError("gemini returned no candidates; the request may have been blocked.")
        candidate = candidates[0]
        text = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
                       if not part.get("thought", False))
        usage = body.get("usageMetadata", {})
        counts = {name: usage[key] for name, key in (
            ("prompt_tokens", "promptTokenCount"), ("completion_tokens", "candidatesTokenCount"),
            ("total_tokens", "totalTokenCount"), ("reasoning_tokens", "thoughtsTokenCount"),
        ) if isinstance(usage.get(key), int)}
        return check_completion(stage, "gemini", Completion(text, candidate.get("finishReason"), counts))


class CompatibleGenerator(_HTTPGenerator):
    @diagnose_generation
    def complete(self, *, stage, messages, schema=None, schema_name="response"):
        payload = {
            "model": self.settings.model, "messages": messages, "temperature": 0,
            "max_completion_tokens": self.settings.max_output_tokens,
        }
        if schema is not None:
            payload["response_format"] = strict_response_format(schema_name, schema)
        body = self._post(
            stage, self.settings.base_url.rstrip("/") + "/chat/completions",
            {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"},
            payload,
        )
        choices = body.get("choices", [])
        if not choices:
            raise GenerationError("openai-compatible provider returned no choices.")
        choice = choices[0]
        usage = {key: value for key, value in body.get("usage", {}).items()
                 if key in {"prompt_tokens", "completion_tokens", "total_tokens"} and isinstance(value, int)}
        reasoning = body.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens")
        if isinstance(reasoning, int):
            usage["reasoning_tokens"] = reasoning
        return check_completion(stage, "openai-compatible", Completion(
            choice.get("message", {}).get("content") or "", choice.get("finish_reason"), usage,
        ))
