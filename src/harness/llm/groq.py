"""Groq SDK adapter; provider-specific request details stay here."""

from groq import Groq

from harness.llm.base import Completion, check_completion
from harness.llm.diagnostics import diagnose_generation
from harness.llm.settings import LLMSettings
from harness.tools.groq_rate_limit import call_groq
from harness.tools.structured_output import strict_response_format


class GroqGenerator:
    def __init__(self, settings: LLMSettings, client=None):
        self.settings = settings
        if client is None:
            settings.require_key()
            client = Groq(api_key=settings.api_key, timeout=settings.timeout, max_retries=0)
        self.client = client

    @diagnose_generation
    def complete(self, *, stage, messages, schema=None, schema_name="response"):
        kwargs = dict(model=self.settings.model, temperature=0, messages=messages,
                      max_completion_tokens=self.settings.max_output_tokens)
        if schema is not None:
            kwargs["response_format"] = strict_response_format(schema_name, schema)
        response = call_groq(stage, self.client.chat.completions.create, **kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        counts = {name: value for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                  if isinstance(value := getattr(usage, name, None), int)}
        reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
        if isinstance(reasoning, int):
            counts["reasoning_tokens"] = reasoning
        return check_completion(stage, "groq", Completion(
            choice.message.content or "", getattr(choice, "finish_reason", None), counts,
        ))
