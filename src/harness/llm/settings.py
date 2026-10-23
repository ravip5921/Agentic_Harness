"""Select a provider without changing agent implementations."""

import math
import os
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str = field(repr=False)
    base_url: str | None = None
    max_output_tokens: int = 4096
    timeout: float = 90.0

    @classmethod
    def from_environment(cls, *, provider: str | None = None, model: str | None = None):
        provider = (provider or os.getenv("LLM_PROVIDER", "groq")).strip().lower()
        defaults = {
            "groq": ("GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-20b"),
            "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.8-flash"),
            "openai-compatible": ("LLM_API_KEY", "LLM_MODEL", ""),
        }
        if provider not in defaults:
            raise ValueError(
                f"Unknown LLM_PROVIDER {provider!r}; choose groq, gemini, or openai-compatible."
            )
        key_var, model_var, default_model = defaults[provider]
        selected_model = model or os.getenv("LLM_MODEL") or os.getenv(model_var) or default_model
        if not selected_model.strip():
            raise ValueError(f"LLM_MODEL is required for {provider}.")
        api_key = os.getenv("LLM_API_KEY") or os.getenv(key_var, "")
        base_url = os.getenv("LLM_BASE_URL") if provider == "openai-compatible" else None
        if provider == "openai-compatible":
            parsed = urlparse(base_url or "")
            if (parsed.scheme not in {"https", "http"} or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError("LLM_BASE_URL must be an HTTP(S) API base URL without credentials or query parameters.")
        try:
            tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))
            timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
        except ValueError as exc:
            raise ValueError("LLM token limit and timeout must be numeric.") from exc
        if tokens <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("LLM token limit and timeout must be positive finite values.")
        return cls(provider, selected_model, api_key, base_url, tokens, timeout)

    def require_key(self):
        if not self.api_key.strip():
            key_var = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
                       "openai-compatible": "LLM_API_KEY"}[self.provider]
            raise ValueError(f"Set {key_var} or LLM_API_KEY for {self.provider}.")
