"""Keep offline mocks independent of the developer's selected live provider."""

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_offline_provider(monkeypatch):
    if os.getenv("RUN_LIVE_GROQ") == "1":
        return
    # Existing SDK fixtures model Groq. Tests for other providers explicitly
    # select their adapter and use fake HTTP transports.
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    for name in ("LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL",
                 "LLM_MAX_OUTPUT_TOKENS", "LLM_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
