"""Compatibility entry point for Groq requests using the shared retry policy."""

from groq import APIStatusError

from harness.llm.retry import RetryInfo, call_with_retry


def call_groq(stage, request, **kwargs):
    def classify(error):
        if isinstance(error, APIStatusError):
            return RetryInfo(error.status_code, str(error), error.response.headers)
        return None

    return call_with_retry(stage, "groq", request, classify, **kwargs)
