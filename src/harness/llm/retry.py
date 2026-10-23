"""Shared bounded retries and a process-wide gate for all model providers."""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from threading import Lock
from time import sleep, time
from typing import Mapping, TypeVar

from harness.events import emit_event


@dataclass(frozen=True)
class RetryInfo:
    status_code: int
    message: str
    headers: Mapping[str, str]


RETRY_PADDING_SECONDS = 2.0
MAX_RATE_LIMIT_RETRIES = 5
MAX_OVERSIZED_RETRIES = 2
_REQUEST_LOCK = Lock()
_Response = TypeVar("_Response")
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|[smhd])", re.IGNORECASE)


def _nonnegative(value) -> float | None:
    try:
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    except (TypeError, ValueError):
        return None


def _retry_delay(error: RetryInfo, retries: int) -> float:
    """Respect the longest valid server hint, with bounded fallback backoff."""
    delays = []
    headers = error.headers
    milliseconds = _nonnegative(headers.get("retry-after-ms"))
    if milliseconds is not None:
        delays.append(milliseconds / 1000)
    retry_after = headers.get("retry-after")
    seconds = _nonnegative(retry_after)
    if seconds is None and retry_after:
        try:
            seconds = max(0, parsedate_to_datetime(retry_after).timestamp() - time())
        except (ValueError, TypeError, OverflowError):
            pass
    if seconds is not None:
        delays.append(seconds)

    hint = re.search(
        r"try again in\s+((?:\d+(?:\.\d+)?\s*(?:ms|[smhd])\s*)+)",
        error.message, re.IGNORECASE,
    )
    if hint:
        units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}
        duration = sum(float(number) * units[unit.lower()]
                       for number, unit in _DURATION_PART.findall(hint.group(1)))
        if math.isfinite(duration):
            delays.append(duration)

    return (max(delays) if delays else min(5 * 2 ** retries, 60)) + RETRY_PADDING_SECONDS


def _token_budget(error: RetryInfo) -> dict:
    """Extract TPM counts only; never expose the raw provider error in progress."""
    match = re.search(
        r"\bTPM\b.*?\bLimit\s+(\d[\d,]*).*?\bRequested\s+(\d[\d,]*)",
        error.message, re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return {}
    return {"tpm_limit": int(match[1].replace(",", "")),
            "requested_tokens": int(match[2].replace(",", ""))}


def call_with_retry(
    stage: str, provider: str, request: Callable[..., _Response],
    classify: Callable[[Exception], RetryInfo | None], **kwargs,
) -> _Response:
    """Retry only rate limits and confirmed oversized TPM errors, preserving causes."""
    prefix = "groq" if provider == "groq" else "llm"
    label = "Groq" if provider == "groq" else provider
    retries = oversized_retries = 0
    with _REQUEST_LOCK:
        while True:
            try:
                response = request(**kwargs)
            except Exception as error:
                info = classify(error)
                if info is None:
                    raise
                budget = _token_budget(info)
                oversized = bool(budget and budget["requested_tokens"] > budget["tpm_limit"])
                if info.status_code != 429 and not (info.status_code == 413 and oversized):
                    raise
                retry_limit = MAX_OVERSIZED_RETRIES if oversized else MAX_RATE_LIMIT_RETRIES
                retry_number = oversized_retries if oversized else retries
                reason = (
                    f"Requested {budget['requested_tokens']} tokens exceeds the TPM limit {budget['tpm_limit']}"
                    if oversized else f"{label} rate limit (429)"
                )
                if retries >= MAX_RATE_LIMIT_RETRIES or (oversized and oversized_retries >= MAX_OVERSIZED_RETRIES):
                    emit_event(
                        stage, f"{prefix}_rate_limit_exhausted",
                        f"{reason}; stopping after {retries} retries",
                        data={**budget, "retries": retries, "oversized": oversized}, level="failure",
                    )
                    raise

                seconds = _retry_delay(info, retries)
                retries += 1
                if oversized:
                    oversized_retries += 1
                data = {**({"provider": provider} if provider != "groq" else {}), **budget, "seconds": seconds, "retry": retry_number + 1,
                        "max_retries": retry_limit, "oversized": oversized}
                emit_event(
                    stage, f"{prefix}_rate_limit_wait_started",
                    f"{reason}; waiting {seconds:g}s before retry {retry_number + 1}/{retry_limit}",
                    data=data, level="start",
                )
                sleep(seconds)
                emit_event(
                    stage, f"{prefix}_rate_limit_retrying",
                    f"Retrying {label} request ({retry_number + 1}/{retry_limit})",
                    data=data, level="start",
                )
            else:
                if retries:
                    emit_event(
                        stage, f"{prefix}_rate_limit_recovered", f"{label} rate limit cleared",
                        data={"retries": retries},
                    )
                return response
