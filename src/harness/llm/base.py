"""Provider-independent text generation contracts."""

from dataclasses import dataclass, field
from typing import Protocol

from harness.events import emit_event
from harness.llm.diagnostics import record_completion


class GenerationError(Exception):
    """A provider could not return a complete usable response."""


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class TextGenerator(Protocol):
    def complete(
        self, *, stage: str, messages: list[dict[str, str]],
        schema: dict | None = None, schema_name: str = "response",
    ) -> Completion: ...


def check_completion(stage: str, provider: str, completion: Completion) -> Completion:
    """Reject interrupted output before an agent can accept or stage it."""
    record_completion(completion)
    if completion.finish_reason not in {None, "stop", "STOP"}:
        emit_event(
            stage, "llm_generation_rejected", "Model response did not finish normally",
            data={"provider": provider, "finish_reason": completion.finish_reason,
                  **completion.usage}, level="failure",
        )
        raise GenerationError(
            f"{provider} response was incomplete or blocked "
            f"(finish_reason={completion.finish_reason}). "
            "Review the output-token limit and provider response status."
        )
    if not completion.text.strip():
        raise GenerationError(f"{provider} returned an empty response.")
    return completion
