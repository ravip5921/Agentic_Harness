"""Best-effort progress events, independent of workflow state and presentation."""

from typing import Literal, TypedDict

from langgraph.config import get_stream_writer


EventLevel = Literal["start", "info", "success", "warning", "failure"]


class ProgressEvent(TypedDict):
    stage: str
    event: str
    message: str
    data: dict
    level: EventLevel


def emit_event(
    stage: str,
    event: str,
    message: str,
    *,
    data: dict | None = None,
    level: EventLevel = "info",
) -> None:
    """Send a small custom event when running inside LangGraph.

    Direct agent/stage calls have no stream writer. Missing context and
    writer failures are deliberately ignored: progress cannot affect results.
    Never include prompts, generated code, or raw exception text in events.
    """
    try:
        payload: ProgressEvent = {
            "stage": stage,
            "event": event,
            "message": message,
            "data": dict(data) if data is not None else {},
            "level": level,
        }
        get_stream_writer()(payload)
    except Exception:
        pass
