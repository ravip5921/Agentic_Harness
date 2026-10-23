"""Rich terminal presentation for informational events and graph updates."""

import re
from math import ceil
from time import monotonic
from collections.abc import Callable

from rich.console import Console
from rich.text import Text

from harness.events import ProgressEvent


# Treat filenames and other runtime text as text, never terminal commands or markup.
_ANSI = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROLS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _text(value: str) -> str:
    return " ".join(_CONTROLS.sub(" ", _ANSI.sub("", value)).split())[:240]


class _RetryCountdown:
    """Rich refreshes the remaining time without extra stream events or lines."""

    def __init__(self, seconds: float):
        self.deadline = monotonic() + seconds

    def __rich__(self) -> Text:
        remaining = max(0, ceil(self.deadline - monotonic()))
        return Text(f"  Groq rate limit: {remaining}s remaining", style="yellow")


class ProgressRenderer:
    """One current-operation indicator on terminals; plain lines in pipes.

    Rendering is best effort, including cleanup, so a presentation error cannot
    stop graph execution or change its final status.
    """

    def __init__(self, console: Console | None = None):
        self.console = console if console is not None else Console(highlight=False)
        self._stage = None
        self._status = None
        self._pending_line = None
        self._enabled = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.finish("FAILED")
        self._safe(self._stop)
        return False

    def _safe(self, operation: Callable, *args) -> None:
        if not self._enabled:
            return
        try:
            operation(*args)
        except Exception:
            self._enabled = False
            try:
                self._stop()
            except Exception:
                pass

    def _stop(self) -> None:
        if self._status is not None:
            status, self._status = self._status, None
            pending_line, self._pending_line = self._pending_line, None
            status.stop()
            if pending_line is not None:
                self.console.print(pending_line)

    def _heading(self, stage: str) -> None:
        if stage != self._stage:
            self._stop()
            self.console.print(Text(f"\n● {_text(stage).upper()}", style="bold cyan"))
            self._stage = stage

    def _line(self, message: str, level: str, seconds: float | None = None) -> None:
        self._stop()
        symbol, style = {
            "start": ("→", "cyan"),
            "info": ("•", "default"),
            "success": ("✓", "green"),
            "warning": ("!", "yellow"),
            "failure": ("✗", "red"),
            "retry": ("↻", "yellow"),
            "wait": ("◷", "yellow"),
        }.get(level, ("•", "default"))
        line = Text(f"  {symbol} {_text(message)}", style=style)
        if level in {"start", "wait"} and self.console.is_interactive:
            current = (
                _RetryCountdown(seconds) if seconds is not None
                else Text(f"  {_text(message)}", style=style)
            )
            self._status = self.console.status(current, spinner="dots", spinner_style=style)
            self._pending_line = line
            self._status.start()
        else:
            self.console.print(line)

    def event(self, event: ProgressEvent) -> None:
        self._safe(self._event, event)

    def _event(self, event: ProgressEvent) -> None:
        self._heading(event["stage"])
        if event["event"] == "groq_rate_limit_wait_started":
            self._line(event["message"], "wait", event["data"]["seconds"])
        elif event["event"] == "groq_rate_limit_retrying":
            self._line(event["message"], "wait")
        else:
            self._line(event["message"], event.get("level", "info"))

    def update(self, update: dict, state: dict) -> None:
        self._safe(self._update, update, state)

    def _update(self, update: dict, state: dict) -> None:
        """Report actual node completion; never infer success from a start event."""
        for node, changes in update.items():
            if node == "retry":
                revision = changes["revision_count"]
                maximum = state["task"].max_dev_revisions
                self._line(f"Retrying Developer (revision {revision}/{maximum})", "retry")
                self._stage = None
                continue

            result_fields = {
                "research": ("research_result", "outcome"),
                "developer": ("development_result", "outcome"),
                "qa": ("qa_result", "verdict"),
                "release": ("release_result", "status"),
            }
            if node not in result_fields or not isinstance(changes, dict):
                continue
            key, field = result_fields[node]
            result = changes.get(key)
            if result is None:
                continue
            self._heading(node)
            status = getattr(result, field).value.upper()
            level = (
                "success" if status in {"COMPLETED", "PASS", "STAGED"}
                else "warning" if status == "BLOCKED" else "failure"
            )
            name = "QA" if node == "qa" else node.capitalize()
            self._line(f"{name} {status}", level)

    def finish(self, status) -> None:
        self._safe(self._finish, status)

    def _finish(self, status) -> None:
        status = getattr(status, "value", status).upper()
        level = "success" if status == "STAGED" else "warning" if status == "BLOCKED" else "failure"
        self._stop()
        self.console.print()
        self._line(f"Workflow completed: {status}", level)
