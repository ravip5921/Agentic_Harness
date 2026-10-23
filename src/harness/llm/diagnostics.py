"""Generation metadata for investigating truncation without logging content."""

import ast
import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import monotonic
from uuid import uuid4

from harness.events import emit_event


_log_path = ContextVar("llm_diagnostics_path", default=None)
_request = ContextVar("llm_diagnostics_request", default=None)


@contextmanager
def diagnostics_to(path: Path):
    """Keep diagnostic artifacts outside staging so failures retain the evidence."""
    token = _log_path.set(path)
    try:
        yield
    finally:
        _log_path.reset(token)


def _record(stage, phase, data, message, level="info"):
    emit_event(stage, f"llm_{phase}", message, data=data, level=level)
    path = _log_path.get()
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "stage": stage, "event": phase, **data,
                }) + "\n")
        except OSError:
            emit_event(stage, "llm_diagnostics_write_failed", "Could not save generation diagnostics",
                       level="warning")


def diagnose_generation(method):
    """Instrument one logical provider request, including responses later rejected."""
    @wraps(method)
    def wrapped(self, *, stage, messages, schema=None, schema_name="response"):
        settings = self.settings
        path = None
        if schema is None:
            for message in reversed(messages):
                if message["role"] == "user":
                    try:
                        context = json.loads(message["content"])
                        candidate = context.get("current_file", {}).get("path")
                        if isinstance(candidate, str):
                            path = candidate
                    except (ValueError, AttributeError, TypeError):
                        pass
                    break
        data = {
            "request_id": uuid4().hex[:12], "provider": settings.provider,
            "model": settings.model, "operation": schema_name if schema is not None else "file_generation",
            "path": path, "max_output_tokens": settings.max_output_tokens,
            "timeout_seconds": settings.timeout,
            "input_characters": sum(len(message["content"]) for message in messages),
            "response_format": "json_schema" if schema is not None else "text",
        }
        label = path or data["operation"]
        _record(stage, "request_started", data,
                f"LLM request {data['request_id']}: {settings.provider}/{settings.model}; "
                f"{label}; format={data['response_format']}; output limit={settings.max_output_tokens} tokens; "
                f"timeout={settings.timeout:g}s; input={data['input_characters']} characters", "start")
        started = monotonic()
        token = _request.set((stage, data, started))
        try:
            return method(self, stage=stage, messages=messages, schema=schema, schema_name=schema_name)
        except Exception as exc:
            _record(stage, "request_failed", {
                **data, "error_type": type(exc).__name__,
                "elapsed_seconds": round(monotonic() - started, 3),
            }, f"LLM request {data['request_id']} failed for {label} ({type(exc).__name__})", "failure")
            raise
        finally:
            _request.reset(token)

    return wrapped


def record_completion(completion):
    """Called before completion validation so truncated output is always recorded."""
    current = _request.get()
    if current is None:
        return
    stage, request, started = current
    reason = completion.finish_reason
    diagnosis = (
        "output_limit_reached" if reason in {"length", "MAX_TOKENS"}
        else "normal_stop_not_completeness_proof" if reason in {"stop", "STOP"}
        else "finish_reason_unreported" if reason is None
        else "non_normal_finish"
    )
    data = {
        **request, "finish_reason": reason, "usage": completion.usage,
        "output_characters": len(completion.text),
        "output_bytes": len(completion.text.encode("utf-8")),
        "output_lines": len(completion.text.splitlines()),
        "elapsed_seconds": round(monotonic() - started, 3), "diagnosis": diagnosis,
    }
    syntax = ""
    if request["path"] and request["path"].endswith(".py"):
        try:
            ast.parse(completion.text)
            data["python_syntax_valid"] = True
            syntax = "; Python syntax=valid"
        except (SyntaxError, ValueError, RecursionError) as exc:
            data.update(python_syntax_valid=False, syntax_error_type=type(exc).__name__,
                        syntax_error_line=getattr(exc, "lineno", None))
            syntax = f"; Python syntax=invalid (line {data['syntax_error_line']})"
    usage = completion.usage
    counts = ", ".join(f"{name}={usage.get(name + '_tokens', 'unreported')}"
                       for name in ("prompt", "completion", "reasoning", "total"))
    _record(stage, "response_received", data,
            f"LLM response {request['request_id']}: {request['path'] or request['operation']}; "
            f"finish={reason or 'unreported'}; {counts}; "
            f"{data['output_characters']} characters/{data['output_bytes']} bytes/{data['output_lines']} lines; "
            f"{data['elapsed_seconds']:g}s; {diagnosis}{syntax}",
            "warning" if diagnosis != "normal_stop_not_completeness_proof" or data.get("python_syntax_valid") is False else "info")
