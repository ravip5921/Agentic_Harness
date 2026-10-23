from io import StringIO
from unittest.mock import Mock

import pytest
from rich.console import Console

from harness.models import QAResult, QAVerdict, RunStatus, TaskRequest
from harness.tools.progress import ProgressRenderer


def test_plain_rendering_events_verdicts_retries_and_final_status():
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    with ProgressRenderer(console) as renderer:
        renderer.event({"stage": "developer", "event": "file_generation_started",
                        "message": "Generating client.py", "level": "start", "data": {}})
        renderer.event({"stage": "developer", "event": "file_generation_completed",
                        "message": "Generated client.py", "level": "success", "data": {}})
        renderer.update({"qa": {"qa_result": QAResult(verdict=QAVerdict.FAIL)}}, {})
        renderer.update({"retry": {"revision_count": 1}}, {"task": TaskRequest(task="Test")})
        renderer.event({"stage": "qa", "event": "assessment_unavailable",
                        "message": "Requirements assessment unavailable", "level": "warning", "data": {}})
        renderer.finish(RunStatus.BLOCKED)

    text = output.getvalue()
    for expected in ("● DEVELOPER", "→ Generating client.py", "✓ Generated client.py",
                     "✗ QA FAIL", "↻ Retrying Developer (revision 1/2)",
                     "! Requirements assessment unavailable", "Workflow completed: BLOCKED"):
        assert expected in text
    assert "\x1b" not in text


@pytest.mark.parametrize("level,style", [("success", "32"), ("warning", "33"), ("failure", "31")])
def test_terminal_colors(level, style):
    output = StringIO()
    renderer = ProgressRenderer(Console(file=output, force_terminal=True, no_color=False,
                                         color_system="standard"))
    renderer.event({"stage": "qa", "event": "check_completed", "message": "Actual result",
                    "data": {}, "level": level})
    assert f"\x1b[{style}m" in output.getvalue()


@pytest.mark.parametrize("level", ["success", "warning", "failure"])
def test_current_operation_is_visible_until_completion_and_cleaned_up(level):
    console = Mock(is_interactive=True)
    status = console.status.return_value
    operations = []
    console.print.side_effect = lambda line: operations.append(("print", str(line)))
    status.stop.side_effect = lambda: operations.append(("stop",))
    with ProgressRenderer(console) as renderer:
        renderer.event({"stage": "research", "event": "specification_started",
                        "message": "Generating specification", "level": "start", "data": {}})
        status.start.assert_called_once()
        status.stop.assert_not_called()
        assert not any("Generating specification" in str(operation) for operation in operations)
        assert "Generating specification" in str(console.status.call_args.args[0])
        renderer.event({"stage": "research", "event": "specification_completed",
                        "message": "Specification result", "level": level, "data": {}})
        status.stop.assert_called_once()
        assert operations[1:3] == [
            ("stop",), ("print", "  → Generating specification"),
        ]
        assert "Specification result" in operations[3][1]
    assert sum(operation == ("print", "  → Generating specification")
               for operation in operations) == 1


def test_next_operation_flushes_previous_line_before_starting_spinner():
    console = Mock(is_interactive=True)
    operations = []
    console.print.side_effect = lambda line: operations.append(("print", str(line)))
    console.status.return_value.start.side_effect = lambda: operations.append(("start",))
    console.status.return_value.stop.side_effect = lambda: operations.append(("stop",))
    with ProgressRenderer(console) as renderer:
        for message in ("Generating implementation", "Planning implementation"):
            renderer.event({"stage": "developer", "event": "operation_started",
                            "message": message, "level": "start", "data": {}})
        assert operations[1:] == [
            ("start",), ("stop",), ("print", "  → Generating implementation"), ("start",),
        ]
    assert operations[-2:] == [("stop",), ("print", "  → Planning implementation")]


def test_runtime_text_is_literal_and_terminal_controls_are_removed():
    output = StringIO()
    renderer = ProgressRenderer(Console(file=output, force_terminal=False, width=120))
    renderer.event({"stage": "developer", "event": "file_generation_started", "data": {},
                    "message": "Generating [red]file.py\x1b[31m\n\x00", "level": "start"})
    assert "Generating [red]file.py" in output.getvalue()
    assert "\x1b" not in output.getvalue()
    assert "\x00" not in output.getvalue()


def test_renderer_failure_does_not_escape_or_prevent_cleanup():
    console = Mock(is_interactive=True)
    renderer = ProgressRenderer(console)
    renderer.event({"stage": "qa", "event": "check_started", "message": "Executing check",
                    "level": "start", "data": {}})
    console.print.side_effect = OSError("output unavailable")
    renderer.finish(RunStatus.STAGED)
    renderer.__exit__(None, None, None)
    console.status.return_value.stop.assert_called_once()
