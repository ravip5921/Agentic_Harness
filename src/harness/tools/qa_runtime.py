"""Check Docker access and provision the harness-owned QA image."""

import subprocess
import tempfile
from pathlib import Path

from harness.events import emit_event
from harness.tools.executor import DEFAULT_IMAGE, QAExecutionError


class QARuntimeError(QAExecutionError):
    """Docker or the trusted QA image is unavailable."""


def prepare_qa_runtime(*, image=DEFAULT_IMAGE, runner=subprocess.run):
    """Fail fast on inaccessible Docker; build a missing default image.

    Builds use only the harness Dockerfile in an empty context, never the
    generated workspace. Custom images must be provisioned by their owner.
    """
    if not isinstance(image, str) or not image or image.startswith("-"):
        raise QARuntimeError("Invalid QA container image.")

    def run(command, description, *, timeout=30, **kwargs):
        try:
            return runner(command, capture_output=True, text=True,
                          check=False, timeout=timeout, **kwargs)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise QARuntimeError(f"{description}: {exc}") from exc

    def require_success(result, description):
        if result.returncode:
            detail = (result.stderr or result.stdout or "No Docker diagnostics.").strip()
            raise QARuntimeError(
                f"{description} (exit code {result.returncode}): {detail[-4000:]}"
            )

    access_error = (
        "Docker access is unavailable. Start Docker Desktop/the Docker daemon "
        "and check the active Docker context and permissions"
    )
    require_success(run(["docker", "info"], access_error), access_error)
    inspected = run(["docker", "image", "inspect", image], "QA image inspection failed")
    if inspected.returncode == 0:
        return

    # A failed inspect can also mean the daemon disappeared. Do not mistake
    # arbitrary inspect failures for a missing image.
    require_success(run(["docker", "info"], access_error), access_error)
    detail = (inspected.stderr or inspected.stdout or "").lower()
    if "no such image" not in detail and "no such object" not in detail:
        require_success(inspected, "QA image inspection failed")
    if image != DEFAULT_IMAGE:
        raise QARuntimeError(f"QA image {image!r} is missing; build or pull this custom image.")

    dockerfile = Path(__file__).resolve().parents[3] / "docker" / "qa-python.Dockerfile"
    if not dockerfile.is_file():
        raise QARuntimeError(
            f"QA image {image!r} is missing and the trusted Dockerfile is unavailable "
            f"at {dockerfile}. Build the QA image from the harness source checkout."
        )
    emit_event("qa", "image_build_started", f"Building missing QA image: {image}", level="start")
    with tempfile.TemporaryDirectory(prefix="harness-qa-image-") as context:
        result = run(
            ["docker", "build", "--file", "-", "--tag", image, context],
            "QA image build failed", timeout=600,
            input=dockerfile.read_text(encoding="utf-8"),
        )
    require_success(result, "QA image build failed")
    require_success(
        run(["docker", "image", "inspect", image], "Built QA image verification failed"),
        "Built QA image verification failed",
    )
    emit_event("qa", "image_build_completed", f"QA image ready: {image}", level="success")
